import os
import random
import tarfile
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import mlflow.pytorch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import requests
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint
)
from pytorch_lightning.utilities import rank_zero_info
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from transformers import (
    Adafactor,
    AdamW,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizerFast,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerFast
)
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.optimization import Adafactor

# huggingface/tokenizers: Disabling parallelism to avoid deadlocks.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

IntList = List[int]
IntListList = List[List[int]]
TEXT_COL_NAME: str = "text"
LABEL_COL_NAME: str = "label"

class LABELS(Enum):
    sports_watch = "sports-watch"
    topic_news = "topic-news"
    dokujo_tsushin = "dokujo-tsushin"
    peachy = "peachy"
    movie_enter = "movie-enter"
    kaden_channel = "kaden-channel"
    livedoor_homme = "livedoor-homme"
    smax = "smax"
    it_life_hack = "it-life-hack"


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


@dataclass
class InputExample:
    guid: str
    text: str
    label: Optional[str]


@dataclass
class InputFeatures:
    input_ids: IntList
    attention_mask: IntList
    label_ids: Optional[IntList]


def download_and_extract_corpus(data_dir: Path) -> Optional[Path]:
    """livedoorコーパスデータのダウンロード"""
    filepath = Path("ldcc.tar")
    url = "https://www.rondhuit.com/download/ldcc-20140209.tar.gz"
    response = requests.get(url)
    if response.ok:
        with open(filepath, "wb") as fp:
            fp.write(response.content)
        with tarfile.open(filepath, "r") as fp:
            fp.extractall(data_dir)
        filepath.unlink()
        return data_dir / "text"
    return None


def make_livedoor_corpus_dataset(data_dir: str = "./data") -> pd.DataFrame:
    # ライブドアコーパスを[カテゴリ, 本文]形式でpd.DataFrameで読み込む
    pdir = Path(data_dir)
    if not (pdir / "text").exists():
        pdir.mkdir(exist_ok=True)
        parent_path = download_and_extract_corpus(Path(data_dir))
    else:
        parent_path = pdir / "text"

    categories = [v.value for v in LABELS]
    docs = []
    for category in categories:
        for p in (parent_path / f"{category}").glob(f"{category}*.txt"):
            with open(p, "r") as f:
                next(f)  # url
                next(f)  # date
                next(f)  # title
                body = "\n".join([line.strip() for line in f if line.strip()])
            docs.append((category, body))

    return pd.DataFrame(docs, columns=[LABEL_COL_NAME, TEXT_COL_NAME])


class SequenceClassificationDataset(Dataset):
    def __init__(
        self,
        data: List[InputExample],
        tokenizer: PreTrainedTokenizerFast,
        max_seq_length: int,
        label_to_id: Dict[str, int]
    ):
        """
        Performs initialization of tokenizer
        :param texts: document texts
        :param labels: labels
        :param tokenizer: bert tokenizer
        :param max_seq_length: maximum length of the document text
        """
        self.examples = data
        texts = [ex.text for ex in self.examples]
        labels = [ex.label for ex in self.examples if ex.label]

        self.encodings = [
            tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=max_seq_length,
                return_token_type_ids=False,
                padding="max_length",
                return_attention_mask=True,
                return_tensors="np",
                truncation=True,
            )
            for text in texts
        ]
        if labels:
            self.features = [
                InputFeatures(
                    input_ids=encoding.input_ids.flatten().tolist(),
                    attention_mask=encoding.attention_mask.flatten().tolist(),
                    label_ids=[label_to_id.get(label, 0)],
                )
                for encoding, label in zip(self.encodings, labels)
            ]
        else:
            self.features = [
                InputFeatures(
                    input_ids=encoding.input_ids.flatten().tolist(),
                    attention_mask=encoding.attention_mask.flatten().tolist(),
                    label_ids=None
                )
                for encoding in self.encodings
            ]
        self._n_features = len(self.features)

    def __len__(self):
        return self._n_features

    def __getitem__(self, idx) -> InputFeatures:
        return self.features[idx]

class InputFeaturesBatch:
    def __init__(self, features: List[InputFeatures]):
        self.input_ids: torch.Tensor
        self.attention_masks: torch.Tensor
        self.label_ids: Optional[torch.Tensor]

        self.n_features = len(features)
        input_ids_list: IntListList = []
        masks_list: IntListList = []
        label_ids: IntListList = []
        for f in features:
            input_ids_list.append(f.input_ids)
            masks_list.append(f.attention_mask)
            if f.label_ids is not None:
                label_ids.append(f.label_ids)
        self.input_ids = torch.LongTensor(input_ids_list)
        self.attention_mask = torch.LongTensor(masks_list)
        if label_ids:
            self.label_ids = torch.LongTensor(label_ids)

    def __len__(self):
        return self.n_features

    def __getitem__(self, item):
        return getattr(self, item)


class SequenceClassificationDataModule(pl.LightningDataModule):
    """
    Prepare dataset and build DataLoader
    """

    def __init__(self, hparams: Namespace):
        self.tokenizer: PreTrainedTokenizerFast
        self.df_train: pd.DataFrame
        self.df_val: pd.DataFrame
        self.df_test: pd.DataFrame
        self.df_org: pd.DataFrame
        self.df_use: pd.DataFrame
        self.label2id: Dict[str, int]

        super().__init__()
        self.max_seq_length = hparams.max_seq_length
        self.cache_dir = hparams.cache_dir
        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)
        self.data_dir = hparams.data_dir
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)

        self.tokenizer_name = hparams.model_name_or_path
        self.train_batch_size = hparams.train_batch_size
        self.eval_batch_size = hparams.eval_batch_size
        self.num_workers = hparams.num_workers
        self.num_samples = hparams.num_samples

    def prepare_data(self):
        """
        Downloads the data and prepare the tokenizer
        """
        self.tokenizer = BertTokenizerFast.from_pretrained(
            self.tokenizer_name, cache_dir=self.cache_dir
        )
        df = make_livedoor_corpus_dataset(self.data_dir)
        self.df_org = df

        # df.sample(frac=1)
        if self.num_samples > 0:
            df = df.iloc[: self.num_samples]
        # label2id =  {k: v for v, k in enumerate(LABELS)}
        self.label2id = {
            k: v for v, k in enumerate(sorted(set(df[LABEL_COL_NAME].values.tolist())))
        }
        self.df_use = df

    def setup(self, stage=None):
        """
        split the data into train, test, validation data
        :param stage: Stage - training or testing
        """

        df = self.df_use

        # NOTE: (fixed) np.random random_state is used by default
        df_train, df_test = train_test_split(
            df, test_size=0.3, stratify=df[LABEL_COL_NAME]
        )
        df_val, df_test = train_test_split(
            df_test, test_size=0.5, stratify=df_test[LABEL_COL_NAME]
        )

        self.train_examples = [
            InputExample(guid=f"train-{i}", text=t, label=l)
            for i, (t, l) in df_train[[TEXT_COL_NAME, LABEL_COL_NAME]].iterrows()
        ]
        self.val_examples = [
            InputExample(guid=f"val-{i}", text=t, label=l)
            for i, (t, l) in df_val[[TEXT_COL_NAME, LABEL_COL_NAME]].iterrows()
        ]
        self.test_examples = [
            InputExample(guid=f"test-{i}", text=t, label=l)
            for i, (t, l) in df_test[[TEXT_COL_NAME, LABEL_COL_NAME]].iterrows()
        ]

        self.train_dataset = self.create_dataset(self.train_examples)
        self.val_dataset = self.create_dataset(self.val_examples)
        self.test_dataset = self.create_dataset(self.test_examples)

        self.dataset_size = len(self.train_dataset)

    def create_dataset(self, data: List[InputExample]) -> SequenceClassificationDataset:
        return SequenceClassificationDataset(
            data,
            tokenizer=self.tokenizer,
            max_seq_length=self.max_seq_length,
            label_to_id=self.label2id,
            # pin_memory=True
        )

    def create_dataloader(
        self,
        ds: SequenceClassificationDataset,
        batch_size: int,
        num_workers: int = 0,
        shuffle: bool = False,
    ) -> DataLoader:
        return DataLoader(
            ds,
            collate_fn=InputFeaturesBatch,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
        )

    def train_dataloader(self):
        return self.create_dataloader(
            self.train_dataset, self.train_batch_size, self.num_workers, shuffle=True
        )

    def val_dataloader(self):
        return self.create_dataloader(
            self.val_dataset, self.eval_batch_size, self.num_workers, shuffle=False
        )

    def test_dataloader(self):
        return self.create_dataloader(
            self.test_dataset, self.eval_batch_size, self.num_workers, shuffle=False
        )

    def total_steps(self) -> int:
        """
        The number of total training steps that will be run. Used for lr scheduler purposes.
        """
        num_devices = max(1, self.hparams.gpus)  # TODO: consider num_tpu_cores
        effective_batch_size = (
            self.hparams.train_batch_size
            * self.hparams.accumulate_grad_batches
            * num_devices
        )
        return (self.dataset_size / effective_batch_size) * self.hparams.max_epochs

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--train_batch_size",
            type=int,
            default=32,
            help="input batch size for training (default: 32)",
        )
        parser.add_argument(
            "--eval_batch_size",
            type=int,
            default=32,
            help="input batch size for validation/test (default: 32)",
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            default=4,
            metavar="N",
            help="number of workers (default: 3)",
        )
        parser.add_argument(
            "--max_seq_length",
            default=256,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--data_dir",
            default="data",
            type=str,
            required=True,
            help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.",
        )
        parser.add_argument(
            "--num_samples",
            type=int,
            default=15000,
            metavar="N",
            help="Number of samples to be used for training and evaluation steps (default: 15000) Maximum:100000",
        )
        return parser


class SequenceClassificationModule(pl.LightningModule):
    """
    Initialize a model and config for token-classification
    """

    def __init__(self, hparams: Union[Dict, Namespace]):
        # NOTE: internal code may pass hparams as dict **kwargs
        if isinstance(hparams, Dict):
            hparams = Namespace(**hparams)

        num_labels = len(LABELS)
        self.scheduler = None
        self.optimizer = None

        super().__init__()
        # Enable to access arguments via self.hparams
        self.save_hyperparameters(hparams)

        self.step_count = 0
        self.output_dir = Path(self.hparams.output_dir)
        self.cache_dir = None
        if self.hparams.cache_dir:
            if not os.path.exists(self.hparams.cache_dir):
                os.mkdir(self.hparams.cache_dir)
            self.cache_dir = self.hparams.cache_dir

        # AutoTokenizer
        # trf>=4.0.0: PreTrainedTokenizerFast by default
        # NOTE: AutoTokenizer doesn't load PreTrainedTokenizerFast...
        self.tokenizer_name = self.hparams.model_name_or_path
        self.tokenizer = BertTokenizerFast.from_pretrained(
            self.tokenizer_name,
            cache_dir=self.cache_dir,
        )

        # AutoConfig
        config_name = self.hparams.model_name_or_path
        self.config: PretrainedConfig = BertConfig.from_pretrained(
            config_name,
            **({"num_labels": num_labels} if num_labels is not None else {}),
            cache_dir=self.cache_dir,
        )
        extra_model_params = (
            "encoder_layerdrop",
            "decoder_layerdrop",
            "dropout",
            "attention_dropout",
        )
        for p in extra_model_params:
            if getattr(self.hparams, p, None):
                assert hasattr(
                    self.config, p
                ), f"model config doesn't have a `{p}` attribute"
                setattr(self.config, p, getattr(self.hparams, p, None))

        # AutoModelForTokenClassification
        self.model: PreTrainedModel = BertForSequenceClassification.from_pretrained(
            self.hparams.model_name_or_path,
            from_tf=bool(".ckpt" in self.hparams.model_name_or_path),
            config=self.config,
            cache_dir=self.cache_dir,
        )
        # for param in self.model.parameters():
        #     param.requires_grad = False

    def forward(self, **inputs) -> SequenceClassifierOutput:
        """ BertForSequenceClassification.forward(
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,)
        """
        return self.model(**inputs)

    def training_step(self, train_batch: InputFeaturesBatch, batch_idx) -> Dict:
        inputs = {
            "input_ids": train_batch.input_ids,
            "attention_mask": train_batch.attention_mask,
            "labels": train_batch.label_ids,
        }
        output: SequenceClassifierOutput = self(**inputs)
        loss = output.loss
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, val_batch: InputFeaturesBatch, batch_idx) -> Dict:
        inputs = {
            "input_ids": val_batch.input_ids,
            "attention_mask": val_batch.attention_mask,
            "labels": val_batch.label_ids,
        }
        output: SequenceClassifierOutput = self(**inputs)
        loss = output.loss
        # self.log("val_step_loss", loss)
        return {"val_step_loss": loss.detach().cpu()}

    def test_step(self, test_batch: InputFeaturesBatch, batch_idx) -> Dict:
        inputs = {
            "input_ids": test_batch.input_ids,
            "attention_mask": test_batch.attention_mask,
            "labels": test_batch.label_ids,
        }
        output: SequenceClassifierOutput = self(**inputs)
        print(output.logits.shape)
        _, y_hat = torch.max(output.logits, dim=1)  # values, indices
        test_acc = accuracy_score(y_hat.cpu(), inputs["labels"].cpu())
        return {"test_acc": torch.tensor(test_acc)}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_step_loss"] for x in outputs]).mean()
        self.log("val_loss", avg_loss, sync_dist=True)

    def test_epoch_end(self, outputs):
        avg_test_acc = torch.stack([x["test_acc"] for x in outputs]).mean()
        self.log("avg_test_acc", avg_test_acc)

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        if self.hparams.adafactor:
            self.optimizer = Adafactor(
                optimizer_grouped_parameters,
                lr=self.hparams.learning_rate,
                scale_parameter=False,
                relative_step=False,
            )
        else:
            self.optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=self.hparams.learning_rate,
                eps=self.hparams.adam_epsilon,
            )
        self.scheduler = {
            "scheduler": ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.2,
                patience=2,
                min_lr=1e-6,
                verbose=True,
            ),
            "monitor": "val_loss",
        }

        return [self.optimizer], [self.scheduler]

    @pl.utilities.rank_zero_only
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]):
        save_path = self.output_dir.joinpath("best_tfmr")
        self.model.config.save_step = self.step_count
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument(
            "--encoder_layerdrop",
            type=float,
            help="Encoder layer dropout probability (Optional). Goes into model.config",
        )
        parser.add_argument(
            "--decoder_layerdrop",
            type=float,
            help="Decoder layer dropout probability (Optional). Goes into model.config",
        )
        parser.add_argument(
            "--dropout",
            type=float,
            help="Dropout probability (Optional). Goes into model.config",
        )
        parser.add_argument(
            "--attention_dropout",
            type=float,
            help="Attention dropout probability (Optional). Goes into model.config",
        )
        parser.add_argument(
            "--weight_decay",
            default=0.0,
            type=float,
            help="Weight decay if we apply some.",
        )
        parser.add_argument(
            "--learning_rate",
            default=5e-5,
            type=float,
            help="The initial learning rate for Adam.",
        )
        parser.add_argument(
            "--adam_epsilon",
            default=1e-8,
            type=float,
            help="Epsilon for Adam optimizer.",
        )
        parser.add_argument("--adafactor", action="store_true")
        return parser


class LoggingCallback(pl.Callback):
    # def on_batch_end(self, trainer, pl_module):
    #     lr_scheduler = trainer.lr_schedulers[0]["scheduler"]
    #     # lrs = {f"lr_group_{i}": lr for i, lr in enumerate(lr_scheduler.get_lr())}
    #     # pl_module.logger.log_metrics(lrs)
    #     pl_module.logger.log_metrics({"last_lr": lr_scheduler._last_lr})

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        rank_zero_info("***** Validation results *****")
        metrics = trainer.callback_metrics
        # Log results
        for key in sorted(metrics):
            rank_zero_info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        rank_zero_info("***** Test results *****")
        metrics = trainer.callback_metrics
        # Log and save results to file
        output_test_results_file = os.path.join(
            pl_module.hparams.output_dir, "test_results.txt"
        )
        with open(output_test_results_file, "w") as writer:
            for key in sorted(metrics):
                rank_zero_info("{} = {}\n".format(key, str(metrics[key])))
                writer.write("{} = {}\n".format(key, str(metrics[key])))


def make_trainer(argparse_args: Namespace):
    """
    Prepare pl.Trainer with callbacks and args
    """

    early_stopping = EarlyStopping(monitor="val_loss", mode="min", verbose=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=argparse_args.output_dir,
        filename="checkpoint-{epoch}-{val_loss:.2f}",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )
    lr_logger = LearningRateMonitor()
    logging_callback = LoggingCallback()

    train_params = {}
    if args.gpus > 1:
        train_params["distributed_backend"] = "ddp"
    train_params["accumulate_grad_batches"] = args.accumulate_grad_batches

    trainer = pl.Trainer.from_argparse_args(
        argparse_args,
        callbacks=[lr_logger, early_stopping, checkpoint_callback, logging_callback],
        **train_params,
    )
    return trainer, checkpoint_callback


def make_model_and_dm(argparse_args: Namespace):
    """
    Prepare pl.LightningDataModule and pl.LightningModule
    """
    dm = SequenceClassificationDataModule(argparse_args)
    dm.prepare_data()
    dm.setup(stage="fit")

    model = SequenceClassificationModule(argparse_args)

    return model, dm


if __name__ == "__main__":

    parser = ArgumentParser(description="Transformers Document Classifier")

    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from huggingface.co",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_predict",
        action="store_true",
        help="Whether to run predictions on the test set.",
    )

    parser = pl.Trainer.add_argparse_args(parent_parser=parser)
    parser = SequenceClassificationModule.add_model_specific_args(parent_parser=parser)
    parser = SequenceClassificationDataModule.add_model_specific_args(
        parent_parser=parser
    )
    args = parser.parse_args()

    # init
    pl.seed_everything(args.seed)
    random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    Path(args.output_dir).mkdir(exist_ok=True)

    mlflow.pytorch.autolog()

    model, dm = make_model_and_dm(args)
    trainer, checkpoint_callback = make_trainer(args)

    # MLflow Autologging is performed here
    trainer.fit(model, dm)

    if args.do_predict:
        # NOTE: load the best checkpoint automatically
        trainer.test()
