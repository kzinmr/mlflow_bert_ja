import os
from argparse import ArgumentParser
import mlflow.pytorch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, AdamW
from typing import Final, Optional
from pathlib import Path
import requests
import tarfile

TEXT_COL_NAME: Final[str] = "text"
LABEL_COL_NAME: Final[str] = "label"
INPUT_IDS: Final[str] = "input_ids"
ATTENTION_MASK: Final[str] = "attention_mask"

PRE_TRAINED_MODEL_NAME: Final[str] = "cl-tohoku/bert-base-japanese"
LABELS: Final[Tuple] = (
    "sports-watch",
    "topic-news",
    "dokujo-tsushin",
    "peachy",
    "movie-enter",
    "kaden-channel",
    "livedoor-homme",
    "smax",
    "it-life-hack",
)
# fix seed
RANDOM_SEED: Final[int] = 42
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def prepare_livedoor_corpus(data_dir: Path) -> Optional[Path]:
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
    # TODO: add livedoor corpus downloader
    # ライブドアコーパスを[カテゴリ, 本文]形式でpd.DataFrameで読み込む
    pdir = Path(data_dir)
    if not (pdir / "text").exists():
        pdir.mkdir(exist_ok=True)
        parent_path = prepare_livedoor_corpus(Path(data_dir))
    else:
        parent_path = pdir / "text"

    categories = LABELS
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


class DocCatDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        """
        Performs initialization of tokenizer
        :param texts: document texts
        :param labels: labels
        :param tokenizer: bert tokenizer
        :param max_length: maximum length of the news text
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.__texts_len = len(texts)

    def __len__(self):
        """
        :return: returns the number of datapoints in the dataframe
        """
        return self.__texts_len

    def __getitem__(self, item):
        """
        Returns the text and the labels of the specified item
        :param item: Index of sample text
        :return: Returns the dictionary of text, input ids, attention mask, labels
        """
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
            truncation=True,
        )

        return {
            TEXT_COL_NAME: text,
            INPUT_IDS: encoding[INPUT_IDS].flatten(),
            ATTENTION_MASK: encoding[ATTENTION_MASK].flatten(),
            LABEL_COL_NAME: torch.tensor(label, dtype=torch.long),
        }


class BertJapaneseDataModule(pl.LightningDataModule):
    def __init__(self, **kwargs):
        """
        Initialization of inherited lightning data module
        """
        super(BertJapaneseDataModule, self).__init__()
        self.PRE_TRAINED_MODEL_NAME = PRE_TRAINED_MODEL_NAME
        self.df_train = None
        self.df_val = None
        self.df_test = None
        self.train_data_loader = None
        self.val_data_loader = None
        self.test_data_loader = None
        self.MAX_LEN = 512
        self.encoding = None
        self.tokenizer = None
        self.args = kwargs
        self.df_org = None
        self.df_use = None
        self.label2id = None

    def prepare_data(self):
        """
        Downloads the data and prepare the tokenizer
        """
        self.tokenizer = BertTokenizer.from_pretrained(self.PRE_TRAINED_MODEL_NAME)
        df = make_livedoor_corpus_dataset()
        self.df_org = df
        df = self.df_all
        df.sample(frac=1)
        df = df.iloc[: self.args["num_samples"]]
        # label2id =  {k: v for v, k in enumerate(LABELS)}
        label2id = {
            k: v for v, k in enumerate(sorted(set(df.label.to_numpy().to_list())))
        }
        df[LABEL_COL_NAME] = df[LABEL_COL_NAME].apply(lambda x: label2id[x])
        self.label2id = label2id
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

        self.df_train = df_train
        self.df_test = df_test
        self.df_val = df_val

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Returns the text and the labels of the specified item
        :param parent_parser: Application specific parser
        :return: Returns the augmented arugument parser
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--batch_size",
            type=int,
            default=16,
            metavar="N",
            help="input batch size for training (default: 16)",
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            default=3,
            metavar="N",
            help="number of workers (default: 3)",
        )
        return parser

    def create_data_loader(self, df, tokenizer, max_len, batch_size, num_workers):
        """
        Generic data loader function
        :param df: Input dataframe
        :param tokenizer: bert tokenizer
        :param max_len: Max length of the news datapoint
        :param batch_size: Batch size for training
        :return: Returns the constructed dataloader
        """
        texts = df[TEXT_COL_NAME].to_numpy()
        labels = df[LABEL_COL_NAME].to_numpy()
        ds = DocCatDataset(
            texts=texts,
            labels=labels,
            tokenizer=tokenizer,
            max_length=max_len,
        )

        return DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    def train_dataloader(self):
        """
        :return: output - Train data loader for the given input
        """
        self.train_data_loader = self.create_data_loader(
            self.df_train,
            self.tokenizer,
            self.MAX_LEN,
            self.args["batch_size"],
            self.args["num_workers"],
        )
        return self.train_data_loader

    def val_dataloader(self):
        """
        :return: output - Validation data loader for the given input
        """
        self.val_data_loader = self.create_data_loader(
            self.df_val,
            self.tokenizer,
            self.MAX_LEN,
            self.args["batch_size"],
            self.args["num_workers"],
        )
        return self.val_data_loader

    def test_dataloader(self):
        """
        :return: output - Test data loader for the given input
        """
        self.test_data_loader = self.create_data_loader(
            self.df_test,
            self.tokenizer,
            self.MAX_LEN,
            self.args["batch_size"],
            self.args["num_workers"],
        )
        return self.test_data_loader


class BertNewsClassifier(pl.LightningModule):
    def __init__(self, **kwargs):
        """
        Initializes the network, optimizer and scheduler
        """
        super(BertNewsClassifier, self).__init__()
        self.PRE_TRAINED_MODEL_NAME = PRE_TRAINED_MODEL_NAME
        self.bert_model = BertModel.from_pretrained(self.PRE_TRAINED_MODEL_NAME)
        for param in self.bert_model.parameters():
            param.requires_grad = False
        self.drop = nn.Dropout(p=0.2)
        # assigning labels
        self.class_names = LABELS
        n_classes = len(self.class_names)

        self.fc1 = nn.Linear(self.bert_model.config.hidden_size, 512)
        self.out = nn.Linear(512, n_classes)

        self.scheduler = None
        self.optimizer = None
        self.args = kwargs

    def forward(self, input_ids, attention_mask):
        """
        :param input_ids: Input data
        :param attention_maks: Attention mask value
        :return: output - Type of news for the given news snippet
        """
        _, pooled_output = self.bert_model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        output = F.relu(self.fc1(pooled_output))
        output = self.drop(output)
        output = self.out(output)
        return output

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Returns the text and the label of the specified item
        :param parent_parser: Application specific parser
        :return: Returns the augmented arugument parser
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--lr",
            type=float,
            default=0.001,
            metavar="LR",
            help="learning rate (default: 0.001)",
        )
        return parser

    def training_step(self, train_batch, batch_idx):
        """
        Training the data as batches and returns training loss on each batch
        :param train_batch Batch data
        :param batch_idx: Batch indices
        :return: output - Training loss
        """
        input_ids = train_batch[INPUT_IDS].to(self.device)
        attention_mask = train_batch[ATTENTION_MASK].to(self.device)
        labels = train_batch[LABEL_COL_NAME].to(self.device)
        output = self.forward(input_ids, attention_mask)
        loss = F.cross_entropy(output, labels)
        self.log("train_loss", loss)
        return {"loss": loss}

    def test_step(self, test_batch, batch_idx):
        """
        Performs test and computes the accuracy of the model
        :param test_batch: Batch data
        :param batch_idx: Batch indices
        :return: output - Testing accuracy
        """
        input_ids = test_batch[INPUT_IDS].to(self.device)
        attention_mask = test_batch[ATTENTION_MASK].to(self.device)
        labels = test_batch[LABEL_COL_NAME].to(self.device)
        output = self.forward(input_ids, attention_mask)
        _, y_hat = torch.max(output, dim=1)
        test_acc = accuracy_score(y_hat.cpu(), labels.cpu())
        return {"test_acc": torch.tensor(test_acc)}

    def validation_step(self, val_batch, batch_idx):
        """
        Performs validation of data in batches
        :param val_batch: Batch data
        :param batch_idx: Batch indices
        :return: output - valid step loss
        """

        input_ids = val_batch[INPUT_IDS].to(self.device)
        attention_mask = val_batch[ATTENTION_MASK].to(self.device)
        labels = val_batch[LABEL_COL_NAME].to(self.device)
        output = self.forward(input_ids, attention_mask)
        loss = F.cross_entropy(output, labels)
        return {"val_step_loss": loss}

    def validation_epoch_end(self, outputs):
        """
        Computes average validation accuracy
        :param outputs: outputs after every epoch end
        :return: output - average valid loss
        """
        avg_loss = torch.stack([x["val_step_loss"] for x in outputs]).mean()
        self.log("val_loss", avg_loss, sync_dist=True)

    def test_epoch_end(self, outputs):
        """
        Computes average test accuracy score
        :param outputs: outputs after every epoch end
        :return: output - average test loss
        """
        avg_test_acc = torch.stack([x["test_acc"] for x in outputs]).mean()
        self.log("avg_test_acc", avg_test_acc)

    def configure_optimizers(self):
        """
        Initializes the optimizer and learning rate scheduler
        :return: output - Initialized optimizer and scheduler
        """
        self.optimizer = AdamW(self.parameters(), lr=self.args["lr"])
        self.scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
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


def make_trainer(argparse_args):
    early_stopping = EarlyStopping(monitor="val_loss", mode="min", verbose=True)

    checkpoint_callback = ModelCheckpoint(
        filepath=os.getcwd(),
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
        prefix="",
    )
    lr_logger = LearningRateMonitor()

    trainer = pl.Trainer.from_argparse_args(
        argparse_args,
        callbacks=[lr_logger, early_stopping],
        checkpoint_callback=checkpoint_callback,
    )
    return trainer


def make_model_and_dm(argparse_args):
    dict_args = vars(argparse_args)

    if "accelerator" in dict_args:
        if dict_args["accelerator"] == "None":
            dict_args["accelerator"] = None
    dm = BertJapaneseDataModule(**dict_args)
    dm.prepare_data()
    dm.setup(stage="fit")

    model = BertNewsClassifier(**dict_args)

    return model, dm


if __name__ == "__main__":
    parser = ArgumentParser(description="Japanese Bert News Classifier Example")

    parser.add_argument(
        "--num_samples",
        type=int,
        default=15000,
        metavar="N",
        help="Number of samples to be used for training and evaluation steps (default: 15000) Maximum:100000",
    )

    parser = pl.Trainer.add_argparse_args(parent_parser=parser)
    parser = BertNewsClassifier.add_model_specific_args(parent_parser=parser)
    parser = BertJapaneseDataModule.add_model_specific_args(parent_parser=parser)

    # Autologging is performed when the fit method of pl.Trainer() is called.
    mlflow.pytorch.autolog()

    args = parser.parse_args()
    model, dm = make_model_and_dm(args)
    trainer = make_trainer(args)

    trainer.fit(model, dm)
    trainer.test()
