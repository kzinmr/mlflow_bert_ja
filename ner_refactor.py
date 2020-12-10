import argparse
import glob
import logging
import os
from argparse import Namespace

import numpy as np
import torch
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset

from lightning_base import BaseTransformer, add_generic_args, generic_train

from dataclasses import dataclass
from enum import Enum
from typing import Final, List, Optional, TextIO, Union
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from transformers.modeling_outputs import TokenClassifierOutput

logger = logging.getLogger(__name__)
PAD_TOKEN_LABEL_ID: Final[int] = CrossEntropyLoss().ignore_index

@dataclass
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    text: str
    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: Optional[List[int]] = None
    label_ids: Optional[List[int]] = None


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"

@dataclass
class InputExample:
    """
    A single training/test example for token classification.
    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        labels: (Optional) list. The labels for each word of the sequence. This should be
        specified for train and dev examples, but not for test examples.
    """

    guid: str
    words: List[str]
    labels: Optional[List[str]]


# en: tokenizer.tokenize: word -> pieces
# ja: tokenizer.tokenize: text -> wordpieces  ??
class NERDataset(Dataset):
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

        TOKEN_TYPE_IDS = None

        return InputFeatures(
            text,
            encoding["input_ids"].flatten(),
            encoding["attention_mask"].flatten(),
            TOKEN_TYPE_IDS,
            torch.tensor(label, dtype=torch.long),
        )

class TokenClassificationDataModule(pl.LightningDataModule):
    def __init__(self, label_idx=-1, **kwargs):
        """
        Initialization of inherited lightning data module
        """
        # in NER datasets, the last column is usually reserved for NER label
        self.label_idx = label_idx        

        super(TokenClassificationDataModule, self).__init__()
        # self.df_train = None
        # self.df_val = None
        # self.df_test = None
        self.train_data_loader = None
        self.val_data_loader = None
        self.test_data_loader = None
        self.max_seq_length = kwargs.max_seq_length
        self.encoding = None
        self.tokenizer = None
        self.args = kwargs


    def read_examples_from_file(
        self, data_dir, mode: Union[Split, str]
    ) -> List[InputExample]:
        if isinstance(mode, Split):
            mode = mode.value
        file_path = os.path.join(data_dir, f"{mode}.txt")
        guid_index = 1
        examples = []
        with open(file_path, encoding="utf-8") as f:
            words = []
            labels = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if words:
                        examples.append(
                            InputExample(
                                guid=f"{mode}-{guid_index}", words=words, labels=labels
                            )
                        )
                        guid_index += 1
                        words = []
                        labels = []
                else:
                    splits = line.split(" ")
                    words.append(splits[0])
                    if len(splits) > 1:
                        labels.append(splits[self.label_idx].replace("\n", ""))
                    else:
                        # Examples could have no label for mode = "test"
                        labels.append("O")
            if words:
                examples.append(
                    InputExample(
                        guid=f"{mode}-{guid_index}", words=words, labels=labels
                    )
                )
        return examples

    @staticmethod
    def convert_examples_to_features(
        examples: List[InputExample],
        label_list: List[str],
        max_seq_length: int,
        tokenizer: PreTrainedTokenizer,
        cls_token_at_end=False,
        cls_token="[CLS]",
        cls_token_segment_id=1,
        sep_token="[SEP]",
        sep_token_extra=False,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        pad_token_label_id=-100,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True,
    ) -> List[InputFeatures]:
        """Loads a data file into a list of `InputFeatures`
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
        """
        # TODO clean up all this to leverage built-in features of tokenizers

        label_map = {label: i for i, label in enumerate(label_list)}

        features: List[InputFatures] = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10_000 == 0:
                logger.info("Writing example %d of %d", ex_index, len(examples))

            tokens = []
            label_ids = []
            for word, label in zip(example.words, example.labels):
                word_tokens = tokenizer.tokenize(word)

                # bert-base-multilingual-cased sometimes output "nothing ([]) when calling tokenize with just a space.
                if len(word_tokens) > 0:
                    tokens.extend(word_tokens)
                    # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                    label_ids.extend(
                        [label_map[label]]   # * len(word_tokens)
                        + [pad_token_label_id] * (len(word_tokens) - 1)
                    )

            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = tokenizer.num_special_tokens_to_add()
            if len(tokens) > max_seq_length - special_tokens_count:
                tokens = tokens[: (max_seq_length - special_tokens_count)]
                label_ids = label_ids[: (max_seq_length - special_tokens_count)]

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids:   0   0   0   0  0     0   0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambiguously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
            if sep_token_extra:
                # roberta uses an extra separator b/w pairs of sentences
                tokens += [sep_token]
                label_ids += [pad_token_label_id]
            segment_ids = [sequence_a_segment_id] * len(tokens)

            if cls_token_at_end:
                tokens += [cls_token]
                label_ids += [pad_token_label_id]
                segment_ids += [cls_token_segment_id]
            else:
                tokens = [cls_token] + tokens
                label_ids = [pad_token_label_id] + label_ids
                segment_ids = [cls_token_segment_id] + segment_ids

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = (
                    [0 if mask_padding_with_zero else 1] * padding_length
                ) + input_mask
                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
                label_ids = ([pad_token_label_id] * padding_length) + label_ids
            else:
                input_ids += [pad_token] * padding_length
                input_mask += [0 if mask_padding_with_zero else 1] * padding_length
                segment_ids += [pad_token_segment_id] * padding_length
                label_ids += [pad_token_label_id] * padding_length

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(label_ids) == max_seq_length

            if ex_index < 5:
                logger.info("*** Example ***")
                logger.info("guid: %s", example.guid)
                logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
                logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
                logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
                logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

            if "token_type_ids" not in tokenizer.model_input_names:
                segment_ids = None

            features.append(
                InputFeatures(
                    input_ids=input_ids,
                    attention_mask=input_mask,
                    token_type_ids=segment_ids,
                    label_ids=label_ids,
                )
            )
        return features


    def prepare_data(self):
        """
        Downloads the data and prepare the tokenizer
        """
        cache_dir = self.hparams.cache_dir if self.hparams.cache_dir else None
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
                self.hparams.tokenizer_name
                if self.hparams.tokenizer_name
                else self.hparams.model_name_or_path,
                cache_dir=cache_dir,
        )

        args = self.hparams
        for mode in ["train", "dev", "test"]:
            cached_features_file = self._feature_file(mode)
            logger.info("Creating features from dataset file at %s", args.data_dir)
            examples = self.read_examples_from_file(
                args.data_dir, mode
            )
            features = self.convert_examples_to_features(
                examples,
                self.labels,
                args.max_seq_length,
                self.tokenizer,
                cls_token_at_end=bool(self.config.model_type in ["xlnet"]),
                cls_token=self.tokenizer.cls_token,
                cls_token_segment_id=2
                if self.config.model_type in ["xlnet"]
                else 0,
                sep_token=self.tokenizer.sep_token,
                sep_token_extra=False,
                pad_on_left=bool(self.config.model_type in ["xlnet"]),
                pad_token=self.tokenizer.pad_token_id,
                pad_token_segment_id=self.tokenizer.pad_token_type_id,
                pad_token_label_id=PAD_TOKEN_LABEL_ID,
            )
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    def _feature_file(self, mode):
        return os.path.join(
            self.hparams.data_dir,
            "cached_{}_{}_{}".format(
                mode,
                list(filter(None, self.hparams.model_name_or_path.split("/"))).pop(),
                str(self.hparams.max_seq_length),
            ),
        )

    def get_dataloader(
        self, mode: int, batch_size: int, num_workers: int, shuffle: bool = False,
    ) -> DataLoader:
        "Load datasets. Called after prepare data."

        cached_features_file = self._feature_file(mode)
        features: List[InputFeatures] = torch.load(cached_features_file)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor(
            [f.attention_mask for f in features], dtype=torch.long
        )
        all_token_type_ids = torch.tensor(
            [f.token_type_ids for f in features], dtype=torch.long
        )
        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
        ds = TensorDataset(
                all_input_ids, all_attention_mask, all_token_type_ids, all_label_ids
        )

        return DataLoader(
            ds,
            batch_size=batch_size, num_workers=num_workers,
        )

    def train_dataloader(self):
        return self.get_dataloader("train", self.hparams.train_batch_size, self.hparams.num_workers, shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader("dev", self.hparams.eval_batch_size, self.hparams.num_workers, shuffle=False)

    def test_dataloader(self):
        return self.get_dataloader("test", self.hparams.eval_batch_size, self.hparams.num_workers, shuffle=False)



    def setup(self, stage=None):
        # """
        # split the data into train, test, validation data
        # :param stage: Stage - training or testing
        # """

        # df = self.df_use

        # # NOTE: (fixed) np.random random_state is used by default
        # df_train, df_test = train_test_split(
        #     df, test_size=0.3, stratify=df[LABEL_COL_NAME]
        # )
        # df_val, df_test = train_test_split(
        #     df_test, test_size=0.5, stratify=df_test[LABEL_COL_NAME]
        # )

        # self.df_train = df_train
        # self.df_test = df_test
        # self.df_val = df_val

    # def create_data_loader(self, df, tokenizer, batch_size, num_workers):
    #     """
    #     Generic data loader function
    #     :param df: Input dataframe
    #     :param tokenizer: bert tokenizer
    #     :param batch_size: Batch size for training
    #     :return: Returns the constructed dataloader
    #     """
    #     texts = df[TEXT_COL_NAME].to_numpy()
    #     labels = df[LABEL_COL_NAME].to_numpy()
    #     ds = NERDataset(
    #         texts=texts,
    #         labels=labels,
    #         tokenizer=tokenizer,
    #         max_length=self.max_seq_length,
    #     )

    #     return DataLoader(
    #         ds, batch_size=batch_size, num_workers=num_workers,
    #     )

    # def train_dataloader(self):
    #     """
    #     :return: output - Train data loader for the given input
    #     """
    #     self.train_data_loader = self.create_data_loader(
    #         self.df_train, self.tokenizer, self.args["train_batch_size"], self.args["num_workers"]
    #     )
    #     return self.train_data_loader

    # def val_dataloader(self):
    #     """
    #     :return: output - Validation data loader for the given input
    #     """
    #     self.val_data_loader = self.create_data_loader(
    #         self.df_val, self.tokenizer, self.args["eval_batch_size"], self.args["num_workers"]
    #     )
    #     return self.val_data_loader

    # def test_dataloader(self):
    #     """
    #     :return: output - Test data loader for the given input
    #     """
    #     self.test_data_loader = self.create_data_loader(
    #         self.df_test, self.tokenizer, self.args["eval_batch_size"], self.args["num_workers"]
    #     )
    #     return self.test_data_loader





    def write_predictions_to_file(
        self, writer: TextIO, test_input_reader: TextIO, preds_list: List
    ):
        example_id = 0
        for line in test_input_reader:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                writer.write(line)
                if not preds_list[example_id]:
                    example_id += 1
            elif preds_list[example_id]:
                output_line = (
                    line.split()[0] + " " + preds_list[example_id].pop(0) + "\n"
                )
                writer.write(output_line)
            else:
                logger.warning(
                    "Maximum sequence length exceeded: No prediction for '%s'.",
                    line.split()[0],
                )

    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):
        """
        Returns the text and the labels of the specified item
        :param parent_parser: Application specific parser
        :return: Returns the augmented arugument parser
        """
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
            default=3,
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
            "--labels",
            default="",
            type=str,
            help="Path to a file containing all labels. If not specified, CoNLL-2003 labels are used.",
        )
        parser.add_argument(
            "--overwrite_cache",
            action="store_true",
            help="Overwrite the cached training and evaluation sets",
        )
        parser.add_argument(
            "--model_name_or_path",
            default=None,
            type=str,
            required=True,
            help="(Common)Path to pretrained model or model identifier from huggingface.co/models",
        )
        parser.add_argument(
            "--tokenizer_name",
            default=None,
            type=str,
            help="(Common)Pretrained tokenizer name or path if not the same as model_name",
        )
        parser.add_argument(
            "--cache_dir",
            default="",
            type=str,
            help="(Common)Where do you want to store the pre-trained models downloaded from huggingface.co",
        )
        parser.add_argument(
            "--data_dir",
            default='ner_data',
            type=str,
            required=True,
            help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.",
        )
        return parser



class TokenClassification(pl.LightningModule):

    def __init__(self, hparams):
        """Initialize a model and config for token-classification"""
        if type(hparams) == dict:
            hparams = Namespace(**hparams)

        self.labels = self._get_labels(hparams.labels)
        self.label_map = {i: label for i, label in enumerate(self.labels)}
        num_labels = len(self.labels)
        self.scheduler = None
        self.optimizer = None

        super(TokenClassification, self).__init__()

        # TODO: move to self.save_hyperparameters()
        # self.save_hyperparameters()
        # can also expand arguments into trainer signature for easier reading

        self.save_hyperparameters(hparams)
        # self.step_count = 0
        self.output_dir = Path(self.hparams.output_dir)
        cache_dir = self.hparams.cache_dir if self.hparams.cache_dir else None

        self.config: PretrainedConfig = AutoConfig.from_pretrained(
            self.hparams.config_name
            if self.hparams.config_name
            else self.hparams.model_name_or_path,
            **({"num_labels": num_labels} if num_labels is not None else {}),
            cache_dir=cache_dir,
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
                setattr(self.config, p, getattr(self.hparams, p))

        self.model: PreTrainedModel = AutoModelForTokenClassification.from_pretrained(
            self.hparams.model_name_or_path,
            from_tf=bool(".ckpt" in self.hparams.model_name_or_path),
            config=self.config,
            cache_dir=cache_dir,
        )
        # for param in self.model.parameters():
        #     param.requires_grad = False

    def _get_labels(self, path: str) -> List[str]:
        with open(path, "r") as f:
            labels = f.read().splitlines()
        if "O" not in labels:
            labels = ["O"] + labels
        return labels

    def forward(self, **inputs):
        """ BertForTokenClassification.forward """
        return self.model(**inputs)

    def training_step(self, train_batch: InputFeatures, batch_idx):
        """
        Training the data as batches and returns training loss on each batch
        :param train_batch Batch data
        :param batch_idx: Batch indices
        :return: output - Training loss
        """
        # XLM and RoBERTa don"t use token_type_ids
        # if self.config.model_type != "distilbert":
        # .to(self.device)
        inputs = {
            "input_ids": train_batch.input_ids,
            "attention_mask": train_batch.attention_mask,
            "token_type_ids": train_batch.token_type_ids if self.config.model_type in ["bert", "xlnet"] else None
            "labels": train_batch.labels
        }
        outputs: TokenClassifierOutput = self(**inputs)
        loss = outputs[0]
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, val_batch: InputFeatures, batch_idx):
        """
        Performs validation of data in batches
        :param val_batch: Batch data
        :param batch_idx: Batch indices
        :return: output - valid step loss
        """
        # XLM and RoBERTa don"t use token_type_ids
        # if self.config.model_type != "distilbert":
        # .to(self.device)
        inputs = {
            "input_ids": val_batch.input_ids,
            "attention_mask": val_batch.attention_mask,
            "token_type_ids": val_batch.token_type_ids if self.config.model_type in ["bert", "xlnet"] else None
            "labels": val_batch.labels
        }
        outputs: TokenClassifierOutput = self(**inputs)
        loss, logits = outputs[:2]
        preds = logits.detach().cpu().numpy()
        target_ids = inputs["labels"].detach().cpu().numpy()
        return {
            "val_loss": loss.detach().cpu(),
            "pred": preds,
            "target": target_ids,
        }

    def test_step(self, test_batch: InputFeatures, batch_idx):
        return self.validation_step(test_batch, batch_idx)

    def _eval_end(self, outputs):
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean()

        preds = np.concatenate([x["pred"] for x in outputs], axis=0)
        preds = np.argmax(preds, axis=2)
        target_ids = np.concatenate([x["target"] for x in outputs], axis=0)

        target_list = [[] for _ in range(target_ids.shape[0])]
        preds_list = [[] for _ in range(target_ids.shape[0])]
        for i in range(target_ids.shape[0]):
            for j in range(target_ids.shape[1]):
                if target_ids[i, j] != PAD_TOKEN_LABEL_ID:
                    target_list[i].append(self.label_map[target_ids[i][j]])
                    preds_list[i].append(self.label_map[preds[i][j]])

        results = {
            "val_loss": val_loss_mean,
            "accuracy_score": accuracy_score(target_list, preds_list),
            "precision": precision_score(target_list, preds_list),
            "recall": recall_score(target_list, preds_list),
            "f1": f1_score(target_list, preds_list),
        }

        ret = {k: v for k, v in results.items()}
        ret["log"] = results
        return ret, preds_list, target_list

    def validation_epoch_end(self, outputs):
        """
        Computes average validation loss
        :param outputs: outputs after every epoch end
        :return: output - average valid loss
        """
        ret, _, _ = self._eval_end(outputs)
        # pytorch_lightning/trainer/logging.py#L139
        logs = ret["log"]
        return {"val_loss": logs["val_loss"], "log": logs, "progress_bar": logs, sync_dist=True}

    def test_epoch_end(self, outputs):
        """
        Computes average test metrics
        :param outputs: outputs after every epoch end
        :return: output - average test metrics
        """
        avg_test_acc = torch.stack([x["test_acc"] for x in outputs]).mean()
        self.log("avg_test_acc", avg_test_acc)
        ret, _, _ = self._eval_end(outputs)
        # pytorch_lightning/trainer/logging.py#L139
        logs = ret["log"]
        return {"test_accuracy": logs["accuracy_score"], "test_precision": logs["precision"], "test_recall": logs["recall"], "test_f1": logs["f1"], "log": logs, "progress_bar": logs}

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
            optimizer = Adafactor(
                optimizer_grouped_parameters,
                lr=self.hparams.learning_rate,
                scale_parameter=False,
                relative_step=False,
            )

        else:
            optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=self.hparams.learning_rate,
                eps=self.hparams.adam_epsilon,
            )
        self.optimizer = optimizer
        # scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
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

    # @pl.utilities.rank_zero_only
    # def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
    #     save_path = self.output_dir.joinpath("best_tfmr")
    #     self.model.config.save_step = self.step_count
    #     self.model.save_pretrained(save_path)
    #     self.tokenizer.save_pretrained(save_path)

    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):
        """
        Returns the text and the label of the specified item
        :param parent_parser: Application specific parser
        :return: Returns the augmented arugument parser
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument(
            "--model_name_or_path",
            default=None,
            type=str,
            required=True,
            help="Path to pretrained model or model identifier from huggingface.co/models",
        )
        parser.add_argument(
            "--config_name",
            default="",
            type=str,
            help="Pretrained config name or path if not the same as model_name",
        )
        # parser.add_argument(
        #     "--tokenizer_name",
        #     default=None,
        #     type=str,
        #     help="Pretrained tokenizer name or path if not the same as model_name",
        # )
        parser.add_argument(
            "--cache_dir",
            default="",
            type=str,
            help="Where do you want to store the pre-trained models downloaded from huggingface.co",
        )
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
            "--learning_rate",
            default=5e-5,
            type=float,
            help="The initial learning rate for Adam.",
        )
        parser.add_argument(
            "--lr_scheduler",
            default="linear",
            choices=arg_to_scheduler_choices,
            metavar=arg_to_scheduler_metavar,
            type=str,
            help="Learning rate scheduler",
        )
        parser.add_argument(
            "--weight_decay",
            default=0.0,
            type=float,
            help="Weight decay if we apply some.",
        )
        parser.add_argument(
            "--adam_epsilon",
            default=1e-8,
            type=float,
            help="Epsilon for Adam optimizer.",
        )
        parser.add_argument(
            "--warmup_steps",
            default=0,
            type=int,
            help="Linear warmup over warmup_steps.",
        )
        parser.add_argument(
            "--num_workers", default=4, type=int, help="kwarg passed to DataLoader"
        )
        parser.add_argument(
            "--num_train_epochs", dest="max_epochs", default=3, type=int
        )

        parser.add_argument("--adafactor", action="store_true")

        parser.add_argument(
            "--gpus",
            default=0,
            type=int,
            help="The number of GPUs allocated for this, it is by default 0 meaning none",
        )
        return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_generic_args(parser, os.getcwd())
    parser = TokenClassification.add_model_specific_args(parser, os.getcwd())
    args = parser.parse_args()

    # If output_dir not provided, a folder will be generated in pwd
    if args.output_dir is None:
        args.output_dir = os.path.join(
            "./results",
            f"{args.task}_{time.strftime('%Y%m%d_%H%M%S')}",
        )
        os.makedirs(args.output_dir)

    model = TokenClassification(args)
    trainer = generic_train(model, args)

    trainer.fit(model)

    if args.do_predict:
        checkpoints = list(
            sorted(
                glob.glob(
                    os.path.join(args.output_dir, "checkpoint-epoch=*.ckpt"),
                    recursive=True,
                )
            )
        )
        model = model.load_from_checkpoint(checkpoints[-1])
        trainer.test(model)
