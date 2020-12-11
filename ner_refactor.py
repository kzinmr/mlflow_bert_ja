import argparse
import glob
import logging
import os
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, field
from enum import Enum
from itertools import product
from pathlib import Path
from typing import Any, Final, List, Optional, Set, TextIO, Union
from typing_extensions import TypedDict

import numpy as np
import pytorch_lightning as pl
import torch

from pytorch_lightning.utilities import rank_zero_info
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset, TensorDataset

from tokenizers import Encoding
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    BatchEncoding,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from transformers.optimization import Adafactor
from transformers.modeling_outputs import TokenClassifierOutput


logger = logging.getLogger(__name__)
PAD_TOKEN_LABEL_ID: Final[int] = CrossEntropyLoss().ignore_index
IntList = List[int]
IntListList = List[IntList]
StrList = List[str]


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


class ExpectedAnnotationShape(TypedDict):
    start: int
    end: int
    label: str


class ExpectedDataItemShape(TypedDict):
    content: str  # The Text to be annotated
    annotations: List[ExpectedAnnotationShape]


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

def read_labels(path: str) -> StrList:
    with open(path, "r") as f:
        labels = f.read().splitlines()
    if "O" not in labels:
        labels = ["O"] + labels
    return labels

def read_examples_from_file(
    data_dir: str,
    mode: Union[Split, str],
    label_idx: int = -1,
    delimiter: str = " ",
) -> List[InputExample]:
    # TODO: InputExample -> ExpectedDataItemShape
    # TokenData -> SpanData
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
                splits = line.strip().split(delimiter)
                words.append(splits[0])
                if len(splits) > 1:
                    labels.append(splits[label_idx])
                else:
                    # for mode = "test"
                    labels.append("O")
        if words:
            examples.append(
                InputExample(guid=f"{mode}-{guid_index}", words=words, labels=labels)
            )
    return examples


@dataclass
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    input_ids: IntList
    attention_mask: IntList
    token_type_ids: Optional[IntList] = None
    label_ids: Optional[IntList] = None


@dataclass
class InputFeaturesBatch:
    input_ids: torch.Tensor = field(init=False)
    attention_mask: torch.Tensor = field(init=False)
    token_type_ids: Optional[torch.Tensor] = field(init=False)
    label_ids: Optional[torch.Tensor] = field(init=False)

    def __getitem__(self, item):
        return getattr(self, item)

    def __post_init__(self, features: List[InputFeatures]):
        input_ids: IntListList = []
        masks: IntListList = []
        token_type_ids: IntListList = []
        label_ids: IntListList = []
        for f in features:
            input_ids.append(f.input_ids)
            masks.append(f.attention_mask)
            if f.token_type_ids is not None:
                token_type_ids.append(f.token_type_ids)
            if f.label_ids is not None:
                label_ids.append(f.label_ids)
        self.input_ids = torch.LongTensor(input_ids)
        self.attention_mask = torch.LongTensor(masks)
        if token_type_ids:
            self.token_type_ids = torch.LongTensor(token_type_ids)
        if label_ids:
            self.label_ids = torch.LongTensor(label_ids)


class LabelSet:
    def __init__(self, labels: StrList):
        self.labels_to_id = {"O": 0}
        self.ids_to_label = {0: "O"}
        for i, (label, s) in enumerate(product(labels, "BILU"), 1):
            l = f"{s}-{label}"
            self.labels_to_id[l] = i
            self.ids_to_label[i] = l

    @staticmethod
    def align_tokens_and_annotations_bilou(
        tokenized: Encoding, annotations: List[ExpectedAnnotationShape]
    ) -> StrList:
        aligned_labels = ["O"] * len(
            tokenized.tokens
        )  # Make a list to store our labels the same length as our tokens
        for anno in annotations:
            annotation_token_ix_set: Set[
                int
            ] = set()  # A set that stores the token indices of the annotation
            for char_ix in range(anno["start"], anno["end"]):
                token_ix = tokenized.char_to_token(char_ix)
                if token_ix is not None:
                    annotation_token_ix_set.add(token_ix)
            if len(annotation_token_ix_set) == 1:
                # If there is only one token
                token_ix = annotation_token_ix_set.pop()
                prefix = "U"  # This annotation spans one token so is prefixed with U for unique
                aligned_labels[token_ix] = f"{prefix}-{anno['label']}"

            else:

                last_token_in_anno_ix = len(annotation_token_ix_set) - 1
                for num, token_ix in enumerate(sorted(annotation_token_ix_set)):
                    if num == 0:
                        prefix = "B"
                    elif num == last_token_in_anno_ix:
                        prefix = "L"  # Its the last token
                    else:
                        prefix = "I"  # We're inside of a multi token annotation
                    aligned_labels[token_ix] = f"{prefix}-{anno['label']}"
        return aligned_labels

    def get_aligned_label_ids_from_annotations(
        self, tokenized_text: Encoding, annotations: List[ExpectedAnnotationShape]
    ) -> IntList:
        # TODO: switch label encoding scheme, align_tokens_and_annotations_bio
        raw_labels = self.align_tokens_and_annotations_bilou(
            tokenized_text, annotations
        )
        return list(map(lambda x: self.labels_to_id.get(x, 0), raw_labels))


def convert_examples_to_features(
    examples: List[InputExample],
    label_list: List[str],
    max_seq_length: int,
    tokenizer: PreTrainedTokenizer,
    is_xlnet: bool = False,
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
    cls_token_at_end = is_xlnet
    cls_token = tokenizer.cls_token
    cls_token_segment_id = 2 if is_xlnet else 0
    sep_token = tokenizer.sep_token
    sep_token_extra = False
    pad_on_left = is_xlnet
    pad_token = tokenizer.pad_token_id
    pad_token_segment_id = tokenizer.pad_token_type_id

    label_map = {label: i for i, label in enumerate(label_list)}

    features: List[InputFeatures] = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10_000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens = []
        label_ids = []
        if example.labels is not None:
            for word, label in zip(example.words, example.labels):
                word_tokens = tokenizer.tokenize(word)

                # bert-base-multilingual-cased sometimes output "nothing ([]) when calling tokenize with just a space.
                if len(word_tokens) > 0:
                    tokens.extend(word_tokens)
                    # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                    label_ids.extend(
                        [label_map[label]]  # * len(word_tokens)
                        + [PAD_TOKEN_LABEL_ID] * (len(word_tokens) - 1)
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
            label_ids += [PAD_TOKEN_LABEL_ID]
            if sep_token_extra:
                # roberta uses an extra separator b/w pairs of sentences
                tokens += [sep_token]
                label_ids += [PAD_TOKEN_LABEL_ID]
            segment_ids = [sequence_a_segment_id] * len(tokens)

            if cls_token_at_end:
                tokens += [cls_token]
                label_ids += [PAD_TOKEN_LABEL_ID]
                segment_ids += [cls_token_segment_id]
            else:
                tokens = [cls_token] + tokens
                label_ids = [PAD_TOKEN_LABEL_ID] + label_ids
                segment_ids = [cls_token_segment_id] + segment_ids

            input_ids: List[int] = tokenizer.convert_tokens_to_ids(tokens)

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
                segment_ids = [pad_token_segment_id] * padding_length + segment_ids
                label_ids = ([PAD_TOKEN_LABEL_ID] * padding_length) + label_ids
            else:
                input_ids += [pad_token] * padding_length
                input_mask += [0 if mask_padding_with_zero else 1] * padding_length
                segment_ids += [pad_token_segment_id] * padding_length
                label_ids += [PAD_TOKEN_LABEL_ID] * padding_length

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


class NERDataset(Dataset):
    def __init__(
        self,
        data: List[InputExample], # List[ExpectedAnnotationShape],
        label_set: LabelSet,
        tokenizer: PreTrainedTokenizerFast,
        tokens_per_batch=32,
    ):
        self.label_set = label_set
        self.tokenizer = tokenizer
        self.texts: StrList = []
        self.annotations: List[List[ExpectedAnnotationShape]] = []
        self.training_examples: List[InputFeatures] = []

        for example in data:
            self.texts.append(example["content"])
            self.annotations.append(example["annotations"])
        ###TOKENIZE All THE DATA
        tokenized_batch: BatchEncoding = self.tokenizer(
            self.texts, add_special_tokens=False
        )
        encodings: List[Encoding] = tokenized_batch.encodings
        # LABEL ALIGNMENT
        aligned_labels: IntListList = [
            label_set.get_aligned_label_ids_from_annotations(encoding, raw_annotations)
            for encoding, raw_annotations in zip(encodings, self.annotations)
        ]

        for encoding, label_ids in zip(encodings, aligned_labels):
            seq_length = len(label_ids)
            for start in range(0, seq_length, tokens_per_batch):
                end = min(start + tokens_per_batch, seq_length)
                n_padding_to_add = max(0, tokens_per_batch - end + start)
                self.training_examples.append(
                    InputFeatures(
                        input_ids=encoding.ids[start:end]
                        + [self.tokenizer.pad_token_id] * n_padding_to_add,
                        label_ids=(
                            label_ids[start:end]
                            + [PAD_TOKEN_LABEL_ID] * n_padding_to_add
                        ),
                        attention_mask=(
                            encoding.attention_mask[start:end] + [0] * n_padding_to_add
                        ),
                    )
                )

    def __len__(self):
        return len(self.training_examples)

    def __getitem__(self, idx) -> InputFeatures:

        return self.training_examples[idx]


class TokenClassificationDataModule(pl.LightningDataModule):
    def __init__(self, **kwargs):
        """
        Initialization of inherited lightning data module
        """
        self.tokenizer: PreTrainedTokenizerFast
        self.train_examples: List[InputExample]
        self.dev_examples: List[InputExample]
        self.test_examples: List[InputExample]
        self.train_dataset: NERDataset
        self.dev_dataset: NERDataset
        self.test_dataset: NERDataset

        super().__init__()
        self.max_seq_length = kwargs["max_seq_length"]
        self.cache_dir = kwargs["cache_dir"]
        self.data_dir = kwargs["data_dir"]
        tn = kwargs["tokenizer_name"]
        self.tokenizer_name = tn if tn else kwargs["model_name_or_path"]
        self.train_batch_size = kwargs["train_batch_size"]
        self.eval_batch_size = kwargs["eval_batch_size"]
        self.num_workers = kwargs["num_workers"]
        self.labels = read_labels(kwargs["labels"])
        self.label_set = LabelSet(self.labels)
        # is_xlnet = bool(self.model_type in ["xlnet"])

    def prepare_data(self):
        """
        Downloads the data and prepare the tokenizer
        """
        # trf>=4.0.0: PreTrainedTokenizerFast by default
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name,
            cache_dir=self.cache_dir,
        )
        self.train_examples = read_examples_from_file(self.data_dir, Split.train)
        self.dev_examples = read_examples_from_file(self.data_dir, Split.dev)
        self.test_examples = read_examples_from_file(self.data_dir, Split.test)

    def setup(self, stage=None):
        """
        split the data into train, test, validation data
        :param stage: Stage - training or testing
        """
        # # NOTE: (fixed) np.random random_state is used by default
        # train, test = train_test_split(
        #     data, test_size=0.3, stratify=data[LABEL_COL_NAME]
        # )
        # dev, test = train_test_split(
        #     test, test_size=0.5, stratify=test[LABEL_COL_NAME]
        # )
        pass

    @property
    def train_dataset(self):
        return NERDataset(
            self.train_examples, self.label_set, self.tokenizer, self.max_seq_length
        )

    @property
    def val_dataset(self):
        return NERDataset(
            self.dev_examples, self.label_set, self.tokenizer, self.max_seq_length
        )

    @property
    def test_dataset(self):
        return NERDataset(
            self.test_examples, self.label_set, self.tokenizer, self.max_seq_length
        )

    @property
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    @property
    def val_dataloader(self):
        return DataLoader(
            self.dev_dataset,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    @property
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    @staticmethod
    def write_predictions_to_file(
        writer: TextIO, test_input_reader: TextIO, preds_list: List
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
            "--labels",
            default="",
            type=str,
            help="Path to a file containing all labels. If not specified, CoNLL-2003 labels are used.",
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
            "--data_dir",
            default="ner_data",
            type=str,
            required=True,
            help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.",
        )
        return parser


class TokenClassificationModule(pl.LightningModule):
    def __init__(self, hparams):
        """Initialize a model and config for token-classification"""
        if type(hparams) == dict:
            hparams = Namespace(**hparams)

        labels = read_labels(hparams.labels)
        num_labels = len(labels)
        self.label_map = {i: label for i, label in enumerate(labels)}
        self.scheduler = None
        self.optimizer = None

        super().__init__()

        # TODO: move to self.save_hyperparameters()
        # self.save_hyperparameters()
        # can also expand arguments into trainer signature for easier reading

        self.save_hyperparameters(hparams)
        self.step_count = 0
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

    def forward(self, **inputs):
        """ BertForTokenClassification.forward """
        return self.model(**inputs)

    def training_step(self, train_batch: InputFeatures, batch_idx) -> Dict:
        """
        Training the data as batches and returns training loss on each batch
        :param train_batch Batch data
        :param batch_idx: Batch indices
        :return: output - Training loss
        """
        # XLM and RoBERTa don"t use token_type_ids
        # if self.config.model_type != "distilbert":
        inputs = {
            "input_ids": train_batch.input_ids.to(self.device),
            "attention_mask": train_batch.attention_mask.to(self.device),
            "token_type_ids": train_batch.token_type_ids.to(self.device)
            if self.config.model_type in ["bert", "xlnet"]
            else None,
            "labels": train_batch.label_ids.to(self.device),
        }
        output: TokenClassifierOutput = self(**inputs)
        loss = output.loss
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, val_batch: InputFeatures, batch_idx) -> Dict:
        """
        Performs validation of data in batches
        :param val_batch: Batch data
        :param batch_idx: Batch indices
        :return: output - valid step loss
        """
        # XLM and RoBERTa don"t use token_type_ids
        # if self.config.model_type != "distilbert":
        inputs = {
            "input_ids": val_batch.input_ids.to(self.device),
            "attention_mask": val_batch.attention_mask.to(self.device),
            "token_type_ids": val_batch.token_type_ids.to(self.device)
            if self.config.model_type in ["bert", "xlnet"]
            else None,
            "labels": val_batch.label_ids.to(self.device),
        }
        output: TokenClassifierOutput = self(**inputs)
        loss, logits = output.loss, output.logits
        preds = logits.detach().cpu().numpy()
        target_ids = inputs["labels"].detach().cpu().numpy()
        self.log("val_loss", loss)
        return {
            "val_loss": loss.detach().cpu(),
            "pred": preds,
            "target": target_ids,
        }

    def test_step(self, test_batch: InputFeatures, batch_idx) -> Dict:
        return self.validation_step(test_batch, batch_idx)

    def _eval_end(self, outputs: List[Dict]):
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

    def validation_epoch_end(self, outputs: List[Dict]) -> Dict:
        """
        Computes average validation loss
        :param outputs: outputs after every epoch end
        :return: output - average valid loss
        """
        ret, _, _ = self._eval_end(outputs)
        # pytorch_lightning/trainer/logging.py#L139
        logs = ret["log"]
        avg_loss = logs["val_loss"]
        self.log("val_loss", avg_loss, sync_dist=True)
        return {"val_loss": avg_loss, "log": logs, "progress_bar": logs}

    def test_epoch_end(self, outputs: List[Dict]) -> Dict:
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
        return {
            "test_accuracy": logs["accuracy_score"],
            "test_precision": logs["precision"],
            "test_recall": logs["recall"],
            "test_f1": logs["f1"],
            "log": logs,
            "progress_bar": logs,
        }

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

    @pl.utilities.rank_zero_only
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]):
        save_path = self.output_dir.joinpath("best_tfmr")
        self.model.config.save_step = self.step_count
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

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


class LoggingCallback(pl.Callback):
    def on_batch_end(self, trainer, pl_module):
        lr_scheduler = trainer.lr_schedulers[0]["scheduler"]
        lrs = {f"lr_group_{i}": lr for i, lr in enumerate(lr_scheduler.get_lr())}
        pl_module.logger.log_metrics(lrs)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        rank_zero_info("***** Validation results *****")
        metrics = trainer.callback_metrics
        # Log results
        for key in sorted(metrics):
            if key not in ["log", "progress_bar"]:
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
                if key not in ["log", "progress_bar"]:
                    rank_zero_info("{} = {}\n".format(key, str(metrics[key])))
                    writer.write("{} = {}\n".format(key, str(metrics[key])))


def add_generic_args(parser, root_dir) -> None:
    #  To allow all pl args uncomment the following line
    #  parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )

    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O2",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--n_tpu_cores", dest="tpu_cores", type=int)
    parser.add_argument(
        "--max_grad_norm",
        dest="gradient_clip_val",
        default=1.0,
        type=float,
        help="Max gradient norm",
    )
    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_predict",
        action="store_true",
        help="Whether to run predictions on the test set.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        dest="accumulate_grad_batches",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.",
    )


def generic_train(
    model: TokenClassificationModule,
    args: Namespace,
    early_stopping_callback=None,
    logger=True,  # can pass WandbLogger() here
    extra_callbacks=[],
    checkpoint_callback=None,
    logging_callback=None,
    **extra_train_kwargs,
):
    pl.seed_everything(args.seed)

    # init model
    odir = Path(model.hparams.output_dir)
    odir.mkdir(exist_ok=True)

    # add custom checkpoints
    if checkpoint_callback is None:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            filepath=args.output_dir,
            prefix="checkpoint",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        )
    if early_stopping_callback:
        extra_callbacks.append(early_stopping_callback)
    if logging_callback is None:
        logging_callback = LoggingCallback()

    train_params = {}

    # TODO: remove with PyTorch 1.6 since pl uses native amp
    if args.fp16:
        train_params["precision"] = 16
        train_params["amp_level"] = args.fp16_opt_level

    if args.gpus > 1:
        train_params["distributed_backend"] = "ddp"

    train_params["accumulate_grad_batches"] = args.accumulate_grad_batches
    train_params["accelerator"] = extra_train_kwargs.get("accelerator", None)
    train_params["profiler"] = extra_train_kwargs.get("profiler", None)

    trainer = pl.Trainer.from_argparse_args(
        args,
        weights_summary=None,
        callbacks=[logging_callback] + extra_callbacks,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        **train_params,
    )

    return trainer


if __name__ == "__main__":
    parser = ArgumentParser()
    add_generic_args(parser, os.getcwd())
    parser = TokenClassificationModule.add_model_specific_args(parser, os.getcwd())
    args = parser.parse_args()

    # If output_dir not provided, a folder will be generated in pwd
    if args.output_dir is None:
        args.output_dir = os.path.join(
            "./results",
            f"{args.task}_{time.strftime('%Y%m%d_%H%M%S')}",
        )
        os.makedirs(args.output_dir)

    model = TokenClassificationModule(args)
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
