import logging
import os
import random
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from enum import Enum
from itertools import product, starmap
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import mlflow.pytorch
import numpy as np
import pytorch_lightning as pl
import requests
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint
)
from pytorch_lightning.utilities import rank_zero_info
from seqeval.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
from seqeval.scheme import BILOU
from tokenizers import Encoding
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from transformers import (  # AutoConfig,; AutoModelForTokenClassification,; AutoTokenizer,
    AdamW,
    BatchEncoding,
    BertConfig,
    BertForTokenClassification,
    BertTokenizerFast,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerFast
)
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.optimization import Adafactor

# huggingface/tokenizers: Disabling parallelism to avoid deadlocks.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)
PAD_TOKEN_LABEL_ID = -100
IntList = List[int]
IntListList = List[IntList]
StrList = List[str]
StrListList = List[StrList]


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


@dataclass
class SpanAnnotation:
    start: int
    end: int
    label: str


@dataclass
class StringSpanExample:
    guid: str
    content: str
    annotations: List[SpanAnnotation]


@dataclass
class TokenLabelExample:
    guid: str
    words: List[str]
    labels: Optional[List[str]]


@dataclass
class InputFeatures:
    input_ids: IntList
    attention_mask: IntList
    label_ids: Optional[IntList] = None


def download_dataset(data_dir: str):
    def _download_data(url, file_path):
        response = requests.get(url)
        if response.ok:
            with open(file_path, "w") as fp:
                fp.write(response.content.decode("utf8"))
            return file_path

    for mode in Split:
        mode = mode.value
        url = f"https://github.com/megagonlabs/UD_Japanese-GSD/releases/download/v2.6-NE/{mode}.bio"
        file_path = os.path.join(data_dir, f"{mode}.txt")
        if _download_data(url, file_path):
            logger.info(f"{mode} data is successfully downloaded")


def is_boundary_line(line: str) -> bool:
    return line.startswith("-DOCSTART-") or line == "" or line == "\n"


def bio2biolu(lines: StrList, label_idx: int = -1, delimiter: str = "\t") -> StrList:
    new_lines = []
    n_lines = len(lines)
    for i, line in enumerate(lines):
        if is_boundary_line(line):
            new_lines.append(line)
        else:
            next_iob = None
            if i < n_lines - 1:
                next_line = lines[i + 1].strip()
                if not is_boundary_line(next_line):
                    next_iob = next_line.split(delimiter)[label_idx][0]

            line = line.strip()
            current_line_content = line.split(delimiter)
            current_label = current_line_content[label_idx]
            word = current_line_content[0]
            tag_type = current_label[2:]
            iob = current_label[0]

            iob_transition = (iob, next_iob)
            current_iob = iob
            if iob_transition == ("B", "I"):
                current_iob = "B"
            elif iob_transition == ("I", "I"):
                current_iob = "I"
            elif iob_transition in {("B", "O"), ("B", "B"), ("B", None)}:
                current_iob = "U"
            elif iob_transition in {("I", "B"), ("I", "O"), ("I", None)}:
                current_iob = "L"
            elif iob == "O":
                current_iob = "O"
            else:
                logger.warning(f"Invalid BIO transition: {iob_transition}")
                if iob not in set("BIOLU"):
                    current_iob = "O"
            biolu = f"{current_iob}-{tag_type}" if current_iob != "O" else "O"
            new_line = f"{word}{delimiter}{biolu}"
            new_lines.append(new_line)
    return new_lines


def read_examples_from_file(
    data_dir: str,
    mode: Union[Split, str],
    label_idx: int = -1,
    delimiter: str = "\t",
    is_bio: bool = True,
) -> List[TokenLabelExample]:
    """
    Read token-wise data like CoNLL2003 from file
    """
    if isinstance(mode, Split):
        mode = mode.value
    file_path = os.path.join(data_dir, f"{mode}.txt")
    guid_index = 1
    examples = []
    with open(file_path, encoding="utf-8") as f:
        lines = [line for line in f]
        if is_bio:
            lines = bio2biolu(lines)
        words = []
        labels = []
        for line in lines:
            if is_boundary_line(line):
                if words:
                    examples.append(
                        TokenLabelExample(
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
                TokenLabelExample(
                    guid=f"{mode}-{guid_index}", words=words, labels=labels
                )
            )
    return examples


def convert_spandata(examples: List[TokenLabelExample]) -> List[StringSpanExample]:
    """
    Convert token-wise data like CoNLL2003 into string-wise span data
    """

    def _get_original_spans(words, text):
        word_spans = []
        start = 0
        for w in words:
            word_spans.append((start, start + len(w)))
            start += len(w)
        assert words == [text[s:e] for s, e in word_spans]
        return word_spans

    new_examples: List[StringSpanExample] = []
    for example in examples:
        words = example.words
        text = "".join(words)
        labels = example.labels
        annotations: List[SpanAnnotation] = []
        if labels:
            word_spans = _get_original_spans(words, text)
            label_span = []
            labeltype = ""
            for span, label in zip(word_spans, labels):
                if label == "O" and label_span and labeltype:
                    start, end = label_span[0][0], label_span[-1][-1]
                    annotations.append(
                        SpanAnnotation(start=start, end=end, label=labeltype)
                    )
                    label_span = []
                elif label != "O":
                    labeltype = label[2:]
                    label_span.append(span)
            if label_span and labeltype:
                start, end = label_span[0][0], label_span[-1][-1]
                annotations.append(
                    SpanAnnotation(start=start, end=end, label=labeltype)
                )

        new_examples.append(
            StringSpanExample(guid=example.guid, content=text, annotations=annotations)
        )
    return new_examples


class LabelTokenAligner:
    """
    Align token-wise BIOLU-labels with subtokens
    """

    def __init__(self, labels_path: str):
        with open(labels_path, "r") as f:
            labels = [l for l in f.read().splitlines() if l and l != "O"]

        self.labels_to_id = {"O": 0}
        self.ids_to_label = {0: "O"}
        for i, (label, s) in enumerate(product(labels, "BILU"), 1):
            l = f"{s}-{label}"
            self.labels_to_id[l] = i
            self.ids_to_label[i] = l

    @staticmethod
    def align_tokens_and_annotations_bilou(
        tokenized: Encoding, annotations: List[SpanAnnotation]
    ) -> StrList:
        """Make subtoken-wise BIOLU-labels aligned with given subtokens
        :param tokenized: output of PreTrainedTokenizerFast
        :param annotations: annotations of string span format
        """
        aligned_labels = ["O"] * len(
            tokenized.tokens
        )  # Make a list to store our labels the same length as our tokens
        for anno in annotations:
            annotation_token_ix_set: Set[int] = set()
            for char_ix in range(anno.start, anno.end):
                token_ix = tokenized.char_to_token(char_ix)
                if token_ix is not None:
                    annotation_token_ix_set.add(token_ix)
            if len(annotation_token_ix_set) == 1:
                token_ix = annotation_token_ix_set.pop()
                prefix = "U"
                aligned_labels[token_ix] = f"{prefix}-{anno.label}"
            else:
                last_token_in_anno_ix = len(annotation_token_ix_set) - 1
                for num, token_ix in enumerate(sorted(annotation_token_ix_set)):
                    if num == 0:
                        prefix = "B"
                    elif num == last_token_in_anno_ix:
                        prefix = "L"
                    else:
                        prefix = "I"
                    aligned_labels[token_ix] = f"{prefix}-{anno.label}"
        return aligned_labels

    def align_labels_with_tokens(
        self, tokenized_text: Encoding, annotations: List[SpanAnnotation]
    ) -> IntList:
        # TODO: switch label encoding scheme, align_tokens_and_annotations_bio
        raw_labels = self.align_tokens_and_annotations_bilou(
            tokenized_text, annotations
        )
        return list(map(lambda x: self.labels_to_id.get(x, 0), raw_labels))


class TokenClassificationDataset(Dataset):
    """
    Build feature dataset so that the model can load
    """

    def __init__(
        self,
        data: List[StringSpanExample],
        tokenizer: PreTrainedTokenizerFast,
        label_token_aligner: LabelTokenAligner,
        tokens_per_batch: int = 32,
    ):
        self.features: List[InputFeatures] = []
        self.examples: List[TokenLabelExample] = []
        guids: StrList = [ex.guid for ex in data]
        texts: StrList = [ex.content for ex in data]
        annotations: List[List[SpanAnnotation]] = [ex.annotations for ex in data]

        # tokenize text into subwords
        # NOTE: add_special_tokens=True is unnecessary for NER (ok?)
        tokenized_batch: BatchEncoding = tokenizer(texts, add_special_tokens=False)
        encodings: List[Encoding] = tokenized_batch.encodings
        # align word-wise labels with subwords
        aligned_label_ids: IntListList = list(
            starmap(
                label_token_aligner.align_labels_with_tokens,
                zip(encodings, annotations),
            )
        )
        # perform manual padding and register features
        for guid, encoding, label_ids in zip(guids, encodings, aligned_label_ids):
            seq_length = len(label_ids)
            for start in range(0, seq_length, tokens_per_batch):
                end = min(start + tokens_per_batch, seq_length)
                n_padding_to_add = max(0, tokens_per_batch - end + start)
                self.features.append(
                    InputFeatures(
                        input_ids=encoding.ids[start:end]
                        + [tokenizer.pad_token_id] * n_padding_to_add,
                        label_ids=(
                            label_ids[start:end]
                            + [PAD_TOKEN_LABEL_ID] * n_padding_to_add
                        ),
                        attention_mask=(
                            encoding.attention_mask[start:end] + [0] * n_padding_to_add
                        ),
                    )
                )
                subwords = encoding.tokens[start:end]
                labels = [
                    label_token_aligner.ids_to_label[i] for i in label_ids[start:end]
                ]
                self.examples.append(
                    TokenLabelExample(guid=guid, words=subwords, labels=labels)
                )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx) -> InputFeatures:
        return self.features[idx]


class InputFeaturesBatch:
    def __init__(self, features: List[InputFeatures]):
        self.input_ids: torch.Tensor
        self.attention_masks: torch.Tensor
        self.label_ids: Optional[torch.Tensor]

        self.n_features = len(features)
        input_ids: IntListList = []
        masks: IntListList = []
        label_ids: IntListList = []
        for f in features:
            input_ids.append(f.input_ids)
            masks.append(f.attention_mask)
            if f.label_ids is not None:
                label_ids.append(f.label_ids)
        self.input_ids = torch.LongTensor(input_ids)
        self.attention_mask = torch.LongTensor(masks)
        if label_ids:
            self.label_ids = torch.LongTensor(label_ids)

    def __len__(self):
        return self.n_features

    def __getitem__(self, item):
        return getattr(self, item)


class TokenClassificationDataModule(pl.LightningDataModule):
    """
    Prepare dataset and build DataLoader
    """

    def __init__(self, hparams: Namespace):
        self.tokenizer: PreTrainedTokenizerFast
        self.train_examples: List[TokenLabelExample]
        self.dev_examples: List[TokenLabelExample]
        self.test_examples: List[TokenLabelExample]
        self.train_data: List[StringSpanExample]
        self.dev_data: List[StringSpanExample]
        self.test_data: List[StringSpanExample]
        self.train_dataset: TokenClassificationDataset
        self.dev_dataset: TokenClassificationDataset
        self.test_dataset: TokenClassificationDataset

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
        self.labels_path = hparams.labels

    def prepare_data(self):
        """
        Downloads the data and prepare the tokenizer
        """
        # trf>=4.0.0: PreTrainedTokenizerFast by default
        # NOTE: AutoTokenizer doesn't load PreTrainedTokenizerFast...
        self.tokenizer = BertTokenizerFast.from_pretrained(
            self.tokenizer_name,
            cache_dir=self.cache_dir,
            tokenize_chinese_chars=False,
            strip_accents=False,
        )
        download_dataset(self.data_dir)
        self.train_examples = read_examples_from_file(self.data_dir, Split.train)
        self.dev_examples = read_examples_from_file(self.data_dir, Split.dev)
        self.test_examples = read_examples_from_file(self.data_dir, Split.test)
        if self.num_samples > 0:
            self.train_examples = self.train_examples[: self.num_samples]
            self.dev_examples = self.dev_examples[: self.num_samples]
            self.test_examples = self.test_examples[: self.num_samples]
        self.train_data = convert_spandata(self.train_examples)
        self.dev_data = convert_spandata(self.dev_examples)
        self.test_data = convert_spandata(self.test_examples)

        if not os.path.exists(self.labels_path):
            all_labels = set()
            for ex in self.train_examples:
                for l in ex.labels:
                    all_labels.add(l)
            for ex in self.dev_examples:
                for l in ex.labels:
                    all_labels.add(l)
            for ex in self.test_examples:
                for l in ex.labels:
                    all_labels.add(l)
            label_types = sorted({l[2:] for l in sorted(all_labels) if l != "O"})
            with open(self.labels_path, "w") as fp:
                fp.write("\n".join(label_types))
        self.label_token_aligner = LabelTokenAligner(self.labels_path)

        self.train_dataset = self.create_dataset(self.train_data)
        self.dev_dataset = self.create_dataset(self.dev_data)
        self.test_dataset = self.create_dataset(self.test_data)

        self.dataset_size = len(self.train_dataset)

    def setup(self, stage=None):
        """
        split the data into train, test, validation data
        but here we assume the dataset is splitted in prior
        """

    def create_dataset(
        self, data: List[StringSpanExample]
    ) -> TokenClassificationDataset:
        return TokenClassificationDataset(
            data,
            self.tokenizer,
            self.label_token_aligner,
            self.max_seq_length,
            # pin_memory=True
        )

    @staticmethod
    def create_dataloader(
        ds: TokenClassificationDataset,
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
            self.dev_dataset, self.eval_batch_size, self.num_workers, shuffle=False
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
            "--labels",
            default="",
            type=str,
            help="Path to a file containing all labels. If not specified, CoNLL-2003 labels are used.",
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
            default=0,
            metavar="N",
            help="Number of samples to be used for training and evaluation steps (default: 15000) Maximum:100000",
        )
        return parser


class TokenClassificationModule(pl.LightningModule):
    """
    Initialize a model and config for token-classification
    """

    def __init__(self, hparams: Union[Dict, Namespace]):
        # NOTE: internal code may pass hparams as dict **kwargs
        if isinstance(hparams, Dict):
            hparams = Namespace(**hparams)

        label_token_aligner = LabelTokenAligner(hparams.labels)
        self.label_ids_to_label = label_token_aligner.ids_to_label
        num_labels = len(self.label_ids_to_label)

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
            tokenize_chinese_chars=False,
            strip_accents=False,
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
            if getattr(self.hparams, p, None) and hasattr(self.config, p):
                setattr(self.config, p, getattr(self.hparams, p, None))

        # AutoModelForTokenClassification
        self.model: PreTrainedModel = BertForTokenClassification.from_pretrained(
            self.hparams.model_name_or_path,
            from_tf=bool(".ckpt" in self.hparams.model_name_or_path),
            config=self.config,
            cache_dir=self.cache_dir,
        )

        self.scheduler = None
        self.optimizer = None

    def forward(self, **inputs) -> TokenClassifierOutput:
        """BertForTokenClassification.forward(
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

    def training_step(
        self, train_batch: InputFeaturesBatch, batch_idx
    ) -> Dict[str, torch.Tensor]:
        # .to(self.device) is not necessary with pl.Traner
        inputs = {
            "input_ids": train_batch.input_ids,
            "attention_mask": train_batch.attention_mask,
            "labels": train_batch.label_ids,
        }
        output: TokenClassifierOutput = self(**inputs)
        loss = output.loss
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(
        self, val_batch: InputFeaturesBatch, batch_idx
    ) -> Dict[str, torch.Tensor]:
        # .to(self.device) is not necessary with pl.Traner
        inputs = {
            "input_ids": val_batch.input_ids,
            "attention_mask": val_batch.attention_mask,
            "labels": val_batch.label_ids,
        }
        output: TokenClassifierOutput = self(**inputs)
        loss = output.loss
        # self.log("val_step_loss", loss)
        return {
            "val_step_loss": loss,
        }

    def test_step(
        self, test_batch: InputFeaturesBatch, batch_idx
    ) -> Dict[str, torch.Tensor]:
        # .to(self.device) is not necessary with pl.Traner
        inputs = {
            "input_ids": test_batch.input_ids,
            "attention_mask": test_batch.attention_mask,
            "labels": test_batch.label_ids,
        }
        output: TokenClassifierOutput = self(**inputs)
        return {
            "pred": output.logits,
            "target": inputs["labels"],
        }

    def validation_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]):
        avg_loss = torch.stack([x["val_step_loss"] for x in outputs]).mean()
        self.log("val_loss", avg_loss, sync_dist=True)

    def test_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]):
        preds = np.concatenate(
            [x["pred"].detach().cpu().numpy() for x in outputs], axis=0
        )
        preds = np.argmax(preds, axis=2)
        target_ids = np.concatenate(
            [x["target"].detach().cpu().numpy() for x in outputs], axis=0
        )

        target_list: StrListList = [[] for _ in range(target_ids.shape[0])]
        preds_list: StrListList = [[] for _ in range(target_ids.shape[0])]
        for i in range(target_ids.shape[0]):
            for j in range(target_ids.shape[1]):
                if target_ids[i][j] != PAD_TOKEN_LABEL_ID:
                    target_list[i].append(self.label_ids_to_label[target_ids[i][j]])
                    preds_list[i].append(self.label_ids_to_label[preds[i][j]])

        results = {
            "accuracy_score": accuracy_score(target_list, preds_list),
            "precision": precision_score(
                target_list, preds_list, mode="strict", scheme=BILOU
            ),
            "recall": recall_score(
                target_list, preds_list, mode="strict", scheme=BILOU
            ),
            "f1": f1_score(target_list, preds_list, mode="strict", scheme=BILOU),
        }

        self.log("test_accuracy", results["accuracy_score"])
        self.log("test_precision", results["precision"])
        self.log("test_recall", results["recall"])
        self.log("test_f1", results["f1"])

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
    dm = TokenClassificationDataModule(argparse_args)
    dm.prepare_data()
    dm.setup(stage="fit")
    # DataModule must be loaded first, because label_types.txt is automatically generated
    model = TokenClassificationModule(argparse_args)

    return model, dm


if __name__ == "__main__":

    parser = ArgumentParser(description="Transformers Token Classifier")

    parser = pl.Trainer.add_argparse_args(parent_parser=parser)
    parser = TokenClassificationModule.add_model_specific_args(parent_parser=parser)
    parser = TokenClassificationDataModule.add_model_specific_args(parent_parser=parser)
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
        # best_model_path = checkpoint_callback.best_model_path
        # model = model.load_from_checkpoint(best_model_path)
