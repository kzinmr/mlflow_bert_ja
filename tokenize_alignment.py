# https://github.com/LightTag/sequence-labeling-with-transformers
import itertools
from dataclasses import dataclass
from typing import List, Any
from typing_extensions import TypedDict

import torch
from tokenizers import Encoding
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast



IntList = List[int]  # A list of token_ids
IntListList = List[IntList]  # A List of List of token_ids, e.g. a Batch


@dataclass
class TrainingExample:
    input_ids: IntList
    attention_masks: IntList
    labels: IntList

class TraingingBatch:
    def __getitem__(self, item):
        return getattr(self, item)

    def __init__(self, examples: List[TrainingExample]):
        self.input_ids: torch.Tensor
        self.attention_masks: torch.Tensor
        self.labels: torch.Tensor
        input_ids: IntListList = []
        masks: IntListList = []
        labels: IntListList = []
        for ex in examples:
            input_ids.append(ex.input_ids)
            masks.append(ex.attention_masks)
            labels.append(ex.labels)
        self.input_ids = torch.LongTensor(input_ids)
        self.attention_masks = torch.LongTensor(masks)
        self.labels = torch.LongTensor(labels)

def align_tokens_and_annotations_bilou(tokenized: Encoding, annotations):
    tokens = tokenized.tokens
    aligned_labels = ["O"] * len(
        tokens
    )  # Make a list to store our labels the same length as our tokens
    for anno in annotations:
        annotation_token_ix_set = (
            set()
        )  # A set that stores the token indices of the annotation
        for char_ix in range(anno["start"], anno["end"]):

            token_ix = tokenized.char_to_token(char_ix)
            if token_ix is not None:
                annotation_token_ix_set.add(token_ix)
        if len(annotation_token_ix_set) == 1:
            # If there is only one token
            token_ix = annotation_token_ix_set.pop()
            prefix = (
                "U"  # This annotation spans one token so is prefixed with U for unique
            )
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

class LabelSet:
    def __init__(self, labels: List[str]):
        self.labels_to_id = {}
        self.ids_to_label = {}
        self.labels_to_id["O"] = 0
        self.ids_to_label[0] = "O"
        num = 0  # in case there are no labels
        # Writing BILU will give us incremntal ids for the labels
        for _num, (label, s) in enumerate(itertools.product(labels, "BILU")):
            num = _num + 1  # skip 0
            l = f"{s}-{label}"
            self.labels_to_id[l] = num
            self.ids_to_label[num] = l

    def get_aligned_label_ids_from_annotations(self, tokenized_text, annotations):
        raw_labels = align_tokens_and_annotations_bilou(tokenized_text, annotations)
        return list(map(self.labels_to_id.get, raw_labels))


class ExpectedAnnotationShape(TypedDict):
    start:int
    end:int
    label :str

class ExpectedDataItemShape(TypedDict):
    content:str # The Text to be annotated
    annotations :List[ExpectedAnnotationShape]

class TrainingDataset(Dataset):
    ''''''
    def __init__(
        self,
        data: Any,
        label_set: LabelSet,
        tokenizer: PreTrainedTokenizerFast,
        tokens_per_batch=32,
        window_stride=None,
    ):
        self.label_set = label_set
        if window_stride is None:
            self.window_stride = tokens_per_batch
        self.tokenizer = tokenizer
        self.texts = []
        self.annotations = []

        for example in data:
            self.texts.append(example["content"])
            self.annotations.append(example["annotations"])
        ###TOKENIZE All THE DATA
        tokenized_batch = self.tokenizer(self.texts, add_special_tokens=False)
        ###ALIGN LABELS ONE EXAMPLE AT A TIME
        aligned_labels = []
        for ix in range(len(tokenized_batch.encodings)):
            encoding = tokenized_batch.encodings[ix]
            raw_annotations = self.annotations[ix]
            aligned = label_set.get_aligned_label_ids_from_annotations(
                encoding, raw_annotations
            )
            aligned_labels.append(aligned)
        ###END OF LABEL ALIGNMENT

        ###MAKE A LIST OF TRAINING EXAMPLES. (This is where we add padding)
        self.training_examples: List[TrainingExample] = []
        empty_label_id = "O"
        for encoding, label in zip(tokenized_batch.encodings, aligned_labels):
            length = len(label)  # How long is this sequence
            for start in range(0, length, self.window_stride):

                end = min(start + tokens_per_batch, length)

                # How much padding do we need ?
                padding_to_add = max(0, tokens_per_batch - end + start)
                self.training_examples.append(
                    TrainingExample(
                        # Record the tokens
                        input_ids=encoding.ids[start:end]  # The ids of the tokens
                        + [self.tokenizer.pad_token_id]
                        * padding_to_add,  # padding if needed
                        labels=(
                            label[start:end]
                            + [-100] * padding_to_add  # padding if needed
                        ),  # -100 is a special token for padding of labels,
                        attention_masks=(
                            encoding.attention_mask[start:end]
                            + [0]
                            * padding_to_add  # 0'd attenetion masks where we added padding
                        ),
                    )
                )

    def __len__(self):
        return len(self.training_examples)

    def __getitem__(self, idx) -> TrainingExample:

        return self.training_examples[idx]


if __name__=='__main__':
    from torch.utils.data.dataloader import DataLoader
    from transformers import BertForTokenClassification, AdamW
    from transformers import BertTokenizerFast,  BatchEncoding
    from tokenizers import Encoding

    # input format
    text = "株式会社XのNLP太郎"
    annotations = [
        dict(start=0,end=5,text="株式会社X",label="Org"),
        dict(start=6,end=11,text="NLP太郎",label="Person"),
    ]
    for anno in annotations:
        print(text[anno['start']:anno['end']], anno['label'])

    example = {'annotations': annotations, 'content': text,}
    

    PRETRAINED_MODEL = 'cl-tohoku/bert-base-japanese'
    tokenizer = BertTokenizerFast.from_pretrained(PRETRAINED_MODEL)


    # tokenize & alignment demo
    tokenized_batch: BatchEncoding = tokenizer(example["content"])
    tokenized_text: Encoding = tokenized_batch[0]
    labels = align_tokens_and_annotations_bilou(tokenized_text, example["annotations"])
    for token, label in zip(tokenized_text.tokens, labels):
        print(token, "-", label)

    # make Dataset
    raw = [example]
    label_set = LabelSet(labels=["Person", "Org"])
    ds = TrainingDataset(
        data=raw, tokenizer=tokenizer, label_set=label_set, tokens_per_batch=16
    )
    dataloader = DataLoader(
        ds,
        collate_fn=TraingingBatch,
        batch_size=4,
        shuffle=True,
    )

    # training demo
    model = BertForTokenClassification.from_pretrained(
        PRETRAINED_MODEL, num_labels=len(ds.label_set.ids_to_label.values())
    )
    optimizer = AdamW(model.parameters(), lr=5e-6)
    for num, batch in enumerate(dataloader):
        output = model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_masks,
            labels=batch.labels,
        )
        loss, logits = output.loss, output.logits
        loss.backward()
        optimizer.step()
        print(loss)
        if num > 20:
            break
