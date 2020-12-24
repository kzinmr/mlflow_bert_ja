import logging
import os
import random
import tempfile
from argparse import ArgumentParser, Namespace
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple, Union

import jsonlines
import MeCab
import mlflow.pytorch
import numpy as np
import pandas as pd
import regex as re
import requests
import sklearn_crfsuite
import textspan
from seqeval.metrics import \
    classification_report as seqeval_classification_report
from seqeval.scheme import BILOU
from sklearn_crfsuite import metrics as crfsuite_metrics

logger = logging.getLogger(__name__)

IntList = List[int]
IntListList = List[IntList]
StrList = List[str]
StrListList = List[StrList]


def seed_everything(seed: int = 1234):
    """乱数固定"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


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
class TokenClassificationExample:
    guid: str
    words: StrList
    labels: StrList


def download_dataset(data_dir: Union[str, Path]):
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
) -> List[TokenClassificationExample]:
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
                        TokenClassificationExample(
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
                TokenClassificationExample(
                    guid=f"{mode}-{guid_index}", words=words, labels=labels
                )
            )
    return examples


def convert_spandata(
    examples: List[TokenClassificationExample],
) -> List[StringSpanExample]:
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
            annotations.append(SpanAnnotation(start=start, end=end, label=labeltype))

        new_examples.append(
            StringSpanExample(guid=example.guid, content=text, annotations=annotations)
        )
    return new_examples


class CharCRF:
    fkeys = [
        "text",
        "word_f",
        "pos_f",
        "finepos_f",
        "bunrui_f",
        "midasi_f",
    ]

    def __init__(self, hparams: Union[Dict, Namespace]):

        if isinstance(hparams, Dict):
            hparams = Namespace(**hparams)

        self.output_dir = Path(hparams.output_dir)
        self.hparams = hparams

        self.model = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=hparams.c1,
            c2=hparams.c2,
            max_iterations=hparams.max_iterations,
            all_possible_transitions=hparams.all_possible_transitions,
        )
        self.tagger = MeCab.Tagger()
        self.tagger.parse("\n")

    @classmethod
    def blank_features(cls) -> Dict[str, float]:
        return {key: 0.0 for key in cls.fkeys}

    def generate_window_features(
        self, features: List[Dict[str, Union[str, float]]]
    ) -> List[Dict[str, float]]:
        window_size = self.hparams.window_size
        feats_new = []
        n_total = len(features)
        window_size = min(n_total, window_size)
        for i, feat in enumerate(features):
            feat_new = feat.copy()
            for j in range(1, window_size + 1):
                feat_prev = features[i - j] if i > 0 else self.blank_features()
                for k, v in feat_prev.items():
                    feat_new[f"-{j}:{k}"] = v
                feat_next = (
                    features[i + j] if i < n_total - j else self.blank_features()
                )
                for k, v in feat_next.items():
                    feat_new[f"+{j}:{k}"] = v
            feats_new.append(feat_new)
        return feats_new

    @staticmethod
    def match_spans(pattern: str, text: str) -> List[Tuple[int, int]]:
        return [m.span() for m in re.finditer(pattern, text)]

    def parse_full(self, text, partial=False):
        # 前処理の差異はalignmentで吸収する
        text_r = re.sub(r"\s+", "", text)
        text_r = re.sub("---thisispageseparator---", "", text_r)
        result = self.tagger.parse(text_r)
        return [l.split("\t") for l in result.splitlines() if l.split("\t")[0] != "EOS"]

    def convert_feature_line(self, text: str) -> List[Dict[str, str]]:
        """文字レベルに対して、品詞など単語レベルの情報をBIO形式や単語表層の形で付与する"""
        # parse and align tokens with characters
        tfs = self.parse_full(text)
        if len(tfs) == 0:
            return []
        ts, fs = zip(*tfs)
        featuremap = {k: v for k, v in enumerate(fs)}
        featuremap[-1] = "*,*,*,*,*,*,*,*,*"
        ts_spans = textspan.get_original_spans(ts, text)
        char2token_index = {
            char_ix: token_ix
            for token_ix, char_spans in enumerate(ts_spans)
            for st, ed in char_spans
            for char_ix in range(st, ed)
        }
        feature_lines = [
            {
                "index": i,
                "word_index": char2token_index.get(i, -1),
                "text": c,
                "features": featuremap[char2token_index.get(i, -1)].split(","),
            }
            for i, c in enumerate(text)
        ]

        # convert feature_lines
        nonemap = {"*": None}
        prev_word_index = -1

        # make word_index2midasi dict
        # NOTE: line_feature['features'][6] では見出しが無いことがある
        word_index2midasi = {-1: ""}
        word_surface = ""
        for line_feature in feature_lines:
            word_index = line_feature["word_index"]
            midasi = line_feature["features"][6]
            midasi = nonemap[midasi] if midasi in nonemap else midasi
            # 見出しが無い場合は単語surfaceで埋める
            if midasi is not None:
                word_surface = midasi
            elif word_index != prev_word_index:
                word_surface = line_feature["text"]
            else:
                word_surface += line_feature["text"]
            prev_word_index = word_index
            word_index2midasi[word_index] = word_surface

        # make word-BIO feature tags from 'word_index' and 'features'
        new_features = []
        for line_feature in feature_lines:
            word_index = line_feature["word_index"]
            if word_index != prev_word_index:
                bioprefix = "B-"
            else:
                bioprefix = "I-"
            prev_word_index = word_index
            pos = line_feature["features"][0]
            pos = nonemap[pos] if pos in nonemap else pos
            finepos = line_feature["features"][1]
            finepos = nonemap[finepos] if finepos in nonemap else finepos
            bunrui = line_feature["features"][2]
            bunrui = nonemap[bunrui] if bunrui in nonemap else bunrui

            new_features.append(
                {
                    "index": line_feature["index"],
                    "word_index": line_feature["word_index"],
                    "text": line_feature["text"],
                    "word_f": bioprefix + "word",
                    "pos_f": bioprefix + pos if pos is not None else "O",
                    "finepos_f": bioprefix + finepos if finepos is not None else "O",
                    "bunrui_f": bioprefix + bunrui if bunrui is not None else "O",
                    "midasi_f": word_index2midasi[word_index],
                    "phrase_f": "O",
                }
            )

        return new_features

    def extract_single_features(self, text: str) -> List[Dict[str, Union[str, float]]]:
        features = self.convert_feature_line(text)
        return [{key: fline[key] for key in self.fkeys} for fline in features]

    def extract_features(self, text: str) -> List[Dict[str, Union[str, float]]]:
        return self.generate_window_features(self.extract_single_features(text))

    @staticmethod
    def get_biolu_from_spans(
        text_length: int, annotations: List[SpanAnnotation]
    ) -> List[str]:
        bio = ["O" for _ in range(text_length)]
        for anno in annotations:
            i, j, et = anno.start, anno.end, anno.label
            if j - i == 1:
                bio[i] = f"U-{et}"
            elif j - i > 1:
                bio[i] = f"B-{et}"
                bio[j - 1] = f"L-{et}"
                if j - i > 2:
                    bio[i + 1 : j - 1] = [f"I-{et}" for _ in range(i + 1, j - 1)]
        return bio

    def create_features(self, data: StringSpanExample) -> Tuple[List[Dict], List[str]]:
        text = data.content
        annotations = data.annotations
        x = self.extract_features(text)
        y = self.get_biolu_from_spans(len(x), annotations)
        return x, y

    def fit(self, dataset: List[StringSpanExample]):
        xys = map(self.create_features, dataset)
        X, y = zip(*xys)
        self.model.fit(X, y)

    def predict(self, texts: StrList) -> StrListList:
        X = list(map(self.extract_features, texts))
        return self.model.predict(X)

    @staticmethod
    def convert_report_df(report_str: str) -> pd.DataFrame:
        lines = list(filter(None, report_str.split("\n")))
        columns = list(filter(None, lines[0].split(" ")))
        metrics = [list(filter(None, ls.split(" "))) for ls in lines[1:]]
        each_metrics = [
            ls
            for ls in metrics
            if ls[0] not in {"accuracy", "macro", "weighted", "micro"}
        ]
        macro_metrics = [["macro"] + ls[2:] for ls in metrics if ls[0] in {"macro"}]
        micro_metrics = [["weighted"] + ls[2:] for ls in metrics if ls[0] == "weighted"]
        all_metrics = each_metrics + macro_metrics + micro_metrics
        index = [ls[0] for ls in all_metrics]
        all_metrics = [list(map(float, ls[1:])) for ls in all_metrics]
        df = pd.DataFrame(all_metrics, index=index)
        df.columns = columns
        return df

    @staticmethod
    def fetch_logged_data(run_id):
        client = mlflow.tracking.MlflowClient()
        data = client.get_run(run_id).data
        tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
        artifacts = [f.path for f in client.list_artifacts(run_id, "model")]
        return data.params, data.metrics, tags, artifacts

    def do_experiment(
        self,
        n_train=-1,
        experiment_name: str = "NER with CRF",
    ):
        """
        # tracking_uri: mlflowサーバー情報
        # experiment_name: 実験名称（COC_株式譲渡など)
        # すでに登録済みであればそのプロジェクトに記録され、新規の場合は新たにプロジェクトが作られる
        ## from: https://blog.hoxo-m.com/entry/mlflow_store
        - Run: １回の試行（e.g. 実験, 学習, ... etc.）
        - Experiment: Runを束ねるグループ
        - Artifact: Runで得られた出力や中間生成物の保管先
        - Artifact Location: Artifactの格納先
        - ある Experiment に対して一つのArtifact Locationが紐づく
        - 複数のExperimentと結びつくようにすることも可能だが、管理面でそのような設計は避けるべき
        """

        # load datset
        train_examples = read_examples_from_file(self.hparams.data_dir, Split.train)
        train_char_examples = convert_spandata(train_examples)
        val_examples = read_examples_from_file(self.hparams.data_dir, Split.dev)
        val_char_examples = convert_spandata(val_examples)
        test_examples = read_examples_from_file(self.hparams.data_dir, Split.test)
        test_char_examples = convert_spandata(test_examples)
        if n_train > 0:
            n_sample = min(n_train, len(train_examples))
            train_examples = random.sample(train_examples, n_sample)

        # chars = [ex.content for ex in train_char_examples]
        # print(chars[:1])
        # print(self.create_features(train_char_examples[0]))

        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        # 使う訓練データの量を徐々に増やす実験を行う
        # mlflowに実験内容を登録
        mlflow.set_experiment(experiment_name)
        mlflow.sklearn.autolog()

        with mlflow.start_run() as run:
            # files
            tmpdir = Path(tempfile.mkdtemp())
            train_path = tmpdir / "train.jsonl"
            val_path = tmpdir / "val.jsonl"
            test_path = tmpdir / "test.jsonl"
            with jsonlines.open(train_path, "w") as writer:
                train_jl = [asdict(ex) for ex in train_char_examples]
                writer.write_all(train_jl)
            with jsonlines.open(val_path, "w") as writer:
                val_jl = [asdict(ex) for ex in val_char_examples]
                writer.write_all(val_jl)
            with jsonlines.open(test_path, "w") as writer:
                test_jl = [asdict(ex) for ex in test_char_examples]
                writer.write_all(test_jl)
            mlflow.log_artifact(train_path)
            mlflow.log_artifact(val_path)
            mlflow.log_artifact(test_path)
            mlflow.log_metric("train_count", len(train_char_examples))
            mlflow.log_metric("val_count", len(val_char_examples))
            mlflow.log_metric("test_count", len(test_char_examples))

            # train
            self.fit(train_char_examples)
            # params, metrics, tags, artifacts = self.fetch_logged_data(run.info.run_id)
            # mlflow.sklearn.log_model(self.model, "model")

            # test
            texts = [ex.content for ex in test_char_examples]
            golds: StrListList = [
                self.get_biolu_from_spans(len(data.content), data.annotations)
                for data in test_char_examples
            ]
            sorted_labels = sorted(set([l for ls in golds for l in ls]))
            preds = self.predict(texts)
            # chunk-wise (seqeval)
            seqeval_report = seqeval_classification_report(
                golds, preds, digits=4, mode="strict", scheme=BILOU
            )
            print(seqeval_report)
            metric_df = CharCRF.convert_report_df(seqeval_report)
            prf = metric_df.loc["weighted"]
            mlflow.log_metric("weighted-f1-score-chunkwise", prf.loc["f1-score"])
            mlflow.log_metric("weighted-precision-chunkwise", prf.loc["precision"])
            mlflow.log_metric("weighted-recall-chunkwise", prf.loc["recall"])
            prf = metric_df.loc["macro"]
            mlflow.log_metric("macro-f1-score-chunkwise", prf.loc["f1-score"])
            mlflow.log_metric("macro-precision-chunkwise", prf.loc["precision"])
            mlflow.log_metric("macro-recall-chunkwise", prf.loc["recall"])
            # tag-wise (sklearn_crfsuite)
            crfsuite_report = crfsuite_metrics.flat_classification_report(
                golds, preds, labels=sorted_labels, digits=4
            )
            print(crfsuite_report)
            metric_df = CharCRF.convert_report_df(crfsuite_report)
            prf = metric_df.loc["weighted"]
            mlflow.log_metric("weighted-f1-score-tagwise", prf.loc["f1-score"])
            mlflow.log_metric("weighted-precision-tagwise", prf.loc["precision"])
            mlflow.log_metric("weighted-recall-tagwise", prf.loc["recall"])
            prf = metric_df.loc["macro"]
            mlflow.log_metric("macro-f1-score-tagwise", prf.loc["f1-score"])
            mlflow.log_metric("macro-precision-tagwise", prf.loc["precision"])
            mlflow.log_metric("macro-recall-tagwise", prf.loc["recall"])


if __name__ == "__main__":

    parser = ArgumentParser(description="Char-CRF Token Classifier")

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--data_dir",
        default="workspace/data",
        type=str,
        required=True,
        help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--c1",
        default=1.0,
        type=float,
        help="Weight decay if we apply some.",
    )
    parser.add_argument(
        "--c2",
        default=1.0,
        type=float,
        help="Weight decay if we apply some.",
    )
    parser.add_argument(
        "--window_size",
        default=20,
        type=int,
    )
    parser.add_argument(
        "--max_iterations",
        default=100,
        type=int,
    )
    parser.add_argument(
        "--all_possible_transitions",
        action="store_true",
    )
    parser.add_argument(
        "--num_samples",
        default=10000,
        type=int,
    )
    args = parser.parse_args()
    # args = parser.parse_args(args=[])  # in jupyter notebook

    # sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
    seed_everything(args.seed)
    Path(args.output_dir).mkdir(exist_ok=True)
    file_path = os.path.join(args.data_dir, f"train.txt")
    if not os.path.exists(file_path):
        download_dataset(args.data_dir)
    crf = CharCRF(args)
    crf.do_experiment(args.num_samples)
