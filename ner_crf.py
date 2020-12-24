import logging
import os
import random
import tempfile
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import jsonlines
import mlflow.pytorch
import numpy as np
import pandas as pd
import requests
import sklearn_crfsuite

from seqeval.metrics import classification_report as seqeval_classification_report

from seqeval.scheme import BILOU
from sklearn_crfsuite import metrics as crfsuite_metrics


logger = logging.getLogger(__name__)

IntList = List[int]
IntListList = List[IntList]
StrList = List[str]
StrListList = List[StrList]

def seed_everything(seed: int = 1234):
    """乱数固定
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


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


class CRF:
    def __init__(self, hparams: Union[Dict, Namespace]):

        if isinstance(hparams, Dict):
            hparams = Namespace(**hparams)

        self.output_dir = Path(hparams.output_dir)
        self.hparams = hparams

        self.model = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=hparams.c1,
            c2=hparams.c2,
            max_iterations=hparams.max_iterations,
            all_possible_transitions=hparams.all_possible_transitions,
        )

    @staticmethod
    def word2features(sent: StrList, i: int) -> Dict:
        word = sent[i]

        features = {
            'bias': 1.0,
            'word.lower()': word.lower(),
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
        }
        if i > 0:
            word1 = sent[i-1][0]
            features.update({
                '-1:word.lower()': word1.lower(),
                '-1:word.istitle()': word1.istitle(),
                '-1:word.isupper()': word1.isupper(),
            })
        else:
            features['BOS'] = True

        if i < len(sent)-1:
            word1 = sent[i+1][0]
            features.update({
                '+1:word.lower()': word1.lower(),
                '+1:word.istitle()': word1.istitle(),
                '+1:word.isupper()': word1.isupper(),
            })
        else:
            features['EOS'] = True

        return features

    def sentence_features(self, tokens: StrList) -> List[Dict]:
        return [self.word2features(tokens, i) for i in range(len(tokens))]

    def fit(self, tokens: StrListList, y: StrListList):
        X: List[List[Dict]] = list(map(self.sentence_features, tokens))
        self.model.fit(X, y)

    def predict(self, tokens: StrListList) -> StrListList:
        X: List[List[Dict]] = list(map(self.sentence_features, tokens))
        return self.model.predict(X)

    @staticmethod
    def convert_report_df(report_str: str) -> pd.DataFrame:
        lines = list(filter(None, report_str.split('\n')))
        columns = list(filter(None, lines[0].split(' ')))
        metrics = [list(filter(None, ls.split(' '))) for ls in lines[1:]]
        each_metrics = [ls for ls in metrics if ls[0] not in {'accuracy', 'macro', 'weighted', 'micro'}]
        macro_metrics = [['macro'] + ls[2:] for ls in metrics if ls[0] in {'macro'}]
        micro_metrics = [['weighted'] + ls[2:] for ls in metrics if ls[0] == 'weighted']
        all_metrics = each_metrics+macro_metrics+micro_metrics
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
        n_train = -1,
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
        val_examples = read_examples_from_file(self.hparams.data_dir, Split.dev)
        test_examples = read_examples_from_file(self.hparams.data_dir, Split.test)
        if n_train > 0:
            n_sample = min(n_train, len(train_examples))
            train_examples = random.sample(train_examples, n_sample)
        datasize = len(train_examples)

        # MLflow Trackingの初期化
        
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
            with jsonlines.open(train_path, 'w') as writer:
                train_jl = [asdict(ex) for ex in train_examples]
                writer.write_all(train_jl)
            with jsonlines.open(val_path, 'w') as writer:
                val_jl = [asdict(ex) for ex in val_examples]
                writer.write_all(val_jl)
            with jsonlines.open(test_path, 'w') as writer:
                test_jl = [asdict(ex) for ex in test_examples]
                writer.write_all(test_jl)
            mlflow.log_artifact(train_path)
            mlflow.log_artifact(val_path)
            mlflow.log_artifact(test_path)
            mlflow.log_metric("train_count", len(train_examples))
            mlflow.log_metric("val_count", len(val_examples))
            mlflow.log_metric("test_count", len(test_examples))
        
            # train
            tokens = [ex.words for ex in train_examples]
            labels = [ex.labels for ex in train_examples]
            self.fit(tokens, labels)
            # params, metrics, tags, artifacts = self.fetch_logged_data(run.info.run_id)
            # mlflow.sklearn.log_model(self.model, "model")

            # test
            tokens = [ex.words for ex in test_examples]
            golds = [ex.labels for ex in test_examples]
            sorted_labels = sorted(set([l for ls in golds for l in ls]))
            preds = self.predict(tokens)
            # chunk-wise (seqeval)
            seqeval_report = seqeval_classification_report(
                golds, preds, digits=4, mode='strict', scheme=BILOU
            )
            print(seqeval_report)
            metric_df = CRF.convert_report_df(seqeval_report)
            prf = metric_df.loc['weighted']
            mlflow.log_metric("weighted-f1-score-chunkwise", prf.loc['f1-score'])
            mlflow.log_metric("weighted-precision-chunkwise", prf.loc['precision'])
            mlflow.log_metric("weighted-recall-chunkwise", prf.loc['recall'])
            prf = metric_df.loc['macro']
            mlflow.log_metric("macro-f1-score-chunkwise", prf.loc['f1-score'])
            mlflow.log_metric("macro-precision-chunkwise", prf.loc['precision'])
            mlflow.log_metric("macro-recall-chunkwise", prf.loc['recall'])
            # tag-wise (sklearn_crfsuite)
            crfsuite_report = crfsuite_metrics.flat_classification_report(
                golds, preds, labels=sorted_labels, digits=4
            )
            print(crfsuite_report)
            metric_df = CRF.convert_report_df(crfsuite_report)
            prf = metric_df.loc['weighted']
            mlflow.log_metric("weighted-f1-score-tagwise", prf.loc['f1-score'])
            mlflow.log_metric("weighted-precision-tagwise", prf.loc['precision'])
            mlflow.log_metric("weighted-recall-tagwise", prf.loc['recall'])
            prf = metric_df.loc['macro']
            mlflow.log_metric("macro-f1-score-tagwise", prf.loc['f1-score'])
            mlflow.log_metric("macro-precision-tagwise", prf.loc['precision'])
            mlflow.log_metric("macro-recall-tagwise", prf.loc['recall'])


if __name__ == "__main__":

    parser = ArgumentParser(description="CRF Token Classifier")

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
    crf = CRF(args)
    crf.do_experiment(args.num_samples)
