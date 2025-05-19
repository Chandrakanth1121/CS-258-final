import argparse
import os
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm
import arff
import re

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, SimpleRNN, Dense
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping



THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR / "data"

# feature‑selection thresholds from Kasongo (Table 4 & Table 8)
THRESH_NSL = 0.0017
THRESH_UNSW = 0.0005

RNN_HP = {
    "lstm": dict(units=150, optimizer="sgd", lr=0.05, epochs=600, batch=128),
    "simplernn": dict(units=60, optimizer="sgd", lr=0.001, epochs=600, batch=128),
}

# classic ML hyper‑parameters (Kilincer Table 4)
ML_HP = {
    "dt": dict(min_samples_leaf=1, random_state=42),
    "knn": dict(n_neighbors=1, metric="euclidean"),
    "svm": dict(kernel="poly", degree=3, C=1, probability=True),
}

# NSL‑KDD categorical columns
NSL_CAT_COLS = [1, 2, 3]    # protocol_type, service, flag
# UNSW‑NB15 categorical columns
UNSW_CAT_COLS = [2, 3, 4]   # proto, service, state

# mapping of NSL attack names → 5‑class grouping (Kasongo Table 2)
NSL_5CLASS_MAP: Dict[str, str] = {
    # normal
    "normal": "normal",
    # DoS
    "back": "dos", "land": "dos", "neptune": "dos", "pod": "dos", "smurf": "dos",
    "teardrop": "dos", "mailbomb": "dos", "processtable": "dos", "udpstorm": "dos",
    # Probe
    "ipsweep": "probe", "nmap": "probe", "portsweep": "probe",
    "saint": "probe", "mscan": "probe",
    # R2L
    "ftp_write": "r2l", "guess_passwd": "r2l", "imap": "r2l", "multihop": "r2l",
    "phf": "r2l", "spy": "r2l", "warezclient": "r2l", "warezmaster": "r2l",
    "sendmail": "r2l", "named": "r2l", "snmpgetattack": "r2l",
    "snmpguess": "r2l", "xlock": "r2l", "xsnoop": "r2l",
    # U2R
    "buffer_overflow": "u2r", "rootkit": "u2r", "loadmodule": "u2r",
    "perl": "u2r", "sqlattack": "u2r", "xterm": "u2r", "ps": "u2r",
}


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    print(f"Downloading {url.split('/')[-1]} …")
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f, tqdm(
        total=int(r.headers["Content-Length"] or 0), unit="B", unit_scale=True
    ) as bar:
        while True:
            chunk = r.read(1024 * 32)
            if not chunk:
                break
            f.write(chunk)
            bar.update(len(chunk))


def load_nsl_kdd() -> pd.DataFrame:
    base = "data/NSL/"
    arff_path = base + "KDDTrain+.arff"
    text = open(arff_path, "r").read()

    def _fix_enum(m):
        tokens = [tok.strip(" '") for tok in m.group(1).split(",")]
        return "{" + ",".join(tokens) + "}"
    text = re.sub(r"\{([^}]*)\}", _fix_enum, text)

    meta = arff.loads(text)     
    feature_cols = [attr[0] for attr in meta["attributes"]][:-1]  # drop 'difficulty'
    columns = feature_cols + ["label", "difficulty"]

    train_df = pd.read_csv(base + "KDDTrain+.txt",
                           names=columns, header=None, index_col=False)
    test_df  = pd.read_csv(base + "KDDTest+.txt",
                           names=columns, header=None, index_col=False)

    df = pd.concat([train_df, test_df], ignore_index=True)

    return df


def load_unsw_nb15() -> pd.DataFrame:
    csv_train = DATA_DIR / "UNSW" / "UNSW_NB15_training-set.csv"
    csv_test = DATA_DIR / "UNSW" / "UNSW_NB15_testing-set.csv"
    train_df = pd.read_csv(csv_train)
    test_df = pd.read_csv(csv_test)
    df = pd.concat([train_df, test_df], ignore_index=True)
    return df


def preprocess(
    df: pd.DataFrame, dataset: str, task: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if dataset == "nsl":
        cat_cols_idx = NSL_CAT_COLS
        threshold = THRESH_NSL
        # binary label
        if task == "binary":
            df["y"] = (df["label"] != "normal").astype(int)
        else:  # 5‑class
            df["y"] = df["label"].map(NSL_5CLASS_MAP)
            df = df[df["y"].notna()]
    elif dataset == "unsw":
        cat_cols_idx = UNSW_CAT_COLS
        threshold = THRESH_UNSW
        if task == "binary":
            df["y"] = (df["attack_cat"] != "Normal").astype(int)
        else:  # 10‑class (original categories)
            df["y"] = df["attack_cat"].str.lower()
            df = df[df["y"].notna()]
    else:
        raise ValueError(dataset)

    if df["y"].dtype == object:
        df["y"] = LabelEncoder().fit_transform(df["y"])

    # drop non‑feature columns
    if dataset == "nsl":
        feature_df = df.drop(columns=["label", "difficulty", "y"])
    else:
        feature_df = df.drop(columns=["attack_cat", "label", "y"])

    # encode categoricals
    le = LabelEncoder()
    for col_idx in cat_cols_idx:
        col = feature_df.columns[col_idx]
        feature_df[col] = le.fit_transform(feature_df[col])

    # scale
    scaler = MinMaxScaler()
    feature_df.iloc[:, :] = scaler.fit_transform(feature_df)

    # XGBoost feature selection
    selector = XGBClassifier(
        n_estimators=400, learning_rate=0.1, max_depth=6, subsample=0.8, n_jobs=-1
    )
    selector.fit(feature_df.values, df["y"].values)
    keep_mask = selector.feature_importances_ >= threshold
    X_reduced = feature_df.values[:, keep_mask]

    print(df["y"])

    # split sets exactly as Kasongo (75 / 25 on entire dataset)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_reduced, df["y"].values, test_size=0.20, random_state=1337, stratify=df["y"]
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.25, random_state=1337, stratify=y_train_full
    )
    return X_train, X_val, X_test, y_train, y_val, y_test



def build_rnn(input_dim: int, model: str, dataset: str, task: str):
    cfg = RNN_HP[model]
    # units may be scalar or dict keyed by dataset
    units = cfg["units"][dataset] if isinstance(cfg["units"], dict) else cfg["units"]

    net = Sequential(name=f"{model.upper()}_{task.upper()}_{dataset.upper()}")
    Cell = {"lstm": LSTM, "simplernn": SimpleRNN}[model]
    net.add(Cell(units, return_sequences=True, input_shape=(1, input_dim)))
    net.add(Cell(units, return_sequences=True))
    net.add(Cell(units))
    out_units = 1 if task == "binary" else (5 if dataset == "nsl" else 10)
    activation = "sigmoid" if task == "binary" else "softmax"
    net.add(Dense(out_units, activation=activation))

    # optimiser
    if cfg["optimizer"] == "sgd":
        opt = SGD(learning_rate=cfg["lr"], momentum=0.9)
    else:
        opt = Adam(learning_rate=cfg["lr"])
    loss = "binary_crossentropy" if task == "binary" else "categorical_crossentropy"
    net.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
    return net, cfg


def build_ml(model: str):
    if model == "dt":
        return DecisionTreeClassifier(**ML_HP["dt"])
    elif model == "knn":
        return KNeighborsClassifier(**ML_HP["knn"])
    elif model == "svm":
        return SVC(**ML_HP["svm"])
    else:
        raise ValueError(model)


class Experiment:
    def __init__(self, dataset: str, task: str):
        self.dataset = dataset
        self.task = task
        if dataset == "nsl":
            self.df = load_nsl_kdd()
        elif dataset == "unsw":
            self.df = load_unsw_nb15()
        else:
            raise ValueError(dataset)
        (
            self.X_train,
            self.X_val,
            self.X_test,
            self.y_train,
            self.y_val,
            self.y_test,
        ) = preprocess(self.df, dataset, task)

    def train_rnn(self, model: str):
        net, cfg = build_rnn(
            input_dim=self.X_train.shape[1],
            model=model,
            dataset=self.dataset,
            task=self.task,
        )
        # reshape to (N, 1, features)
        X_train_rnn = self.X_train[:, np.newaxis, :]
        X_val_rnn = self.X_val[:, np.newaxis, :]
        y_train = (
            tf.keras.utils.to_categorical(self.y_train)
            if self.task != "binary"
            else self.y_train
        )
        y_val = (
            tf.keras.utils.to_categorical(self.y_val)
            if self.task != "binary"
            else self.y_val
        )

        early = EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True, verbose=1
        )
        net.fit(
            X_train_rnn,
            y_train,
            validation_data=(X_val_rnn, y_val),
            epochs=cfg["epochs"],
            batch_size=cfg["batch"],
            callbacks=[early],
            verbose=2,
        )
        self.model = net
        self.model_name = model
        return self

    def train_ml(self, model: str):
        clf = build_ml(model)
        clf.fit(self.X_train, self.y_train)
        self.model = clf
        self.model_name = model
        return self

    def evaluate(self, average: str = "weighted",report_path: str | None = None) -> Dict[str, float]:
        if isinstance(self.model, Sequential):
            # RNN
            X_test = self.X_test[:, np.newaxis, :]
            y_pred_prob = self.model.predict(X_test, verbose=0)
            if self.task == "binary":
                y_pred = (y_pred_prob.flatten() >= 0.5).astype(int)
                roc = roc_auc_score(self.y_test, y_pred_prob)
            else:
                y_pred = np.argmax(y_pred_prob, axis=1)
                roc = np.nan
        else:
            # classic ML
            if hasattr(self.model, "predict_proba"):
                y_pred_prob = self.model.predict_proba(self.X_test)
            else:
                y_pred_prob = None
            y_pred = self.model.predict(self.X_test)
            roc = (
                roc_auc_score(self.y_test, y_pred_prob[:, 1])
                if y_pred_prob is not None and self.task == "binary"
                else np.nan
            )

        acc = accuracy_score(self.y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            self.y_test, y_pred, average=average, zero_division=0
        )
        cm = confusion_matrix(self.y_test, y_pred)
        metrics = dict(accuracy=acc, precision=prec, recall=rec, f1=f1, roc_auc=roc)
        print("── metrics ─────────────────────────────────")
        for k, v in metrics.items():
            print(f"{k:9s}: {v:.4f}")
        print("Confusion matrix:\n", cm)

        if report_path:
            os.makedirs(Path(report_path).parent, exist_ok=True)
            with open(report_path, "w") as f:
                f.write(f"Intrusion-replication report:\n")
                f.write(f"Dataset : {self.dataset}\n")
                f.write(f"Task    : {self.task}\n")
                f.write(f"Model   : {self.model_name}\n\n")
                for k, v in metrics.items():
                    f.write(f"{k:9s}: {v:.4f}\n")
                f.write("\nConfusion matrix:\n")
                f.write(np.array2string(cm, separator=" ", formatter={"int": lambda x: f"{x:4d}"}))
                f.write("\n")
            print(f"Report saved to {report_path}")
        return metrics


def cli():
    p = argparse.ArgumentParser(
        description="Intrusion‑detection replication of Kasongo & Kilincer papers"
    )
    p.add_argument(
        "--dataset",
        choices=["nsl", "unsw"],
        required=True,
        help="Which dataset to run on",
    )
    p.add_argument(
        "--task",
        choices=["binary", "multiclass"],
        default="binary",
        help="Binary vs multi‑class classification",
    )
    p.add_argument(
        "--model",
        choices=["lstm", "simplernn", "dt", "knn", "svm"],
        required=True,
        help="Which model to train",
    )
    p.add_argument(
        "--report",
        metavar="FILE",
        help="If given, save a txt report with metrics and the confusion "
             "matrix to this path.",
    )
    args = p.parse_args()

    exp = Experiment(dataset=args.dataset, task=args.task)
    if args.model in ("lstm", "simplernn"):
        exp.train_rnn(args.model)
    else:
        exp.train_ml(args.model)
    exp.evaluate(report_path=args.report)


if __name__ == "__main__":
    cli()

