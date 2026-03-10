"""
RNN baseline with Cross-Entropy training, evaluated using AP / AUPRC
====================================================================

What this script does
---------------------
This script keeps the SAME pipeline as before:
- same train/test files
- same window construction
- same RNN model
- same CrossEntropyLoss
- same Adam optimizer
- same training loop

The main change is in the EVALUATION stage.

Why we are changing the evaluation
----------------------------------
Our data is class-imbalanced. In such settings, accuracy can look good even if the
model is not doing well on the more difficult / less represented class. F1 can also
be useful, but it is not the main quantity we want to compare for this specific study.

The comparison we want to make later is:
1) standard Cross-Entropy training
vs
2) AUPRC-oriented training

So the evaluation metric should match that goal.

For this reason, we compute:
- AP / AUPRC for each class using one-vs-rest
- Macro AUPRC = average AP across the 3 classes

Why one-vs-rest?
----------------
The paper's formal setup is binary classification, but our task has 3 classes:
- walk
- bus
- car

So we adapt the evaluation as follows:
- walk vs rest
- bus vs rest
- car vs rest

Then we average the three AP values to get a macro AUPRC score.

What will indicate the better method later?
-------------------------------------------
When we later train the AUPRC-oriented approach, we will compare:
- AP/AUPRC for walk
- AP/AUPRC for bus
- AP/AUPRC for car
- Macro AUPRC

The better method is the one that gives higher Macro AUPRC on the SAME test set,
with the SAME model/data split.

What we still keep as secondary statistics
------------------------------------------
Even though AUPRC is the main comparison metric, we still print:
- accuracy
- confusion matrix
- TP / FP / FN / TN per class
- support per class

These are useful diagnostics, but they are NOT the main comparison metric for the
Cross-Entropy baseline vs the later AUPRC-oriented method.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    average_precision_score,
    precision_recall_curve,
)


def find_cellmob_root() -> Path:
    current = Path(__file__).resolve().parent

    for candidate in [current] + list(current.parents):
        if candidate.name == "CellMob" and (candidate / "Data").exists():
            return candidate

    raise FileNotFoundError(
        "Could not locate the 'CellMob' root folder containing the 'Data' directory. "
        "Make sure this script is stored somewhere inside the CellMob project."
    )


CELLMOB_ROOT = find_cellmob_root()
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = CELLMOB_ROOT / "Data"
SEPARATED_DATA_DIR = DATA_DIR / "data(raw_but_seperated)"

TRAIN_DIR = SEPARATED_DATA_DIR / "zdata_train"
TEST_DIR = SEPARATED_DATA_DIR / "zdata_test"

OUTPUT_DIR = SCRIPT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 5
MAX_WINDOW_SPAN_SECONDS = 3.5
BATCH_SIZE = 64
HIDDEN_SIZE = 64
NUM_LAYERS = 1
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TIME_COLUMN = "time"
FEATURE_COLUMNS = [
    "rsrp1", "rsrp2", "rsrp3", "rsrp4",
    "rssi1", "rssi2", "rssi3", "rssi4",
    "rsrq1", "rsrq2", "rsrq3", "rsrq4",
]

LABEL_MAP = {
    "walk": 0,
    "bus": 1,
    "car": 2,
}
IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

TRAIN_FILES = {
    "walk": TRAIN_DIR / "walk_kaust_cleaned.csv",
    "bus": TRAIN_DIR / "bus_colored_kaust_cleaned.csv",
    "car": TRAIN_DIR / "car_kaust_cleaned.csv",
}

TEST_FILES = {
    "walk": TEST_DIR / "walk_kaust_cleaned.csv",
    "bus": TEST_DIR / "bus_colored_kaust_cleaned.csv",
    "car": TEST_DIR / "car_kaust_cleaned.csv",
}


def time_to_seconds(t) -> float:
    s = str(t).strip()

    if s == "" or s.lower() == "nan":
        raise ValueError(f"Invalid time value: {t!r}")

    parts = s.split(":")
    if len(parts) != 3:
        raise ValueError(f"Bad time format: {t!r}")

    hh = int(parts[0])
    mm = int(parts[1])

    sec_part = parts[2]
    if "." in sec_part:
        ss_str, frac_str = sec_part.split(".", 1)
        ss = int(ss_str)
        micro = int(frac_str.ljust(6, "0")[:6])
    else:
        ss = int(sec_part)
        micro = 0

    return hh * 3600 + mm * 60 + ss + micro / 1_000_000


def build_windows_from_file(csv_path: Path, class_name: str):
    df = pd.read_csv(csv_path)

    required_cols = [TIME_COLUMN] + FEATURE_COLUMNS
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path.name} is missing columns: {missing}")

    df = df.dropna(subset=required_cols).reset_index(drop=True)

    df[TIME_COLUMN] = df[TIME_COLUMN].astype(str).str.strip()
    df["time_seconds"] = df[TIME_COLUMN].apply(time_to_seconds)

    features = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    times = df["time_seconds"].to_numpy(dtype=np.float64)

    X = []
    y = []

    total_candidate_windows = 0
    kept_windows = 0
    rejected_windows = 0

    for i in range(len(df) - WINDOW_SIZE + 1):
        total_candidate_windows += 1

        start_time = times[i]
        end_time = times[i + WINDOW_SIZE - 1]
        span = end_time - start_time

        if span <= MAX_WINDOW_SPAN_SECONDS:
            window = features[i:i + WINDOW_SIZE]
            X.append(window)
            y.append(LABEL_MAP[class_name])
            kept_windows += 1
        else:
            rejected_windows += 1

    if len(X) == 0:
        X = np.empty((0, WINDOW_SIZE, len(FEATURE_COLUMNS)), dtype=np.float32)
        y = np.empty((0,), dtype=np.int64)
    else:
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int64)

    stats = {
        "file": csv_path.name,
        "class_name": class_name,
        "rows": len(df),
        "candidate_windows": total_candidate_windows,
        "kept_windows": kept_windows,
        "rejected_windows": rejected_windows,
    }

    return X, y, stats


def build_dataset(file_dict):
    X_all = []
    y_all = []
    all_stats = []

    for class_name, path in file_dict.items():
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        X, y, stats = build_windows_from_file(path, class_name)
        X_all.append(X)
        y_all.append(y)
        all_stats.append(stats)

        print(
            f"{path.name} | rows={stats['rows']} | "
            f"candidate_windows={stats['candidate_windows']} | "
            f"kept={stats['kept_windows']} | rejected={stats['rejected_windows']}"
        )

    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)

    return X_all, y_all, all_stats


class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity="tanh",
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.rnn(x)
        last_hidden = out[:, -1, :]
        logits = self.fc(last_hidden)
        return logits


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)

    return total_loss / len(loader.dataset)


def plot_pr_curves(pr_curves, save_path: Path):
    plt.figure(figsize=(8, 6))

    for class_name, curve_data in pr_curves.items():
        plt.plot(
            curve_data["recall"],
            curve_data["precision"],
            label=f"{class_name} (AP={curve_data['ap']:.4f})"
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves (one-vs-rest)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


@torch.no_grad()
def evaluate(model, loader):
    model.eval()

    all_logits = []
    all_preds = []
    all_true = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        logits = model(X_batch)

        preds = torch.argmax(logits, dim=1).cpu().numpy()

        all_logits.append(logits.cpu().numpy())
        all_preds.extend(preds)
        all_true.extend(y_batch.numpy())

    y_true = np.array(all_true)
    y_pred = np.array(all_preds)
    logits_all = np.vstack(all_logits)

    probs = torch.softmax(torch.tensor(logits_all), dim=1).numpy()

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    total = cm.sum()
    per_class_counts = {}

    for cls_idx in range(cm.shape[0]):
        tp = cm[cls_idx, cls_idx]
        fp = cm[:, cls_idx].sum() - tp
        fn = cm[cls_idx, :].sum() - tp
        tn = total - tp - fp - fn

        per_class_counts[IDX_TO_LABEL[cls_idx]] = {
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "TN": int(tn),
            "Support": int(cm[cls_idx, :].sum()),
        }

    ap_per_class = {}
    pr_curves = {}

    for cls_idx in [0, 1, 2]:
        y_true_bin = (y_true == cls_idx).astype(int)
        y_score = probs[:, cls_idx]

        ap = average_precision_score(y_true_bin, y_score)
        precision_curve, recall_curve, _ = precision_recall_curve(y_true_bin, y_score)

        class_name = IDX_TO_LABEL[cls_idx]
        ap_per_class[class_name] = float(ap)
        pr_curves[class_name] = {
            "precision": precision_curve,
            "recall": recall_curve,
            "ap": float(ap),
        }

    macro_ap = float(np.mean(list(ap_per_class.values())))

    return {
        "accuracy": float(acc),
        "confusion_matrix": cm,
        "per_class_counts": per_class_counts,
        "ap_per_class": ap_per_class,
        "macro_ap": macro_ap,
        "pr_curves": pr_curves,
    }


def print_class_distribution(name, y):
    unique, counts = np.unique(y, return_counts=True)
    print(f"\n{name} class distribution:")
    for u, c in zip(unique, counts):
        print(f"  {IDX_TO_LABEL[int(u)]}: {int(c)}")


def main():
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"Train directory does not exist: {TRAIN_DIR}")
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Test directory does not exist: {TEST_DIR}")

    print(f"CellMob root: {CELLMOB_ROOT}")
    print(f"Train directory: {TRAIN_DIR}")
    print(f"Test directory: {TEST_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")

    print("Building KAUST train dataset...")
    X_train, y_train, _ = build_dataset(TRAIN_FILES)

    print("\nBuilding KAUST test dataset...")
    X_test, y_test, _ = build_dataset(TEST_FILES)

    print("\nFinal shapes:")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_test :", X_test.shape)
    print("y_test :", y_test.shape)

    print_class_distribution("Train", y_train)
    print_class_distribution("Test", y_test)

    train_dataset = SequenceDataset(X_train, y_train)
    test_dataset = SequenceDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = RNNClassifier(
        input_size=len(FEATURE_COLUMNS),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=3,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\nTraining on {DEVICE}...\n")
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        print(f"Epoch {epoch:02d}/{NUM_EPOCHS} - Train Loss: {train_loss:.6f}")

    print("\nEvaluating on test set...")
    results = evaluate(model, test_loader)

    print("\n==================== RESULTS ====================")
    print("Primary comparison metrics for later CE vs AUPRC-method comparison:")
    print(f"Macro AP / Macro AUPRC: {results['macro_ap']:.6f}")

    print("\nAP / AUPRC per class (one-vs-rest):")
    for class_name in ["walk", "bus", "car"]:
        print(f"{class_name:>5} | AP: {results['ap_per_class'][class_name]:.6f}")

    print("\nSecondary diagnostic statistics:")
    print(f"Accuracy: {results['accuracy']:.6f}")

    print("\nConfusion Matrix [rows=true, cols=pred]:")
    print(results["confusion_matrix"])

    print("\nTP / FP / FN / TN per class:")
    for class_name, counts in results["per_class_counts"].items():
        print(
            f"{class_name:>5} | "
            f"TP={counts['TP']} | FP={counts['FP']} | "
            f"FN={counts['FN']} | TN={counts['TN']} | "
            f"Support={counts['Support']}"
        )

    pr_plot_path = OUTPUT_DIR / "pr_curves_ce_baseline.png"
    plot_pr_curves(results["pr_curves"], pr_plot_path)
    print(f"\nPR curves saved as: {pr_plot_path}")

    model_path = OUTPUT_DIR / "rnn_kaust_3class_ce_baseline.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved as: {model_path}")


if __name__ == "__main__":
    main()