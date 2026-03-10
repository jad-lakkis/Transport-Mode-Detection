from pathlib import Path
import random
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
    ConfusionMatrixDisplay,
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


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

WINDOW_SIZE = 5
MAX_WINDOW_SPAN_SECONDS = 3.5

BATCH_SIZE = 64
HIDDEN_SIZE = 64
NUM_LAYERS = 1
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CELLMOB_ROOT = find_cellmob_root()
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = CELLMOB_ROOT / "Data"
KAUST_6400_DIR = DATA_DIR / "6400 KAUST"

OUTPUT_DIR = SCRIPT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
    "walk": KAUST_6400_DIR / "walk_train_kaust_standardized.csv",
    "bus": KAUST_6400_DIR / "bus_train_kaust_standardized.csv",
    "car": KAUST_6400_DIR / "car_train_kaust_standardized.csv",
}

TEST_FILES = {
    "walk": KAUST_6400_DIR / "walk_test_kaust_standardized_6400windows.csv",
    "bus": KAUST_6400_DIR / "bus_test_kaust_standardized_6400windows.csv",
    "car": KAUST_6400_DIR / "car_test_kaust_standardized_6400windows.csv",
}

CLASS_WEIGHTS = torch.tensor([1.0, 11.0, 5.0], dtype=torch.float32)

LOAD_EXISTING_MODEL = False

EXISTING_MODEL_PATH = OUTPUT_DIR / "rnn_kaust_3class_ce_baseline_6400.pth"
NEW_MODEL_PATH = OUTPUT_DIR / "rnn_kaust_3class_ce_baseline_6400_weighted.pth"
PR_PLOT_PATH = OUTPUT_DIR / "pr_curves_ce_baseline_kaust_6400_weighted.png"
CM_PLOT_PATH = OUTPUT_DIR / "confusion_matrix_ce_baseline_kaust_6400_weighted.png"


def time_to_seconds(t):
    s = str(t).strip()

    if s == "" or s.lower() == "nan":
        raise ValueError(f"Invalid time value {t}")

    parts = s.split(":")

    if len(parts) != 3:
        raise ValueError(f"Bad time format {t}")

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


def build_windows_from_file(csv_path, class_name):
    df = pd.read_csv(csv_path, skipinitialspace=True)

    df.columns = [c.strip() for c in df.columns]

    required_cols = [TIME_COLUMN] + FEATURE_COLUMNS

    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(f"{csv_path.name} missing columns {missing}")

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

        span = times[i + WINDOW_SIZE - 1] - times[i]

        if 0 <= span <= MAX_WINDOW_SPAN_SECONDS:
            X.append(features[i:i + WINDOW_SIZE])
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
        "rows": len(df),
        "candidate_windows": total_candidate_windows,
        "kept_windows": kept_windows,
        "rejected_windows": rejected_windows,
    }

    return X, y, stats


def build_dataset(file_dict):
    X_all = []
    y_all = []

    for class_name, path in file_dict.items():
        if not path.exists():
            raise FileNotFoundError(path)

        X, y, stats = build_windows_from_file(path, class_name)

        X_all.append(X)
        y_all.append(y)

        print(
            f"{path.name} | rows={stats['rows']} | "
            f"candidate_windows={stats['candidate_windows']} | "
            f"kept={stats['kept_windows']} | rejected={stats['rejected_windows']}"
        )

    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)

    return X_all, y_all


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
        return self.fc(last_hidden)


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


def plot_pr_curves(pr_curves, save_path):
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


def plot_confusion_matrix(cm, save_path):
    fig, ax = plt.subplots(figsize=(6, 6))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["walk", "bus", "car"]
    )

    disp.plot(ax=ax, cmap="Blues", colorbar=False)

    ax.set_title("Confusion Matrix")

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
        "ap_per_class": ap_per_class,
        "macro_ap": macro_ap,
        "pr_curves": pr_curves,
    }


def main():
    if not KAUST_6400_DIR.exists():
        raise FileNotFoundError(f"Data directory does not exist: {KAUST_6400_DIR}")

    print(f"CellMob root: {CELLMOB_ROOT}")
    print(f"6400 KAUST directory: {KAUST_6400_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")

    print("Building KAUST train dataset")

    X_train, y_train = build_dataset(TRAIN_FILES)

    print("\nBuilding KAUST test dataset")

    X_test, y_test = build_dataset(TEST_FILES)

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

    if LOAD_EXISTING_MODEL and EXISTING_MODEL_PATH.exists():
        model.load_state_dict(torch.load(EXISTING_MODEL_PATH, map_location=DEVICE))
        print(f"Loaded existing model {EXISTING_MODEL_PATH}")

    criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(DEVICE))

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\nUsing class weights {CLASS_WEIGHTS.tolist()}")

    print(f"\nTraining on {DEVICE}")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        print(f"Epoch {epoch:02d}/{NUM_EPOCHS} Train Loss {train_loss:.6f}")

    print("\nEvaluating on test set")

    results = evaluate(model, test_loader)

    print("\n================ RESULTS ================")

    print("Macro AUPRC", results["macro_ap"])

    for cls in ["walk", "bus", "car"]:
        print(cls, results["ap_per_class"][cls])

    print("\nAccuracy", results["accuracy"])

    print("\nConfusion Matrix")

    print(results["confusion_matrix"])

    plot_pr_curves(results["pr_curves"], PR_PLOT_PATH)

    plot_confusion_matrix(results["confusion_matrix"], CM_PLOT_PATH)

    torch.save(model.state_dict(), NEW_MODEL_PATH)

    print("\nModel saved", NEW_MODEL_PATH)


if __name__ == "__main__":
    main()