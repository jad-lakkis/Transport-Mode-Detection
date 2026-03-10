from pathlib import Path
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    average_precision_score,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from libauc.losses import APLoss
from libauc.optimizers import SOAP
from libauc.sampler import DualSampler


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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WINDOW_SIZE = 5
MAX_WINDOW_SPAN_SECONDS = 3.5

BATCH_SIZE = 64
HIDDEN_SIZE = 64
NUM_LAYERS = 1
NUM_EPOCHS = 15

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

SOAP_MARGIN = 1.0
SOAP_GAMMA = 0.9
SOAP_SAMPLING_RATE = 0.5

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
    "car": 1,
}
IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

TRAIN_FILES = {
    "walk": KAUST_6400_DIR / "walk_train_kaust_standardized.csv",
    "car": KAUST_6400_DIR / "car_train_kaust_standardized.csv",
}

TEST_FILES = {
    "walk": KAUST_6400_DIR / "walk_test_kaust_standardized_6400windows.csv",
    "car": KAUST_6400_DIR / "car_test_kaust_standardized_6400windows.csv",
}

SOAP_MODEL_PATH = OUTPUT_DIR / "rnn_car_vs_walk_soap_only.pth"
SOAP_CM_PLOT_PATH = OUTPUT_DIR / "cm_car_vs_walk_soap_only.png"
CM_COMPARE_PLOT_PATH = OUTPUT_DIR / "cm_car_vs_walk_ce_vs_soap.png"

CE_CONFUSION_MATRIX = None


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
    df = pd.read_csv(csv_path, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]

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

    for class_name, path in file_dict.items():
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

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


class IndexedSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.targets = np.array(y, dtype=np.int64)
        self.indices = torch.arange(len(y), dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.indices[idx]


class BinaryRNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity="tanh",
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        last_hidden = out[:, -1, :]
        return self.fc(last_hidden).squeeze(1)


def train_one_epoch_soap(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0.0
    total_seen = 0

    for X_batch, y_batch, index_batch in loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        index_batch = index_batch.to(DEVICE)

        logits = model(X_batch)
        probs = torch.sigmoid(logits)

        loss = loss_fn(probs, y_batch, index_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = X_batch.size(0)
        total_loss += loss.item() * bs
        total_seen += bs

    return total_loss / total_seen


@torch.no_grad()
def evaluate(model, loader, threshold=0.5):
    model.eval()

    all_probs = []
    all_true = []

    for X_batch, y_batch, _ in loader:
        X_batch = X_batch.to(DEVICE)
        logits = model(X_batch)
        probs = torch.sigmoid(logits).cpu().numpy()

        all_probs.append(probs)
        all_true.append(y_batch.numpy())

    y_true = np.concatenate(all_true).astype(int)
    y_prob = np.concatenate(all_probs)
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    ap_car = average_precision_score(y_true, y_prob)

    y_true_walk = 1 - y_true
    y_prob_walk = 1.0 - y_prob
    ap_walk = average_precision_score(y_true_walk, y_prob_walk)

    macro_ap = (ap_car + ap_walk) / 2.0

    return {
        "accuracy": float(acc),
        "confusion_matrix": cm,
        "ap_car": float(ap_car),
        "ap_walk": float(ap_walk),
        "macro_ap": float(macro_ap),
    }


def plot_single_confusion_matrix(cm, title, save_path: Path):
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["walk", "car"]
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_comparison_confusion_matrix(cm_ce, cm_soap, save_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    disp1 = ConfusionMatrixDisplay(
        confusion_matrix=cm_ce,
        display_labels=["walk", "car"]
    )
    disp1.plot(ax=axes[0], cmap="Blues", colorbar=False)
    axes[0].set_title("Cross-Entropy")

    disp2 = ConfusionMatrixDisplay(
        confusion_matrix=cm_soap,
        display_labels=["walk", "car"]
    )
    disp2.plot(ax=axes[1], cmap="Blues", colorbar=False)
    axes[1].set_title("SOAP")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def print_class_distribution(name, y):
    unique, counts = np.unique(y, return_counts=True)
    print(f"\n{name} class distribution:")
    for u, c in zip(unique, counts):
        print(f"  {IDX_TO_LABEL[int(u)]}: {int(c)}")


def main():
    if not KAUST_6400_DIR.exists():
        raise FileNotFoundError(f"Data directory does not exist: {KAUST_6400_DIR}")

    print(f"CellMob root: {CELLMOB_ROOT}")
    print(f"6400 KAUST directory: {KAUST_6400_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")

    print("Building walk/car train dataset...")
    X_train, y_train = build_dataset(TRAIN_FILES)

    print("\nBuilding walk/car test dataset...")
    X_test, y_test = build_dataset(TEST_FILES)

    print("\nFinal shapes:")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_test :", X_test.shape)
    print("y_test :", y_test.shape)

    print_class_distribution("Train", y_train)
    print_class_distribution("Test", y_test)

    train_dataset = IndexedSequenceDataset(X_train, y_train)
    test_dataset = IndexedSequenceDataset(X_test, y_test)

    sampler = DualSampler(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampling_rate=SOAP_SAMPLING_RATE
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = BinaryRNNClassifier(
        input_size=len(FEATURE_COLUMNS),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
    ).to(DEVICE)

    loss_fn = APLoss(
        data_len=len(train_dataset),
        gamma=SOAP_GAMMA,
        margin=SOAP_MARGIN,
        surr_loss="squared_hinge",
    )

    optimizer = SOAP(
        model.parameters(),
        lr=LEARNING_RATE,
        mode="adam",
        weight_decay=WEIGHT_DECAY,
    )

    print(f"\nTraining SOAP from scratch on {DEVICE}...\n")
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch_soap(model, train_loader, optimizer, loss_fn)
        results = evaluate(model, test_loader)

        print(
            f"SOAP Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Test AP(car): {results['ap_car']:.6f} | "
            f"Test Macro AP: {results['macro_ap']:.6f} | "
            f"Test Acc: {results['accuracy']:.6f}"
        )

    final_results = evaluate(model, test_loader)

    print("\n==================== SOAP RESULTS ====================")
    print(f"AP / AUPRC (car positive):  {final_results['ap_car']:.6f}")
    print(f"AP / AUPRC (walk positive): {final_results['ap_walk']:.6f}")
    print(f"Binary Macro AP:            {final_results['macro_ap']:.6f}")
    print(f"Accuracy:                   {final_results['accuracy']:.6f}")
    print("Confusion Matrix [rows=true, cols=pred]:")
    print(final_results["confusion_matrix"])

    torch.save(model.state_dict(), SOAP_MODEL_PATH)
    print(f"\nSaved SOAP model to: {SOAP_MODEL_PATH}")

    plot_single_confusion_matrix(
        final_results["confusion_matrix"],
        "SOAP Confusion Matrix",
        SOAP_CM_PLOT_PATH
    )
    print(f"Saved SOAP confusion matrix to: {SOAP_CM_PLOT_PATH}")

    if CE_CONFUSION_MATRIX is not None:
        plot_comparison_confusion_matrix(
            CE_CONFUSION_MATRIX,
            final_results["confusion_matrix"],
            CM_COMPARE_PLOT_PATH
        )
        print(f"Saved CE vs SOAP comparison to: {CM_COMPARE_PLOT_PATH}")
    else:
        print("\nCE_CONFUSION_MATRIX is still None.")
        print("Paste your CE confusion matrix into CE_CONFUSION_MATRIX to save the comparison image.")


if __name__ == "__main__":
    main()