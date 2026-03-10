from pathlib import Path
import random
import time
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split

try:
    from libauc.losses import APLoss
    from libauc.optimizers import SOAP
    from libauc.sampler import DualSampler
except ImportError as e:
    raise ImportError("LibAUC is not installed. Run: pip install -U libauc") from e


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

PRETRAIN_EPOCHS = 10
FINETUNE_EPOCHS = 10

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 2e-4

AP_GAMMA = 0.2
AP_MARGIN = 0.7
AP_SURR_LOSS = "squared_hinge"

PRETRAIN_LR_DECAY_EPOCHS = [5, 8]
FINETUNE_LR_DECAY_EPOCHS = [5, 8]
LR_DECAY_FACTOR = 0.1

VAL_SIZE = 0.15
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

BEST_SAMPLING_BY_CLASS = {
    "walk": 0.80,
    "bus": 0.15,
    "car": 0.30,
}

BCE_POS_WEIGHT_BY_CLASS = {
    "walk": 1.0,
    "bus": 1.0,
    "car": 2.5,
}

LOAD_EXISTING_MODELS = True


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


def print_class_distribution(name, y):
    unique, counts = np.unique(y, return_counts=True)
    print(f"\n{name} class distribution:")
    for u, c in zip(unique, counts):
        print(f"  {IDX_TO_LABEL[int(u)]}: {int(c)}")


class IndexedBinarySequenceDataset(Dataset):
    def __init__(self, X, y_binary):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y_binary, dtype=torch.float32)
        self.targets = np.asarray(y_binary, dtype=np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index], index


class BinaryRNNScorer(nn.Module):
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
        logits = self.fc(last_hidden).squeeze(1)
        return logits


def make_binary_labels(y_multiclass, target_class):
    return (y_multiclass == target_class).astype(np.int64)


def state_dict_to_cpu(state_dict):
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}


@torch.no_grad()
def evaluate_binary_model(model, X_eval, y_eval_multiclass, target_class):
    model.eval()

    probs_all = []
    batch_size = 256

    for start in range(0, len(X_eval), batch_size):
        end = min(start + batch_size, len(X_eval))
        X_batch = torch.tensor(X_eval[start:end], dtype=torch.float32, device=DEVICE)
        logits = model(X_batch)
        probs = torch.sigmoid(logits).cpu().numpy()
        probs_all.append(probs)

    probs = np.concatenate(probs_all, axis=0)
    y_true_bin = (y_eval_multiclass == target_class).astype(np.int64)

    ap = average_precision_score(y_true_bin, probs)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true_bin, probs)

    return {
        "probs": probs,
        "ap": float(ap),
        "precision_curve": precision_curve,
        "recall_curve": recall_curve,
    }


def plot_pr_curves(results_by_class, save_path: Path, title: str):
    plt.figure(figsize=(8, 6))
    for class_name in ["walk", "bus", "car"]:
        entry = results_by_class[class_name]
        plt.plot(
            entry["recall_curve"],
            entry["precision_curve"],
            label=f"{class_name} (AP={entry['ap']:.4f})"
        )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_confusion_matrix(cm, save_path: Path):
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


def multiclass_diagnostics_from_ovr(results_by_class, y_true_multiclass):
    class_order = ["walk", "bus", "car"]
    prob_matrix = np.column_stack([results_by_class[c]["probs"] for c in class_order])

    y_pred = np.argmax(prob_matrix, axis=1)
    y_true = y_true_multiclass

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    total = cm.sum()
    per_class_counts = {}

    for cls_idx in range(3):
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

    return {
        "accuracy": float(acc),
        "confusion_matrix": cm,
        "per_class_counts": per_class_counts,
    }


def pretrain_binary_model_bce(
    X_train,
    y_train_multiclass,
    target_class,
    X_val,
    y_val_multiclass,
):
    class_name = IDX_TO_LABEL[target_class]
    print(f"\n==================== BCE PRETRAIN: {class_name} vs rest ====================")

    y_binary = make_binary_labels(y_train_multiclass, target_class)
    train_dataset = IndexedBinarySequenceDataset(X_train, y_binary)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    model = BinaryRNNScorer(
        input_size=len(FEATURE_COLUMNS),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
    ).to(DEVICE)

    existing_final_path = OUTPUT_DIR / f"rnn_kaust_6400_pretrain_aploss_{class_name}_vs_rest_best.pth"
    existing_bce_path = OUTPUT_DIR / f"rnn_kaust_6400_bce_{class_name}_vs_rest_best.pth"

    if LOAD_EXISTING_MODELS:
        if existing_bce_path.exists():
            model.load_state_dict(torch.load(existing_bce_path, map_location=DEVICE))
            print(f"Loaded previous BCE model: {existing_bce_path}")
        elif existing_final_path.exists():
            model.load_state_dict(torch.load(existing_final_path, map_location=DEVICE))
            print(f"Loaded previous final OvR model: {existing_final_path}")

    pos_weight_value = BCE_POS_WEIGHT_BY_CLASS[class_name]
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=DEVICE)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=PRETRAIN_LR_DECAY_EPOCHS,
        gamma=LR_DECAY_FACTOR,
    )

    best_val_ap = -1.0
    best_state = state_dict_to_cpu(model.state_dict())
    start_time = time.time()

    print(f"Using BCE pos_weight for {class_name}: {pos_weight_value}")

    for epoch in range(1, PRETRAIN_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        sample_count = 0

        for X_batch, y_batch, _ in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_n = X_batch.size(0)
            running_loss += loss.item() * batch_n
            sample_count += batch_n

        avg_loss = running_loss / sample_count

        val_result = evaluate_binary_model(
            model=model,
            X_eval=X_val,
            y_eval_multiclass=y_val_multiclass,
            target_class=target_class,
        )
        val_ap = val_result["ap"]

        if val_ap > best_val_ap:
            best_val_ap = val_ap
            best_state = state_dict_to_cpu(model.state_dict())

        print(
            f"[BCE] Epoch {epoch:02d}/{PRETRAIN_EPOCHS} | "
            f"Loss: {avg_loss:.6f} | Val AP: {val_ap:.6f} | "
            f"Best Val AP: {best_val_ap:.6f}"
        )
        scheduler.step()

    model.load_state_dict(best_state)
    elapsed = time.time() - start_time

    bce_save_path = OUTPUT_DIR / f"rnn_kaust_6400_bce_{class_name}_vs_rest_best.pth"
    torch.save(model.state_dict(), bce_save_path)
    print(f"Saved BCE model: {bce_save_path}")

    return model, elapsed, best_val_ap


def finetune_binary_model_aploss(
    pretrained_model,
    X_train,
    y_train_multiclass,
    target_class,
    X_val,
    y_val_multiclass,
):
    class_name = IDX_TO_LABEL[target_class]
    sampling_rate = BEST_SAMPLING_BY_CLASS[class_name]

    print(f"\n==================== APLoss FINE-TUNE: {class_name} vs rest ====================")
    print(f"gamma={AP_GAMMA}, margin={AP_MARGIN}, sampling_rate={sampling_rate}")

    y_binary = make_binary_labels(y_train_multiclass, target_class)
    train_dataset = IndexedBinarySequenceDataset(X_train, y_binary)

    sampler = DualSampler(
        train_dataset,
        batch_size=BATCH_SIZE,
        labels=train_dataset.targets,
        sampling_rate=sampling_rate,
        random_seed=SEED,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
        shuffle=False,
    )

    model = copy.deepcopy(pretrained_model).to(DEVICE)

    loss_fn = APLoss(
        data_len=len(train_dataset),
        gamma=AP_GAMMA,
        margin=AP_MARGIN,
        surr_loss=AP_SURR_LOSS,
    )

    optimizer = SOAP(
        model.parameters(),
        lr=LEARNING_RATE,
        mode="adam",
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=FINETUNE_LR_DECAY_EPOCHS,
        gamma=LR_DECAY_FACTOR,
    )

    best_val_ap = -1.0
    best_state = state_dict_to_cpu(model.state_dict())
    start_time = time.time()

    for epoch in range(1, FINETUNE_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        sample_count = 0

        for X_batch, y_batch, index in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            index = index.to(DEVICE)

            logits = model(X_batch)
            probs = torch.sigmoid(logits)
            loss = loss_fn(probs, y_batch, index)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_n = X_batch.size(0)
            running_loss += loss.item() * batch_n
            sample_count += batch_n

        avg_loss = running_loss / sample_count

        val_result = evaluate_binary_model(
            model=model,
            X_eval=X_val,
            y_eval_multiclass=y_val_multiclass,
            target_class=target_class,
        )
        val_ap = val_result["ap"]

        if val_ap > best_val_ap:
            best_val_ap = val_ap
            best_state = state_dict_to_cpu(model.state_dict())

        print(
            f"[APLoss] Epoch {epoch:02d}/{FINETUNE_EPOCHS} | "
            f"Loss: {avg_loss:.6f} | Val AP: {val_ap:.6f} | "
            f"Best Val AP: {best_val_ap:.6f}"
        )
        scheduler.step()

    model.load_state_dict(best_state)
    elapsed = time.time() - start_time
    return model, elapsed, best_val_ap


def main():
    if not KAUST_6400_DIR.exists():
        raise FileNotFoundError(f"Data directory does not exist: {KAUST_6400_DIR}")

    print(f"CellMob root: {CELLMOB_ROOT}")
    print(f"6400 KAUST directory: {KAUST_6400_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")

    print("Building KAUST 6400 train dataset...")
    X_train_full, y_train_full = build_dataset(TRAIN_FILES)

    print("\nBuilding KAUST 6400 test dataset...")
    X_test, y_test = build_dataset(TEST_FILES)

    print("\nFinal shapes:")
    print("X_train_full:", X_train_full.shape)
    print("y_train_full:", y_train_full.shape)
    print("X_test      :", X_test.shape)
    print("y_test      :", y_test.shape)

    print_class_distribution("Train full", y_train_full)
    print_class_distribution("Test", y_test)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=VAL_SIZE,
        random_state=SEED,
        stratify=y_train_full,
    )

    models = {}
    bce_times = {}
    aploss_times = {}
    best_bce_val_aps = {}
    best_aploss_val_aps = {}

    for target_class in [0, 1, 2]:
        class_name = IDX_TO_LABEL[target_class]

        pretrained_model, bce_time_seconds, best_bce_val_ap = pretrain_binary_model_bce(
            X_train=X_train,
            y_train_multiclass=y_train,
            target_class=target_class,
            X_val=X_val,
            y_val_multiclass=y_val,
        )

        final_model, aploss_time_seconds, best_aploss_val_ap = finetune_binary_model_aploss(
            pretrained_model=pretrained_model,
            X_train=X_train,
            y_train_multiclass=y_train,
            target_class=target_class,
            X_val=X_val,
            y_val_multiclass=y_val,
        )

        models[class_name] = final_model
        bce_times[class_name] = bce_time_seconds
        aploss_times[class_name] = aploss_time_seconds
        best_bce_val_aps[class_name] = best_bce_val_ap
        best_aploss_val_aps[class_name] = best_aploss_val_ap

        model_path = OUTPUT_DIR / f"rnn_kaust_6400_pretrain_aploss_{class_name}_vs_rest_best_car_penalty.pth"
        torch.save(final_model.state_dict(), model_path)
        print(f"Saved final model: {model_path}")

    results_by_class = {}
    for target_class in [0, 1, 2]:
        class_name = IDX_TO_LABEL[target_class]
        results_by_class[class_name] = evaluate_binary_model(
            model=models[class_name],
            X_eval=X_test,
            y_eval_multiclass=y_test,
            target_class=target_class,
        )

    macro_auprc = float(np.mean([
        results_by_class["walk"]["ap"],
        results_by_class["bus"]["ap"],
        results_by_class["car"]["ap"],
    ]))

    diagnostics = multiclass_diagnostics_from_ovr(results_by_class, y_test)

    print("\n==================== PRETRAIN + APLoss RESULTS ====================")
    print(f"Macro AP / Macro AUPRC: {macro_auprc:.6f}")

    print("\nAP / AUPRC per class (one-vs-rest):")
    for class_name in ["walk", "bus", "car"]:
        print(f"{class_name:>5} | AP: {results_by_class[class_name]['ap']:.6f}")

    print("\nBest validation AP after BCE pretraining:")
    for class_name in ["walk", "bus", "car"]:
        print(f"{class_name:>5} | BCE Val AP: {best_bce_val_aps[class_name]:.6f}")

    print("\nBest validation AP after APLoss fine-tuning:")
    for class_name in ["walk", "bus", "car"]:
        print(f"{class_name:>5} | APLoss Val AP: {best_aploss_val_aps[class_name]:.6f}")

    print("\nAccuracy:")
    print(f"{diagnostics['accuracy']:.6f}")

    print("\nConfusion Matrix [rows=true, cols=pred]:")
    print(diagnostics["confusion_matrix"])

    print("\nTP / FP / FN / TN per class:")
    for class_name, counts in diagnostics["per_class_counts"].items():
        print(
            f"{class_name:>5} | "
            f"TP={counts['TP']} | FP={counts['FP']} | "
            f"FN={counts['FN']} | TN={counts['TN']} | "
            f"Support={counts['Support']}"
        )

    pr_plot_path = OUTPUT_DIR / "pr_curves_pretrain_aploss_ovr_kaust_6400_car_penalty.png"
    plot_pr_curves(
        results_by_class,
        pr_plot_path,
        title="BCE Pretrain + APLoss Fine-Tune PR Curves on KAUST 6400 Test (Car Penalty)",
    )
    print(f"\nPR curves saved as: {pr_plot_path}")

    cm_plot_path = OUTPUT_DIR / "confusion_matrix_pretrain_aploss_kaust_6400_car_penalty.png"
    plot_confusion_matrix(diagnostics["confusion_matrix"], cm_plot_path)
    print(f"Confusion matrix saved as: {cm_plot_path}")


if __name__ == "__main__":
    main()