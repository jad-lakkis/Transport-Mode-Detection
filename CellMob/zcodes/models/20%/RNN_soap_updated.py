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
)
from sklearn.model_selection import train_test_split

try:
    from libauc.losses import APLoss
    from libauc.optimizers import SOAP
    from libauc.sampler import DualSampler
except ImportError as e:
    raise ImportError("LibAUC is not installed. Run: pip install -U libauc") from e


"""
RNN + BCE pretraining, then LibAUC APLoss + SOAP fine-tuning (one-vs-rest)
===========================================================================

What this script does
---------------------
This script applies the recommended LibAUC training flow to your 3-class KAUST task.

Because LibAUC's APLoss + SOAP + DualSampler workflow is binary, we adapt the
3-class problem using one-vs-rest:

- walk vs rest
- bus  vs rest
- car  vs rest

For each class, we do TWO stages:

Stage 1: BCE pretraining
- train a binary RNN scorer with sigmoid + BCELoss
- select the best checkpoint using validation AP for that one-vs-rest task

Stage 2: APLoss + SOAP fine-tuning
- start from the BCE-pretrained checkpoint
- fine-tune with LibAUC's APLoss + SOAP
- again keep the best checkpoint using validation AP

Why this matches the LibAUC recommendation
------------------------------------------
The official LibAUC AUPRC tutorial recommends starting from a pretrained
cross-entropy / BCE checkpoint before the AUPRC-maximization stage, because this
can significantly improve performance. It also uses sigmoid + BCELoss in the
pretraining stage, and then APLoss + SOAP for AUPRC optimization.

What we compare at the end
--------------------------
Main comparison:
- AP / AUPRC for walk vs rest
- AP / AUPRC for bus  vs rest
- AP / AUPRC for car  vs rest
- Macro AUPRC = average of the 3 AP values

Secondary diagnostics:
- multiclass accuracy
- confusion matrix
- TP / FP / FN / TN per class

Important note
--------------
This code does NOT sweep hyperparameters.
It uses your best hyperparameters found from the previous sweep:

- gamma = 0.1
- margin = 0.6
- walk_sampling = 0.80
- bus_sampling  = 0.15
- car_sampling  = 0.25

Also, this script uses the same validation split logic as before and evaluates
the final models on the test set.
"""


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

PRETRAIN_EPOCHS = 20
FINETUNE_EPOCHS = 20

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 2e-4

AP_GAMMA = 0.1
AP_MARGIN = 0.6
AP_SURR_LOSS = "squared_hinge"

PRETRAIN_LR_DECAY_EPOCHS = [10, 15]
FINETUNE_LR_DECAY_EPOCHS = [10, 15]
LR_DECAY_FACTOR = 0.1

VAL_SIZE = 0.15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CELLMOB_ROOT = find_cellmob_root()
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = CELLMOB_ROOT / "Data"
SEPARATED_DATA_DIR = DATA_DIR / "data(raw_but_seperated)"

TRAIN_DIR = SEPARATED_DATA_DIR / "zdata_train"
TEST_DIR = SEPARATED_DATA_DIR / "zdata_test"

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
    "walk": TRAIN_DIR / "walk_kaust_cleaned.csv",
    "bus": TRAIN_DIR / "bus_colored_kaust_cleaned.csv",
    "car": TRAIN_DIR / "car_kaust_cleaned.csv",
}

TEST_FILES = {
    "walk": TEST_DIR / "walk_kaust_cleaned.csv",
    "bus": TEST_DIR / "bus_colored_kaust_cleaned.csv",
    "car": TEST_DIR / "car_kaust_cleaned.csv",
}

BEST_SAMPLING_BY_CLASS = {
    "walk": 0.80,
    "bus": 0.15,
    "car": 0.25,
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


def multiclass_diagnostics_from_ovr(results_by_class, y_true_multiclass):
    class_order = ["walk", "bus", "car"]
    prob_matrix = np.column_stack([results_by_class[c]["probs"] for c in class_order])

    y_pred = np.argmax(prob_matrix, axis=1)
    y_true = y_true_multiclass

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

    return {
        "accuracy": float(acc),
        "confusion_matrix": cm,
        "per_class_counts": per_class_counts,
        "y_pred": y_pred,
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
    pos_count = int(y_binary.sum())
    neg_count = int(len(y_binary) - pos_count)

    print(f"Target class: {class_name}")
    print(f"  Positives in train: {pos_count}")
    print(f"  Negatives in train: {neg_count}")

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

    loss_fn = nn.BCELoss()
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
    pretrain_start = time.time()

    for epoch in range(1, PRETRAIN_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        sample_count = 0

        for X_batch, y_batch, _ in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            logits = model(X_batch)
            probs = torch.sigmoid(logits)
            loss = loss_fn(probs, y_batch)

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

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[BCE] Epoch {epoch:02d}/{PRETRAIN_EPOCHS} | "
            f"Loss: {avg_loss:.6f} | "
            f"Val AP: {val_ap:.6f} | "
            f"Best Val AP: {best_val_ap:.6f} | "
            f"LR: {current_lr:.6f}"
        )

        scheduler.step()

    model.load_state_dict(best_state)
    pretrain_time_seconds = time.time() - pretrain_start

    return model, pretrain_time_seconds, best_val_ap


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
    print(f"Using best hyperparameters:")
    print(f"  gamma          : {AP_GAMMA}")
    print(f"  margin         : {AP_MARGIN}")
    print(f"  sampling_rate  : {sampling_rate}")

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
    finetune_start = time.time()

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

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[APLoss] Epoch {epoch:02d}/{FINETUNE_EPOCHS} | "
            f"Loss: {avg_loss:.6f} | "
            f"Val AP: {val_ap:.6f} | "
            f"Best Val AP: {best_val_ap:.6f} | "
            f"LR: {current_lr:.6f}"
        )

        scheduler.step()

    model.load_state_dict(best_state)
    finetune_time_seconds = time.time() - finetune_start

    return model, finetune_time_seconds, best_val_ap


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
    X_train_full, y_train_full, _ = build_dataset(TRAIN_FILES)

    print("\nBuilding KAUST test dataset...")
    X_test, y_test, _ = build_dataset(TEST_FILES)

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

    print("\nAfter train/validation split:")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_val  :", X_val.shape)
    print("y_val  :", y_val.shape)

    print_class_distribution("Train", y_train)
    print_class_distribution("Validation", y_val)

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

        model_path = OUTPUT_DIR / f"rnn_kaust_pretrain_aploss_{class_name}_vs_rest_best.pth"
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

    total_bce_time = float(sum(bce_times.values()))
    total_aploss_time = float(sum(aploss_times.values()))
    total_time = total_bce_time + total_aploss_time

    print("\n==================== PRETRAIN + APLoss RESULTS ====================")
    print("Primary comparison metrics:")
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

    print("\nSecondary diagnostics:")
    print(f"Accuracy: {diagnostics['accuracy']:.6f}")

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

    print("\nTiming:")
    for class_name in ["walk", "bus", "car"]:
        print(
            f"{class_name:>5} | "
            f"BCE: {bce_times[class_name]:.2f}s | "
            f"APLoss: {aploss_times[class_name]:.2f}s | "
            f"Total: {bce_times[class_name] + aploss_times[class_name]:.2f}s"
        )
    print(f"Total BCE time   : {total_bce_time:.2f} seconds")
    print(f"Total APLoss time: {total_aploss_time:.2f} seconds")
    print(f"Grand total time : {total_time:.2f} seconds")

    pr_plot_path = OUTPUT_DIR / "pr_curves_pretrain_aploss_ovr.png"
    plot_pr_curves(
        results_by_class,
        pr_plot_path,
        title="BCE Pretrain + APLoss Fine-Tune Precision-Recall Curves on Test",
    )
    print(f"\nPR curves saved as: {pr_plot_path}")


if __name__ == "__main__":
    main()