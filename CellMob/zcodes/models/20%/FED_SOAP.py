from pathlib import Path
from copy import deepcopy
import random
import time

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
    precision_recall_fscore_support,
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


# =========================================================
# Reproducibility
# =========================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# =========================================================
# Paths
# =========================================================
CELLMOB_ROOT = find_cellmob_root()
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = CELLMOB_ROOT / "Data"
SEPARATED_DATA_DIR = DATA_DIR / "data(raw_but_seperated)"

TRAIN_DIR = SEPARATED_DATA_DIR / "zdata_train"
TEST_DIR = SEPARATED_DATA_DIR / "zdata_test"

OUTPUT_ROOT = SCRIPT_DIR / "outputs_federated_models"
RUN_OUTPUT_DIR = OUTPUT_ROOT / "fedavg_soap_kaust_jeddah_mekkah_ovr"
RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# Data / Model / Training Hyperparameters
# =========================================================
WINDOW_SIZE = 5
MAX_WINDOW_SPAN_SECONDS = 3.5

BATCH_SIZE = 64
HIDDEN_SIZE = 64
NUM_LAYERS = 1

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 2e-4

GLOBAL_ROUNDS_BCE = 20
GLOBAL_ROUNDS_AP = 20
LOCAL_EPOCHS = 1

AP_GAMMA = 0.2
AP_MARGIN = 0.7
AP_SURR_LOSS = "squared_hinge"

VAL_SIZE = 0.15

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

CITY_FILES = {
    "kaust": {
        "train": {
            "walk": TRAIN_DIR / "walk_kaust_cleaned.csv",
            "bus": TRAIN_DIR / "bus_colored_kaust_cleaned.csv",
            "car": TRAIN_DIR / "car_kaust_cleaned.csv",
        },
        "test": {
            "walk": TEST_DIR / "walk_kaust_cleaned.csv",
            "bus": TEST_DIR / "bus_colored_kaust_cleaned.csv",
            "car": TEST_DIR / "car_kaust_cleaned.csv",
        },
    },
    "jeddah": {
        "train": {
            "walk": TRAIN_DIR / "walk_jeddah_cleaned.csv",
            "bus": TRAIN_DIR / "bus_jeddah_cleaned.csv",
            "car": TRAIN_DIR / "car_jeddah_cleaned.csv",
        },
        "test": {
            "walk": TEST_DIR / "walk_jeddah_cleaned.csv",
            "bus": TEST_DIR / "bus_jeddah_cleaned.csv",
            "car": TEST_DIR / "car_jeddah_cleaned.csv",
        },
    },
    "mekkah": {
        "train": {
            "walk": TRAIN_DIR / "walk_mekkah_cleaned.csv",
            "bus": TRAIN_DIR / "bus_mekkah_cleaned.csv",
            "car": TRAIN_DIR / "car_mekkah_cleaned.csv",
        },
        "test": {
            "walk": TEST_DIR / "walk_mekkah_cleaned.csv",
            "bus": TEST_DIR / "bus_mekkah_cleaned.csv",
            "car": TEST_DIR / "car_mekkah_cleaned.csv",
        },
    },
}

# These were used in your centralized SOAP style earlier; tune if needed.
BEST_SAMPLING_BY_CLASS = {
    "walk": 0.80,
    "bus": 0.15,
    "car": 0.30,
}

BCE_POS_WEIGHT_BY_CLASS = {
    "walk": 1.0,
    "bus": 1.5,
    "car": 1.5,
}


# =========================================================
# Utilities
# =========================================================
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


def load_dataframe(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required_cols = [TIME_COLUMN] + FEATURE_COLUMNS
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path.name} is missing columns: {missing}")

    df = df.dropna(subset=required_cols).reset_index(drop=True)
    df[TIME_COLUMN] = df[TIME_COLUMN].astype(str).str.strip()
    df["time_seconds"] = df[TIME_COLUMN].apply(time_to_seconds)

    return df


def fit_global_standardizer(city_files):
    frames = []

    for _, split_files in city_files.items():
        for _, path in split_files["train"].items():
            df = load_dataframe(path)
            frames.append(df[FEATURE_COLUMNS].copy())

    full_train_df = pd.concat(frames, axis=0, ignore_index=True)

    mean = full_train_df.mean()
    std = full_train_df.std(ddof=0)
    std = std.replace(0, 1.0)

    return mean, std


def apply_standardization(df: pd.DataFrame, mean: pd.Series, std: pd.Series) -> pd.DataFrame:
    out = df.copy()
    out[FEATURE_COLUMNS] = (out[FEATURE_COLUMNS] - mean) / std
    return out


def build_windows_from_df(df: pd.DataFrame, class_name: str):
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


def build_dataset_from_files(file_dict, mean, std):
    X_all = []
    y_all = []
    all_stats = []

    for class_name, path in file_dict.items():
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        df = load_dataframe(path)
        df = apply_standardization(df, mean, std)

        X, y, stats = build_windows_from_df(df, class_name)

        X_all.append(X)
        y_all.append(y)
        all_stats.append((class_name, path.name, stats))

        print(
            f"{path.name} | class={class_name} | rows={stats['rows']} | "
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


def make_binary_labels(y_multiclass, target_class):
    return (y_multiclass == target_class).astype(np.int64)


def state_dict_to_cpu(state_dict):
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}


# =========================================================
# Dataset / Model
# =========================================================
class IndexedBinarySequenceDataset(Dataset):
    def __init__(self, X, y_binary):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y_binary, dtype=torch.float32)
        self.targets = np.asarray(y_binary, dtype=np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], idx


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


# =========================================================
# Federated aggregation
# =========================================================
def fedavg_weighted_aggregate(local_state_dicts, client_names, client_sample_counts, verbose=True):
    total_samples = sum(client_sample_counts)
    if total_samples == 0:
        raise ValueError("Total client samples is zero during aggregation.")

    if verbose:
        print("\nServer aggregation coefficients:")
        for name, n in zip(client_names, client_sample_counts):
            print(f"  {name}: {n / total_samples:.6f}")

    aggregated_state = {}

    for key in local_state_dicts[0].keys():
        aggregated_param = torch.zeros_like(local_state_dicts[0][key], dtype=torch.float32)

        for state_dict, n_samples in zip(local_state_dicts, client_sample_counts):
            coeff = n_samples / total_samples
            aggregated_param += state_dict[key].float() * coeff

        aggregated_state[key] = aggregated_param

    return aggregated_state


# =========================================================
# Training
# =========================================================
def train_one_epoch_bce(model, loader, optimizer, pos_weight_value):
    model.train()
    total_loss = 0.0
    total_samples = 0

    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for X_batch, y_batch, _ in loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        n = X_batch.size(0)
        total_loss += loss.item() * n
        total_samples += n

    return total_loss / total_samples


def train_one_epoch_aploss(model, loader, optimizer, aploss_fn):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for X_batch, y_batch, index in loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        index = index.to(DEVICE)

        optimizer.zero_grad()
        logits = model(X_batch)
        probs = torch.sigmoid(logits)
        loss = aploss_fn(probs, y_batch, index)
        loss.backward()
        optimizer.step()

        n = X_batch.size(0)
        total_loss += loss.item() * n
        total_samples += n

    return total_loss / total_samples


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


def federated_bce_pretrain_for_class(
    target_class,
    client_train_binary,
    X_val_global,
    y_val_global,
):
    class_name = IDX_TO_LABEL[target_class]
    print(f"\n==================== FEDERATED BCE PRETRAIN: {class_name} vs rest ====================")

    global_model = BinaryRNNScorer(
        input_size=len(FEATURE_COLUMNS),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
    ).to(DEVICE)

    best_val_ap = -1.0
    best_state = state_dict_to_cpu(global_model.state_dict())
    round_loss_history = []

    client_names = list(client_train_binary.keys())
    client_sample_counts = [client_train_binary[c]["num_train"] for c in client_names]

    pos_weight_value = BCE_POS_WEIGHT_BY_CLASS[class_name]
    print(f"Using BCE pos_weight for {class_name}: {pos_weight_value}")

    start_time = time.time()

    for round_idx in range(1, GLOBAL_ROUNDS_BCE + 1):
        print(f"\n--- BCE Global Round {round_idx:02d}/{GLOBAL_ROUNDS_BCE} ---")

        local_state_dicts = []
        local_avg_losses = []

        for city_name in client_names:
            local_model = BinaryRNNScorer(
                input_size=len(FEATURE_COLUMNS),
                hidden_size=HIDDEN_SIZE,
                num_layers=NUM_LAYERS,
            ).to(DEVICE)
            local_model.load_state_dict(deepcopy(global_model.state_dict()))

            optimizer = torch.optim.Adam(
                local_model.parameters(),
                lr=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
            )

            local_loader = client_train_binary[city_name]["bce_loader"]

            epoch_losses = []
            for _ in range(LOCAL_EPOCHS):
                local_loss = train_one_epoch_bce(
                    model=local_model,
                    loader=local_loader,
                    optimizer=optimizer,
                    pos_weight_value=pos_weight_value,
                )
                epoch_losses.append(local_loss)

            local_avg_loss = float(np.mean(epoch_losses))
            local_avg_losses.append(local_avg_loss)
            local_state_dicts.append(deepcopy(local_model.state_dict()))

            print(f"Client {city_name.upper()} | BCE local avg loss: {local_avg_loss:.6f}")

        aggregated_state = fedavg_weighted_aggregate(
            local_state_dicts=local_state_dicts,
            client_names=client_names,
            client_sample_counts=client_sample_counts,
            verbose=True,
        )
        global_model.load_state_dict(aggregated_state)

        round_avg_loss = float(np.mean(local_avg_losses))
        round_loss_history.append(round_avg_loss)

        val_result = evaluate_binary_model(
            model=global_model,
            X_eval=X_val_global,
            y_eval_multiclass=y_val_global,
            target_class=target_class,
        )
        val_ap = val_result["ap"]

        if val_ap > best_val_ap:
            best_val_ap = val_ap
            best_state = state_dict_to_cpu(global_model.state_dict())

        print(
            f"Round {round_idx:02d} | BCE avg client loss: {round_avg_loss:.6f} | "
            f"Val AP: {val_ap:.6f} | Best Val AP: {best_val_ap:.6f}"
        )

    global_model.load_state_dict(best_state)
    elapsed = time.time() - start_time

    return global_model, elapsed, best_val_ap, round_loss_history


def federated_aploss_finetune_for_class(
    target_class,
    init_model,
    client_train_binary,
    X_val_global,
    y_val_global,
):
    class_name = IDX_TO_LABEL[target_class]
    sampling_rate = BEST_SAMPLING_BY_CLASS[class_name]

    print(f"\n==================== FEDERATED APLoss/SOAP: {class_name} vs rest ====================")
    print(f"gamma={AP_GAMMA}, margin={AP_MARGIN}, sampling_rate={sampling_rate}")

    global_model = deepcopy(init_model).to(DEVICE)

    best_val_ap = -1.0
    best_state = state_dict_to_cpu(global_model.state_dict())
    round_loss_history = []

    client_names = list(client_train_binary.keys())
    client_sample_counts = [client_train_binary[c]["num_train"] for c in client_names]

    start_time = time.time()

    for round_idx in range(1, GLOBAL_ROUNDS_AP + 1):
        print(f"\n--- APLoss Global Round {round_idx:02d}/{GLOBAL_ROUNDS_AP} ---")

        local_state_dicts = []
        local_avg_losses = []

        for city_name in client_names:
            local_model = BinaryRNNScorer(
                input_size=len(FEATURE_COLUMNS),
                hidden_size=HIDDEN_SIZE,
                num_layers=NUM_LAYERS,
            ).to(DEVICE)
            local_model.load_state_dict(deepcopy(global_model.state_dict()))

            local_dataset = client_train_binary[city_name]["dataset_for_ap"]

            aploss_fn = APLoss(
                data_len=len(local_dataset),
                gamma=AP_GAMMA,
                margin=AP_MARGIN,
                surr_loss=AP_SURR_LOSS,
            )

            optimizer = SOAP(
                local_model.parameters(),
                lr=LEARNING_RATE,
                mode="adam",
                weight_decay=WEIGHT_DECAY,
            )

            local_loader = client_train_binary[city_name]["ap_loader"]

            epoch_losses = []
            for _ in range(LOCAL_EPOCHS):
                local_loss = train_one_epoch_aploss(
                    model=local_model,
                    loader=local_loader,
                    optimizer=optimizer,
                    aploss_fn=aploss_fn,
                )
                epoch_losses.append(local_loss)

            local_avg_loss = float(np.mean(epoch_losses))
            local_avg_losses.append(local_avg_loss)
            local_state_dicts.append(deepcopy(local_model.state_dict()))

            print(f"Client {city_name.upper()} | APLoss local avg loss: {local_avg_loss:.6f}")

        aggregated_state = fedavg_weighted_aggregate(
            local_state_dicts=local_state_dicts,
            client_names=client_names,
            client_sample_counts=client_sample_counts,
            verbose=True,
        )
        global_model.load_state_dict(aggregated_state)

        round_avg_loss = float(np.mean(local_avg_losses))
        round_loss_history.append(round_avg_loss)

        val_result = evaluate_binary_model(
            model=global_model,
            X_eval=X_val_global,
            y_eval_multiclass=y_val_global,
            target_class=target_class,
        )
        val_ap = val_result["ap"]

        if val_ap > best_val_ap:
            best_val_ap = val_ap
            best_state = state_dict_to_cpu(global_model.state_dict())

        print(
            f"Round {round_idx:02d} | APLoss avg client loss: {round_avg_loss:.6f} | "
            f"Val AP: {val_ap:.6f} | Best Val AP: {best_val_ap:.6f}"
        )

    global_model.load_state_dict(best_state)
    elapsed = time.time() - start_time

    return global_model, elapsed, best_val_ap, round_loss_history


# =========================================================
# Diagnostics
# =========================================================
def multiclass_diagnostics_from_ovr(results_by_class, y_true_multiclass):
    class_order = ["walk", "bus", "car"]
    prob_matrix = np.column_stack([results_by_class[c]["probs"] for c in class_order])

    y_pred = np.argmax(prob_matrix, axis=1)
    y_true = y_true_multiclass

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        average=None,
        zero_division=0,
    )

    per_class_counts = {}
    per_class_metrics = {}

    total = cm.sum()

    for cls_idx in range(3):
        tp = cm[cls_idx, cls_idx]
        fp = cm[:, cls_idx].sum() - tp
        fn = cm[cls_idx, :].sum() - tp
        tn = total - tp - fp - fn

        class_name = IDX_TO_LABEL[cls_idx]

        per_class_counts[class_name] = {
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "TN": int(tn),
            "Support": int(cm[cls_idx, :].sum()),
        }

        per_class_metrics[class_name] = {
            "precision": float(precision[cls_idx]),
            "recall": float(recall[cls_idx]),
            "f1": float(f1[cls_idx]),
            "support": int(support[cls_idx]),
            "accuracy_ovr": float((tp + tn) / total if total > 0 else 0.0),
        }

    macro_precision = float(np.mean([per_class_metrics[c]["precision"] for c in ["walk", "bus", "car"]]))
    macro_recall = float(np.mean([per_class_metrics[c]["recall"] for c in ["walk", "bus", "car"]]))
    macro_f1 = float(np.mean([per_class_metrics[c]["f1"] for c in ["walk", "bus", "car"]]))

    return {
        "accuracy": float(acc),
        "confusion_matrix": cm,
        "per_class_counts": per_class_counts,
        "per_class_metrics": per_class_metrics,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    }


def plot_training_loss(loss_history, save_path: Path, title: str):
    plt.figure(figsize=(8, 5))
    rounds = np.arange(1, len(loss_history) + 1)
    plt.plot(rounds, loss_history, marker="o")
    plt.xlabel("Global Round")
    plt.ylabel("Average Local Train Loss")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


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


def plot_confusion_matrix_figure(cm, save_path: Path):
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


def save_metrics_summary(
    results_by_class,
    diagnostics,
    best_bce_val_aps,
    best_aploss_val_aps,
    bce_times,
    aploss_times,
    bce_round_histories,
    aploss_round_histories,
    client_info,
    save_path: Path,
):
    macro_auprc = float(np.mean([results_by_class[c]["ap"] for c in ["walk", "bus", "car"]]))

    lines = []
    lines.append("==================== FEDERATED SOAP RESULTS ====================")
    lines.append(f"Overall Accuracy: {diagnostics['accuracy']:.6f}")
    lines.append(f"Macro Precision: {diagnostics['macro_precision']:.6f}")
    lines.append(f"Macro Recall: {diagnostics['macro_recall']:.6f}")
    lines.append(f"Macro F1-score: {diagnostics['macro_f1']:.6f}")
    lines.append(f"Macro AP / Macro AUPRC: {macro_auprc:.6f}")
    lines.append("")

    lines.append("AP / AUPRC per class:")
    for class_name in ["walk", "bus", "car"]:
        lines.append(f"{class_name:>5} | AP: {results_by_class[class_name]['ap']:.6f}")

    lines.append("")
    lines.append("Best validation AP after federated BCE pretraining:")
    for class_name in ["walk", "bus", "car"]:
        lines.append(f"{class_name:>5} | BCE Val AP: {best_bce_val_aps[class_name]:.6f}")

    lines.append("")
    lines.append("Best validation AP after federated APLoss/SOAP:")
    for class_name in ["walk", "bus", "car"]:
        lines.append(f"{class_name:>5} | APLoss Val AP: {best_aploss_val_aps[class_name]:.6f}")

    lines.append("")
    lines.append("Per-class classification metrics:")
    for class_name in ["walk", "bus", "car"]:
        m = diagnostics["per_class_metrics"][class_name]
        lines.append(
            f"{class_name:>5} | "
            f"Precision={m['precision']:.6f} | "
            f"Recall={m['recall']:.6f} | "
            f"F1={m['f1']:.6f} | "
            f"One-vs-Rest Accuracy={m['accuracy_ovr']:.6f} | "
            f"Support={m['support']}"
        )

    lines.append("")
    lines.append("Confusion Matrix [rows=true, cols=pred]:")
    lines.append(str(diagnostics["confusion_matrix"]))

    lines.append("")
    lines.append("TP / FP / FN / TN per class:")
    for class_name, counts in diagnostics["per_class_counts"].items():
        lines.append(
            f"{class_name:>5} | "
            f"TP={counts['TP']} | FP={counts['FP']} | "
            f"FN={counts['FN']} | TN={counts['TN']} | "
            f"Support={counts['Support']}"
        )

    lines.append("")
    lines.append("Client training sample counts:")
    for city_name, info in client_info.items():
        lines.append(
            f"{city_name:>7} | train_windows={info['train_windows']} | "
            f"val_windows={info['val_windows']} | test_windows={info['test_windows']}"
        )

    lines.append("")
    lines.append("Training time (seconds):")
    for class_name in ["walk", "bus", "car"]:
        lines.append(
            f"{class_name:>5} | BCE={bce_times[class_name]:.2f} | "
            f"APLoss/SOAP={aploss_times[class_name]:.2f}"
        )

    lines.append("")
    lines.append("BCE round losses:")
    for class_name in ["walk", "bus", "car"]:
        lines.append(f"[{class_name}]")
        for i, loss in enumerate(bce_round_histories[class_name], start=1):
            lines.append(f"Round {i:02d}: {loss:.6f}")

    lines.append("")
    lines.append("APLoss round losses:")
    for class_name in ["walk", "bus", "car"]:
        lines.append(f"[{class_name}]")
        for i, loss in enumerate(aploss_round_histories[class_name], start=1):
            lines.append(f"Round {i:02d}: {loss:.6f}")

    save_path.write_text("\n".join(lines), encoding="utf-8")


# =========================================================
# Main
# =========================================================
def main():
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"Train directory does not exist: {TRAIN_DIR}")
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Test directory does not exist: {TEST_DIR}")

    print(f"CellMob root: {CELLMOB_ROOT}")
    print(f"Train directory: {TRAIN_DIR}")
    print(f"Test directory: {TEST_DIR}")
    print(f"Run output directory: {RUN_OUTPUT_DIR}")

    global_mean, global_std = fit_global_standardizer(CITY_FILES)

    print("\nShared global standardization statistics computed from ALL TRAINING ROWS only.")
    print("Global mean (first few entries):")
    print(global_mean.head())
    print("\nGlobal std (first few entries):")
    print(global_std.head())

    client_train_binary = {}
    client_info = {}

    X_val_global_parts = []
    y_val_global_parts = []

    X_test_all = []
    y_test_all = []

    for city_name, split_files in CITY_FILES.items():
        print(f"\n==================== Building client: {city_name.upper()} ====================")

        print(f"Building {city_name} TRAIN dataset...")
        X_train_city_full, y_train_city_full, _ = build_dataset_from_files(
            split_files["train"], global_mean, global_std
        )

        print(f"\nBuilding {city_name} TEST dataset...")
        X_test_city, y_test_city, _ = build_dataset_from_files(
            split_files["test"], global_mean, global_std
        )

        print(f"\n{city_name.upper()} shapes before train/val split:")
        print("X_train_full:", X_train_city_full.shape)
        print("y_train_full:", y_train_city_full.shape)
        print("X_test      :", X_test_city.shape)
        print("y_test      :", y_test_city.shape)

        X_train_city, X_val_city, y_train_city, y_val_city = train_test_split(
            X_train_city_full,
            y_train_city_full,
            test_size=VAL_SIZE,
            random_state=SEED,
            stratify=y_train_city_full,
        )

        print(f"\n{city_name.upper()} shapes after train/val split:")
        print("X_train:", X_train_city.shape)
        print("y_train:", y_train_city.shape)
        print("X_val  :", X_val_city.shape)
        print("y_val  :", y_val_city.shape)
        print("X_test :", X_test_city.shape)
        print("y_test :", y_test_city.shape)

        print_class_distribution(f"{city_name.upper()} Train", y_train_city)
        print_class_distribution(f"{city_name.upper()} Val", y_val_city)
        print_class_distribution(f"{city_name.upper()} Test", y_test_city)

        client_train_binary[city_name] = {
            "X_train": X_train_city,
            "y_train_multiclass": y_train_city,
            "X_val": X_val_city,
            "y_val_multiclass": y_val_city,
            "X_test": X_test_city,
            "y_test_multiclass": y_test_city,
        }

        client_info[city_name] = {
            "train_windows": len(X_train_city),
            "val_windows": len(X_val_city),
            "test_windows": len(X_test_city),
        }

        X_val_global_parts.append(X_val_city)
        y_val_global_parts.append(y_val_city)

        X_test_all.append(X_test_city)
        y_test_all.append(y_test_city)

    X_val_global = np.concatenate(X_val_global_parts, axis=0)
    y_val_global = np.concatenate(y_val_global_parts, axis=0)

    X_test_global = np.concatenate(X_test_all, axis=0)
    y_test_global = np.concatenate(y_test_all, axis=0)

    print("\n==================== Combined Global Validation/Test Sets ====================")
    print("X_val_global :", X_val_global.shape)
    print("y_val_global :", y_val_global.shape)
    print("X_test_global:", X_test_global.shape)
    print("y_test_global:", y_test_global.shape)

    print_class_distribution("Global Val", y_val_global)
    print_class_distribution("Global Test", y_test_global)

    models = {}
    bce_times = {}
    aploss_times = {}
    best_bce_val_aps = {}
    best_aploss_val_aps = {}
    bce_round_histories = {}
    aploss_round_histories = {}

    # Prepare class-specific client datasets/loaders
    for target_class in [0, 1, 2]:
        class_name = IDX_TO_LABEL[target_class]

        print(f"\n==============================================================")
        print(f"Preparing federated binary datasets for class: {class_name}")
        print(f"==============================================================")

        per_city_binary = {}

        for city_name in CITY_FILES.keys():
            X_train_city = client_train_binary[city_name]["X_train"]
            y_train_city_multiclass = client_train_binary[city_name]["y_train_multiclass"]
            y_train_binary = make_binary_labels(y_train_city_multiclass, target_class)

            dataset_for_bce = IndexedBinarySequenceDataset(X_train_city, y_train_binary)
            bce_loader = DataLoader(
                dataset_for_bce,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=0,
            )

            dataset_for_ap = IndexedBinarySequenceDataset(X_train_city, y_train_binary)
            sampler = DualSampler(
                dataset_for_ap,
                batch_size=BATCH_SIZE,
                labels=dataset_for_ap.targets,
                sampling_rate=BEST_SAMPLING_BY_CLASS[class_name],
                random_seed=SEED,
            )
            ap_loader = DataLoader(
                dataset_for_ap,
                batch_size=BATCH_SIZE,
                sampler=sampler,
                shuffle=False,
                num_workers=0,
            )

            positives = int(y_train_binary.sum())
            negatives = int(len(y_train_binary) - positives)

            print(
                f"{city_name.upper()} | total={len(y_train_binary)} | "
                f"positive={positives} | negative={negatives}"
            )

            per_city_binary[city_name] = {
                "dataset_for_ap": dataset_for_ap,
                "bce_loader": bce_loader,
                "ap_loader": ap_loader,
                "num_train": len(dataset_for_bce),
            }

        pretrained_model, bce_time_sec, best_bce_val_ap, bce_loss_history = federated_bce_pretrain_for_class(
            target_class=target_class,
            client_train_binary=per_city_binary,
            X_val_global=X_val_global,
            y_val_global=y_val_global,
        )

        final_model, aploss_time_sec, best_ap_val_ap, aploss_loss_history = federated_aploss_finetune_for_class(
            target_class=target_class,
            init_model=pretrained_model,
            client_train_binary=per_city_binary,
            X_val_global=X_val_global,
            y_val_global=y_val_global,
        )

        models[class_name] = final_model
        bce_times[class_name] = bce_time_sec
        aploss_times[class_name] = aploss_time_sec
        best_bce_val_aps[class_name] = best_bce_val_ap
        best_aploss_val_aps[class_name] = best_ap_val_ap
        bce_round_histories[class_name] = bce_loss_history
        aploss_round_histories[class_name] = aploss_loss_history

        model_path = RUN_OUTPUT_DIR / f"fedavg_soap_{class_name}_vs_rest_best.pth"
        torch.save(final_model.state_dict(), model_path)
        print(f"Saved model: {model_path}")

        plot_training_loss(
            bce_loss_history,
            RUN_OUTPUT_DIR / f"bce_round_loss_{class_name}.png",
            title=f"Federated BCE Loss per Round ({class_name} vs rest)",
        )
        plot_training_loss(
            aploss_loss_history,
            RUN_OUTPUT_DIR / f"aploss_round_loss_{class_name}.png",
            title=f"Federated APLoss/SOAP Loss per Round ({class_name} vs rest)",
        )

    results_by_class = {}
    for target_class in [0, 1, 2]:
        class_name = IDX_TO_LABEL[target_class]
        results_by_class[class_name] = evaluate_binary_model(
            model=models[class_name],
            X_eval=X_test_global,
            y_eval_multiclass=y_test_global,
            target_class=target_class,
        )

    macro_auprc = float(np.mean([
        results_by_class["walk"]["ap"],
        results_by_class["bus"]["ap"],
        results_by_class["car"]["ap"],
    ]))

    diagnostics = multiclass_diagnostics_from_ovr(results_by_class, y_test_global)

    print("\n==================== FEDERATED SOAP RESULTS ====================")
    print(f"Overall Accuracy: {diagnostics['accuracy']:.6f}")
    print(f"Macro Precision: {diagnostics['macro_precision']:.6f}")
    print(f"Macro Recall: {diagnostics['macro_recall']:.6f}")
    print(f"Macro F1-score: {diagnostics['macro_f1']:.6f}")
    print(f"Macro AP / Macro AUPRC: {macro_auprc:.6f}")

    print("\nAP / AUPRC per class:")
    for class_name in ["walk", "bus", "car"]:
        print(f"{class_name:>5} | AP: {results_by_class[class_name]['ap']:.6f}")

    print("\nBest validation AP after federated BCE pretraining:")
    for class_name in ["walk", "bus", "car"]:
        print(f"{class_name:>5} | BCE Val AP: {best_bce_val_aps[class_name]:.6f}")

    print("\nBest validation AP after federated APLoss/SOAP:")
    for class_name in ["walk", "bus", "car"]:
        print(f"{class_name:>5} | APLoss Val AP: {best_aploss_val_aps[class_name]:.6f}")

    print("\nPer-class metrics:")
    for class_name in ["walk", "bus", "car"]:
        m = diagnostics["per_class_metrics"][class_name]
        print(
            f"{class_name:>5} | "
            f"Precision={m['precision']:.6f} | "
            f"Recall={m['recall']:.6f} | "
            f"F1={m['f1']:.6f} | "
            f"One-vs-Rest Accuracy={m['accuracy_ovr']:.6f} | "
            f"Support={m['support']}"
        )

    print("\nConfusion Matrix [rows=true, cols=pred]:")
    print(diagnostics["confusion_matrix"])

    pr_plot_path = RUN_OUTPUT_DIR / "pr_curves_federated_soap_ovr.png"
    cm_plot_path = RUN_OUTPUT_DIR / "confusion_matrix_federated_soap.png"
    metrics_txt_path = RUN_OUTPUT_DIR / "metrics_summary_federated_soap.txt"

    plot_pr_curves(
        results_by_class,
        pr_plot_path,
        title="Federated SOAP Precision-Recall Curves (one-vs-rest)",
    )
    plot_confusion_matrix_figure(diagnostics["confusion_matrix"], cm_plot_path)
    save_metrics_summary(
        results_by_class=results_by_class,
        diagnostics=diagnostics,
        best_bce_val_aps=best_bce_val_aps,
        best_aploss_val_aps=best_aploss_val_aps,
        bce_times=bce_times,
        aploss_times=aploss_times,
        bce_round_histories=bce_round_histories,
        aploss_round_histories=aploss_round_histories,
        client_info=client_info,
        save_path=metrics_txt_path,
    )

    print(f"\nSaved PR curves plot: {pr_plot_path}")
    print(f"Saved confusion matrix plot: {cm_plot_path}")
    print(f"Saved metrics summary: {metrics_txt_path}")


if __name__ == "__main__":
    main()