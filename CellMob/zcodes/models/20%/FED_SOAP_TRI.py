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
    from libauc.losses import meanAveragePrecisionLoss
    from libauc.optimizers import SOAP
    from libauc.sampler import TriSampler
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
RUN_OUTPUT_DIR = OUTPUT_ROOT / "fedavg_multiclass_maploss_soap_kaust_jeddah_mekkah"
RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# Hyperparameters
# =========================================================
WINDOW_SIZE = 5
MAX_WINDOW_SPAN_SECONDS = 3.5

NUM_CLASSES = 3
BATCH_SIZE_CE = 64

# For TriSampler:
# total batch size in SOAP stage = BATCH_SIZE_PER_TASK * NUM_SAMPLED_TASKS
BATCH_SIZE_PER_TASK = 32
NUM_SAMPLED_TASKS = NUM_CLASSES

HIDDEN_SIZE = 64
NUM_LAYERS = 1

PRETRAIN_GLOBAL_ROUNDS = 20
SOAP_GLOBAL_ROUNDS = 20
LOCAL_EPOCHS = 1

PRETRAIN_LR = 5e-4
SOAP_LR = 1e-3
WEIGHT_DECAY = 2e-4

MAP_GAMMA = 0.9
MAP_MARGIN = 0.7
MAP_SURR_LOSS = "squared_hinge"
MAP_TOP_K = -1

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


def compute_global_class_weights(client_data_dict, num_classes=3) -> torch.Tensor:
    all_y = []
    for city_name in client_data_dict:
        all_y.append(client_data_dict[city_name]["y_train"])
    y = np.concatenate(all_y, axis=0)

    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    total = counts.sum()

    if np.any(counts == 0):
        raise ValueError(f"At least one class has zero samples in training data. Counts: {counts}")

    weights = total / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def to_one_hot(y, num_classes):
    return np.eye(num_classes, dtype=np.float32)[y]


def state_dict_to_cpu(state_dict):
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}


def print_class_distribution(name, y):
    unique, counts = np.unique(y, return_counts=True)
    print(f"\n{name} class distribution:")
    for u, c in zip(unique, counts):
        print(f"  {IDX_TO_LABEL[int(u)]}: {int(c)}")


# =========================================================
# Dataset classes
# =========================================================
class SequenceDatasetCE(Dataset):
    def __init__(self, X, y_int):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y_int, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


class SequenceDatasetTri(Dataset):
    """
    Compatible with TriSampler.
    If index is a tuple (sample_id, task_id), it returns that tuple back.
    """
    def __init__(self, X, y_int, num_classes):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_int = torch.tensor(y_int, dtype=torch.long)
        self.y_oh = torch.tensor(to_one_hot(y_int, num_classes), dtype=torch.float32)
        self.num_classes = num_classes

        # TriSampler expects labels array.
        # For single-label multiclass, one-hot targets are the cleanest representation.
        self.targets = self.y_oh.numpy()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        if isinstance(index, tuple):
            sample_id, task_id = index
        else:
            sample_id = index
            task_id = -1  # unused in normal access
        return self.X[sample_id], self.y_int[sample_id], self.y_oh[sample_id], (sample_id, task_id)


# =========================================================
# Model
# =========================================================
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


# =========================================================
# FedAvg
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
# Local training
# =========================================================
def train_one_epoch_ce(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for X_batch, y_batch in loader:
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


def train_one_epoch_maploss(model, loader, optimizer, map_loss_fn):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for X_batch, _, y_oh_batch, index_info in loader:
        sample_ids, task_ids = index_info

        X_batch = X_batch.to(DEVICE)
        y_oh_batch = y_oh_batch.to(DEVICE)
        sample_ids = sample_ids.to(DEVICE)
        task_ids = task_ids.to(DEVICE)

        optimizer.zero_grad()
        logits = model(X_batch)

        # task_ids is supplied by TriSampler and indicates sampled tasks/classes in the batch
        loss = map_loss_fn(logits, y_oh_batch, sample_ids, task_ids)
        loss.backward()
        optimizer.step()

        n = X_batch.size(0)
        total_loss += loss.item() * n
        total_samples += n

    return total_loss / total_samples


# =========================================================
# Evaluation
# =========================================================
@torch.no_grad()
def evaluate_multiclass(model, X_eval, y_eval):
    model.eval()

    all_logits = []
    batch_size = 256

    for start in range(0, len(X_eval), batch_size):
        end = min(start + batch_size, len(X_eval))
        X_batch = torch.tensor(X_eval[start:end], dtype=torch.float32, device=DEVICE)
        logits = model(X_batch)
        all_logits.append(logits.cpu().numpy())

    logits_all = np.vstack(all_logits)
    probs = torch.softmax(torch.tensor(logits_all), dim=1).numpy()

    y_pred = np.argmax(probs, axis=1)
    y_true = y_eval

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        average=None,
        zero_division=0,
    )

    total = cm.sum()
    per_class_counts = {}
    per_class_metrics = {}

    for cls_idx in range(cm.shape[0]):
        tp = cm[cls_idx, cls_idx]
        fp = cm[:, cls_idx].sum() - tp
        fn = cm[cls_idx, :].sum() - tp
        tn = total - tp - fp - fn

        class_name = IDX_TO_LABEL[cls_idx]
        one_vs_rest_accuracy = (tp + tn) / total if total > 0 else 0.0

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
            "accuracy_ovr": float(one_vs_rest_accuracy),
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
    macro_f1 = float(np.mean([per_class_metrics[c]["f1"] for c in ["walk", "bus", "car"]]))
    macro_precision = float(np.mean([per_class_metrics[c]["precision"] for c in ["walk", "bus", "car"]]))
    macro_recall = float(np.mean([per_class_metrics[c]["recall"] for c in ["walk", "bus", "car"]]))

    return {
        "accuracy": float(acc),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
        "per_class_counts": per_class_counts,
        "per_class_metrics": per_class_metrics,
        "ap_per_class": ap_per_class,
        "macro_ap": macro_ap,
        "pr_curves": pr_curves,
    }


# =========================================================
# Federated stages
# =========================================================
def federated_ce_pretrain(client_train_data, class_weights, X_val_global, y_val_global):
    print("\n==================== FEDERATED CE PRETRAIN ====================")

    global_model = RNNClassifier(
        input_size=len(FEATURE_COLUMNS),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

    best_val_macro_ap = -1.0
    best_state = state_dict_to_cpu(global_model.state_dict())
    round_loss_history = []

    client_names = list(client_train_data.keys())
    client_sample_counts = [client_train_data[c]["num_train"] for c in client_names]

    start_time = time.time()

    for round_idx in range(1, PRETRAIN_GLOBAL_ROUNDS + 1):
        print(f"\n--- CE Global Round {round_idx:02d}/{PRETRAIN_GLOBAL_ROUNDS} ---")

        local_state_dicts = []
        local_avg_losses = []

        for city_name in client_names:
            local_model = RNNClassifier(
                input_size=len(FEATURE_COLUMNS),
                hidden_size=HIDDEN_SIZE,
                num_layers=NUM_LAYERS,
                num_classes=NUM_CLASSES,
            ).to(DEVICE)
            local_model.load_state_dict(deepcopy(global_model.state_dict()))

            optimizer = torch.optim.Adam(
                local_model.parameters(),
                lr=PRETRAIN_LR,
                weight_decay=WEIGHT_DECAY,
            )

            local_loader = client_train_data[city_name]["ce_loader"]

            epoch_losses = []
            for _ in range(LOCAL_EPOCHS):
                local_loss = train_one_epoch_ce(local_model, local_loader, optimizer, criterion)
                epoch_losses.append(local_loss)

            local_avg_loss = float(np.mean(epoch_losses))
            local_avg_losses.append(local_avg_loss)
            local_state_dicts.append(deepcopy(local_model.state_dict()))

            print(f"Client {city_name.upper()} | CE local avg loss: {local_avg_loss:.6f}")

        aggregated_state = fedavg_weighted_aggregate(
            local_state_dicts=local_state_dicts,
            client_names=client_names,
            client_sample_counts=client_sample_counts,
            verbose=True,
        )
        global_model.load_state_dict(aggregated_state)

        round_avg_loss = float(np.mean(local_avg_losses))
        round_loss_history.append(round_avg_loss)

        val_result = evaluate_multiclass(global_model, X_val_global, y_val_global)
        val_macro_ap = val_result["macro_ap"]

        if val_macro_ap > best_val_macro_ap:
            best_val_macro_ap = val_macro_ap
            best_state = state_dict_to_cpu(global_model.state_dict())

        print(
            f"Round {round_idx:02d} | CE avg client loss: {round_avg_loss:.6f} | "
            f"Val Macro AP: {val_macro_ap:.6f} | Best Val Macro AP: {best_val_macro_ap:.6f}"
        )

    global_model.load_state_dict(best_state)
    elapsed = time.time() - start_time
    return global_model, elapsed, best_val_macro_ap, round_loss_history


def federated_maploss_soap_finetune(client_train_data, init_model, X_val_global, y_val_global):
    print("\n==================== FEDERATED MULTICLASS mAPLoss + SOAP ====================")

    global_model = deepcopy(init_model).to(DEVICE)

    best_val_macro_ap = -1.0
    best_state = state_dict_to_cpu(global_model.state_dict())
    round_loss_history = []

    client_names = list(client_train_data.keys())
    client_sample_counts = [client_train_data[c]["num_train"] for c in client_names]

    start_time = time.time()

    for round_idx in range(1, SOAP_GLOBAL_ROUNDS + 1):
        print(f"\n--- SOAP Global Round {round_idx:02d}/{SOAP_GLOBAL_ROUNDS} ---")

        local_state_dicts = []
        local_avg_losses = []

        for city_name in client_names:
            local_model = RNNClassifier(
                input_size=len(FEATURE_COLUMNS),
                hidden_size=HIDDEN_SIZE,
                num_layers=NUM_LAYERS,
                num_classes=NUM_CLASSES,
            ).to(DEVICE)
            local_model.load_state_dict(deepcopy(global_model.state_dict()))

            local_dataset = client_train_data[city_name]["tri_dataset"]

            loss_fn = meanAveragePrecisionLoss(
                data_len=len(local_dataset),
                num_labels=NUM_CLASSES,
                margin=MAP_MARGIN,
                gamma=MAP_GAMMA,
                top_k=MAP_TOP_K,
                surr_loss=MAP_SURR_LOSS,
                device=DEVICE,
            )

            optimizer = SOAP(
                local_model.parameters(),
                lr=SOAP_LR,
                mode="adam",
                weight_decay=WEIGHT_DECAY,
            )

            local_loader = client_train_data[city_name]["soap_loader"]

            epoch_losses = []
            for _ in range(LOCAL_EPOCHS):
                local_loss = train_one_epoch_maploss(local_model, local_loader, optimizer, loss_fn)
                epoch_losses.append(local_loss)

            local_avg_loss = float(np.mean(epoch_losses))
            local_avg_losses.append(local_avg_loss)
            local_state_dicts.append(deepcopy(local_model.state_dict()))

            print(f"Client {city_name.upper()} | SOAP local avg loss: {local_avg_loss:.6f}")

        aggregated_state = fedavg_weighted_aggregate(
            local_state_dicts=local_state_dicts,
            client_names=client_names,
            client_sample_counts=client_sample_counts,
            verbose=True,
        )
        global_model.load_state_dict(aggregated_state)

        round_avg_loss = float(np.mean(local_avg_losses))
        round_loss_history.append(round_avg_loss)

        val_result = evaluate_multiclass(global_model, X_val_global, y_val_global)
        val_macro_ap = val_result["macro_ap"]

        if val_macro_ap > best_val_macro_ap:
            best_val_macro_ap = val_macro_ap
            best_state = state_dict_to_cpu(global_model.state_dict())

        print(
            f"Round {round_idx:02d} | SOAP avg client loss: {round_avg_loss:.6f} | "
            f"Val Macro AP: {val_macro_ap:.6f} | Best Val Macro AP: {best_val_macro_ap:.6f}"
        )

    global_model.load_state_dict(best_state)
    elapsed = time.time() - start_time
    return global_model, elapsed, best_val_macro_ap, round_loss_history


# =========================================================
# Plotting / reporting
# =========================================================
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
    plt.title("Precision-Recall Curves (one-vs-rest evaluation)")
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
    results,
    ce_loss_history,
    soap_loss_history,
    class_weights,
    best_val_macro_ap_ce,
    best_val_macro_ap_soap,
    ce_time_sec,
    soap_time_sec,
    client_info,
    save_path: Path,
):
    lines = []
    lines.append("==================== FEDERATED MULTICLASS SOAP RESULTS ====================")
    lines.append(f"Class weights used in CE pretraining: {class_weights.tolist()}")
    lines.append(f"Best validation Macro AP after CE pretraining: {best_val_macro_ap_ce:.6f}")
    lines.append(f"Best validation Macro AP after SOAP fine-tuning: {best_val_macro_ap_soap:.6f}")
    lines.append(f"Overall Accuracy: {results['accuracy']:.6f}")
    lines.append(f"Macro Precision: {results['macro_precision']:.6f}")
    lines.append(f"Macro Recall: {results['macro_recall']:.6f}")
    lines.append(f"Macro F1-score: {results['macro_f1']:.6f}")
    lines.append(f"Macro AP / Macro AUPRC: {results['macro_ap']:.6f}")
    lines.append("")

    lines.append("Per-class classification metrics:")
    for class_name in ["walk", "bus", "car"]:
        m = results["per_class_metrics"][class_name]
        lines.append(
            f"{class_name:>5} | "
            f"Precision={m['precision']:.6f} | "
            f"Recall={m['recall']:.6f} | "
            f"F1={m['f1']:.6f} | "
            f"One-vs-Rest Accuracy={m['accuracy_ovr']:.6f} | "
            f"Support={m['support']}"
        )

    lines.append("")
    lines.append("AP / AUPRC per class:")
    for class_name in ["walk", "bus", "car"]:
        lines.append(f"{class_name:>5} | AP: {results['ap_per_class'][class_name]:.6f}")

    lines.append("")
    lines.append("Confusion Matrix [rows=true, cols=pred]:")
    lines.append(str(results["confusion_matrix"]))
    lines.append("")

    lines.append("TP / FP / FN / TN per class:")
    for class_name, counts in results["per_class_counts"].items():
        lines.append(
            f"{class_name:>5} | "
            f"TP={counts['TP']} | FP={counts['FP']} | "
            f"FN={counts['FN']} | TN={counts['TN']} | "
            f"Support={counts['Support']}"
        )

    lines.append("")
    lines.append("Client sample counts:")
    for city_name, info in client_info.items():
        lines.append(
            f"{city_name:>7} | train_windows={info['train_windows']} | "
            f"val_windows={info['val_windows']} | test_windows={info['test_windows']}"
        )

    lines.append("")
    lines.append(f"CE pretraining time (sec): {ce_time_sec:.2f}")
    lines.append(f"SOAP fine-tuning time (sec): {soap_time_sec:.2f}")

    lines.append("")
    lines.append("CE round losses:")
    for i, loss in enumerate(ce_loss_history, start=1):
        lines.append(f"Round {i:02d}: {loss:.6f}")

    lines.append("")
    lines.append("SOAP round losses:")
    for i, loss in enumerate(soap_loss_history, start=1):
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
    print(global_mean.head())
    print(global_std.head())

    client_data = {}
    client_info = {}

    X_val_global_parts = []
    y_val_global_parts = []

    X_test_all = []
    y_test_all = []

    for city_name, split_files in CITY_FILES.items():
        print(f"\n==================== Building client: {city_name.upper()} ====================")

        print(f"Building {city_name} TRAIN dataset...")
        X_train_full, y_train_full, _ = build_dataset_from_files(
            split_files["train"], global_mean, global_std
        )

        print(f"\nBuilding {city_name} TEST dataset...")
        X_test_city, y_test_city, _ = build_dataset_from_files(
            split_files["test"], global_mean, global_std
        )

        X_train_city, X_val_city, y_train_city, y_val_city = train_test_split(
            X_train_full,
            y_train_full,
            test_size=VAL_SIZE,
            random_state=SEED,
            stratify=y_train_full,
        )

        print(f"\n{city_name.upper()} shapes after split:")
        print("X_train:", X_train_city.shape)
        print("y_train:", y_train_city.shape)
        print("X_val  :", X_val_city.shape)
        print("y_val  :", y_val_city.shape)
        print("X_test :", X_test_city.shape)
        print("y_test :", y_test_city.shape)

        print_class_distribution(f"{city_name.upper()} Train", y_train_city)
        print_class_distribution(f"{city_name.upper()} Val", y_val_city)
        print_class_distribution(f"{city_name.upper()} Test", y_test_city)

        ce_dataset = SequenceDatasetCE(X_train_city, y_train_city)
        ce_loader = DataLoader(
            ce_dataset,
            batch_size=BATCH_SIZE_CE,
            shuffle=True,
            num_workers=0,
        )

        tri_dataset = SequenceDatasetTri(X_train_city, y_train_city, NUM_CLASSES)

        tri_sampler = TriSampler(
            tri_dataset,
            batch_size_per_task=BATCH_SIZE_PER_TASK,
            num_sampled_tasks=NUM_SAMPLED_TASKS,
            sampling_rate=0.5,
            mode="classification",
            labels=tri_dataset.targets,
            shuffle=True,
            random_seed=SEED,
        )

        soap_loader = DataLoader(
            tri_dataset,
            batch_size=BATCH_SIZE_PER_TASK * NUM_SAMPLED_TASKS,
            sampler=tri_sampler,
            shuffle=False,
            num_workers=0,
        )

        client_data[city_name] = {
            "X_train": X_train_city,
            "y_train": y_train_city,
            "X_val": X_val_city,
            "y_val": y_val_city,
            "X_test": X_test_city,
            "y_test": y_test_city,
            "ce_loader": ce_loader,
            "tri_dataset": tri_dataset,
            "soap_loader": soap_loader,
            "num_train": len(X_train_city),
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

    class_weights = compute_global_class_weights(client_data, num_classes=NUM_CLASSES)
    print(f"\nGlobal CE class weights: {class_weights.tolist()}")

    # Stage 1: CE pretraining
    pretrained_model, ce_time_sec, best_val_macro_ap_ce, ce_loss_history = federated_ce_pretrain(
        client_train_data=client_data,
        class_weights=class_weights,
        X_val_global=X_val_global,
        y_val_global=y_val_global,
    )

    # Stage 2: multiclass LibAUC fine-tuning
    final_model, soap_time_sec, best_val_macro_ap_soap, soap_loss_history = federated_maploss_soap_finetune(
        client_train_data=client_data,
        init_model=pretrained_model,
        X_val_global=X_val_global,
        y_val_global=y_val_global,
    )

    print("\nEvaluating on global test set...")
    results = evaluate_multiclass(final_model, X_test_global, y_test_global)

    print("\n==================== RESULTS ====================")
    print(f"Overall Accuracy: {results['accuracy']:.6f}")
    print(f"Macro Precision: {results['macro_precision']:.6f}")
    print(f"Macro Recall: {results['macro_recall']:.6f}")
    print(f"Macro F1-score: {results['macro_f1']:.6f}")
    print(f"Macro AP / Macro AUPRC: {results['macro_ap']:.6f}")

    print("\nPer-class metrics:")
    for class_name in ["walk", "bus", "car"]:
        m = results["per_class_metrics"][class_name]
        print(
            f"{class_name:>5} | "
            f"Precision={m['precision']:.6f} | "
            f"Recall={m['recall']:.6f} | "
            f"F1={m['f1']:.6f} | "
            f"One-vs-Rest Accuracy={m['accuracy_ovr']:.6f} | "
            f"Support={m['support']}"
        )

    print("\nAP / AUPRC per class:")
    for class_name in ["walk", "bus", "car"]:
        print(f"{class_name:>5} | AP: {results['ap_per_class'][class_name]:.6f}")

    print("\nConfusion Matrix [rows=true, cols=pred]:")
    print(results["confusion_matrix"])

    loss_plot_ce_path = RUN_OUTPUT_DIR / "ce_round_loss.png"
    loss_plot_soap_path = RUN_OUTPUT_DIR / "soap_round_loss.png"
    pr_plot_path = RUN_OUTPUT_DIR / "pr_curves.png"
    cm_plot_path = RUN_OUTPUT_DIR / "confusion_matrix.png"
    metrics_txt_path = RUN_OUTPUT_DIR / "metrics_summary.txt"
    model_path = RUN_OUTPUT_DIR / "fedavg_multiclass_maploss_soap_model.pth"

    plot_training_loss(
        ce_loss_history,
        loss_plot_ce_path,
        title="Federated CE Pretraining Loss per Round",
    )
    plot_training_loss(
        soap_loss_history,
        loss_plot_soap_path,
        title="Federated mAPLoss + SOAP Loss per Round",
    )
    plot_pr_curves(results["pr_curves"], pr_plot_path)
    plot_confusion_matrix_figure(results["confusion_matrix"], cm_plot_path)
    save_metrics_summary(
        results=results,
        ce_loss_history=ce_loss_history,
        soap_loss_history=soap_loss_history,
        class_weights=class_weights,
        best_val_macro_ap_ce=best_val_macro_ap_ce,
        best_val_macro_ap_soap=best_val_macro_ap_soap,
        ce_time_sec=ce_time_sec,
        soap_time_sec=soap_time_sec,
        client_info=client_info,
        save_path=metrics_txt_path,
    )

    torch.save(final_model.state_dict(), model_path)

    print(f"\nSaved CE loss plot: {loss_plot_ce_path}")
    print(f"Saved SOAP loss plot: {loss_plot_soap_path}")
    print(f"Saved PR curves plot: {pr_plot_path}")
    print(f"Saved confusion matrix plot: {cm_plot_path}")
    print(f"Saved metrics summary: {metrics_txt_path}")
    print(f"Saved model: {model_path}")


if __name__ == "__main__":
    main()