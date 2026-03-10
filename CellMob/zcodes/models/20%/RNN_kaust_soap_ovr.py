from pathlib import Path
import random
import time
import copy
from itertools import product

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
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 2e-4

AP_SURR_LOSS = "squared_hinge"

LR_DECAY_EPOCHS = [10, 15]
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

GAMMA_LIST = [0.1]
MARGIN_LIST = [0.6]

WALK_SAMPLING_LIST = [0.75, 0.80]
BUS_SAMPLING_LIST = [0.10, 0.15]
CAR_SAMPLING_LIST = [0.20, 0.25]

EVALUATE_EVERY_CONFIG_ON_TEST = False


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


def train_libauc_binary_model(
    X_train,
    y_train_multiclass,
    target_class,
    X_val,
    y_val_multiclass,
    gamma,
    margin,
    sampling_rate,
):
    class_name = IDX_TO_LABEL[target_class]

    y_binary = make_binary_labels(y_train_multiclass, target_class)
    pos_count = int(y_binary.sum())
    neg_count = int(len(y_binary) - pos_count)
    pos_ratio = float(pos_count / len(y_binary))

    print(f"\n-------------------- {class_name} vs rest --------------------")
    print(f"  Positives in train: {pos_count}")
    print(f"  Negatives in train: {neg_count}")
    print(f"  Positive ratio:     {pos_ratio:.6f}")
    print(f"  Sampling rate:      {sampling_rate:.6f}")
    print(f"  Gamma:              {gamma:.6f}")
    print(f"  Margin:             {margin:.6f}")

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

    model = BinaryRNNScorer(
        input_size=len(FEATURE_COLUMNS),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
    ).to(DEVICE)

    loss_fn = APLoss(
        data_len=len(train_dataset),
        gamma=gamma,
        margin=margin,
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
        milestones=LR_DECAY_EPOCHS,
        gamma=LR_DECAY_FACTOR,
    )

    best_val_ap = -1.0
    best_state = state_dict_to_cpu(model.state_dict())
    train_start = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
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
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"APLoss: {avg_loss:.6f} | "
            f"Val AP: {val_ap:.6f} | "
            f"Best Val AP: {best_val_ap:.6f} | "
            f"LR: {current_lr:.6f}"
        )

        scheduler.step()

    model.load_state_dict(best_state)
    train_time_seconds = time.time() - train_start

    return model, train_time_seconds, best_val_ap


def run_single_config(
    config_id,
    gamma,
    margin,
    walk_sampling,
    bus_sampling,
    car_sampling,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test=None,
    y_test=None,
):
    print("\n============================================================")
    print(f"CONFIG {config_id}")
    print(
        f"gamma={gamma}, margin={margin}, "
        f"walk_sampling={walk_sampling}, bus_sampling={bus_sampling}, car_sampling={car_sampling}"
    )
    print("============================================================")

    sampling_by_class = {
        "walk": walk_sampling,
        "bus": bus_sampling,
        "car": car_sampling,
    }

    models = {}
    train_times = {}
    val_results_by_class = {}

    for target_class in [0, 1, 2]:
        class_name = IDX_TO_LABEL[target_class]

        model, train_time_seconds, best_val_ap = train_libauc_binary_model(
            X_train=X_train,
            y_train_multiclass=y_train,
            target_class=target_class,
            X_val=X_val,
            y_val_multiclass=y_val,
            gamma=gamma,
            margin=margin,
            sampling_rate=sampling_by_class[class_name],
        )

        models[class_name] = model
        train_times[class_name] = train_time_seconds

        val_results_by_class[class_name] = evaluate_binary_model(
            model=model,
            X_eval=X_val,
            y_eval_multiclass=y_val,
            target_class=target_class,
        )

    val_macro_ap = float(np.mean([
        val_results_by_class["walk"]["ap"],
        val_results_by_class["bus"]["ap"],
        val_results_by_class["car"]["ap"],
    ]))

    row = {
        "config_id": config_id,
        "gamma": gamma,
        "margin": margin,
        "walk_sampling": walk_sampling,
        "bus_sampling": bus_sampling,
        "car_sampling": car_sampling,
        "val_ap_walk": val_results_by_class["walk"]["ap"],
        "val_ap_bus": val_results_by_class["bus"]["ap"],
        "val_ap_car": val_results_by_class["car"]["ap"],
        "val_macro_ap": val_macro_ap,
        "train_time_walk_sec": train_times["walk"],
        "train_time_bus_sec": train_times["bus"],
        "train_time_car_sec": train_times["car"],
        "train_time_total_sec": train_times["walk"] + train_times["bus"] + train_times["car"],
    }

    print("\nVALIDATION RESULTS FOR THIS CONFIG")
    print(f"  AP_walk      : {row['val_ap_walk']:.6f}")
    print(f"  AP_bus       : {row['val_ap_bus']:.6f}")
    print(f"  AP_car       : {row['val_ap_car']:.6f}")
    print(f"  Macro AUPRC  : {row['val_macro_ap']:.6f}")
    print(f"  Total time(s): {row['train_time_total_sec']:.2f}")

    test_results_by_class = None
    if EVALUATE_EVERY_CONFIG_ON_TEST and X_test is not None and y_test is not None:
        test_results_by_class = {}
        for target_class in [0, 1, 2]:
            class_name = IDX_TO_LABEL[target_class]
            test_results_by_class[class_name] = evaluate_binary_model(
                model=models[class_name],
                X_eval=X_test,
                y_eval_multiclass=y_test,
                target_class=target_class,
            )
        row["test_ap_walk"] = test_results_by_class["walk"]["ap"]
        row["test_ap_bus"] = test_results_by_class["bus"]["ap"]
        row["test_ap_car"] = test_results_by_class["car"]["ap"]
        row["test_macro_ap"] = float(np.mean([
            test_results_by_class["walk"]["ap"],
            test_results_by_class["bus"]["ap"],
            test_results_by_class["car"]["ap"],
        ]))
        print(f"  TEST Macro AUPRC (not for selection): {row['test_macro_ap']:.6f}")

    return row, models, val_results_by_class, test_results_by_class


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

    total_configs = (
        len(GAMMA_LIST)
        * len(MARGIN_LIST)
        * len(WALK_SAMPLING_LIST)
        * len(BUS_SAMPLING_LIST)
        * len(CAR_SAMPLING_LIST)
    )

    print("\n============================================================")
    print(f"TOTAL CONFIGURATIONS TO TRY: {total_configs}")
    print("============================================================")

    all_rows = []

    best_val_macro = -1.0
    best_config_row = None
    best_models_state = None

    config_id = 0

    for gamma, margin, walk_sampling, bus_sampling, car_sampling in product(
        GAMMA_LIST,
        MARGIN_LIST,
        WALK_SAMPLING_LIST,
        BUS_SAMPLING_LIST,
        CAR_SAMPLING_LIST,
    ):
        config_id += 1

        row, models, _, _ = run_single_config(
            config_id=config_id,
            gamma=gamma,
            margin=margin,
            walk_sampling=walk_sampling,
            bus_sampling=bus_sampling,
            car_sampling=car_sampling,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
        )

        all_rows.append(row)

        if row["val_macro_ap"] > best_val_macro:
            best_val_macro = row["val_macro_ap"]
            best_config_row = row.copy()
            best_models_state = {
                class_name: state_dict_to_cpu(models[class_name].state_dict())
                for class_name in ["walk", "bus", "car"]
            }

        del models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_df = pd.DataFrame(all_rows)
    summary_df = summary_df.sort_values("val_macro_ap", ascending=False).reset_index(drop=True)

    results_csv_path = OUTPUT_DIR / "libauc_sweep_results.csv"
    summary_df.to_csv(results_csv_path, index=False)

    avg_val_macro = float(summary_df["val_macro_ap"].mean())

    print("\n============================================================")
    print("SWEEP FINISHED")
    print("============================================================")
    print(f"Average validation Macro AUPRC across all configs: {avg_val_macro:.6f}")
    print(f"Best validation Macro AUPRC: {best_config_row['val_macro_ap']:.6f}")
    print("\nBest configuration:")
    print(best_config_row)
    print(f"\nFull sweep results saved to: {results_csv_path}")

    print("\nTop 10 configurations by validation Macro AUPRC:")
    print(summary_df.head(10).to_string(index=False))

    best_models = {}
    for class_name in ["walk", "bus", "car"]:
        model = BinaryRNNScorer(
            input_size=len(FEATURE_COLUMNS),
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
        ).to(DEVICE)
        model.load_state_dict(best_models_state[class_name])
        model.eval()
        best_models[class_name] = model

    best_test_results_by_class = {}
    for target_class in [0, 1, 2]:
        class_name = IDX_TO_LABEL[target_class]
        best_test_results_by_class[class_name] = evaluate_binary_model(
            model=best_models[class_name],
            X_eval=X_test,
            y_eval_multiclass=y_test,
            target_class=target_class,
        )

    best_test_macro_ap = float(np.mean([
        best_test_results_by_class["walk"]["ap"],
        best_test_results_by_class["bus"]["ap"],
        best_test_results_by_class["car"]["ap"],
    ]))

    diagnostics = multiclass_diagnostics_from_ovr(best_test_results_by_class, y_test)

    print("\n==================== BEST-CONFIG TEST RESULTS ====================")
    print("Best config selected by validation Macro AUPRC")
    print(f"Validation Macro AUPRC: {best_config_row['val_macro_ap']:.6f}")
    print(f"Test Macro AUPRC:       {best_test_macro_ap:.6f}")

    print("\nTest AP / AUPRC per class (one-vs-rest):")
    for class_name in ["walk", "bus", "car"]:
        print(f"{class_name:>5} | AP: {best_test_results_by_class[class_name]['ap']:.6f}")

    print("\nSecondary diagnostics on test:")
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

    pr_plot_path = OUTPUT_DIR / "pr_curves_libauc_best_config_test.png"
    plot_pr_curves(
        best_test_results_by_class,
        pr_plot_path,
        title="LibAUC Best-Config Precision-Recall Curves on Test",
    )
    print(f"\nBest-config PR curves saved to: {pr_plot_path}")


if __name__ == "__main__":
    main()