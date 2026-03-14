from pathlib import Path
from copy import deepcopy
import numpy as np
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

FIXED_DATA_DIR = DATA_DIR / "fixed_32k_windows"
TRAIN_DIR = FIXED_DATA_DIR / "train"
TEST_DIR = FIXED_DATA_DIR / "test"

OUTPUT_ROOT = SCRIPT_DIR / "outputs_federated_models"
RUN_OUTPUT_DIR = OUTPUT_ROOT / "fedavg_fixed_32k_windows_corrected"
RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Hyperparameters
# -----------------------------
BATCH_SIZE = 128
HIDDEN_SIZE = 128
NUM_LAYERS = 2
LEARNING_RATE = 1e-3

GLOBAL_ROUNDS = 15
LOCAL_EPOCHS = 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_MAP = {
    "walk": 0,
    "bus": 1,
    "car": 2,
}
IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

CITY_FILES = {
    "kaust": {
        "train": {
            "walk": TRAIN_DIR / "walk_kaust_train_25600windows.npz",
            "bus": TRAIN_DIR / "bus_kaust_train_25600windows.npz",
            "car": TRAIN_DIR / "car_kaust_train_25600windows.npz",
        },
        "test": {
            "walk": TEST_DIR / "walk_kaust_test_6400windows.npz",
            "bus": TEST_DIR / "bus_kaust_test_6400windows.npz",
            "car": TEST_DIR / "car_kaust_test_6400windows.npz",
        },
    },
    "jeddah": {
        "train": {
            "walk": TRAIN_DIR / "walk_jeddah_train_25600windows.npz",
            "bus": TRAIN_DIR / "bus_jeddah_train_25600windows.npz",
            "car": TRAIN_DIR / "car_jeddah_train_25600windows.npz",
        },
        "test": {
            "walk": TEST_DIR / "walk_jeddah_test_6400windows.npz",
            "bus": TEST_DIR / "bus_jeddah_test_6400windows.npz",
            "car": TEST_DIR / "car_jeddah_test_6400windows.npz",
        },
    },
    "mekkah": {
        "train": {
            "walk": TRAIN_DIR / "walk_mekkah_train_25600windows.npz",
            "bus": TRAIN_DIR / "bus_mekkah_train_25600windows.npz",
            "car": TRAIN_DIR / "car_mekkah_train_25600windows.npz",
        },
        "test": {
            "walk": TEST_DIR / "walk_mekkah_test_6400windows.npz",
            "bus": TEST_DIR / "bus_mekkah_test_6400windows.npz",
            "car": TEST_DIR / "car_mekkah_test_6400windows.npz",
        },
    },
}


def load_npz_dataset(npz_path: Path):
    if not npz_path.exists():
        raise FileNotFoundError(f"File not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)

    if X.ndim != 3:
        raise ValueError(f"{npz_path.name}: expected X to have 3 dims, got shape {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"{npz_path.name}: expected y to have 1 dim, got shape {y.shape}")
    if len(X) != len(y):
        raise ValueError(f"{npz_path.name}: X and y length mismatch: {len(X)} vs {len(y)}")

    return X, y


def build_city_dataset(file_dict):
    X_all = []
    y_all = []

    for class_name, path in file_dict.items():
        X, y = load_npz_dataset(path)

        print(f"{path.name} | class={class_name} | X shape={X.shape} | y shape={y.shape}")

        X_all.append(X)
        y_all.append(y)

    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)

    return X_all, y_all


def compute_class_weights_from_labels(y: np.ndarray, num_classes: int = 3) -> torch.Tensor:
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    total = counts.sum()

    if np.any(counts == 0):
        raise ValueError(f"At least one class has zero samples in training data. Counts: {counts}")

    weights = total / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


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


def weighted_mean(values, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    return float(np.sum(values * weights) / np.sum(weights))


def plot_training_loss(loss_history, save_path: Path):
    plt.figure(figsize=(8, 5))
    rounds = np.arange(1, len(loss_history) + 1)
    plt.plot(rounds, loss_history, marker="o")
    plt.xlabel("Global Round")
    plt.ylabel("Weighted Avg Local Train Loss")
    plt.title("Federated Training Loss vs Global Round")
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
    plt.title("Precision-Recall Curves (one-vs-rest)")
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


def plot_ap_bar(ap_per_class, macro_ap, save_path: Path):
    class_names = ["walk", "bus", "car"]
    values = [ap_per_class[c] for c in class_names]

    plt.figure(figsize=(7, 5))
    bars = plt.bar(class_names, values)
    plt.axhline(macro_ap, linestyle="--", label=f"Macro AP = {macro_ap:.4f}")
    plt.ylabel("Average Precision (AP)")
    plt.title("AP / AUPRC per Class")
    plt.ylim(0, 1.05)
    plt.legend()

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.4f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_per_class_metrics(per_class_metrics, save_path: Path):
    class_names = ["walk", "bus", "car"]
    precision_vals = [per_class_metrics[c]["precision"] for c in class_names]
    recall_vals = [per_class_metrics[c]["recall"] for c in class_names]
    f1_vals = [per_class_metrics[c]["f1"] for c in class_names]

    x = np.arange(len(class_names))
    width = 0.25

    plt.figure(figsize=(9, 5))
    b1 = plt.bar(x - width, precision_vals, width, label="Precision")
    b2 = plt.bar(x, recall_vals, width, label="Recall")
    b3 = plt.bar(x + width, f1_vals, width, label="F1-score")

    plt.xticks(x, class_names)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Per-Class Precision / Recall / F1")
    plt.legend()

    for bars in [b1, b2, b3]:
        for bar in bars:
            value = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.02,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

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

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        average=None,
        zero_division=0,
    )

    per_class_metrics = {}
    for cls_idx in [0, 1, 2]:
        class_name = IDX_TO_LABEL[cls_idx]
        per_class_metrics[class_name] = {
            "precision": float(precision[cls_idx]),
            "recall": float(recall[cls_idx]),
            "f1": float(f1[cls_idx]),
            "support": int(support[cls_idx]),
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
    macro_precision = float(np.mean([per_class_metrics[c]["precision"] for c in ["walk", "bus", "car"]]))
    macro_recall = float(np.mean([per_class_metrics[c]["recall"] for c in ["walk", "bus", "car"]]))
    macro_f1 = float(np.mean([per_class_metrics[c]["f1"] for c in ["walk", "bus", "car"]]))

    return {
        "accuracy": float(acc),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
        "per_class_metrics": per_class_metrics,
        "ap_per_class": ap_per_class,
        "macro_ap": macro_ap,
        "pr_curves": pr_curves,
    }


def print_class_distribution(name, y):
    unique, counts = np.unique(y, return_counts=True)
    print(f"\n{name} class distribution:")
    for u, c in zip(unique, counts):
        print(f"  {IDX_TO_LABEL[int(u)]}: {int(c)}")


def save_metrics_summary(results, round_loss_history, client_info, global_class_weights, save_path: Path):
    lines = []
    lines.append("==================== FEDERATED RESULTS ====================")
    lines.append(f"Shared global class weights: {global_class_weights.tolist()}")
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
    lines.append("Client training sample counts:")
    for city_name, info in client_info.items():
        lines.append(f"{city_name:>7} | train_windows={info['train_windows']} | test_windows={info['test_windows']}")

    lines.append("")
    lines.append("Weighted average local train loss per global round:")
    for i, loss in enumerate(round_loss_history, start=1):
        lines.append(f"Round {i:02d}: {loss:.6f}")

    save_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"Train directory does not exist: {TRAIN_DIR}")
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Test directory does not exist: {TEST_DIR}")

    print(f"CellMob root: {CELLMOB_ROOT}")
    print(f"Train directory: {TRAIN_DIR}")
    print(f"Test directory: {TEST_DIR}")
    print(f"Run output directory: {RUN_OUTPUT_DIR}")

    client_dataloaders = {}
    client_sample_counts = []
    client_info = {}

    X_train_all = []
    y_train_all = []
    X_test_all = []
    y_test_all = []

    for city_name, split_files in CITY_FILES.items():
        print(f"\n==================== Building client: {city_name.upper()} ====================")

        print(f"Building {city_name} TRAIN dataset...")
        X_train_city, y_train_city = build_city_dataset(split_files["train"])

        print(f"\nBuilding {city_name} TEST dataset...")
        X_test_city, y_test_city = build_city_dataset(split_files["test"])

        print(f"\n{city_name.upper()} shapes:")
        print("X_train:", X_train_city.shape)
        print("y_train:", y_train_city.shape)
        print("X_test :", X_test_city.shape)
        print("y_test :", y_test_city.shape)

        print_class_distribution(f"{city_name.upper()} Train", y_train_city)
        print_class_distribution(f"{city_name.upper()} Test", y_test_city)

        train_dataset_city = SequenceDataset(X_train_city, y_train_city)
        test_dataset_city = SequenceDataset(X_test_city, y_test_city)

        train_loader_city = DataLoader(train_dataset_city, batch_size=BATCH_SIZE, shuffle=True)
        test_loader_city = DataLoader(test_dataset_city, batch_size=BATCH_SIZE, shuffle=False)

        client_dataloaders[city_name] = {
            "train_loader": train_loader_city,
            "test_loader": test_loader_city,
        }

        client_sample_counts.append(len(train_dataset_city))
        client_info[city_name] = {
            "train_windows": len(train_dataset_city),
            "test_windows": len(test_dataset_city),
        }

        X_train_all.append(X_train_city)
        y_train_all.append(y_train_city)
        X_test_all.append(X_test_city)
        y_test_all.append(y_test_city)

    X_train_global = np.concatenate(X_train_all, axis=0)
    y_train_global = np.concatenate(y_train_all, axis=0)
    X_test_global = np.concatenate(X_test_all, axis=0)
    y_test_global = np.concatenate(y_test_all, axis=0)

    global_class_weights = compute_class_weights_from_labels(y_train_global, num_classes=3)

    print("\n==================== Combined Global Sets ====================")
    print("X_train_global:", X_train_global.shape)
    print("y_train_global:", y_train_global.shape)
    print("X_test_global :", X_test_global.shape)
    print("y_test_global :", y_test_global.shape)

    print_class_distribution("Global Train", y_train_global)
    print_class_distribution("Global Test", y_test_global)

    print(f"\nShared global class weights: {global_class_weights.tolist()}")

    global_test_dataset = SequenceDataset(X_test_global, y_test_global)
    global_test_loader = DataLoader(global_test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    global_model = RNNClassifier(
        input_size=X_train_global.shape[2],
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=3,
    ).to(DEVICE)

    print(f"\nStarting federated training on {DEVICE} ...")
    print(f"Clients: {list(CITY_FILES.keys())}")
    print(f"Global rounds: {GLOBAL_ROUNDS}")
    print(f"Local epochs per round: {LOCAL_EPOCHS}")

    round_loss_history = []

    for round_idx in range(1, GLOBAL_ROUNDS + 1):
        print(f"\n==================== Global Round {round_idx:02d}/{GLOBAL_ROUNDS} ====================")

        local_state_dicts = []
        local_avg_losses = []

        for city_name in CITY_FILES.keys():
            print(f"\n--- Client: {city_name.upper()} ---")

            local_model = RNNClassifier(
                input_size=X_train_global.shape[2],
                hidden_size=HIDDEN_SIZE,
                num_layers=NUM_LAYERS,
                num_classes=3,
            ).to(DEVICE)

            local_model.load_state_dict(deepcopy(global_model.state_dict()))

            optimizer = torch.optim.Adam(local_model.parameters(), lr=LEARNING_RATE)
            criterion = nn.CrossEntropyLoss(weight=global_class_weights.to(DEVICE))

            local_loader = client_dataloaders[city_name]["train_loader"]

            epoch_losses = []
            for local_epoch in range(1, LOCAL_EPOCHS + 1):
                local_loss = train_one_epoch(local_model, local_loader, optimizer, criterion)
                epoch_losses.append(local_loss)
                print(
                    f"Client {city_name.upper()} | "
                    f"Local Epoch {local_epoch:02d}/{LOCAL_EPOCHS} | "
                    f"Loss: {local_loss:.6f}"
                )

            local_avg_loss = float(np.mean(epoch_losses))
            local_avg_losses.append(local_avg_loss)

            local_state_dicts.append(deepcopy(local_model.state_dict()))

        aggregated_state = fedavg_weighted_aggregate(
            local_state_dicts=local_state_dicts,
            client_names=list(CITY_FILES.keys()),
            client_sample_counts=client_sample_counts,
            verbose=True,
        )
        global_model.load_state_dict(aggregated_state)

        # corrected: weighted average loss per round, consistent with FedAvg
        round_weighted_loss = weighted_mean(local_avg_losses, client_sample_counts)
        round_loss_history.append(round_weighted_loss)

        print(f"\nRound {round_idx:02d} weighted average local loss: {round_weighted_loss:.6f}")

    print("\nEvaluating final global federated model on the combined global test set...")
    results = evaluate(global_model, global_test_loader)

    print("\n==================== FEDERATED RESULTS ====================")
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
            f"Support={m['support']}"
        )

    print("\nAP / AUPRC per class:")
    for class_name in ["walk", "bus", "car"]:
        print(f"{class_name:>5} | AP: {results['ap_per_class'][class_name]:.6f}")

    print("\nConfusion Matrix [rows=true, cols=pred]:")
    print(results["confusion_matrix"])

    loss_plot_path = RUN_OUTPUT_DIR / "federated_training_loss.png"
    pr_plot_path = RUN_OUTPUT_DIR / "pr_curves.png"
    cm_plot_path = RUN_OUTPUT_DIR / "confusion_matrix.png"
    ap_bar_plot_path = RUN_OUTPUT_DIR / "ap_per_class.png"
    class_metrics_plot_path = RUN_OUTPUT_DIR / "per_class_precision_recall_f1.png"
    metrics_txt_path = RUN_OUTPUT_DIR / "metrics_summary.txt"
    model_path = RUN_OUTPUT_DIR / "fedavg_global_rnn_fixed_32k_windows_corrected.pth"

    plot_training_loss(round_loss_history, loss_plot_path)
    plot_pr_curves(results["pr_curves"], pr_plot_path)
    plot_confusion_matrix_figure(results["confusion_matrix"], cm_plot_path)
    plot_ap_bar(results["ap_per_class"], results["macro_ap"], ap_bar_plot_path)
    plot_per_class_metrics(results["per_class_metrics"], class_metrics_plot_path)
    save_metrics_summary(results, round_loss_history, client_info, global_class_weights, metrics_txt_path)

    torch.save(global_model.state_dict(), model_path)

    print(f"\nSaved federated training loss plot: {loss_plot_path}")
    print(f"Saved PR curves plot: {pr_plot_path}")
    print(f"Saved confusion matrix plot: {cm_plot_path}")
    print(f"Saved AP bar plot: {ap_bar_plot_path}")
    print(f"Saved per-class metrics plot: {class_metrics_plot_path}")
    print(f"Saved metrics summary: {metrics_txt_path}")
    print(f"Saved global model: {model_path}")


if __name__ == "__main__":
    main()