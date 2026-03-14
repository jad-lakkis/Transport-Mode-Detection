from pathlib import Path
import json
import re
import numpy as np
import pandas as pd


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
DATA_DIR = CELLMOB_ROOT / "Data"

# Input: cleaned row-level files before any class-wise standardization
INPUT_DIR = DATA_DIR / "zdata_unfinished"

# Output
OUTPUT_ROOT = DATA_DIR / "fixed_32k_windows"
TRAIN_OUT_DIR = OUTPUT_ROOT / "train"
TEST_OUT_DIR = OUTPUT_ROOT / "test"

WINDOW_SIZE = 5
MAX_WINDOW_SPAN_SECONDS = 3.5

TOTAL_WINDOWS_PER_CITY_CLASS = 32000
TEST_RATIO = 0.20
TEST_WINDOWS = int(TOTAL_WINDOWS_PER_CITY_CLASS * TEST_RATIO)   # 6400
TRAIN_WINDOWS = TOTAL_WINDOWS_PER_CITY_CLASS - TEST_WINDOWS     # 25600

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

# Only the 3 cities and 3 labels you asked for
CITY_CLASS_FILES = {
    "kaust": {
        "walk": INPUT_DIR / "walk_kaust_cleaned.csv",
        "bus": INPUT_DIR / "bus_colored_kaust_cleaned.csv",
        "car": INPUT_DIR / "car_kaust_cleaned.csv",
    },
    "jeddah": {
        "walk": INPUT_DIR / "walk_jeddah_cleaned.csv",
        "bus": INPUT_DIR / "bus_jeddah_cleaned.csv",
        "car": INPUT_DIR / "car_jeddah_cleaned.csv",
    },
    "mekkah": {
        "walk": INPUT_DIR / "walk_mekkah_cleaned.csv",
        "bus": INPUT_DIR / "bus_mekkah_cleaned.csv",
        "car": INPUT_DIR / "car_mekkah_cleaned.csv",
    },
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


def clean_numeric_value(x):
    s = str(x).strip()
    s = s.replace(",", "")
    s = re.sub(r"[^0-9.\-+]", "", s)
    return s


def load_dataframe(csv_path: Path) -> pd.DataFrame:
    """
    Load one cleaned CSV and force all feature columns to numeric.
    This prevents object/string columns from breaking standardization.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path, skipinitialspace=True)

    required_cols = [TIME_COLUMN] + FEATURE_COLUMNS
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path.name} is missing columns: {missing}")

    df = df[required_cols].copy()
    df.columns = [c.strip() for c in df.columns]

    df[TIME_COLUMN] = df[TIME_COLUMN].astype(str).str.strip()

    for col in FEATURE_COLUMNS:
        df[col] = df[col].astype(str).map(clean_numeric_value)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[TIME_COLUMN] + FEATURE_COLUMNS).reset_index(drop=True)
    df["time_seconds"] = df[TIME_COLUMN].apply(time_to_seconds)

    return df


def valid_window_starts(time_seconds: np.ndarray) -> np.ndarray:
    starts = []

    for i in range(len(time_seconds) - WINDOW_SIZE + 1):
        span = time_seconds[i + WINDOW_SIZE - 1] - time_seconds[i]
        if 0 <= span <= MAX_WINDOW_SPAN_SECONDS:
            starts.append(i)

    return np.array(starts, dtype=np.int64)


def choose_fixed_train_test_starts(starts: np.ndarray):
    """
    Choose exact numbers of valid windows:
      - first TRAIN_WINDOWS for train
      - last TEST_WINDOWS for test

    and make sure the underlying row spans do not overlap.
    """
    total_needed = TRAIN_WINDOWS + TEST_WINDOWS

    if len(starts) < total_needed:
        raise ValueError(
            f"Not enough valid windows. Needed {total_needed}, found {len(starts)}."
        )

    train_starts = starts[:TRAIN_WINDOWS]
    test_starts = starts[-TEST_WINDOWS:]

    train_last_start = int(train_starts[-1])
    test_first_start = int(test_starts[0])

    # Train uses rows up to train_last_start + WINDOW_SIZE - 1
    # Test starts at test_first_start
    if train_last_start + WINDOW_SIZE > test_first_start:
        raise ValueError(
            "Cannot create exact non-overlapping train/test window subsets with the current counts. "
            f"train_last_start={train_last_start}, test_first_start={test_first_start}, "
            f"WINDOW_SIZE={WINDOW_SIZE}. Reduce counts or use files with more valid windows."
        )

    return train_starts, test_starts


def row_subset_for_window_starts(df: pd.DataFrame, starts: np.ndarray) -> pd.DataFrame:
    """
    Keep the minimal contiguous row block that contains the chosen windows.
    """
    start_row = int(starts[0])
    end_row_exclusive = int(starts[-1] + WINDOW_SIZE)
    return df.iloc[start_row:end_row_exclusive].copy().reset_index(drop=True)


def build_windows_from_subset(df_subset: pd.DataFrame, class_name: str):
    """
    Rebuild valid windows from the chosen subset after standardization.
    """
    features = df_subset[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    times = df_subset["time_seconds"].to_numpy(dtype=np.float64)

    X = []
    y = []

    for i in range(len(df_subset) - WINDOW_SIZE + 1):
        span = times[i + WINDOW_SIZE - 1] - times[i]
        if 0 <= span <= MAX_WINDOW_SPAN_SECONDS:
            X.append(features[i:i + WINDOW_SIZE])
            y.append(LABEL_MAP[class_name])

    if len(X) == 0:
        X = np.empty((0, WINDOW_SIZE, len(FEATURE_COLUMNS)), dtype=np.float32)
        y = np.empty((0,), dtype=np.int64)
    else:
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int64)

    return X, y


def fit_global_standardizer(train_row_subsets: list[pd.DataFrame]):
    """
    Fit ONE shared scaler using ONLY the selected training rows
    from all city/class subsets combined.
    """
    combined_train = pd.concat(
        [df[FEATURE_COLUMNS].copy() for df in train_row_subsets],
        axis=0,
        ignore_index=True,
    )

    for col in FEATURE_COLUMNS:
        combined_train[col] = pd.to_numeric(combined_train[col], errors="coerce")

    if combined_train[FEATURE_COLUMNS].isna().any().any():
        bad_counts = combined_train[FEATURE_COLUMNS].isna().sum()
        raise ValueError(
            "NaNs appeared in combined_train after numeric conversion.\n"
            f"{bad_counts[bad_counts > 0]}"
        )

    mean = combined_train[FEATURE_COLUMNS].mean()
    std = combined_train[FEATURE_COLUMNS].std(ddof=0)
    std = std.replace(0, 1.0)

    return mean, std


def apply_standardization(df: pd.DataFrame, mean: pd.Series, std: pd.Series) -> pd.DataFrame:
    out = df.copy()
    out[FEATURE_COLUMNS] = (out[FEATURE_COLUMNS] - mean) / std
    return out


def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory does not exist: {INPUT_DIR}")

    TRAIN_OUT_DIR.mkdir(parents=True, exist_ok=True)
    TEST_OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"CellMob root: {CELLMOB_ROOT}")
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output root: {OUTPUT_ROOT}")
    print(f"Train windows per (city,class): {TRAIN_WINDOWS}")
    print(f"Test windows per (city,class): {TEST_WINDOWS}")

    selected = {}
    train_row_subsets = []
    summary_rows = []

    # Step 1: load only the requested 3 cities x 3 labels and choose fixed windows
    for city_name, class_files in CITY_CLASS_FILES.items():
        selected[city_name] = {}

        for class_name, path in class_files.items():
            print(f"\nProcessing {city_name} / {class_name}: {path.name}")

            df = load_dataframe(path)
            starts = valid_window_starts(df["time_seconds"].to_numpy())

            print(f"Total rows: {len(df)}")
            print(f"Total valid windows available: {len(starts)}")

            train_starts, test_starts = choose_fixed_train_test_starts(starts)

            train_df_subset = row_subset_for_window_starts(df, train_starts)
            test_df_subset = row_subset_for_window_starts(df, test_starts)

            selected[city_name][class_name] = {
                "train_starts": train_starts,
                "test_starts": test_starts,
                "train_df_subset": train_df_subset,
                "test_df_subset": test_df_subset,
            }

            train_row_subsets.append(train_df_subset)

            summary_rows.append({
                "city": city_name,
                "class": class_name,
                "source_file": path.name,
                "total_rows_in_source": len(df),
                "total_valid_windows_in_source": int(len(starts)),
                "selected_train_windows": int(len(train_starts)),
                "selected_test_windows": int(len(test_starts)),
                "train_subset_rows": int(len(train_df_subset)),
                "test_subset_rows": int(len(test_df_subset)),
            })

    # Step 2: fit one shared scaler on all selected training rows only
    global_mean, global_std = fit_global_standardizer(train_row_subsets)

    scaler_info = {
        "feature_means": {k: float(v) for k, v in global_mean.to_dict().items()},
        "feature_stds": {k: float(v) for k, v in global_std.to_dict().items()},
        "window_size": WINDOW_SIZE,
        "max_window_span_seconds": MAX_WINDOW_SPAN_SECONDS,
        "total_windows_per_city_class": TOTAL_WINDOWS_PER_CITY_CLASS,
        "train_windows_per_city_class": TRAIN_WINDOWS,
        "test_windows_per_city_class": TEST_WINDOWS,
    }

    # Step 3: standardize subsets, rebuild exact windows, save NPZ files
    for city_name in selected:
        for class_name in selected[city_name]:
            info = selected[city_name][class_name]

            train_std = apply_standardization(info["train_df_subset"], global_mean, global_std)
            test_std = apply_standardization(info["test_df_subset"], global_mean, global_std)

            X_train, y_train = build_windows_from_subset(train_std, class_name)
            X_test, y_test = build_windows_from_subset(test_std, class_name)

            if len(X_train) != TRAIN_WINDOWS:
                raise RuntimeError(
                    f"{city_name}/{class_name}: expected {TRAIN_WINDOWS} train windows, got {len(X_train)}."
                )

            if len(X_test) != TEST_WINDOWS:
                raise RuntimeError(
                    f"{city_name}/{class_name}: expected {TEST_WINDOWS} test windows, got {len(X_test)}."
                )

            train_out = TRAIN_OUT_DIR / f"{class_name}_{city_name}_train_25600windows.npz"
            test_out = TEST_OUT_DIR / f"{class_name}_{city_name}_test_6400windows.npz"

            np.savez_compressed(
                train_out,
                X=X_train,
                y=y_train,
                city=city_name,
                label=class_name,
            )

            np.savez_compressed(
                test_out,
                X=X_test,
                y=y_test,
                city=city_name,
                label=class_name,
            )

            print(f"Saved train: {train_out} | X shape = {X_train.shape}")
            print(f"Saved test : {test_out} | X shape = {X_test.shape}")

    # Step 4: save metadata
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = OUTPUT_ROOT / "fixed_32k_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    scaler_json = OUTPUT_ROOT / "global_standardizer_info.json"
    scaler_json.write_text(json.dumps(scaler_info, indent=2), encoding="utf-8")

    label_map_json = OUTPUT_ROOT / "label_map.json"
    label_map_json.write_text(json.dumps(LABEL_MAP, indent=2), encoding="utf-8")

    readme_txt = OUTPUT_ROOT / "README.txt"
    readme_txt.write_text(
        "\n".join([
            "Fixed 32k window dataset",
            "========================",
            "",
            f"Cities: {', '.join(CITY_CLASS_FILES.keys())}",
            "Classes: walk, bus, car",
            f"Window size: {WINDOW_SIZE}",
            f"Max valid window span: {MAX_WINDOW_SPAN_SECONDS} seconds",
            f"Total windows per (city,class): {TOTAL_WINDOWS_PER_CITY_CLASS}",
            f"Train windows per (city,class): {TRAIN_WINDOWS}",
            f"Test windows per (city,class): {TEST_WINDOWS}",
            "",
            "Standardization:",
            "- One shared global scaler",
            "- Fit ONLY on selected training rows from all city/class subsets combined",
            "- Applied to all train/test subsets",
            "",
            "Files:",
            "- train/*.npz : exact train windows",
            "- test/*.npz  : exact test windows",
            "- fixed_32k_summary.csv",
            "- global_standardizer_info.json",
            "- label_map.json",
        ]),
        encoding="utf-8",
    )

    print(f"\nDone. Output folder: {OUTPUT_ROOT}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Scaler info : {scaler_json}")
    print(f"Label map   : {label_map_json}")
    print(f"README      : {readme_txt}")


if __name__ == "__main__":
    main()