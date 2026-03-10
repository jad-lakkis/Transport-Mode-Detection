from pathlib import Path
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
INPUT_DIR = DATA_DIR / "zdata_unfinished"
OUTPUT_DIR = DATA_DIR / "6400 KAUST"

CLASS_FILES = {
    "bus": "bus_colored_kaust_cleaned.csv",
    "car": "car_kaust_cleaned.csv",
    "walk": "walk_kaust_cleaned.csv",
}

TIME_COL = "time"
FEATURE_COLS = [
    "rsrp1", "rsrp2", "rsrp3", "rsrp4",
    "rssi1", "rssi2", "rssi3", "rssi4",
    "rsrq1", "rsrq2", "rsrq3", "rsrq4",
]

WINDOW_SIZE = 5
MAX_SPAN_SECONDS = 3.5
N_TEST_WINDOWS = 6400


def time_to_seconds(x):
    s = str(x).strip()
    h, m, sec = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(sec)


def clean_numeric_value(x):
    s = str(x).strip()
    s = s.replace(",", "")
    s = re.sub(r"[^0-9.\-+]", "", s)
    return s


def load_file(path):
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]

    needed = [TIME_COL] + FEATURE_COLS
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")

    df = df[needed].copy()
    df[TIME_COL] = df[TIME_COL].astype(str).str.strip()

    for col in FEATURE_COLS:
        df[col] = df[col].astype(str).map(clean_numeric_value)
        bad_mask = (df[col] == "") | (df[col] == "-") | (df[col] == "+") | (df[col] == ".")
        if bad_mask.any():
            bad_rows = df.index[bad_mask].tolist()[:10]
            raise ValueError(f"{path.name}: invalid values in column {col} at rows {bad_rows}")
        df[col] = pd.to_numeric(df[col], errors="raise")

    df["time_seconds"] = df[TIME_COL].apply(time_to_seconds)
    return df


def valid_window_starts(time_seconds):
    starts = []
    n = len(time_seconds)

    for i in range(n - WINDOW_SIZE + 1):
        span = time_seconds[i + WINDOW_SIZE - 1] - time_seconds[i]
        if 0 <= span <= MAX_SPAN_SECONDS:
            starts.append(i)

    return np.array(starts, dtype=int)


def standardize_with_train_only(train_df, test_df):
    mean = train_df[FEATURE_COLS].mean()
    std = train_df[FEATURE_COLS].std(ddof=0)
    std = std.replace(0, 1.0)

    train_out = train_df.copy()
    test_out = test_df.copy()

    train_out[FEATURE_COLS] = (train_out[FEATURE_COLS] - mean) / std
    test_out[FEATURE_COLS] = (test_out[FEATURE_COLS] - mean) / std

    return train_out, test_out


def process_class(class_name, filename):
    input_path = INPUT_DIR / filename
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    print(f"\nProcessing {class_name}: {input_path.name}")
    df = load_file(input_path)

    starts = valid_window_starts(df["time_seconds"].to_numpy())
    total_valid = len(starts)
    print(f"Total rows: {len(df)}")
    print(f"Total valid windows: {total_valid}")

    if total_valid < N_TEST_WINDOWS:
        raise ValueError(
            f"{input_path.name} has only {total_valid} valid windows, but {N_TEST_WINDOWS} are required."
        )

    test_start_row = int(starts[-N_TEST_WINDOWS])

    train_df = df.iloc[:test_start_row].copy()
    test_df = df.iloc[test_start_row:].copy()

    if len(train_df) < WINDOW_SIZE:
        raise ValueError(f"{input_path.name}: training split became too small.")

    train_valid = len(valid_window_starts(train_df["time_seconds"].to_numpy()))
    test_valid = len(valid_window_starts(test_df["time_seconds"].to_numpy()))

    if test_valid != N_TEST_WINDOWS:
        raise RuntimeError(
            f"{input_path.name}: expected {N_TEST_WINDOWS} test windows, got {test_valid}."
        )

    train_std, test_std = standardize_with_train_only(train_df, test_df)

    train_std = train_std.drop(columns=["time_seconds"])
    test_std = test_std.drop(columns=["time_seconds"])

    train_out = OUTPUT_DIR / f"{class_name}_train_kaust_standardized.csv"
    test_out = OUTPUT_DIR / f"{class_name}_test_kaust_standardized_6400windows.csv"

    train_std.to_csv(train_out, index=False)
    test_std.to_csv(test_out, index=False)

    print(f"Train rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"Train valid windows: {train_valid}")
    print(f"Test valid windows: {test_valid}")
    print(f"Saved: {train_out}")
    print(f"Saved: {test_out}")


def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory does not exist: {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"CellMob root: {CELLMOB_ROOT}")
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")

    for class_name, filename in CLASS_FILES.items():
        process_class(class_name, filename)

    print(f"\nDone. Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()