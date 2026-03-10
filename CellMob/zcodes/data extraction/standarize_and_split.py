from pathlib import Path
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
SEPARATED_DATA_DIR = DATA_DIR / "data(raw_but_seperated)"

input_folder = SEPARATED_DATA_DIR / "zdata_unfinished"
train_output_folder = SEPARATED_DATA_DIR / "zdata_train"
test_output_folder = SEPARATED_DATA_DIR / "zdata_test"

train_output_folder.mkdir(parents=True, exist_ok=True)
test_output_folder.mkdir(parents=True, exist_ok=True)

time_col = "time"
feature_cols = [
    "rsrp1", "rsrp2", "rsrp3", "rsrp4",
    "rssi1", "rssi2", "rssi3", "rssi4",
    "rsrq1", "rsrq2", "rsrq3", "rsrq4"
]


def main():
    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_folder}")

    print(f"CellMob root: {CELLMOB_ROOT}")
    print(f"Input folder: {input_folder}")
    print(f"Train output folder: {train_output_folder}")
    print(f"Test output folder: {test_output_folder}")

    for file in sorted(input_folder.glob("*.csv")):
        print(f"Processing {file.name}")

        df = pd.read_csv(file)

        required_cols = [time_col] + feature_cols
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"{file.name} is missing required columns: {missing}")

        for col in feature_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=feature_cols).reset_index(drop=True)

        n = len(df)
        split_idx = int(0.8 * n)

        if split_idx == 0 or split_idx == n:
            raise ValueError(
                f"{file.name} does not contain enough rows for an 80/20 split after cleaning."
            )

        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

        train_mean = train_df[feature_cols].mean()
        train_std = train_df[feature_cols].std(ddof=0)
        train_std = train_std.replace(0, 1)

        train_df[feature_cols] = (train_df[feature_cols] - train_mean) / train_std
        test_df[feature_cols] = (test_df[feature_cols] - train_mean) / train_std

        train_out = train_output_folder / file.name
        test_out = test_output_folder / file.name

        train_df.to_csv(train_out, index=False)
        test_df.to_csv(test_out, index=False)

        print(f"  train rows: {len(train_df)} -> {train_out}")
        print(f"  test rows : {len(test_df)} -> {test_out}")

    print("\nDone.")


if __name__ == "__main__":
    main()