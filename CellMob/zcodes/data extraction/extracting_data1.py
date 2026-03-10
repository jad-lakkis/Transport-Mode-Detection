from pathlib import Path
import pandas as pd
import numpy as np


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
INPUT_ROOT = DATA_DIR / "orignal_raw_data"
OUTPUT_ROOT = DATA_DIR / "zdata_unfinished"

RSRP_all = [f"RSRP/antenna port - {i}" for i in range(1, 15)]
RSSI_all = [f"E-UTRAN carrier RSSI/antenna port - {i}" for i in range(1, 15)]
RSRQ_all = [f"RSRQ/antenna port - {i}" for i in range(1, 15)]


def process_folder(folder: Path):
    output_file = OUTPUT_ROOT / f"{folder.name}_cleaned.csv"

    if output_file.exists():
        print(f"Skipping {folder.name} (already done)")
        return

    print(f"\nProcessing {folder.name}")

    rows = []

    for file in sorted(folder.glob("*.csv")):
        print("  ", file.name)

        df = pd.read_csv(file, low_memory=False)

        rsrp_cols = [c for c in RSRP_all if c in df.columns]
        rssi_cols = [c for c in RSSI_all if c in df.columns]
        rsrq_cols = [c for c in RSRQ_all if c in df.columns]

        if "Time" not in df.columns:
            raise ValueError(f"{file} is missing required column: Time")

        for _, row in df.iterrows():
            rsrp = pd.to_numeric(row[rsrp_cols], errors="coerce").dropna().values
            rssi = pd.to_numeric(row[rssi_cols], errors="coerce").dropna().values
            rsrq = pd.to_numeric(row[rsrq_cols], errors="coerce").dropna().values

            if len(rsrp) < 2 or len(rssi) < 2 or len(rsrq) < 2:
                continue

            rsrp = rsrp[:4]
            rssi = rssi[:4]
            rsrq = rsrq[:4]

            def fill4(arr):
                arr = list(arr)
                m = float(np.mean(arr))
                while len(arr) < 4:
                    arr.append(m)
                return arr

            rsrp = fill4(rsrp)
            rssi = fill4(rssi)
            rsrq = fill4(rsrq)

            rows.append([
                row["Time"],
                *rsrp,
                *rssi,
                *rsrq
            ])

    columns = [
        "time",
        "rsrp1", "rsrp2", "rsrp3", "rsrp4",
        "rssi1", "rssi2", "rssi3", "rssi4",
        "rsrq1", "rsrq2", "rsrq3", "rsrq4"
    ]

    clean = pd.DataFrame(rows, columns=columns)
    clean.to_csv(output_file, index=False)

    print(f"Saved {output_file}")
    print("Rows:", len(clean))


def main():
    if not INPUT_ROOT.exists():
        raise FileNotFoundError(f"Input directory does not exist: {INPUT_ROOT}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"CellMob root: {CELLMOB_ROOT}")
    print(f"Input root: {INPUT_ROOT}")
    print(f"Output root: {OUTPUT_ROOT}")

    for folder in sorted(INPUT_ROOT.iterdir()):
        if not folder.is_dir():
            continue
        process_folder(folder)


if __name__ == "__main__":
    main()