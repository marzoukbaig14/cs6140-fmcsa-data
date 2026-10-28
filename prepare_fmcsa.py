"""Build the FMCSA modeling snapshot for the CS6140 midterm.

Reads the full Vehicle Inspection File export, keeps inspections where the
vehicle itself gets checked (levels 1, 2, 5, 6) during federal fiscal year
2026 (Oct 1, 2025 to Sep 30, 2026), and saves a reproducible random sample
plus a JSON file of counts that the notebook reports.

Run from ~/cs6140-midterm with the venv active:
    python prepare_fmcsa.py
"""
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path.home() / "fmcsa_raw" / "vehicle_inspection_full.csv"
OUT_DIR = Path.home() / "cs6140-fmcsa-data"
OUT_FILE = OUT_DIR / "fmcsa_vehicle_inspections_fy2026_sample.csv.gz"
COUNTS_FILE = OUT_DIR / "prep_counts.json"
SOURCE_URL = "https://data.transportation.gov/Trucking-and-Motorcoaches/Vehicle-Inspection-File/fx4q-ay7w"

VEHICLE_LEVELS = [1, 2, 5, 6]          # levels where the vehicle is inspected
START = pd.Timestamp("2025-10-01")     # federal fiscal year 2026
END = pd.Timestamp("2026-09-30")
SAMPLE_ROWS = 200_000
SEED = 42
CHUNK = 250_000
REQUIRED = ["INSP_DATE", "INSP_LEVEL_ID", "VEHICLE_OOS_TOTAL"]


def norm_cols(df):
    df.columns = [c.strip().upper().replace(" ", "_") for c in df.columns]
    return df


def parse_dates(s):
    s = s.astype("string").str.strip()
    d = pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    if d.isna().mean() > 0.01:
        d = d.fillna(pd.to_datetime(s, errors="coerce", format="mixed"))
    return d


def read_chunks():
    return pd.read_csv(RAW, dtype=str, chunksize=CHUNK,
                       keep_default_na=False, na_values=[""])


def main():
    if not RAW.exists():
        sys.exit(f"Missing {RAW}. Download the full file first.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- Pass 1: count everything ----------
    print("Pass 1 of 2: counting rows (a few minutes)...")
    full_rows, columns = 0, None
    level_counts = {}
    oos_full = 0
    date_fail = 0
    dmin = dmax = None
    target_rows = target_oos = 0

    for i, ch in enumerate(read_chunks()):
        ch = norm_cols(ch)
        if i == 0:
            columns = list(ch.columns)
            missing = [c for c in REQUIRED if c not in columns]
            if missing:
                print("Columns found:", columns)
                sys.exit(f"Required columns missing: {missing}. Check the input files and try again.")
            print(f"  {len(columns)} columns. Sample INSP_DATE values: "
                  f"{ch['INSP_DATE'].head(3).tolist()}")

        full_rows += len(ch)
        lvl = pd.to_numeric(ch["INSP_LEVEL_ID"], errors="coerce")
        for k, v in lvl.value_counts(dropna=False).items():
            key = "missing" if pd.isna(k) else str(int(k))
            level_counts[key] = level_counts.get(key, 0) + int(v)

        oos = pd.to_numeric(ch["VEHICLE_OOS_TOTAL"], errors="coerce").fillna(0) > 0
        oos_full += int(oos.sum())

        d = parse_dates(ch["INSP_DATE"])
        date_fail += int(d.isna().sum())
        if d.notna().any():
            cmin, cmax = d.min(), d.max()
            dmin = cmin if dmin is None else min(dmin, cmin)
            dmax = cmax if dmax is None else max(dmax, cmax)

        keep = lvl.isin(VEHICLE_LEVELS) & d.between(START, END)
        target_rows += int(keep.sum())
        target_oos += int((keep & oos).sum())
        print(f"  {full_rows:,} rows read", end="\r")
    print()

    fail_rate = date_fail / full_rows
    if fail_rate > 0.01:
        sys.exit(f"{fail_rate:.1%} of dates failed to parse. Check the input files and try again.")
    if target_rows < SAMPLE_ROWS:
        sys.exit(f"Only {target_rows:,} rows in the filtered year. Check the input files and try again.")

    # ---------- Pass 2: sample ----------
    print("Pass 2 of 2: sampling (a few minutes)...")
    rng = np.random.default_rng(SEED)
    p = min(1.0, 1.2 * SAMPLE_ROWS / target_rows)
    parts = []
    for ch in read_chunks():
        ch = norm_cols(ch)
        lvl = pd.to_numeric(ch["INSP_LEVEL_ID"], errors="coerce")
        d = parse_dates(ch["INSP_DATE"])
        keep = (lvl.isin(VEHICLE_LEVELS) & d.between(START, END)).to_numpy()
        sub = ch[keep]
        parts.append(sub[rng.random(len(sub)) < p])
    pool = pd.concat(parts, ignore_index=True)
    sample = pool.sample(n=SAMPLE_ROWS, random_state=SEED).reset_index(drop=True)
    sample.to_csv(OUT_FILE, index=False, compression="gzip")
    sample_oos = float((pd.to_numeric(sample["VEHICLE_OOS_TOTAL"], errors="coerce")
                        .fillna(0) > 0).mean())

    counts = {
        "snapshot_date": date.today().isoformat(),
        "source_url": SOURCE_URL,
        "full_rows": full_rows,
        "full_columns": len(columns),
        "columns": columns,
        "insp_date_min": str(dmin.date()) if dmin is not None else None,
        "insp_date_max": str(dmax.date()) if dmax is not None else None,
        "date_parse_failures": date_fail,
        "level_counts_full": level_counts,
        "vehicle_oos_rate_full": oos_full / full_rows,
        "filter_levels": VEHICLE_LEVELS,
        "filter_start": str(START.date()),
        "filter_end": str(END.date()),
        "filtered_rows": target_rows,
        "vehicle_oos_rate_filtered": target_oos / target_rows,
        "sample_rows": SAMPLE_ROWS,
        "sample_seed": SEED,
        "vehicle_oos_rate_sample": sample_oos,
    }
    COUNTS_FILE.write_text(json.dumps(counts, indent=2))
    (OUT_DIR / "SNAPSHOT_DATE.txt").write_text(date.today().isoformat() + "\n")

    print("\nDONE")
    print(f"  Full file:            {full_rows:,} rows x {len(columns)} columns")
    print(f"  Date range:           {counts['insp_date_min']} to {counts['insp_date_max']}")
    print(f"  Date parse failures:  {date_fail:,}")
    print(f"  Vehicle OOS rate (all rows):         {counts['vehicle_oos_rate_full']:.1%}")
    print(f"  Levels 1/2/5/6 in FY2026:            {target_rows:,} rows")
    print(f"  Vehicle OOS rate (that subset):      {counts['vehicle_oos_rate_filtered']:.1%}")
    print(f"  Sample saved:         {SAMPLE_ROWS:,} rows -> {OUT_FILE}")
    print(f"  Vehicle OOS rate (sample):           {sample_oos:.1%}")


if __name__ == "__main__":
    main()
