"""Build carrier history features for the CS6140 midterm.

For every carrier (DOT_NUMBER) that appears in the FY2026 sample, count its
inspections from BEFORE FY2026 (everything up to Sep 30, 2025). Because every
counted inspection happened before every sampled inspection, these features
are known when an inspection starts and cannot leak the outcome.

Outputs (in ~/cs6140-fmcsa-data/):
  carrier_history.csv.gz       one row per sampled carrier
  carrier_history_meta.json    period, counts, and definitions

Run from ~/cs6140-midterm with the venv active:
    python prepare_carrier_history.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

RAW = Path.home() / "fmcsa_raw" / "vehicle_inspection_full.csv"
DATA_DIR = Path.home() / "cs6140-fmcsa-data"
SAMPLE = DATA_DIR / "fmcsa_vehicle_inspections_fy2026_sample.csv.gz"
OUT = DATA_DIR / "carrier_history.csv.gz"
META = DATA_DIR / "carrier_history_meta.json"

HIST_END = pd.Timestamp("2025-09-30")   # day before FY2026 starts
VEHICLE_LEVELS = [1, 2, 5, 6]
NEEDED = {"DOT_NUMBER", "INSP_DATE", "INSP_LEVEL_ID", "VEHICLE_OOS_TOTAL"}
CHUNK = 1_000_000


def main():
    for p in (RAW, SAMPLE):
        if not p.exists():
            sys.exit(f"Missing {p}. Check the input files and try again.")

    sample_dots = (pd.read_csv(SAMPLE, usecols=["DOT_NUMBER"], dtype=str)["DOT_NUMBER"]
                   .str.strip().replace("", pd.NA).dropna())
    dot_set = set(sample_dots)
    print(f"Sample: {len(sample_dots):,} rows with a DOT number, {len(dot_set):,} unique carriers")

    parts, rows_read, rows_used = [], 0, 0
    dmin = dmax = None
    reader = pd.read_csv(RAW, dtype=str, chunksize=CHUNK,
                         usecols=lambda c: c.strip().upper() in NEEDED)
    for ch in reader:
        ch.columns = [c.strip().upper() for c in ch.columns]
        rows_read += len(ch)
        dot = ch["DOT_NUMBER"].str.strip()
        d = pd.to_datetime(ch["INSP_DATE"].str.strip(), format="%Y%m%d", errors="coerce")
        keep = dot.isin(dot_set) & (d <= HIST_END)
        if not keep.any():
            print(f"  {rows_read:,} rows read", end="\r")
            continue
        sub_d = d[keep]
        dmin = sub_d.min() if dmin is None else min(dmin, sub_d.min())
        dmax = sub_d.max() if dmax is None else max(dmax, sub_d.max())
        lvl = pd.to_numeric(ch.loc[keep, "INSP_LEVEL_ID"], errors="coerce")
        oos = pd.to_numeric(ch.loc[keep, "VEHICLE_OOS_TOTAL"], errors="coerce").fillna(0) > 0
        veh = lvl.isin(VEHICLE_LEVELS)
        g = pd.DataFrame({
            "DOT_NUMBER": dot[keep].values,
            "carrier_prior_insp_all": 1,
            "carrier_prior_vehicle_insp": veh.astype(int).values,
            "carrier_prior_vehicle_oos": (veh & oos).astype(int).values,
        }).groupby("DOT_NUMBER").sum()
        parts.append(g)
        rows_used += int(keep.sum())
        print(f"  {rows_read:,} rows read", end="\r")
    print()

    hist = (pd.concat(parts).groupby(level=0).sum()
            if parts else pd.DataFrame(columns=["carrier_prior_insp_all",
                                                "carrier_prior_vehicle_insp",
                                                "carrier_prior_vehicle_oos"]))
    hist = hist.reindex(sorted(dot_set), fill_value=0)
    hist.index.name = "DOT_NUMBER"
    hist = hist.reset_index()
    hist.to_csv(OUT, index=False, compression="gzip")

    with_hist = int((hist["carrier_prior_vehicle_insp"] > 0).sum())
    tot_insp = int(hist["carrier_prior_vehicle_insp"].sum())
    tot_oos = int(hist["carrier_prior_vehicle_oos"].sum())
    meta = {
        "history_period_start": str(dmin.date()) if dmin is not None else None,
        "history_period_end": str(HIST_END.date()),
        "definition": ("Counts of each carrier's inspections dated on or before "
                       "Sep 30, 2025, i.e. strictly before every FY2026 inspection "
                       "in the sample. carrier_prior_vehicle_* use levels 1, 2, 5, 6 only."),
        "full_file_rows_read": rows_read,
        "history_rows_used": rows_used,
        "sample_carriers": len(dot_set),
        "carriers_with_prior_vehicle_inspections": with_hist,
        "prior_vehicle_inspections_total": tot_insp,
        "prior_vehicle_oos_rate_pooled": (tot_oos / tot_insp) if tot_insp else None,
        "columns": list(hist.columns),
    }
    META.write_text(json.dumps(meta, indent=2))

    print("\nDONE")
    print(f"  Rows read from full file:          {rows_read:,}")
    print(f"  History rows used (pre-FY2026):    {rows_used:,}")
    print(f"  History period:                    {meta['history_period_start']} to {meta['history_period_end']}")
    print(f"  Sample carriers:                   {len(dot_set):,}")
    print(f"  ...with prior vehicle inspections: {with_hist:,} ({with_hist / len(dot_set):.1%})")
    if tot_insp:
        print(f"  Pooled prior vehicle OOS rate:     {tot_oos / tot_insp:.1%}")
    print(f"  Saved -> {OUT}")


if __name__ == "__main__":
    main()
