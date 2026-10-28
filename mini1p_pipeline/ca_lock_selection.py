"""Freeze a confirmed whole-day selection: every candidate that is OFF now becomes reviewed = 1
(keep 0), so re-running the post script (transfer + duplicate resolution) cannot switch it back on
(2026-10-21: 5941 #140, a rescued cell, came back once its duplicate partner was excluded).
Kept cells are not touched (their attention marks stay). Backup *_before_lock_<time>.csv.

  python ca_lock_selection.py --day-root <mini1p_out/20261019/<m>_d1>
"""
import argparse
import datetime
import os
import shutil

import pandas as pd


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--day-root", required=True)
    a = ap.parse_args(argv)
    tag = os.path.basename(os.path.normpath(a.day_root))
    csv = os.path.join(a.day_root, "4_cell_review", f"{tag}_cell_review.csv")
    R = pd.read_csv(csv)
    shutil.copy(csv, csv.replace(".csv", f"_before_lock_{datetime.datetime.now():%Y%m%d_%H%M}.csv"))
    if "reviewed" not in R:
        R["reviewed"] = 0
    R["reviewed"] = R["reviewed"].fillna(0).astype(int)
    R["note"] = R.get("note", pd.Series([""] * len(R))).fillna("").astype(str)
    off = (R.keep != 1) & (R.reviewed != 1)
    R.loc[off, "reviewed"] = 1
    R.loc[off, "note"] = R.loc[off, "note"].where(R.loc[off, "note"] == "", R.loc[off, "note"] + "; ") + \
        f"off in the selection you confirmed {datetime.date.today().isoformat()}"
    R.to_csv(csv, index=False)
    print(f"{tag}: {int(off.sum())} candidates locked off, {int((R.keep == 1).sum())} kept unchanged -> {csv}")


if __name__ == "__main__":
    main()
