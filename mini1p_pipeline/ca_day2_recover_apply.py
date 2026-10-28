"""ca_day2_recover_apply.py - after rescue_footprints.m + extract_cell_review (4_cell_review_recover): the Day-2 review
table gets the NEW QC numbers of every component (all traces were re-fitted jointly, so every number changes a little),
the recovered rows are marked (recovered_day1_id), and your own Day-2 decisions (reviewed = 1) are kept as they were.
The review pages of 4_cell_review_recover replace the old ones. Backup: <tag>_cell_review_before_recover_apply.csv

  python ca_day2_recover_apply.py --mouse 5939
"""
import argparse
import glob
import os
import shutil

import numpy as np
import pandas as pd
import scipy.io as sio

BASE = r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mouse", required=True)
    a = ap.parse_args(argv)
    t2 = f"{a.mouse}_d2"
    root = os.path.join(BASE, "mini1p_out", "20261021", t2)
    rv, nw = os.path.join(root, "4_cell_review"), os.path.join(root, "4_cell_review_recover")
    csv = os.path.join(rv, f"{t2}_cell_review.csv")
    R = pd.read_csv(csv)
    N = pd.read_csv(os.path.join(nw, f"{t2}_cell_review.csv"))
    if len(N) != len(R):
        raise SystemExit(f"{t2}: new review {len(N)} rows, table {len(R)}")
    rec = sio.loadmat(os.path.join(root, "3_extract", f"{t2}_extract.mat"), variable_names=["recovered"],
                      squeeze_me=True, struct_as_record=False)["recovered"]
    ids = np.atleast_1d(rec.ids).astype(int); d1 = np.atleast_1d(rec.day1_id).astype(int)
    shutil.copy(csv, csv.replace(".csv", "_before_recover_apply.csv"))
    out = N.copy()
    for c in R.columns:                       # columns only the later steps write (decisions, tracking) come along
        if c not in out:
            out[c] = R[c]
    yours = R.get("reviewed", pd.Series([0] * len(R))).fillna(0).astype(int).values == 1
    for c in ("keep", "note", "decision_source"):
        if c in R:
            out.loc[yours, c] = R.loc[yours, c].values
    out["recovered_day1_id"] = np.nan
    for i, d in zip(ids, d1):
        out.loc[out.id == i, "recovered_day1_id"] = d
        out.loc[out.id == i, "note"] = f"recovered: Day-1 #{d} footprint moved to Day 2 (not found by the Day-2 EXTRACT)"
    out.to_csv(csv, index=False)
    for p in glob.glob(os.path.join(nw, "*.png")):
        shutil.copy(p, os.path.join(rv, os.path.basename(p)))
    v = out.verdict.astype(str)
    r_ = out[out.recovered_day1_id.notna()]
    print(f"{t2}: review numbers renewed for {len(out)} components ({len(r_)} recovered: "
          f"{int((r_.verdict == 'cell').sum())} cell, {int((r_.verdict == 'check').sum())} check, "
          f"{int((r_.verdict == 'not_cell').sum())} not_cell); all: {int((v == 'cell').sum())} cell, "
          f"{int((v == 'check').sum())} check, {int((v == 'not_cell').sum())} not_cell")


if __name__ == "__main__":
    main()
