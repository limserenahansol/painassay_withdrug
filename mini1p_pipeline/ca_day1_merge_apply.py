"""ca_day1_merge_apply.py - your decision of 2026-10-22 in the Day-1 review table after merge_split_soma.m:
1191 Day 1 #1 + #2 (the brightest neuron of the field, cut into two by EXTRACT and left out by my rule) = ONE neuron,
kept as #1 (reviewed, your decision); #2 = merged into #1 (not kept). Centre of #1 from the merged footprint.
Backup: <tag>_cell_review_before_merge_20261022.csv

  python ca_day1_merge_apply.py --mouse 1191 --keep 1 --merge 2
"""
import argparse
import os
import shutil

import scipy.io as sio
import pandas as pd

BASE = r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mouse", required=True)
    ap.add_argument("--keep", type=int, required=True)
    ap.add_argument("--merge", type=int, nargs="+", required=True)
    a = ap.parse_args(argv)
    t1 = f"{a.mouse}_d1"
    root = os.path.join(BASE, "mini1p_out", "20261019", t1)
    csv = os.path.join(root, "4_cell_review", f"{t1}_cell_review.csv")
    shutil.copy(csv, csv.replace(".csv", "_before_merge_20261022.csv"))
    R = pd.read_csv(csv)
    q = sio.loadmat(os.path.join(root, "3_extract", f"{t1}_extract.mat"), variable_names=["cell_qc"], squeeze_me=True,
                    struct_as_record=False)["cell_qc"]
    cx, cy = q.centroid_xy[a.keep - 1]
    for c in ("note", "decision_source", "attention"):
        if c in R:
            R[c] = R[c].fillna("").astype(str)
    k = R.id == a.keep
    R.loc[k, "keep"] = 1
    R.loc[k, "decision_source"] = "user"
    R.loc[k, "reviewed"] = 1
    R.loc[k, "row"], R.loc[k, "col"] = float(cy), float(cx)
    R.loc[k, "note"] = (f"your decision 2026-10-22: the brightest neuron of the field; EXTRACT had cut it into "
                        f"#{a.keep} + #{' + #'.join(map(str, a.merge))} -> merged into #{a.keep} (footprint = sum, traces re-fitted)")
    if "attention" in R:
        R.loc[k, "attention"] = ""
    for j in a.merge:
        kj = R.id == j
        R.loc[kj, "keep"] = 0
        R.loc[kj, "decision_source"] = "user"
        R.loc[kj, "reviewed"] = 1
        R.loc[kj, "note"] = f"merged into #{a.keep} (one soma; your decision 2026-10-22)"
        if "attention" in R:
            R.loc[kj, "attention"] = ""
    R.to_csv(csv, index=False)
    print(f"{t1}: #{a.keep} kept (merged with #{a.merge}), centre row {cy:.1f} col {cx:.1f}; "
          f"{int((R.keep == 1).sum())} kept neurons now")


if __name__ == "__main__":
    main()
