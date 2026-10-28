"""ca_apply_crossday.py - Day-2 candidates that ca_crossday_match.py paired with a Day-1 confirmed neuron are kept as
that neuron (green on the maps: decision_source 'user', note 'Day-1 neuron #k'), even when the footprint overlap
alone (ca_transfer_decisions, cosine >= 0.5) missed them (shift, rotation, small z change). A day1_id column is
written for the longitudinal analysis. Other candidates are not touched. Backup *_before_crossday_<time>.csv.

  python ca_apply_crossday.py --mouse 5941
"""
import argparse
import datetime
import os
import shutil

import pandas as pd

BASE = r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mouse", required=True)
    a = ap.parse_args(argv)
    tag = f"{a.mouse}_d2"
    csv = os.path.join(BASE, "mini1p_out", "20261021", tag, "4_cell_review", f"{tag}_cell_review.csv")
    P = pd.read_csv(os.path.join(BASE, "analysis", "crossday", f"{a.mouse}_pairs_all.csv"))
    R = pd.read_csv(csv)
    shutil.copy(csv, csv.replace(".csv", f"_before_crossday_{datetime.datetime.now():%Y%m%d_%H%M}.csv"))
    R["day1_id"] = pd.NA
    R["note"] = R.get("note", pd.Series([""] * len(R))).fillna("").astype(str)
    R["decision_source"] = R.get("decision_source", pd.Series([""] * len(R))).fillna("").astype(str)
    n_new = 0
    for p in P.itertuples():
        k = R.index[R.id == p.day2_id]
        if not len(k):
            continue
        k = k[0]
        if R.at[k, "keep"] != 1:
            n_new += 1
        R.at[k, "keep"] = 1
        R.at[k, "day1_id"] = int(p.day1_id)
        R.at[k, "decision_source"] = "user"
        tail = f"Day-1 neuron #{int(p.day1_id)} ({p.accepted_by}, {p.dist_px:.1f} px)"
        R.at[k, "note"] = tail if not R.at[k, "note"] else R.at[k, "note"] + "; " + tail
    R.to_csv(csv, index=False)
    print(f"{tag}: {len(P)} Day-2 candidates are Day-1 neurons ({n_new} of them were not kept by the footprint overlap); "
          f"{int((R.keep == 1).sum())} kept now -> {csv}")


if __name__ == "__main__":
    main()
