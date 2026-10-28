"""Exclude ROIs you name on the whole-day max projection - no EXTRACT re-run.

Sets keep = 0, reviewed = 1, decision_source 'user', note 'excluded by you <date>' for the given
ids in <tag>_cell_review.csv (backup *_before_exclusion_<time>.csv). reviewed = 1 makes the
decision final: ca_transfer_decisions keeps "your decision on the whole day" when the post
script runs again. --keep-ids does the opposite (keep = 1, reviewed = 1) for a cell you want back.

  python ca_exclude_rois.py --day-root <mini1p_out/20261019/<m>_d1> --ids 12 40 88 [--keep-ids 7]
"""
import argparse
import datetime
import os
import shutil

import pandas as pd


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--day-root", required=True)
    ap.add_argument("--ids", nargs="*", type=int, default=[])
    ap.add_argument("--keep-ids", nargs="*", type=int, default=[])
    a = ap.parse_args(argv)
    tag = os.path.basename(os.path.normpath(a.day_root))
    csv = os.path.join(a.day_root, "4_cell_review", f"{tag}_cell_review.csv")
    R = pd.read_csv(csv)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    shutil.copy(csv, csv.replace(".csv", f"_before_exclusion_{stamp}.csv"))
    for c in ("reviewed",):
        if c not in R:
            R[c] = 0
    for c in ("note", "decision_source"):
        if c not in R:
            R[c] = ""
        R[c] = R[c].fillna("").astype(str)
    bad = [i for i in a.ids + a.keep_ids if i not in set(R.id.astype(int))]
    if bad:
        raise SystemExit(f"ids not in {tag}: {bad}")
    day = datetime.date.today().isoformat()
    for i in a.ids:
        k = R.index[R.id.astype(int) == i][0]
        R.at[k, "keep"] = 0; R.at[k, "reviewed"] = 1; R.at[k, "decision_source"] = "user"
        R.at[k, "note"] = f"excluded by you {day}"
    for i in a.keep_ids:
        k = R.index[R.id.astype(int) == i][0]
        R.at[k, "keep"] = 1; R.at[k, "reviewed"] = 1; R.at[k, "decision_source"] = "user"
        R.at[k, "note"] = f"kept by you {day}"
    R.to_csv(csv, index=False)
    print(f"{tag}: excluded {len(a.ids)}, kept back {len(a.keep_ids)} -> {int((R.keep == 1).sum())} kept ({csv})")


if __name__ == "__main__":
    main()
