"""Your preview marks and preview keeps -> the whole-day run.

1. registers the preview mean image to the day mean image (the same NCC registration as
   ca_transfer_decisions) and writes every mark in day coordinates (marks_day_<tag>.csv)
2. writes a reference table of the final preview decision in which only the cells kept there
   count as your decisions (decision_source 'user', reviewed 1); the deleted ones stay
   unreviewed, so a day candidate is never deleted just because a 1-min footprint looked bad

  python ca_marks_to_day.py --preview <preview3min/<rid>> --variant v4_marks --day-mat <day extract.mat>
      --marks <csv> [<csv> ...] --out-dir <analysis/curation_marks_20261021>
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ca_transfer_decisions import load_extract, register  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--day-mat", required=True)
    ap.add_argument("--marks", nargs="*", default=[])
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    rid = os.path.basename(os.path.normpath(a.preview))
    tag = rid.rsplit("_", 1)[0]
    vd = os.path.join(a.preview, a.variant)
    pm = os.path.join(vd, "3_extract", f"{tag}_extract.mat")
    _, mean_p = load_extract(pm)
    _, mean_d = load_extract(a.day_mat)
    dy, dx, peak = register(mean_p, mean_d)
    print(f"{rid} -> day: day pixel = preview pixel - ({dy}, {dx}), NCC peak {peak:.2f}")
    ms = [pd.read_csv(f) for f in a.marks if os.path.isfile(f) and os.path.getsize(f) > 10]
    M = pd.concat(ms, ignore_index=True) if ms else pd.DataFrame(columns=["fov_row", "fov_col"])
    M = M.copy()
    M["preview_row"], M["preview_col"] = M.fov_row, M.fov_col
    M["fov_row"], M["fov_col"] = M.preview_row - dy, M.preview_col - dx
    M["mark"] = np.arange(1, len(M) + 1)
    out_m = os.path.join(a.out_dir, f"marks_day_{tag}.csv")
    M[["mark", "fov_row", "fov_col", "preview_row", "preview_col"]].to_csv(out_m, index=False)
    R = pd.read_csv(os.path.join(vd, "4_review", f"{tag}_cell_review.csv"))
    R["decision_source"] = np.where(R.keep == 1, "user", "")
    R["reviewed"] = (R.keep == 1).astype(int)
    out_r = os.path.join(vd, "4_review", f"{tag}_cell_review_user.csv")
    R.to_csv(out_r, index=False)
    print(f"   {len(M)} marks -> {out_m}; {int((R.keep == 1).sum())} preview keeps as your decisions -> {out_r}")


if __name__ == "__main__":
    main()
