"""Do the RESCUED cells (rescue_missing_cells.m) carry the signal of the cell you kept?
For every rescued cell: the correlation of its whole-day trace with its trace in the reference
run you reviewed (same frames; slow trends removed with a 30-s running mean), the best
correlation with any OTHER whole-day component (a rescued cell whose signal sits in another
component would show r_other > r_with_ref), and the skew of both traces (about 0 = no clear
transients: a quiet cell, not a failed rescue).

  python ca_rescue_check.py --day-root <mini1p_out/20261019/5941_d1>
      --ref-mat <test_5941_5min/4c_extract_seeded_loose/5941_d1_extract.mat> --ref-name 5min_test
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.ndimage import uniform_filter1d


def load(p):
    M = sio.loadmat(p, squeeze_me=True, struct_as_record=False, variable_names=["frame_table", "temporal_raw"])
    ft = M["frame_table"]
    FT = pd.DataFrame({k: np.atleast_1d(getattr(ft, k)) for k in ft._fieldnames})
    FT["recording_id"] = FT.recording_id.astype(str)
    return np.asarray(M["temporal_raw"], float), FT


def hp(x):
    return x - uniform_filter1d(x, 900, mode="nearest")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--day-root", required=True)
    ap.add_argument("--ref-mat", required=True)
    ap.add_argument("--ref-name", default="5min_test")
    a = ap.parse_args(argv)
    tag = os.path.basename(os.path.normpath(a.day_root))
    R = pd.read_csv(os.path.join(a.day_root, "4_cell_review", f"{tag}_cell_review.csv"))
    Tt, Ft = load(a.ref_mat)
    Td, Fd = load(os.path.join(a.day_root, "3_extract", f"{tag}_extract.mat"))
    rows = []
    for rid in Ft.recording_id.unique():
        sel = np.where(Fd.recording_id == rid)[0]
        fd = Fd.isx_frame.values[sel]
        ft_ = Ft.isx_frame.values[(Ft.recording_id == rid).values]
        rt = np.where((Ft.recording_id == rid).values)[0]
        common = np.intersect1d(fd, ft_)
        common = common[common >= 450]                       # after the 15-s switch-on
        ia, ib = np.searchsorted(fd, common), np.searchsorted(ft_, common)
        D = Td[sel[ia]]
        for _, r in R[R.verdict == "rescued"].iterrows():
            mm = re.search(rf"{re.escape(a.ref_name)} #(\d+)", str(r.note))
            if not mm:
                continue
            tid = int(mm.group(1))
            b = Tt[rt[ib], tid - 1]
            j0 = int(r.id) - 1
            ok = np.isfinite(D[:, j0]) & np.isfinite(b)
            bh = hp(b[ok])
            cc = np.full(D.shape[1], np.nan)
            for j in range(D.shape[1]):
                if j != j0 and np.isfinite(D[ok, j]).all():
                    cc[j] = np.corrcoef(hp(D[ok, j]), bh)[0, 1]
            jb = int(np.nanargmax(cc))
            kb = R.loc[R.id == jb + 1, "keep"].values
            rows.append(dict(recording=rid, day_id=int(r.id), keep=int(r.keep), ref=a.ref_name, ref_id=tid,
                             r_with_ref=np.corrcoef(hp(D[ok, j0]), bh)[0, 1],
                             skew_ref=float(pd.Series(bh).skew()), skew_day=float(pd.Series(hp(D[ok, j0])).skew()),
                             best_other_day_id=jb + 1, r_best_other=cc[jb], best_other_kept=int(kb[0]) if len(kb) else -1,
                             n_frames=int(ok.sum())))
    X = pd.DataFrame(rows)
    out = os.path.join(a.day_root, "5_analysis", "decisions", f"{tag}_rescued_vs_{a.ref_name}.csv")
    X.round(3).to_csv(out, index=False)
    print(X.round(2).to_string(index=False))
    print("->", out)


if __name__ == "__main__":
    main()
