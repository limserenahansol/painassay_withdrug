"""Your screenshot marks are decisions: the candidate at each mark is pre-filled keep = 1.

A candidate is 'at' a mark if the mark lies inside its 30 % outline, else the nearest centroid
within --max-px (10). Writes keep = 1, note 'your mark <date>' into the review csv (a backup
<csv>_before_marks.csv is kept); the GUI reads keep from that csv when it opens.

  python ca_apply_marks.py --run <preview3min/<rid>/v2_loose> --marks <marks csv>
"""
import argparse
import os
import shutil

import numpy as np
import pandas as pd
import scipy.io as sio


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="folder with 3_extract and 4_review")
    ap.add_argument("--marks", required=True)
    ap.add_argument("--max-px", type=float, default=10.0)
    ap.add_argument("--note", default="your mark 2026-10-21")
    a = ap.parse_args(argv)
    rev = [f for f in os.listdir(os.path.join(a.run, "4_review")) if f.endswith("_cell_review.csv")][0]
    tag = rev.replace("_cell_review.csv", "")
    csv = os.path.join(a.run, "4_review", rev)
    R = pd.read_csv(csv)
    S = np.asarray(sio.loadmat(os.path.join(a.run, "3_extract", f"{tag}_extract.mat"),
                               variable_names=["spatial_weights"])["spatial_weights"], np.float32)
    mx = S.reshape(-1, S.shape[2]).max(0)
    M = pd.read_csv(a.marks)
    if not os.path.isfile(csv.replace(".csv", "_before_marks.csv")):
        shutil.copy(csv, csv.replace(".csv", "_before_marks.csv"))
    if "note" not in R:
        R["note"] = ""
    R["note"] = R["note"].fillna("").astype(str)
    done = []
    for _, m in M.iterrows():
        r, c = int(round(m.fov_row)), int(round(m.fov_col))
        hit = []
        if 0 <= r < S.shape[0] and 0 <= c < S.shape[1]:
            hit = list(np.where(S[r, c, :] >= 0.3 * np.maximum(mx, 1e-12))[0])
        if hit:
            j = max(hit, key=lambda i: S[r, c, i] / max(mx[i], 1e-12))
        else:
            d = np.hypot(R.row - m.fov_row, R.col - m.fov_col)
            j = int(np.argmin(d))
            if d.iloc[j] > a.max_px:
                done.append((int(m.mark), None, "no candidate there"))
                continue
        was = int(R.at[j, "keep"])
        R.at[j, "keep"] = 1
        R.at[j, "note"] = (R.at[j, "note"] + "; " if R.at[j, "note"] else "") + a.note
        done.append((int(m.mark), int(R.at[j, "id"]), f"keep {was} -> 1 ({R.at[j, 'verdict']})"))
    R.to_csv(csv, index=False)
    for k, i, s in done:
        print(f"mark {k}: " + (f"#{i} {s}" if i else s))
    print(f"{tag}: {int((R.keep == 1).sum())} kept -> {csv}")


if __name__ == "__main__":
    main()
