"""Old vs loose preview of one recording, side by side, with your marks.

Background: the max dF/F projection of the 1-min preview. Outlines at 30 % of each footprint
peak, green = keep, red = delete (the rule's pre-filled decision), numbers = GUI ids. Yellow
circles = the positions you circled on the screenshots. For every mark: is it now on a KEPT
candidate (inside its 30 % outline, or centroid within 8 px)?

  python ca_preview_compare.py --run <preview3min/<rid>> [--marks <marks csv>] --out <png>
"""
import argparse
import os

import numpy as np
import pandas as pd
import scipy.io as sio
from skimage.segmentation import find_boundaries

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(xd, rd, tag):
    S = np.asarray(sio.loadmat(os.path.join(xd, f"{tag}_extract.mat"), variable_names=["spatial_weights"])["spatial_weights"], np.float32)
    R = pd.read_csv(os.path.join(rd, f"{tag}_cell_review.csv"))
    return S, R


def covered(S, R, r, c):
    """kept candidate ids whose 30 % outline contains (r, c) or whose centroid is within 8 px"""
    out = []
    rr, cc = int(round(r)), int(round(c))
    for j in range(S.shape[2]):
        if R.keep.iloc[j] != 1:
            continue
        f = S[:, :, j]
        m = f.max()
        inside = 0 <= rr < f.shape[0] and 0 <= cc < f.shape[1] and m > 0 and f[rr, cc] >= 0.3 * m
        if inside or np.hypot(R.row.iloc[j] - r, R.col.iloc[j] - c) <= 8:
            out.append(int(R.id.iloc[j]))
    return out


def panel(ax, bg, S, R, marks, title):
    ax.imshow(bg, cmap="gray", vmin=np.percentile(bg, 1), vmax=np.percentile(bg, 99.7))
    for j in range(S.shape[2]):
        f = S[:, :, j]
        if f.max() <= 0:
            continue
        col = "#2ecc40" if R.keep.iloc[j] == 1 else "#ff4136"
        ax.contour((f >= 0.3 * f.max()).astype(float), levels=[0.5], colors=[col], linewidths=0.7)
        ax.text(R.col.iloc[j] + 5, R.row.iloc[j] - 5, str(int(R.id.iloc[j])), color=col, fontsize=5.5, fontweight="bold")
    if marks is not None:
        for k, m in marks.iterrows():
            ax.add_patch(plt.Circle((m.fov_col, m.fov_row), 13, fill=False, ec="yellow", lw=1.4))
    ax.set_title(title, fontsize=9)
    ax.set_axis_off()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--marks", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rid = os.path.basename(os.path.normpath(a.run))
    tag = rid.rsplit("_", 1)[0]
    S0, R0 = load(os.path.join(a.run, "3_extract"), os.path.join(a.run, "4_review"), tag)
    S1, R1 = load(os.path.join(a.run, "v2_loose", "3_extract"), os.path.join(a.run, "v2_loose", "4_review"), tag)
    P = sio.loadmat(os.path.join(a.run, "3_extract", f"{tag}_projections.mat"), variable_names=["max_dff"])
    bg = np.asarray(P["max_dff"], float)
    marks = pd.read_csv(a.marks) if a.marks and os.path.isfile(a.marks) else None
    rows = []
    if marks is not None:
        for _, m in marks.iterrows():
            rows.append(dict(mark=int(m.mark), fov_row=m.fov_row, fov_col=m.fov_col,
                             kept_before=";".join(map(str, covered(S0, R0, m.fov_row, m.fov_col))),
                             kept_now=";".join(map(str, covered(S1, R1, m.fov_row, m.fov_col)))))
    M = pd.DataFrame(rows)
    n0 = int((R0.keep == 1).sum()); n1 = int((R1.keep == 1).sum())
    fig, ax = plt.subplots(1, 2, figsize=(16, 7.6))
    hit0 = int((M.kept_before != "").sum()) if len(M) else 0
    hit1 = int((M.kept_now != "").sum()) if len(M) else 0
    panel(ax[0], bg, S0, R0, marks, f"BEFORE: {len(R0)} candidates, {n0} kept (green)" +
          (f"; your marks on a kept cell {hit0}/{len(M)}" if len(M) else ""))
    panel(ax[1], bg, S1, R1, marks, f"NEW (v3): {len(R1)} candidates, {n1} kept (green)" +
          (f"; your marks on a kept cell {hit1}/{len(M)}" if len(M) else ""))
    fig.suptitle(f"{rid}, first 1 min: max dF/F, outlines at 30 % of each footprint (green keep, red delete; numbers = GUI ids); "
                 "yellow circles = your marks", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(a.out, dpi=150); plt.close(fig)
    if len(M):
        M.to_csv(a.out.replace(".png", "_marks.csv"), index=False)
        print(M.to_string(index=False))
    print(f"{rid}: before {len(R0)} candidates / {n0} kept -> loose {len(R1)} / {n1} kept; marks on kept {hit0} -> {hit1} of {len(M)}")
    print("->", a.out)


if __name__ == "__main__":
    main()
