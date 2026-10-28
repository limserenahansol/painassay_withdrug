"""Old rule vs strict rule on the max dF/F projection (whole day).

left  the max dF/F projection alone
right green = your decision, cyan = rescued, ORANGE = kept by the old rule AND the strict rule,
      MAGENTA dashed = kept by the old rule but dropped by the strict rule (numbered), thin red = deleted

  python ca_maxproj_rule_diff.py --day-root <mini1p_out/20261019/<m>_d1> --strict <decisions.csv of a
      ca_transfer_decisions --strict-new --no-write run> --out <png>
"""
import argparse
import os

import numpy as np
import pandas as pd
import scipy.io as sio

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--day-root", required=True)
    ap.add_argument("--strict", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--crop", nargs=4, type=int, default=None, metavar=("R0", "R1", "C0", "C1"), help="zoom to rows R0:R1, cols C0:C1")
    a = ap.parse_args(argv)
    tag = os.path.basename(os.path.normpath(a.day_root))
    xd = os.path.join(a.day_root, "3_extract")
    S = np.asarray(sio.loadmat(os.path.join(xd, f"{tag}_extract.mat"), variable_names=["spatial_weights"])["spatial_weights"], np.float32)
    R = pd.read_csv(os.path.join(a.day_root, "4_cell_review", f"{tag}_cell_review.csv"))
    D = pd.read_csv(a.strict).set_index("id")
    bg = np.asarray(sio.loadmat(os.path.join(xd, f"{tag}_projections.mat"), variable_names=["max_dff"])["max_dff"], float)
    lo, hi = np.percentile(bg, [1, 99.8])
    img = np.clip((bg - lo) / (hi - lo), 0, 1) ** 0.7
    H, W = bg.shape
    src = R.get("decision_source", pd.Series([""] * len(R))).astype(str)
    if a.crop:
        r0, r1, c0, c1 = a.crop
        H, W = r1 - r0, c1 - c0
    fig, ax = plt.subplots(1, 2, figsize=(2 * 10.5, 10.5 * H / W + 1.2))
    for k in (0, 1):
        ax[k].imshow(img, cmap="gray", vmin=0, vmax=1)
        ax[k].set_axis_off()
    n = dict(user=0, rescued=0, both=0, dropped=0)
    for j in range(S.shape[2]):
        f = S[:, :, j]
        if f.max() <= 0:
            continue
        cid = int(R.id.iloc[j])
        style = None
        if R.keep.iloc[j] != 1:
            ax[1].contour((f >= 0.3 * f.max()).astype(float), levels=[0.5], colors=["#ff4136"], linewidths=0.35, alpha=0.45)
            continue
        if R.verdict.iloc[j] == "rescued":
            col, ls, kind = "#00d0ff", "solid", "rescued"
        elif src.iloc[j] == "user":
            col, ls, kind = "#2ecc40", "solid", "user"
        elif cid in D.index and int(D.at[cid, "keep"]) == 1:
            col, ls, kind = "#ff9f1a", "solid", "both"
        else:
            col, ls, kind = "#ff2bd6", "dashed", "dropped"
        n[kind] += 1
        ax[1].contour((f >= 0.3 * f.max()).astype(float), levels=[0.5], colors=[col], linewidths=1.5 if kind in ("both", "dropped") else 1.0,
                      linestyles=ls)
        inside = (not a.crop) or (a.crop[0] <= R.row.iloc[j] < a.crop[1] and a.crop[2] <= R.col.iloc[j] < a.crop[3])
        if kind in ("both", "dropped") and inside:
            ax[1].text(R.col.iloc[j] + 8, R.row.iloc[j] - 8, str(cid), color=col, fontsize=9 if a.crop else 8,
                       fontweight="bold", clip_on=True)
    if a.crop:
        for k in (0, 1):
            ax[k].set_xlim(c0, c1); ax[k].set_ylim(r1, r0)
    ax[0].set_title(f"{tag} whole day: max dF/F projection (no outlines)" + (f", zoom rows {r0}-{r1}, cols {c0}-{c1}" if a.crop else ""), fontsize=11)
    ax[1].set_title(f"green = your decision ({n['user']}), cyan = rescued ({n['rescued']}); old-rule cells: "
                    f"ORANGE = strict rule keeps too ({n['both']}), MAGENTA dashed = strict rule drops ({n['dropped']})", fontsize=10)
    fig.tight_layout()
    fig.savefig(a.out, dpi=140); plt.close(fig)
    print(n, "->", a.out)


if __name__ == "__main__":
    main()
