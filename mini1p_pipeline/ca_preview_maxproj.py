"""The preview decision on the max projection (not a GUI frame): left the max dF/F projection of
the 1-min preview alone, right the same image with every candidate outline (green = keep with its
number, red = delete, thin) and your marks (yellow circles).

  python ca_preview_maxproj.py --run <preview3min/<rid>> --variant v4_marks [--marks a.csv b.csv] --out <png>
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
    ap.add_argument("--run", required=True)
    ap.add_argument("--variant", default="v4_marks")
    ap.add_argument("--marks", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rid = os.path.basename(os.path.normpath(a.run))
    tag = rid.rsplit("_", 1)[0]
    vd = os.path.join(a.run, a.variant)
    S = np.asarray(sio.loadmat(os.path.join(vd, "3_extract", f"{tag}_extract.mat"), variable_names=["spatial_weights"])["spatial_weights"], np.float32)
    R = pd.read_csv(os.path.join(vd, "4_review", f"{tag}_cell_review.csv"))
    P = sio.loadmat(os.path.join(a.run, "3_extract", f"{tag}_projections.mat"), variable_names=["max_dff"])
    bg = np.asarray(P["max_dff"], float)
    lo, hi = np.percentile(bg, [1, 99.8])
    img = np.clip((bg - lo) / (hi - lo), 0, 1) ** 0.7
    ms = [pd.read_csv(f) for f in a.marks if os.path.isfile(f) and os.path.getsize(f) > 10]
    M = pd.concat(ms, ignore_index=True) if ms else pd.DataFrame(columns=["fov_row", "fov_col"])
    H, W = bg.shape
    fig, ax = plt.subplots(1, 2, figsize=(2 * 9.5, 9.5 * H / W + 0.9))
    for k in (0, 1):
        ax[k].imshow(img, cmap="gray", vmin=0, vmax=1)
        ax[k].set_axis_off()
    for j in range(S.shape[2]):
        f = S[:, :, j]
        if f.max() <= 0:
            continue
        keep = R.keep.iloc[j] == 1
        ax[1].contour((f >= 0.3 * f.max()).astype(float), levels=[0.5], colors=["#2ecc40" if keep else "#ff4136"],
                      linewidths=1.0 if keep else 0.4, alpha=1.0 if keep else 0.55)
        if keep:
            ax[1].text(R.col.iloc[j] + 7, R.row.iloc[j] - 7, str(int(R.id.iloc[j])), color="#2ecc40", fontsize=7, fontweight="bold")
    mx = S.reshape(-1, S.shape[2]).max(0)
    n_miss = 0
    for _, m in M.iterrows():
        rr, cc = int(round(m.fov_row)), int(round(m.fov_col))
        on = np.where(S[rr, cc, :] >= 0.3 * np.maximum(mx, 1e-12))[0] if 0 <= rr < H and 0 <= cc < W else []
        d = np.hypot(R.row - m.fov_row, R.col - m.fov_col)
        ok = any(R.keep.iloc[j] == 1 for j in on) or bool(((d <= 10) & (R.keep == 1)).any())
        if ok:
            ax[1].add_patch(plt.Circle((m.fov_col, m.fov_row), 14, fill=False, ec="yellow", lw=1.6))
        else:
            n_miss += 1
            ax[1].add_patch(plt.Circle((m.fov_col, m.fov_row), 14, fill=False, ec="orange", lw=1.6, ls="--"))
            ax[1].text(m.fov_col + 15, m.fov_row + 15, "?", color="orange", fontsize=10, fontweight="bold")
    n_keep = int((R.keep == 1).sum())
    ax[0].set_title(f"{rid}: max dF/F projection, first 1 min (no outlines)", fontsize=10)
    ax[1].set_title(f"same image: {n_keep} kept (green, numbered), {len(R) - n_keep} deleted (thin red)\n"
                    f"your {len(M)} marks: yellow = on a kept cell, orange dashed ? = no EXTRACT component there yet ({n_miss})", fontsize=10)
    fig.tight_layout()
    fig.savefig(a.out, dpi=130); plt.close(fig)
    print(f"{rid} {a.variant}: {len(R)} candidates, {n_keep} kept -> {a.out}")


if __name__ == "__main__":
    main()
