"""Whole-day cells on the max dF/F projection, coloured by where each keep decision came from.

left  the day's max dF/F projection alone (both sessions, the projection window of the run)
right the same image with every candidate outline at 30 % of its peak:
      green  kept by YOUR decision (matched to a cell you kept, or your mark)
      cyan   rescued (your kept cell, footprint put back)
      orange kept by the RULE only - you have not seen these (numbered)
      thin red  deleted
Look at the morphology here: a neuron is a compact round body on the max projection.

  python ca_day_maxproj.py --day-root <mini1p_out/20261019/<m>_d1> [--out <png>]
"""
import argparse
import os

import numpy as np
import pandas as pd
import scipy.io as sio

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import safe_io  # noqa: E402,F401  (a figure save retries on a transient Windows write error)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--day-root", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--quadrants", action="store_true", help="also 4 zoomed quarters (plain | outlines + numbers) to pick ROIs")
    ap.add_argument("--number-deleted", action="store_true",
                    help="quarters also number the deleted (red) candidates, so one can be named to add back (*_ids.png)")
    a = ap.parse_args(argv)
    tag = os.path.basename(os.path.normpath(a.day_root))
    xd = os.path.join(a.day_root, "3_extract")
    S = np.asarray(sio.loadmat(os.path.join(xd, f"{tag}_extract.mat"), variable_names=["spatial_weights"])["spatial_weights"], np.float32)
    R = pd.read_csv(os.path.join(a.day_root, "4_cell_review", f"{tag}_cell_review.csv"))
    P = sio.loadmat(os.path.join(xd, f"{tag}_projections.mat"), variable_names=["max_dff"])
    bg = np.asarray(P["max_dff"], float)
    if bg.shape != S.shape[:2]:
        raise SystemExit(f"projection {bg.shape} vs footprints {S.shape[:2]}")
    lo, hi = np.percentile(bg, [1, 99.8])
    img = np.clip((bg - lo) / (hi - lo), 0, 1) ** 0.7
    H, W = bg.shape
    src = R.get("decision_source", pd.Series([""] * len(R))).astype(str)
    ver = R.verdict.astype(str)
    fig, ax = plt.subplots(1, 2, figsize=(2 * 10, 10 * H / W + 1.0))
    for k in (0, 1):
        ax[k].imshow(img, cmap="gray", vmin=0, vmax=1)
        ax[k].set_axis_off()
    n = {"user": 0, "rescued": 0, "rule": 0, "deleted": 0}
    for j in range(S.shape[2]):
        f = S[:, :, j]
        if f.max() <= 0:
            continue
        if R.keep.iloc[j] != 1:
            col, lw, al, kind = "#ff4136", 0.4, 0.5, "deleted"
        elif ver.iloc[j] == "rescued":
            col, lw, al, kind = "#00d0ff", 1.1, 1.0, "rescued"
        elif src.iloc[j] == "user":
            col, lw, al, kind = "#2ecc40", 1.1, 1.0, "user"
        else:
            col, lw, al, kind = "#ff9f1a", 1.3, 1.0, "rule"
        n[kind] += 1
        ax[1].contour((f >= 0.3 * f.max()).astype(float), levels=[0.5], colors=[col], linewidths=lw, alpha=al)
        if kind != "deleted":
            ax[1].text(R.col.iloc[j] + 7, R.row.iloc[j] - 7, str(int(R.id.iloc[j])), color=col, fontsize=7,
                       fontweight="bold" if kind == "rule" else "normal")
    ax[0].set_title(f"{tag} whole day: max dF/F projection (no outlines)", fontsize=11)
    ax[1].set_title(f"kept {n['user'] + n['rescued'] + n['rule']}: green = your decision ({n['user']}), cyan = rescued ({n['rescued']}), "
                    f"ORANGE = rule only, not seen by you ({n['rule']}); thin red = deleted ({n['deleted']})", fontsize=10)
    fig.tight_layout()
    out = a.out or os.path.join(a.day_root, "5_analysis", f"{tag}_maxproj_decisions.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    if a.quadrants:
        quads(a, tag, S, R, src, ver, img, out, a.number_deleted)
    rule = R[(R.keep == 1) & (src != "user") & (ver != "rescued")]
    print(f"{tag}: user {n['user']}, rescued {n['rescued']}, rule-only {n['rule']}, deleted {n['deleted']} -> {out}")
    print("rule-only ids:", " ".join(str(int(i)) for i in rule.id))


def quads(a, tag, S, R, src, ver, img, out, number_deleted=False):
    """4 overlapping quarters, each: the max projection alone | the same with outlines and numbers"""
    H, W = img.shape
    mh, mw = H // 2, W // 2
    pad = 40
    boxes = [("top-left", 0, mh + pad, 0, mw + pad), ("top-right", 0, mh + pad, mw - pad, W),
             ("bottom-left", mh - pad, H, 0, mw + pad), ("bottom-right", mh - pad, H, mw - pad, W)]
    for name, r0, r1, c0, c1 in boxes:
        fig, ax = plt.subplots(1, 2, figsize=(2 * 9, 9 * (r1 - r0) / (c1 - c0) + 0.8))
        for k in (0, 1):
            ax[k].imshow(img[r0:r1, c0:c1], cmap="gray", vmin=0, vmax=1, extent=(c0 - 0.5, c1 - 0.5, r1 - 0.5, r0 - 0.5))
            ax[k].set_axis_off()
        for j in range(S.shape[2]):
            f = S[r0:r1, c0:c1, j]
            full = S[:, :, j]
            if full.max() <= 0 or f.max() < 0.3 * full.max():
                continue
            if R.keep.iloc[j] != 1:
                ax[1].contour(np.arange(c0, c1), np.arange(r0, r1), (f >= 0.3 * full.max()).astype(float), levels=[0.5],
                              colors=["#ff4136"], linewidths=0.8 if number_deleted else 0.5, alpha=0.8 if number_deleted else 0.6)
                if number_deleted and r0 <= R.row.iloc[j] < r1 and c0 <= R.col.iloc[j] < c1:
                    ax[1].text(R.col.iloc[j], R.row.iloc[j], str(int(R.id.iloc[j])), color="#ff6f61", fontsize=8,
                               ha="center", va="center", clip_on=True)
                continue
            if ver.iloc[j] == "rescued":
                col = "#00d0ff"
            elif src.iloc[j] == "user":
                col = "#2ecc40"
            else:
                col = "#ff9f1a"
            ax[1].contour(np.arange(c0, c1), np.arange(r0, r1), (f >= 0.3 * full.max()).astype(float), levels=[0.5],
                          colors=[col], linewidths=1.4)
            if r0 <= R.row.iloc[j] < r1 and c0 <= R.col.iloc[j] < c1:
                ax[1].text(R.col.iloc[j] + 6, R.row.iloc[j] - 6, str(int(R.id.iloc[j])), color=col, fontsize=10,
                           fontweight="bold", clip_on=True)
        ax[0].set_title(f"{tag} {name} quarter: max dF/F projection", fontsize=11)
        ax[1].set_title("green = your decision, cyan = rescued, orange = rule only (attention); red = deleted"
                        + (" (numbered: name one to add it back)" if number_deleted else ""), fontsize=10)
        fig.tight_layout()
        fig.savefig(out.replace(".png", f"_{name}" + ("_ids" if number_deleted else "") + ".png"), dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
