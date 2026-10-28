"""ca_missed_cells.py - bright compact bodies on the max projection that no kept neuron covers (what you circled on Day 1).

On Day 1 you circled cells the detection had missed (analysis\\curation_marks_20261021\\marks_day_<tag>.csv); they became
forced seeds. This finds such places automatically for any day, for you to look at (never added automatically):
  seed image   Cn x PNR (the image that seeds EXTRACT), Gaussian 1.5 px, inside the lens
  spots        local maxima >= 1 radius apart, above median + K x robust SD of the image
  soma shape   the connected region above half way between the local ring and the peak: area 0.3-3 x a soma of
               radius 10 px, solidity >= 0.8 (a compact round body, your Day-1 criterion)
  covered      a KEPT footprint (>= 30 % of its peak, dilated 2 px) contains the spot -> not listed; a deleted
               candidate there -> listed as 'a candidate exists (deleted)', none -> 'no candidate (missed)'
--validate: on the Day-1 runs BEFORE your marks (3_extract_v1_20261020, 4_cell_review_v1_20261020) of 5937 / 5939 /
1191: how many of your marks the list contains (within 8 px), and how long the list is.
Writes analysis\\ca_selfcheck\\missed_<tag>.csv and missed_<tag>.png (max dF/F, kept outlines, numbered spots)

  python ca_missed_cells.py --day-root <mini1p_out\\20261021\\5941_d2> [--k 3]
  python ca_missed_cells.py --validate
"""
import argparse
import os

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.ndimage import binary_dilation, gaussian_filter, label, maximum_filter
from skimage.measure import regionprops

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553"
RAD = 10


def find(xd, rd, tag, k=3.0, out_png=None):
    S = np.asarray(sio.loadmat(os.path.join(xd, f"{tag}_extract.mat"), variable_names=["spatial_weights"])["spatial_weights"], np.float32)
    M = sio.loadmat(os.path.join(xd, f"{tag}_extract.mat"), variable_names=["movie_mask"])
    lens = np.asarray(M["movie_mask"], bool) if "movie_mask" in M else np.ones(S.shape[:2], bool)
    P = sio.loadmat(os.path.join(xd, f"{tag}_projections.mat"), variable_names=["Cn", "PNR", "max_dff"])
    seed = gaussian_filter(np.asarray(P["Cn"], float) * np.asarray(P["PNR"], float), 1.5)
    mx = np.asarray(P["max_dff"], float)
    R = pd.read_csv(os.path.join(rd, f"{tag}_cell_review.csv"))
    keep = R.keep.values == 1
    cover_k = np.zeros(S.shape[:2], bool); cover_a = np.zeros(S.shape[:2], bool)
    for j in range(S.shape[2]):
        f = S[:, :, j]
        if f.max() <= 0:
            continue
        b = f >= 0.3 * f.max()
        cover_a |= b
        if keep[j]:
            cover_k |= b
    cover_k = binary_dilation(cover_k, iterations=2); cover_a = binary_dilation(cover_a, iterations=2)
    inner = binary_dilation(~lens, iterations=RAD) == 0
    v = seed[inner]
    med = np.median(v); sd = 1.4826 * np.median(np.abs(v - med))
    pk = (seed == maximum_filter(seed, size=2 * RAD + 1)) & inner & (seed > med + k * sd)
    rows = []
    yy, xx = np.mgrid[0:seed.shape[0], 0:seed.shape[1]]
    for r0, c0 in zip(*np.nonzero(pk)):
        if cover_k[r0, c0]:
            continue
        d = np.hypot(yy - r0, xx - c0)
        ring = (d >= 2 * RAD) & (d <= 3 * RAD)
        bg = np.median(seed[ring])
        thr = bg + 0.5 * (seed[r0, c0] - bg)
        L, _ = label((seed >= thr) & (d <= 2 * RAD))
        reg = L == L[r0, c0]
        if not reg.any():
            continue
        pr = regionprops(reg.astype(int))[0]
        ar = pr.area / (np.pi * RAD ** 2)
        if not (0.3 <= ar <= 3.0 and pr.solidity >= 0.8):
            continue
        rows.append(dict(row=int(r0) + 1, col=int(c0) + 1, seed_value=round(float((seed[r0, c0] - med) / sd), 1),
                         area_ratio=round(ar, 2), solidity=round(pr.solidity, 2),
                         status="a candidate exists (deleted)" if cover_a[r0, c0] else "no candidate (missed)"))
    T = pd.DataFrame(rows).sort_values("seed_value", ascending=False).reset_index(drop=True) if rows else pd.DataFrame(
        columns=["row", "col", "seed_value", "area_ratio", "solidity", "status"])
    T.insert(0, "spot", np.arange(1, len(T) + 1))
    if out_png:
        lo, hi = np.percentile(mx, [1, 99.8])
        fig, ax = plt.subplots(1, 2, figsize=(20, 10 * mx.shape[0] / mx.shape[1] + 0.8))
        for a in ax:
            a.imshow(np.clip((mx - lo) / (hi - lo), 0, 1) ** 0.7, cmap="gray"); a.set_axis_off()
        for j in np.flatnonzero(keep):
            f = S[:, :, j]
            if f.max() > 0:
                ax[1].contour(f >= 0.3 * f.max(), levels=[0.5], colors=["#2ecc40"], linewidths=0.6)
        for r in T.itertuples():
            col = "#ff4136" if r.status.startswith("no") else "#ffdc00"
            ax[1].plot(r.col - 1, r.row - 1, "o", ms=14, mfc="none", mec=col, mew=1.6)
            ax[1].text(r.col + 9, r.row - 9, str(r.spot), color=col, fontsize=9, fontweight="bold")
        ax[0].set_title(f"{tag}: max dF/F projection", fontsize=12)
        ax[1].set_title(f"green = kept neurons; circles = compact bright bodies no kept neuron covers ({len(T)}): red = no "
                        f"candidate (missed), yellow = a candidate exists but was deleted", fontsize=11)
        fig.tight_layout(); fig.savefig(out_png, dpi=100); plt.close(fig)
    return T


def validate(k):
    out = []
    for m in ("5937", "5939", "1191"):
        tag = f"{m}_d1"
        root = os.path.join(BASE, "mini1p_out", "20261019", tag)
        xd, rd = os.path.join(root, "3_extract_v1_20261020"), os.path.join(root, "4_cell_review_v1_20261020")
        if not os.path.isdir(xd):
            continue
        T = find(xd, rd, tag, k)
        Mk = pd.read_csv(os.path.join(BASE, "analysis", "curation_marks_20261021", f"marks_day_{tag}.csv"))
        hit = 0
        for r in Mk.itertuples():
            if len(T) and (np.hypot(T.row - r.fov_row, T.col - r.fov_col) <= 8).any():
                hit += 1
        out.append(dict(mouse=m, k=k, your_marks=len(Mk), found=hit, recall=round(hit / max(len(Mk), 1), 2),
                        listed=len(T), missed_listed=int((T.status == "no candidate (missed)").sum()) if len(T) else 0))
    return pd.DataFrame(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--day-root", default="")
    ap.add_argument("--k", type=float, default=3.0)
    ap.add_argument("--validate", action="store_true")
    a = ap.parse_args(argv)
    od = os.path.join(BASE, "analysis", "ca_selfcheck"); os.makedirs(od, exist_ok=True)
    if a.validate:
        V = pd.concat([validate(k) for k in (2.0, 3.0, 4.0)])
        V.to_csv(os.path.join(od, "missed_cells_validation_day1.csv"), index=False)
        print(V.to_string(index=False))
        return
    tag = os.path.basename(os.path.normpath(a.day_root))
    T = find(os.path.join(a.day_root, "3_extract"), os.path.join(a.day_root, "4_cell_review"), tag, a.k,
             os.path.join(od, f"missed_{tag}.png"))
    T.to_csv(os.path.join(od, f"missed_{tag}.csv"), index=False)
    print(f"{tag}: {len(T)} compact bright bodies not covered by a kept neuron "
          f"({int((T.status == 'no candidate (missed)').sum()) if len(T) else 0} with no candidate) -> {od}")


if __name__ == "__main__":
    main()
