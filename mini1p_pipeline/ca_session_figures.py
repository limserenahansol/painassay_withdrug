"""Per-session Ca figures of one mouse-day (after the cell check):

  <tag>_cellmap.png                 the selected cells (green, numbered) and the rejected candidates
                                    (red) on the max dF/F projection and on the mean image
  <recording>_all_traces_p<k>.png   every selected cell's local dF/F over the whole session (one row
                                    per cell, scaled to its own 99th percentile of the day), the
                                    schedule blocks shaded, behaviour under it: deliveries (by
                                    stimulus), withdrawal / flinch taps, movement, grooming, escape
  <recording>_heatmap.png           the same as a z-score heatmap; pre and post share the cell order
                                    (sorted by mean z in the stimulus blocks of the baseline session)

  python ca_session_figures.py --extract <tag>_extract.mat --keep <tag>_cell_review.csv
      --projections <tag>_projections.mat --traces-dir DIR --scoring-dir FINAL_DIR --out-dir OUT
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import scipy.io as sio
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import safe_io  # noqa: E402,F401  (a figure save retries on a transient Windows write error)

FPS = 30.0
BLOCKS = [("baseline", 0, 180), ("stim1", 180, 360), ("stim2", 390, 570), ("stim3", 600, 780), ("stim4", 810, 990)]
SCOL = {"Light touch": "#5dade2", "Mild touch": "#f5b041", "Pin prick": "#e74c3c", "Heat": "#8e44ad"}


def contour_xy(fp, level=0.5):
    from skimage import measure
    m = fp / max(fp.max(), 1e-12)
    cs = measure.find_contours(m, level)
    return max(cs, key=len) if cs else None


def scoring_for(scoring_dir, mouse, phase):
    for p in glob.glob(os.path.join(scoring_dir, "ScoringAB_*.mat")):
        S = sio.loadmat(p, squeeze_me=True)
        ph = "pre" if str(S["phase"]).lower().startswith("base") else "post"
        if str(S["mouseID"]) == mouse and ph == phase:
            return S
    return None


def behaviour(S, n):
    """behaviour in isx frames: camera / scoring frame n (1-based) was exposed during isx_frame n
    (0-based, K = 0), so a mark at frame n is at t = n / FPS and per-frame vector index i is
    camera frame i + 1 (see the spans() call: + 1)"""
    if S is None:
        return None
    sc = np.atleast_1d(S["score"]).astype(int)
    re = np.atleast_2d(S["reflexEvents"]).astype(int).reshape(-1, 2)
    stim = [str(x) for x in np.atleast_1d(S["stimNames"])]
    dF, dT = np.atleast_1d(S["dFrames"]).astype(int), np.atleast_1d(S["dTypes"]).astype(int)
    return dict(deliv=[(f, stim[k - 1] if 1 <= k <= 4 else "?") for f, k in zip(dF, dT)], W=re[re[:, 1] == 1, 0],
                F=re[re[:, 1] == 2, 0], move=np.asarray(S["moveVec"]).astype(bool).ravel() if "moveVec" in S else None,
                groom=np.asarray(S["groomVec"]).astype(bool).ravel() if "groomVec" in S else None, esc=sc == 4, stim=stim)


def spans(m):
    d = np.diff(np.r_[0, np.asarray(m, int), 0])
    return zip(np.where(d == 1)[0], np.where(d == -1)[0])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", required=True)
    ap.add_argument("--keep", required=True)
    ap.add_argument("--projections", required=True)
    ap.add_argument("--traces-dir", required=True)
    ap.add_argument("--scoring-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--per-page", type=int, default=60)
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    M = sio.loadmat(a.extract, squeeze_me=True, variable_names=["spatial_weights", "mouse_id"])
    Sp = np.asarray(M["spatial_weights"], float)
    if Sp.ndim == 2:
        Sp = Sp[:, :, None]
    tag = os.path.basename(a.extract).replace("_extract.mat", "")
    mouse = tag.split("_")[0]
    R = pd.read_csv(a.keep)
    keep = R.id[R.keep == 1].values.astype(int)
    drop = R.id[R.keep != 1].values.astype(int)
    P = sio.loadmat(a.projections, squeeze_me=True)
    bg1 = np.asarray(P["max_dff"], float) if "max_dff" in P else np.asarray(P["max_bgsub"], float)
    bg2 = np.asarray(P["mean_img"], float)

    # ---- cell map ---------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(20, 9))
    for k, (bg, nm) in enumerate(((bg1, "max dF/F projection"), (bg2, "mean image"))):
        lo, hi = np.nanpercentile(bg, [1, 99.7])
        ax[k].imshow(bg, cmap="gray", vmin=lo, vmax=hi)
        for c in drop:
            xy = contour_xy(Sp[:, :, c - 1])
            if xy is not None:
                ax[k].plot(xy[:, 1], xy[:, 0], color="#ff5050", lw=0.6, alpha=0.7)
        for c in keep:
            xy = contour_xy(Sp[:, :, c - 1])
            if xy is not None:
                ax[k].plot(xy[:, 1], xy[:, 0], color="#2ecc71", lw=1.0)
                ax[k].text(xy[:, 1].mean(), xy[:, 0].mean(), str(c), color="yellow", fontsize=6, ha="center", va="center")
        ax[k].set_title(f"{nm}: {len(keep)} selected cells (green), {len(drop)} rejected candidates (red)", fontsize=11)
        ax[k].axis("off")
    fig.suptitle(f"{tag}: cells used for analysis (same footprints and numbers in the baseline and the injection session)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out_dir, f"{tag}_cellmap.png"), dpi=110); plt.close(fig)

    # ---- traces -------------------------------------------------------------------------------
    sess = {}
    for f in sorted(glob.glob(os.path.join(a.traces_dir, f"{tag}_*_traces.npz"))):
        rid = os.path.basename(f).replace("_traces.npz", "")
        sess[rid] = dict(np.load(f))
    if not sess:
        print("no trace files"); return 1
    # common scale and order
    sm6 = lambda A: pd.DataFrame(A).rolling(6, center=True, min_periods=1).mean().values   # 0.2 s, display only
    for s in sess.values():
        s["dff_s"] = sm6(s["dff"])
    allD = np.concatenate([s["dff_s"] for s in sess.values()], 0)
    scale = np.nanpercentile(allD, 99, axis=0)
    scale[~(scale > 0)] = 1
    ids = list(sess.values())[0]["cell_ids"]
    base_rid = next((r for r in sess if r.endswith("pre")), list(sess)[0])
    sb = sess[base_rid]
    inb = np.zeros(len(sb["t_s"]), bool)
    for _, s0, s1 in BLOCKS[1:]:
        inb |= (sb["t_s"] >= s0) & (sb["t_s"] < s1)
    order = np.argsort(-np.nanmean(sb["z"][inb], 0))
    for rid, s in sess.items():
        ph = rid.split("_")[-1]
        B = behaviour(scoring_for(a.scoring_dir, mouse, ph), len(s["t_s"]))
        t = s["t_s"]
        D = s["dff_s"] / scale
        n_c = D.shape[1]
        pages = [order[i:i + a.per_page] for i in range(0, n_c, a.per_page)]
        for pg, idx in enumerate(pages, 1):
            h = 0.16 * len(idx) + 2.6
            fig = plt.figure(figsize=(22, h))
            axT = fig.add_axes([0.05, 1.9 / h, 0.93, 1 - 2.4 / h])
            axB = fig.add_axes([0.05, 0.35 / h, 0.93, 1.35 / h], sharex=axT)
            for r, j in enumerate(idx):
                axT.plot(t, np.nan_to_num(D[:, j]) * 0.9 + (len(idx) - 1 - r), lw=0.35, color="k")
            for k, (bn, s0, s1) in enumerate(BLOCKS[1:], 1):
                stn = B["stim"][k - 1] if B else bn
                axT.axvspan(s0, s1, color=SCOL.get(stn, "0.9"), alpha=0.08, lw=0)
                axT.text((s0 + s1) / 2, len(idx) + 0.3, stn, ha="center", fontsize=9, color=SCOL.get(stn, "k"))
            axT.set_yticks(range(len(idx)), [str(ids[j]) for j in idx[::-1]], fontsize=5)
            axT.set_ylim(-1, len(idx) + 1.2)
            axT.set_ylabel("cell (local dF/F, 0.2-s mean, scaled to its 99th percentile)")
            if B:
                rows = [("deliveries", 5), ("withdrawal", 4), ("flinch", 3), ("escape/rear", 2), ("movement", 1), ("grooming", 0)]
                for f, st in B["deliv"]:
                    axB.plot([f / FPS] * 2, [4.6, 5.4], color=SCOL.get(st, "k"), lw=1)
                axB.plot(B["W"] / FPS, np.full(len(B["W"]), 4), "|", color="#d4ac0d", ms=6)
                axB.plot(B["F"] / FPS, np.full(len(B["F"]), 3), "|", color="0.4", ms=6)
                for yy, v, col in ((2, B["esc"], "#27ae60"), (1, B["move"], "#17a2b8"), (0, B["groom"], "#d6338a")):
                    if v is not None:
                        for s0, s1 in spans(v):
                            axB.add_patch(Rectangle(((s0 + 1) / FPS, yy - 0.35), (s1 - s0) / FPS, 0.7, color=col, lw=0))
                axB.set_yticks([y for _, y in rows], [nm for nm, _ in rows], fontsize=7)
                axB.set_ylim(-0.6, 5.6)
            axB.set_xlabel("time in session (s)")
            axT.set_xlim(0, t[-1])
            fig.suptitle(f"{rid}: all {n_c} selected cells, whole session" + (f" (page {pg}/{len(pages)})" if len(pages) > 1 else "") +
                         " - same cell order in baseline and injection (sorted by stimulus-block activity of the baseline)", fontsize=11)
            fig.savefig(os.path.join(a.out_dir, f"{rid}_all_traces_p{pg}.png"), dpi=100); plt.close(fig)
        # heatmap
        Z = s["z"][:, order]
        fig, ax = plt.subplots(figsize=(22, 0.06 * n_c + 2.5))
        im = ax.imshow(np.clip(Z.T, -2, 8), aspect="auto", cmap="magma", vmin=-2, vmax=8,
                       extent=[t[0], t[-1], n_c - 0.5, -0.5], interpolation="nearest")
        for _, s0, s1 in BLOCKS[1:]:
            ax.axvline(s0, color="w", lw=0.6, ls=":"); ax.axvline(s1, color="w", lw=0.6, ls=":")
        if B:
            for f, st in B["deliv"]:
                ax.plot([f / FPS] * 2, [-0.5, -2.5], color=SCOL.get(st, "w"), lw=0.8, clip_on=False)
        ax.set_yticks(range(n_c), [str(ids[j]) for j in order], fontsize=4)
        ax.set_xlabel("time in session (s)"); ax.set_ylabel("cell")
        plt.colorbar(im, ax=ax, fraction=0.015, label="z (local dF/F / robust SD)")
        ax.set_title(f"{rid}: z-scored activity of all {n_c} selected cells (ticks on top = deliveries coloured by stimulus)", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out_dir, f"{rid}_heatmap.png"), dpi=100); plt.close(fig)
        print(rid, "figures done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
