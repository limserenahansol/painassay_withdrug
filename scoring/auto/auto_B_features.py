"""auto_B_features.py  -  per-frame features from the SIDE view, for training
an automatic behaviour classifier.

Why the side view and why silhouettes
    From the WT QC: the side view segments the mouse in 60/60 sampled frames at
    2.8 SD contrast and solidity 0.82, and carries 73 motion-energy bouts per
    10 min against the bottom view's 3. The bottom view images the mouse as a
    saturated silhouette with no internal detail and a mesh pitch (~20 px) the
    same size as a paw (~25 px), so it cannot support paw-level scoring at all.

    Background subtraction does NOT work here: the mouse is almost stationary
    over 28 min (centroid scatter SD 19 x 14 px against a 200-300 px body), so a
    temporal-median plate already contains the animal. This module therefore
    uses an absolute threshold inside a fixed ROI, the settings validated in QC.

What it produces
    One row per frame, with shape / posture / motion features plus rolling
    statistics over +/-0.5 s and +/-2 s. These target the four AFFECTIVE codes
    that score_AB_dual_view.m writes, all of which are posture questions:
        1  paw attending              head oriented to the paw, no mouth contact
        2  licking or biting          mouth on the paw
        3  guarding (> 2 s)           slow, sustained posture change
        4  escape / rearing           large body height and eccentricity change
    The two REFLEXIVE measures - paw withdrawal and flinch - are NOT reachable:
    at 30 fps a withdrawal is 1.5-4.5 frames. That is a rig limit, not a
    feature-engineering one; raise the frame rate, do not add features.

    If you later add DeepLabCut, append its keypoint columns to this CSV and
    auto_B_train.py will pick them up automatically - no code change needed.

USAGE
    python auto_B_features.py <side_video> [--out DIR] [--roi y0 y1 x0 x1]
                              [--thr 75] [--step 1]

OUTPUT
    Features_<vid>.csv      one row per analysed frame
    Features_<vid>.png      diagnostic: segmentation on 4 frames + feature traces

Hansol Lim - HEAL mini1p / SBI-553
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np

# QC-validated side-view settings (see qc/WT_RECORDING_QC.md)
ROI = (90, 370, 110, 600)      # y0, y1, x0, x1 - excludes the dark cloth fold
THR = 75                       # mouse body ~55 grey, cloth fold 92-117
AMIN, AMAX = 6000, 70000

BASE_COLS = ["frame", "time_s", "found", "area", "w", "h", "aspect", "solidity",
             "ecc", "cx", "cy", "top_y", "bot_y", "extent", "perim",
             "hull_defect_max", "hull_defect_sum", "n_defects",
             "low_contour_rough", "motion", "motion_roi_frac"]


def segment(gray, thr=THR, amin=AMIN, amax=AMAX):
    m = (gray < thr).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if amin <= cv2.contourArea(c) <= amax]
    if not cnts:
        return None, m
    return max(cnts, key=cv2.contourArea), m


def shape_features(c, gray_shape):
    """Geometry of one contour. Names match BASE_COLS."""
    a = cv2.contourArea(c)
    x, y, w, h = cv2.boundingRect(c)
    hull = cv2.convexHull(c)
    ha = cv2.contourArea(hull)
    sol = a / ha if ha > 0 else 0.0
    perim = cv2.arcLength(c, True)
    if len(c) >= 5:
        (_, _), (MA, ma), _ = cv2.fitEllipse(c)
        ecc = max(MA, ma) / max(min(MA, ma), 1.0)
    else:
        ecc = 1.0
    M = cv2.moments(c)
    cx = M["m10"] / M["m00"] if M["m00"] else float(x + w / 2)
    cy = M["m01"] / M["m00"] if M["m00"] else float(y + h / 2)

    # convexity defects: a lifted paw or a lowered head breaks the outline
    dmax = dsum = 0.0
    ndef = 0
    hull_i = cv2.convexHull(c, returnPoints=False)
    if hull_i is not None and len(hull_i) > 3:
        try:
            defects = cv2.convexityDefects(c, np.sort(hull_i[:, 0])[:, None])
            if defects is not None:
                d = defects[:, 0, 3] / 256.0
                d = d[d > 3]
                if d.size:
                    dmax, dsum, ndef = float(d.max()), float(d.sum()), int(d.size)
        except cv2.error:
            pass

    # roughness of the LOWER outline: where paws are, in a side view
    pts = c[:, 0, :]
    lower = pts[pts[:, 1] > (y + 0.75 * h)]
    rough = float(lower[:, 1].std()) if len(lower) > 4 else 0.0

    return dict(area=a, w=w, h=h, aspect=w / max(h, 1), solidity=sol, ecc=ecc,
                cx=cx, cy=cy, top_y=float(y), bot_y=float(y + h),
                extent=a / max(w * h, 1), perim=perim,
                hull_defect_max=dmax, hull_defect_sum=dsum, n_defects=ndef,
                low_contour_rough=rough)


def rolling(df_cols, names, fps, wins=(0.5, 2.0)):
    """Centred rolling mean / SD / range for each named column."""
    import pandas as pd
    out = {}
    for wsec in wins:
        k = max(3, int(round(wsec * fps)) | 1)      # odd window
        tag = f"{wsec:g}s".replace(".", "p")
        for n in names:
            s = pd.Series(df_cols[n], dtype=float)
            r = s.rolling(k, center=True, min_periods=1)
            out[f"{n}_mean_{tag}"] = r.mean().to_numpy()
            out[f"{n}_sd_{tag}"] = r.std().fillna(0).to_numpy()
            out[f"{n}_rng_{tag}"] = (r.max() - r.min()).fillna(0).to_numpy()
    return out


def run(path, outdir, roi=ROI, thr=THR, step=1):
    import pandas as pd
    y0, y1, x0, x1 = roi
    vid = os.path.splitext(os.path.basename(path))[0]
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ntot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"=== {os.path.basename(path)} ===")
    print(f"  {fps:.2f} fps, {ntot} frames, ROI y{y0}-{y1} x{x0}-{x1}, I<{thr}")

    rows, prev, tiles = [], None, []
    i = 0
    nfound = 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        if i % step == 0:
            g = cv2.cvtColor(im[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
            gb = cv2.GaussianBlur(g, (5, 5), 0).astype(np.int16)
            if prev is None:
                motion, mfrac = 0.0, 0.0
            else:
                d = np.abs(gb - prev)
                motion = float(d.mean())
                mfrac = float((d > 25).mean())
            prev = gb

            c, mask = segment(g, thr)
            r = dict(frame=i, time_s=i / fps, motion=motion, motion_roi_frac=mfrac)
            if c is None:
                r["found"] = 0
                for k in BASE_COLS:
                    r.setdefault(k, np.nan)
            else:
                nfound += 1
                r["found"] = 1
                r.update(shape_features(c, g.shape))
            rows.append(r)

            if len(tiles) < 4 and c is not None and i > ntot * 0.2 \
                    and i % max(1, ntot // 5) < step:
                v = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
                cv2.drawContours(v, [c], -1, (0, 0, 255), 2)
                bx, by, bw, bh = cv2.boundingRect(c)
                cv2.rectangle(v, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
                cv2.putText(v, f"f{i}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (0, 255, 255), 2)
                tiles.append(cv2.resize(v, (330, 190)))
        i += 1
        if i % 9000 == 0:
            print(f"    {i}/{ntot}", end="\r", flush=True)
    cap.release()
    print(f"    {i}/{ntot} frames read      ")

    df = pd.DataFrame(rows)
    # forward/backward fill the geometry on the rare frames with no contour, so
    # the rolling windows are not poisoned by NaN
    geo = [c for c in BASE_COLS if c not in
           ("frame", "time_s", "found", "motion", "motion_roi_frac")]
    df[geo] = df[geo].ffill().bfill()

    # derived kinematics
    dt = step / fps
    df["speed"] = np.hypot(df["cx"].diff().fillna(0), df["cy"].diff().fillna(0)) / dt
    df["d_area"] = df["area"].diff().fillna(0) / dt
    df["d_top_y"] = df["top_y"].diff().fillna(0) / dt      # rearing signature
    df["d_ecc"] = df["ecc"].diff().fillna(0) / dt
    df["height_norm"] = (df["bot_y"] - df["top_y"]) / df["w"].clip(lower=1)

    roll_on = ["area", "h", "aspect", "solidity", "ecc", "top_y",
               "low_contour_rough", "hull_defect_max", "motion", "speed",
               "height_norm"]
    df = pd.concat([df, pd.DataFrame(rolling(df, roll_on, fps / step))], axis=1)

    os.makedirs(outdir, exist_ok=True)
    csvp = os.path.join(outdir, f"Features_{vid}.csv")
    df.to_csv(csvp, index=False)
    print(f"  detected in {nfound}/{len(df)} analysed frames "
          f"({100 * nfound / max(len(df), 1):.1f}%)")
    print(f"  {len(df)} rows x {df.shape[1]} columns")
    print(f"  wrote {csvp}")

    _diag(df, tiles, os.path.join(outdir, f"Features_{vid}.png"), vid)
    return csvp


def _diag(df, tiles, png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(4, 4, height_ratios=[1.1, 1, 1, 1], hspace=.45)
    for j, t in enumerate(tiles[:4]):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(cv2.cvtColor(t, cv2.COLOR_BGR2RGB))
        ax.axis("off")
    sub = df.iloc[::max(1, len(df) // 6000)]
    panels = [("height_norm", "height / width", "rearing", "#1C6E8C"),
              ("low_contour_rough", "outline roughness", "paw / guarding", "#B45309"),
              ("motion", "motion energy", "activity", "#2E8B6F")]
    for k, (col, lab, note, col_c) in enumerate(panels):
        ax = fig.add_subplot(gs[k + 1, :])
        ax.plot(sub["time_s"], sub[col], lw=.6, color=col_c)
        ax.set_ylabel(lab, fontsize=9)
        ax.grid(alpha=.3)
        ax.text(0.995, 0.93, note, transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color=col_c,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=col_c, lw=.8))
        if k == len(panels) - 1:
            ax.set_xlabel("time (s)")
    fig.suptitle(f"side-view features  -  {title}", fontsize=12)
    fig.savefig(png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", help="side-view video, or a folder of them")
    ap.add_argument("--out", default=None)
    ap.add_argument("--roi", nargs=4, type=int, default=list(ROI),
                    metavar=("Y0", "Y1", "X0", "X1"))
    ap.add_argument("--thr", type=int, default=THR)
    ap.add_argument("--step", type=int, default=1,
                    help="analyse every Nth frame (1 = every frame)")
    a = ap.parse_args()

    if os.path.isdir(a.target):
        vids = sorted(f for f in os.listdir(a.target)
                      if f.lower().endswith((".avi", ".mp4", ".mov")))
        paths = [os.path.join(a.target, v) for v in vids]
        outdir = a.out or os.path.join(a.target, "output")
    else:
        paths = [a.target]
        outdir = a.out or os.path.join(os.path.dirname(a.target) or ".", "output")
    for p in paths:
        run(p, outdir, tuple(a.roi), a.thr, a.step)


if __name__ == "__main__":
    main()
