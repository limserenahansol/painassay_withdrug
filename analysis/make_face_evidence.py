"""make_face_evidence.py  -  the face zoom, Day 1 vs Day 2, and its verdict.

WHY THIS SCRIPT EXISTS
    A grimace-like face was observed on the drug day: narrowed, triangular
    eyes and ears rotated back. Those are two of the five Mouse Grimace Scale
    action units (Langford et al., Nat Methods 2010) - orbital tightening and
    ear position - so the observation is worth following up properly.

    This script zooms the head region in both days so the footage can be
    judged directly, and measures whether it could support that scoring at
    all. It cannot, and the panel says so on its face rather than leaving a
    blurry crop to be over-interpreted.

WHAT IS MEASURED
    Internal contrast inside the silhouette: the standard deviation and the
    5-95 percentile range of grey values within the animal. The Mouse Grimace
    Scale needs a resolvable eye aperture and ear angle. If the animal carries
    only a few grey levels of internal structure, no amount of stretching adds
    the missing information - it only amplifies sensor noise, which the
    right-hand columns of the panel demonstrate.

USAGE
    python make_face_evidence.py --day1-side <folder> --day2-side <folder> \\
                                 --out <folder>

Hansol Lim - HEAL mini1p / SBI-553
"""
from __future__ import annotations

import argparse
import glob
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROI = dict(y0=150, y1=430, x0=60, x1=680)
DARK_PCT = 8.0
C2 = "#C0483B"
plt.rcParams.update({"font.size": 11, "figure.dpi": 130})


def silhouette(gray):
    band = gray[ROI["y0"]:ROI["y1"], ROI["x0"]:ROI["x1"]]
    thr = np.percentile(band, DARK_PCT)
    m = cv2.morphologyEx((band < thr).astype(np.uint8), cv2.MORPH_OPEN,
                         np.ones((5, 5), np.uint8))
    n, lab, st, cent = cv2.connectedComponentsWithStats(m, 8)
    if n < 2:
        return None
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    if st[i, cv2.CC_STAT_AREA] < 800:
        return None
    x, y, w, h = (st[i, cv2.CC_STAT_LEFT], st[i, cv2.CC_STAT_TOP],
                  st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT])
    return dict(x=int(x + ROI["x0"]), y=int(y + ROI["y0"]),
                w=int(w), h=int(h), mask=(lab == i))


def best_frame(path, times):
    """Pick the sampled frame with the MOST internal structure - the fairest
    possible case for the footage, not a cherry-picked bad one."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    best = None
    for t in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, fr = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        s = silhouette(g)
        if s is None:
            continue
        inner = g[s["y"] + s["h"] // 5:s["y"] + 4 * s["h"] // 5,
                  s["x"] + s["w"] // 5:s["x"] + 4 * s["w"] // 5]
        if inner.size == 0:
            continue
        sd = float(np.std(inner))
        rng = float(np.percentile(inner, 95) - np.percentile(inner, 5))
        if best is None or sd > best["sd"]:
            best = dict(t=t, gray=g, sil=s, sd=sd, rng=rng)
    cap.release()
    return best


def treatments(crop):
    a = cv2.resize(crop, (300, 300), interpolation=cv2.INTER_CUBIC)
    lo, hi = np.percentile(a, 2), np.percentile(a, 98)
    b = np.clip((a.astype(np.float32) - lo) * 255.0 / max(hi - lo, 1),
                0, 255).astype(np.uint8)
    c = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(6, 6)).apply(b)
    c = cv2.fastNlMeansDenoising(c, None, 12, 7, 21)
    return a, b, c


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--day1-side", required=True)
    ap.add_argument("--day2-side", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mouse", default="female1",
                    help="which animal to show (default female1, present "
                         "both days)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    times = [120, 330, 420, 700, 780, 900, 1100, 1180, 1300, 1450, 1600]

    picks = {}
    for lab, folder in (("Day 1  no drug", a.day1_side),
                        ("Day 2  drug", a.day2_side)):
        # startswith, not "in": "male2" is a substring of "female2", so a
        # containment test silently returns the wrong animal
        vids = [v for v in sorted(glob.glob(os.path.join(folder, "*.avi")))
                if os.path.basename(v).lower().startswith(a.mouse.lower())]
        if not vids:
            raise SystemExit(f"no {a.mouse} video in {folder}")
        print(f"{lab}: {os.path.basename(vids[0])}")
        b = best_frame(vids[0], times)
        if b is None:
            raise SystemExit(f"no silhouette found in {vids[0]}")
        print(f"    best of {len(times)} sampled frames: t = {b['t']} s, "
              f"internal SD {b['sd']:.1f}, 5-95 range {b['rng']:.0f} "
              f"grey levels")
        picks[lab] = b

    fig = plt.figure(figsize=(15.2, 7.6))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.55, 1, 1, 1],
                          hspace=.22, wspace=.10)
    for r, (lab, b) in enumerate(picks.items()):
        g, s = b["gray"], b["sil"]
        ax = fig.add_subplot(gs[r, 0])
        ax.imshow(g, cmap="gray", vmin=0, vmax=255)
        # Crop the WHOLE animal, with a margin. An earlier version guessed the
        # head as the upper third of the blob, which put the box on the flank:
        # the mouse lies horizontal, so the head is at one END, and which end
        # changes between frames. Cropping the whole animal cannot be wrong
        # about that and the head is included either way.
        pad = int(.12 * max(s["w"], s["h"]))
        x0, y0 = max(s["x"] - pad, 0), max(s["y"] - pad, 0)
        x1 = min(s["x"] + s["w"] + pad, g.shape[1])
        y1 = min(s["y"] + s["h"] + pad, g.shape[0])
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                   color=C2, lw=2.0))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(f"{lab}\nt = {b['t']} s", fontsize=12,
                      fontweight="bold")
        if r == 0:
            ax.set_title("side view, full frame  (720 × 480, backlit)",
                         fontsize=11)
        crop = g[y0:y1, x0:x1]
        tr = treatments(crop)
        for c, (img, nm) in enumerate(zip(
                tr, ("animal zoomed, as recorded", "contrast stretched",
                     "+ CLAHE, denoised"))):
            ax = fig.add_subplot(gs[r, c + 1])
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_xlabel(f"internal contrast:  SD {b['sd']:.1f} / 255,"
                              f"  5–95 range {b['rng']:.0f} / 255",
                              fontsize=9, color="#333")
            if r == 0:
                ax.set_title(nm, fontsize=11)

    fig.suptitle(
        "Requested face zoom, Day 1 vs Day 2, same mouse, best of 11 sampled "
        "frames per day (the most internal structure, not the worst)\n"
        "VERDICT: orbital tightening cannot be scored - the eye is not "
        "resolved at all. Ear ANGLE is partly visible as an outline in "
        "favourable frames,\n"
        "but not reliably enough to grade. Stretching only amplifies sensor "
        "noise and mesh texture; the detail was never recorded.",
        fontsize=11.5)
    fig.text(.5, .008,
             "TO SCORE THE MOUSE GRIMACE SCALE:  front or 3/4 view  ·  "
             "front-lit, not backlit  ·  head filling ≥ 300 × 300 px  ·  "
             "≥ 8-bit mono is fine, colour not required.  "
             "A USB macro camera aimed at the cylinder would do it.",
             ha="center", fontsize=10.5, color=C2, fontweight="bold")
    p = os.path.join(a.out, "FACE_zoom_day1_vs_day2.png")
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
