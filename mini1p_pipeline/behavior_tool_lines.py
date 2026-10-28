"""Thin-tool features of the bottom view: the von Frey filament, the pin shaft, the heat probe
and their holders are thin bright/dark LINES that the blurred silhouette features cannot see.

Per frame (camera frame n = row n-1):
  z      = |frame - mouse-free background| / (1.48 MAD + 4), at full resolution (the static
           mesh drops out because it is in the background; the background is the per-pixel
           median of 300 frames with the mouse and 20 px around it masked)
  line   = max over 8 orientations of (mean of z on a 21-px line - mean of z in a 7-px band
           around it): high on thin straight things, ~0 on blobs (the mouse's shadow) and
           on flat areas
  near   = 5-40 px from the mouse silhouette, far = 40-100 px
  columns: line_near_p99, line_near_cnt (pixels with line > 1.5), line_far_p99,
           line_far_cnt, z_far_mean
Validated on 5941 SBI: at the user's delivery taps vs other stimulus-block frames, single-frame
AUC 0.84 (far p99), 0.78-0.94 per stimulus.

  python behavior_tool_lines.py --bottom B.avi --features <tag>_bfeatures.csv --out <tag>_btool.npz
"""
import argparse
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import behavior_features as bf  # noqa: E402

HALF = 180          # crop half-size around the mouse centroid


def kernels():
    K = []
    for ang in np.arange(0, 180, 22.5):
        k = np.zeros((25, 25), np.float32)
        dx, dy = np.cos(np.radians(ang)), np.sin(np.radians(ang))
        for t in np.linspace(-10, 10, 41):
            k[int(round(12 + t * dy)), int(round(12 + t * dx))] = 1
        band = cv2.dilate(k, np.ones((7, 7), np.uint8))
        K.append(k / k.sum() - band / band.sum())
    return K


def background(path, n, k=300):
    c = cv2.VideoCapture(path)
    x0, y0, x1, y1 = bf.B_ARENA
    arena = np.zeros((480, 720), bool)
    arena[y0:y1, x0:x1] = True
    raw, blur = [], []
    for j in np.linspace(0, n - 1, k).astype(int):
        c.set(cv2.CAP_PROP_POS_FRAMES, int(j))
        ok, f = c.read()
        if ok:
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            raw.append(g.astype(np.float32))
            blur.append(cv2.GaussianBlur(g, (0, 0), 3).astype(np.float32))
    raw, blur = np.stack(raw), np.stack(blur)
    bg90 = np.percentile(blur, 90, axis=0)
    for i in range(len(raw)):
        m = ((blur[i] < bf.B_DARK) & (bg90 - blur[i] > bf.B_BG_DROP) & arena).astype(np.uint8)
        raw[i][cv2.dilate(m, np.ones((41, 41), np.uint8)) > 0] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(raw, axis=0)
        mad = np.nanmedian(np.abs(raw - med), axis=0)
    med = np.where(np.isfinite(med), med, 255).astype(np.float32)
    mad = np.where(np.isfinite(mad), mad, 50).astype(np.float32)
    return med, mad, bg90.astype(np.float32)


def work(args):
    path, f0, f1, cxs, cys, med, mad, bg90 = args
    K = kernels()
    x0, y0, x1, y1 = bf.B_ARENA
    arena = np.zeros((480, 720), bool)
    arena[y0:y1, x0:x1] = True
    c = cv2.VideoCapture(path)
    c.set(cv2.CAP_PROP_POS_FRAMES, int(f0))
    out = np.full((f1 - f0, 5), np.nan, np.float32)
    k11, k81, k201 = (np.ones((s, s), np.uint8) for s in (11, 81, 201))
    for i in range(f0, f1):
        ok, im = c.read()
        if not ok:
            break
        cx, cy = cxs[i - f0], cys[i - f0]
        if not (np.isfinite(cx) and np.isfinite(cy)):
            continue
        xa, ya = int(np.clip(cx - HALF, 0, 720 - 2 * HALF)), int(np.clip(cy - HALF, 0, max(0, 480 - 2 * HALF)))
        xb, yb = xa + 2 * HALF, min(480, ya + 2 * HALF)
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)[ya:yb, xa:xb]
        sm = cv2.GaussianBlur(g, (0, 0), 3)
        m = ((sm < bf.B_DARK) & (bg90[ya:yb, xa:xb] - sm > bf.B_BG_DROP) & arena[ya:yb, xa:xb]).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        mm, _, _ = bf.pick(m, np.array([cx - xa, cy - ya]))
        if mm is None:
            continue
        z = np.clip(np.abs(g.astype(np.float32) - med[ya:yb, xa:xb]) / (1.4826 * mad[ya:yb, xa:xb] + 4), 0, 10)
        R = np.max([cv2.filter2D(z, -1, k) for k in K], axis=0)
        mm8 = mm.astype(np.uint8)
        d11, d81 = cv2.dilate(mm8, k11) > 0, cv2.dilate(mm8, k81) > 0
        near = d81 & ~d11
        far = (cv2.dilate(mm8, k201) > 0) & ~d81
        if near.any():
            out[i - f0, 0] = np.percentile(R[near], 99)
            out[i - f0, 1] = (R[near] > 1.5).sum()
        if far.any():
            out[i - f0, 2] = np.percentile(R[far], 99)
            out[i - f0, 3] = (R[far] > 1.5).sum()
            out[i - f0, 4] = z[far].mean()
    return f0, out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bottom", required=True)
    ap.add_argument("--features", required=True, help="<tag>_bfeatures.csv (mouse centroid per frame)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    F = pd.read_csv(a.features)
    n = len(F)
    cx = F.b_cx.interpolate(limit_direction="both").values
    cy = F.b_cy.interpolate(limit_direction="both").values
    med, mad, bg90 = background(a.bottom, n)
    edges = np.linspace(0, n, a.workers * 4 + 1).astype(int)
    jobs = [(a.bottom, e0, e1, cx[e0:e1], cy[e0:e1], med, mad, bg90) for e0, e1 in zip(edges[:-1], edges[1:])]
    out = np.full((n, 5), np.nan, np.float32)
    with ProcessPoolExecutor(a.workers) as ex:
        for f0, o in ex.map(work, jobs):
            out[f0:f0 + len(o)] = o
            print(f"  frames {f0}-{f0 + len(o)} done", flush=True)
    names = np.array(["line_near_p99", "line_near_cnt", "line_far_p99", "line_far_cnt", "z_far_mean"])
    np.savez_compressed(a.out, X=out, names=names)
    print(f"behavior_tool_lines: {n} frames -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
