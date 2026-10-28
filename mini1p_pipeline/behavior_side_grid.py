"""Side-camera motion on a 3 x 3 grid laid over the mouse (grooming features).

Grooming = the body stays put while ONE part of the mouse (head + forepaws) moves fast and
rhythmically (face-washing strokes; body licking moves the head in the same way). Because the
head can be on either side of the side-view blob, the blob's bounding box is cut into 3 x 3
cells and the motion is measured per cell.

The mouse is segmented as "darker than this session's own background" (per-pixel 90th
percentile of 200 frames), not with a fixed grey level: the side view of 1191 is dark (grey ~75)
and 5939 SBI has glare late in the session, so a fixed threshold fails there.

Per frame (camera frame n = row n-1):
  sg_me0..8    mean |frame difference| in each cell (row-major, top-left first), inside the
               dilated blob
  sg_cx, sg_cy blob centroid; sg_h, sg_w bounding box; sg_top, sg_bot, sg_area
  sg_dark      mean darkness of the blob relative to the background (contrast check)
  sg_bright    mean grey of the box (frames far from the session median = lights off / glare)

  python behavior_side_grid.py --side S.avi --n FRAMES --out <tag>_bsgrid.npz
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import behavior_features as bf  # noqa: E402

BOX = (180, 40, 600, 455)          # x0 y0 x1 y1: the cylinder, floor included


def background(path, n, k=200):
    c = cv2.VideoCapture(path)
    x0, y0, x1, y1 = BOX
    fr = []
    for j in np.linspace(0, n - 1, k).astype(int):
        c.set(cv2.CAP_PROP_POS_FRAMES, int(j))
        ok, f = c.read()
        if ok:
            fr.append(cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1], (0, 0), 2))
    return np.percentile(np.stack(fr), 90, axis=0).astype(np.float32)


def work(args):
    path, f0, f1, bg = args
    x0, y0, x1, y1 = BOX
    drop = np.maximum(25.0, 0.30 * bg)
    c = cv2.VideoCapture(path)
    c.set(cv2.CAP_PROP_POS_FRAMES, int(max(0, f0 - 1)))
    prev = None
    if f0 > 0:
        ok, im = c.read()
        prev = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1] if ok else None
    out = np.full((f1 - f0, 18), np.nan, np.float32)
    for i in range(f0, f1):
        ok, im = c.read()
        if not ok:
            break
        roi = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1]
        sm = cv2.GaussianBlur(roi, (0, 0), 2).astype(np.float32)
        out[i - f0, 17] = roi.mean()
        m = ((bg - sm) > drop).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        msk, st, cen = bf.largest(m)
        if msk is not None and prev is not None and st[cv2.CC_STAT_AREA] > 1500:
            x, y, w, h = st[0], st[1], st[2], st[3]
            d = cv2.absdiff(roi, prev).astype(np.float32)
            m8 = cv2.dilate(msk.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(np.float32)
            xs = np.linspace(x, x + w, 4).astype(int)
            ys = np.linspace(y, y + h, 4).astype(int)
            v = []
            for r in range(3):
                for q in range(3):
                    mm = m8[ys[r]:ys[r + 1], xs[q]:xs[q + 1]]
                    v.append((d[ys[r]:ys[r + 1], xs[q]:xs[q + 1]] * mm).sum() / max(mm.sum(), 1))
            dark = float(((bg - sm)[msk]).mean() / max(bg[msk].mean(), 1))
            out[i - f0, :17] = v + [cen[0] + x0, cen[1] + y0, h, w, y + y0, y + h + y0, st[cv2.CC_STAT_AREA], dark]
        prev = roi
    return f0, out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args(argv)
    bg = background(a.side, a.n)
    edges = np.linspace(0, a.n, a.workers * 4 + 1).astype(int)
    out = np.full((a.n, 18), np.nan, np.float32)
    with ProcessPoolExecutor(a.workers) as ex:
        for f0, o in ex.map(work, [(a.side, e0, e1, bg) for e0, e1 in zip(edges[:-1], edges[1:])]):
            out[f0:f0 + len(o)] = o
    names = np.array([f"sg_me{k}" for k in range(9)] + ["sg_cx", "sg_cy", "sg_h", "sg_w", "sg_top", "sg_bot",
                                                         "sg_area", "sg_dark", "sg_bright"])
    np.savez_compressed(a.out, X=out, names=names, bg=bg.astype(np.uint8), box=np.array(BOX))
    print(f"behavior_side_grid: {a.n} frames, {np.isfinite(out[:, 0]).mean() * 100:.1f}% with a mouse -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
