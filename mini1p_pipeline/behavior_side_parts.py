"""Side-camera body-part motion for grooming (and sitting up): the mouse blob of the side view
(dark on the white background, behavior_features.S_BOX / S_DARK) is cut into its upper 45 %
(head, forepaws when the mouse sits up) and lower 55 % (trunk, hind legs).

Per frame (camera frame n = row n-1):
  sp_top, sp_bot, sp_h, sp_w     blob bounding box (y of top / bottom, height, width)
  sp_up_me, sp_low_me            mean |frame difference| in the upper / lower part of the box
  sp_up_w                        width of the blob at 25 % of its height (forepaws together
                                 in front of the chest = narrow)
Grooming = forepaws stroking the face: the upper part moves rhythmically (3-8 Hz strokes)
while the lower part and the floor position stay still (see behavior_auto_annotate.py).

  python behavior_side_parts.py --side S.avi --n FRAMES --out <tag>_bside.npz
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import behavior_features as bf  # noqa: E402


def work(args):
    path, f0, f1 = args
    sx0, sy0, sx1, sy1 = bf.S_BOX
    c = cv2.VideoCapture(path)
    c.set(cv2.CAP_PROP_POS_FRAMES, int(max(0, f0 - 1)))
    prev = None
    if f0 > 0:
        ok, im = c.read()
        prev = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if ok else None
    out = np.full((f1 - f0, 7), np.nan, np.float32)
    for i in range(f0, f1):
        ok, im = c.read()
        if not ok:
            break
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        roi = g[sy0:sy1, sx0:sx1]
        ms = cv2.morphologyEx((roi < bf.S_DARK).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        msk, st, _ = bf.largest(ms)
        if msk is not None and prev is not None:
            x, y, w, h = st[0], st[1], st[2], st[3]
            yc = y + int(round(0.45 * h))
            d = cv2.absdiff(roi, prev[sy0:sy1, sx0:sx1])
            m8 = cv2.dilate(msk.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
            up = m8.copy(); up[yc:, :] = False
            lo = m8.copy(); lo[:yc, :] = False
            r25 = msk[min(msk.shape[0] - 1, y + int(0.25 * h)), :]
            out[i - f0] = [y + sy0, y + h + sy0, h, w, d[up].mean() if up.any() else np.nan,
                           d[lo].mean() if lo.any() else np.nan, r25.sum()]
        prev = g
    return f0, out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args(argv)
    edges = np.linspace(0, a.n, a.workers * 4 + 1).astype(int)
    out = np.full((a.n, 7), np.nan, np.float32)
    with ProcessPoolExecutor(a.workers) as ex:
        for f0, o in ex.map(work, [(a.side, e0, e1) for e0, e1 in zip(edges[:-1], edges[1:])]):
            out[f0:f0 + len(o)] = o
    names = np.array(["sp_top", "sp_bot", "sp_h", "sp_w", "sp_up_me", "sp_low_me", "sp_up_w"])
    np.savez_compressed(a.out, X=out, names=names)
    print(f"behavior_side_parts: {a.n} frames -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
