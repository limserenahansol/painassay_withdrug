"""ca_yellow_marks.py - the places you circled in yellow on a screenshot of a tracking map -> Day-2 pixel positions.

The screenshot is matched to the tracking map it was taken from (SIFT + RANSAC similarity), each yellow stroke
(R > 200, G > 170, B < 90; strokes 6 px apart joined) becomes one mark at its centre, and the mark is converted from the
map figure to Day-2 pixels through the Day-2 panel (the grey image block in the right half of the figure).

  python ca_yellow_marks.py --mouse 5939 --shot <png/webp> [--shot ...]
Writes analysis\\crossday\\<m>_yellow_marks.csv (mark, day2_row, day2_col, shot) and <m>_yellow_marks_check.png
(the marks drawn back on the map, to check the mapping).
"""
import argparse
import os

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, label

BASE = r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553"


def right_panel_box(O):
    """pixel box (x0, y0, x1, y1) of the Day-2 image in the right half of the map figure"""
    H, W = O.shape[:2]
    half = O[:, W // 2:]
    nonwhite = (half.min(axis=2) < 245)
    rows = np.flatnonzero(nonwhite.mean(1) > 0.5)
    # the image is the longest run of rows that are mostly grey
    runs = np.split(rows, np.flatnonzero(np.diff(rows) > 1) + 1)
    rr = max(runs, key=len)
    cols = np.flatnonzero(nonwhite[rr].mean(0) > 0.5)
    runs = np.split(cols, np.flatnonzero(np.diff(cols) > 1) + 1)
    cc = max(runs, key=len)
    return W // 2 + cc[0], rr[0], W // 2 + cc[-1] + 1, rr[-1] + 1


def match(shot, O):
    g1 = cv2.cvtColor(shot, cv2.COLOR_RGB2GRAY); g2 = cv2.cvtColor(O, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create(6000)
    k1, d1 = sift.detectAndCompute(g1, None); k2, d2 = sift.detectAndCompute(g2, None)
    mm = cv2.BFMatcher().knnMatch(d1, d2, k=2)
    good = [a for a, b in mm if a.distance < 0.75 * b.distance]
    P = np.float32([k1[a.queryIdx].pt for a in good]); Q = np.float32([k2[a.trainIdx].pt for a in good])
    M, inl = cv2.estimateAffinePartial2D(P, Q, method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=5000)
    return M, int(inl.sum()), len(good)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mouse", required=True)
    ap.add_argument("--shot", nargs="+", required=True)
    ap.add_argument("--map", default="")
    a = ap.parse_args(argv)
    m = a.mouse
    cd = os.path.join(BASE, "analysis", "crossday")
    O = cv2.cvtColor(cv2.imread(a.map or os.path.join(cd, f"{m}_tracking_map.png")), cv2.COLOR_BGR2RGB)
    import scipy.io as sio
    t2 = f"{m}_d2"
    H2, W2 = sio.loadmat(os.path.join(BASE, "mini1p_out", "20261021", t2, "3_extract", f"{t2}_projections.mat"),
                         variable_names=["max_dff"])["max_dff"].shape
    x0, y0, x1, y1 = right_panel_box(O)
    rows = []
    for sp in a.shot:
        from PIL import Image
        S = np.asarray(Image.open(sp).convert("RGB"))
        M, ninl, ngood = match(S, O)
        yel = (S[:, :, 0] > 200) & (S[:, :, 1] > 170) & (S[:, :, 2] < 90)
        L, n = label(binary_dilation(yel, iterations=3))
        for k in range(1, n + 1):
            yy, xx = np.nonzero((L == k) & yel)
            if len(yy) < 15:
                continue
            p = np.array([xx.mean(), yy.mean(), 1.0]) @ M.T          # map figure pixel
            col = (p[0] - x0) / (x1 - x0) * W2 - 0.5
            row = (p[1] - y0) / (y1 - y0) * H2 - 0.5
            rows.append(dict(mark=len(rows) + 1, day2_row=round(row + 1, 1), day2_col=round(col + 1, 1),
                             fig_x=round(p[0], 1), fig_y=round(p[1], 1), n_px=len(yy), shot=os.path.basename(sp),
                             sift_inliers=ninl))
        print(f"{m} {os.path.basename(sp)}: {ninl} of {ngood} SIFT matches inliers, scale {np.hypot(*M[0, :2]):.3f}; "
              f"{sum(r['shot'] == os.path.basename(sp) for r in rows)} yellow marks")
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(cd, f"{m}_yellow_marks.csv"), index=False)
    chk = O.copy()
    cv2.rectangle(chk, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 2)
    for r in rows:
        cv2.circle(chk, (int(r["fig_x"]), int(r["fig_y"])), 22, (255, 220, 0), 3)
        cv2.putText(chk, str(r["mark"]), (int(r["fig_x"]) + 24, int(r["fig_y"])), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 220, 0), 2)
    cv2.imwrite(os.path.join(cd, f"{m}_yellow_marks_check.png"), cv2.cvtColor(chk, cv2.COLOR_RGB2BGR))
    print(f"{m}: {len(rows)} marks -> {cd}")


if __name__ == "__main__":
    main()
