"""Your marks on a cell_curation_gui screenshot -> FOV coordinates of that run.

The GUI draws the field at 0.5x (prepare_curation_data scale) with every candidate outlined at
30 % of its footprint peak (green keep / red delete). This redraws those outlines from the run's
extract.mat, finds the scale and offset that put them on the red + green pixels of the screenshot
(scale search + FFT cross-correlation), and maps every yellow mark (your pen) to FOV pixels.
For each mark: the nearest candidate (centroid distance, and whether the mark lies on its
footprint), its verdict / keep / reasons from the review csv.

  python ca_marks_from_screenshot.py --run <preview3min/<rid>> --png <screenshot> --out <csv>
"""
import argparse
import os

import numpy as np
import pandas as pd
import scipy.io as sio
from PIL import Image
from scipy.ndimage import binary_dilation, label, find_objects, center_of_mass
from skimage.segmentation import find_boundaries

SC = 0.5


def screenshot_masks(png):
    im = np.asarray(Image.open(png).convert("RGB")).astype(int)
    R, G, B = im[..., 0], im[..., 1], im[..., 2]
    red = (R > 170) & (G < 130) & (B < 130)
    green = (G > 140) & (R < 140) & (B < 140)
    yellow = (R > 190) & (G > 160) & (B < 110)
    return im, red | green, yellow


def model_outline(S, H, W):
    """boundary pixels of every footprint (>= 0.3 max) in display coordinates"""
    pts = []
    for c in range(S.shape[2]):
        f = S[:, :, c]
        if f.max() <= 0:
            continue
        m = f >= 0.3 * f.max()
        b = find_boundaries(m, mode="inner")
        r, cc = np.where(b)
        pts.append(np.c_[r, cc] * SC)
    return np.vstack(pts)


def fit(pts, mask, scales):
    from numpy.fft import rfft2, irfft2
    Hs, Ws = mask.shape
    Mf = rfft2(binary_dilation(mask, iterations=1).astype(float), s=(2 * Hs, 2 * Ws))
    best = (-1, None, None)
    for s in scales:
        q = np.round(pts * s).astype(int)
        h, w = q[:, 0].max() + 3, q[:, 1].max() + 3
        if h > 2 * Hs or w > 2 * Ws:
            continue
        T = np.zeros((h, w))
        T[q[:, 0], q[:, 1]] = 1
        T = binary_dilation(T > 0, iterations=1).astype(float)
        Tf = rfft2(T[::-1, ::-1], s=(2 * Hs, 2 * Ws))
        cc = irfft2(Mf * Tf, s=(2 * Hs, 2 * Ws))
        i = np.unravel_index(np.argmax(cc), cc.shape)
        score = cc[i] / T.sum()
        if score > best[0]:
            dy, dx = i[0] - (h - 1), i[1] - (w - 1)
            best = (score, s, (dy, dx))
    return best


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="preview3min/<rid> folder")
    ap.add_argument("--png", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-px", type=int, default=25, help="smallest yellow mark (screen px)")
    ap.add_argument("--tag", default="", help="<mouse>_d1 when --run is not named <tag>_<session> (e.g. a v2_loose folder)")
    a = ap.parse_args(argv)
    rid = os.path.basename(os.path.normpath(a.run))
    tag = a.tag or rid.rsplit("_", 1)[0]
    M = sio.loadmat(os.path.join(a.run, "3_extract", f"{tag}_extract.mat"), variable_names=["spatial_weights"])
    S = np.asarray(M["spatial_weights"], np.float32)
    R = pd.read_csv(os.path.join(a.run, "4_review", f"{tag}_cell_review.csv"))
    im, mask, yellow = screenshot_masks(a.png)
    pts = model_outline(S, *S.shape[:2])
    scales = np.arange(1.5, 2.6, 0.01)
    score, s, (dy, dx) = fit(pts, mask, scales)
    print(f"{rid}: screenshot {mask.shape[::-1]}, display scale {s:.2f} screen px per GUI px, offset ({dx}, {dy}), "
          f"outline overlap {score:.2f}")
    lab, n = label(binary_dilation(yellow, iterations=2))
    rows = []
    for k, sl in enumerate(find_objects(lab), start=1):
        comp = lab[sl] == k
        if comp.sum() < a.min_px:
            continue
        cy, cx = center_of_mass(lab == k)
        hgt, wid = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        # screen -> GUI display -> FOV
        fr, fc = ((cy - dy) / s) / SC, ((cx - dx) / s) / SC
        d = np.hypot(R.row - fr, R.col - fc)
        j = int(np.argmin(d))
        rr, c0 = int(round(fr)), int(round(fc))
        inside = []
        if 0 <= rr < S.shape[0] and 0 <= c0 < S.shape[1]:
            f = S[rr, c0, :]
            mx = S.reshape(-1, S.shape[2]).max(0)
            inside = [int(i) + 1 for i in np.where(f >= 0.3 * np.maximum(mx, 1e-12))[0]]
        rows.append(dict(mark=len(rows) + 1, screen_x=round(cx, 1), screen_y=round(cy, 1), mark_w_px=wid, mark_h_px=hgt,
                         fov_row=round(fr, 1), fov_col=round(fc, 1), nearest_id=int(R.id.iloc[j]),
                         nearest_dist_px=round(float(d.iloc[j]), 1), on_footprint_of=";".join(map(str, inside)),
                         nearest_keep=int(R.keep.iloc[j]), nearest_verdict=R.verdict.iloc[j], nearest_reasons=R.reasons.iloc[j]))
    X = pd.DataFrame(rows)
    X.to_csv(a.out, index=False)
    print(X[["mark", "fov_row", "fov_col", "nearest_id", "nearest_dist_px", "on_footprint_of", "nearest_keep", "nearest_verdict",
             "nearest_reasons"]].to_string(index=False))
    # check image: the redrawn outlines on the screenshot
    chk = im.copy()
    q = np.round(pts * s).astype(int) + np.array([dy, dx])
    ok = (q[:, 0] >= 0) & (q[:, 0] < chk.shape[0]) & (q[:, 1] >= 0) & (q[:, 1] < chk.shape[1])
    chk[q[ok, 0], q[ok, 1]] = [0, 200, 255]
    Image.fromarray(chk.astype(np.uint8)).save(a.out.replace(".csv", "_alignment.png"))
    print("->", a.out)


if __name__ == "__main__":
    main()
