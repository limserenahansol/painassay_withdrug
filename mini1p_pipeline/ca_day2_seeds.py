"""ca_day2_seeds.py - the Day-1 confirmed neurons as forced seeds for the Day-2 detection.

The Day-1 field of view is registered onto the Day-2 one with the same high-pass + normalised cross-correlation as
ca_transfer_decisions.register, now searched over a small rotation and scale too (the baseplate can be re-mounted a
little turned; a translation-only fit puts the seeds of neurons far from the centre off by several px):
  coarse grid  rotation -6..6 deg (0.5 deg), scale 0.95..1.05 (0.025), translation by NCC at each
  fine grid    +-0.5 deg (0.1 deg), +-0.02 scale (0.005) around the best
The rotation / scale is only used when it raises the correlation clearly (by >= 0.01) over translation only.
SENSOR PATTERN (found 2026-10-22 on 5937): the camera's fixed pixel pattern (hot pixels, dust) does not move with the
tissue; with a fine high-pass (1.5 / 12 px) it can dominate the correlation, and the 'registration' then equals the
difference of the two lens crops (5937: (+2, +23) = crop rows 22 -> 24, cols 274 -> 297; tissue was (-3, -16)).
Now: 5 x 5 median filter (removes single-pixel patterns), then a cell-scale band-pass (3 / 15 px).
Images: Day-1 EXTRACT mean image vs the Day-2 mean image (<tag>_pre_mc.h5).
Each Day-1 kept neuron's centroid (cell_review row / col, 1-based) is mapped into Day-2 coordinates; neurons that
land outside the Day-2 frame are dropped. Writes analysis\\day2_seeds\\marks_day2_<tag>.csv (mark, fov_row, fov_col,
day1_id - what RUN_ca_overnight_20261021.m reads as extra_rc), day2_registration_<m>.json and a QC image: the Day-2
mean image with the mapped Day-1 outlines (if the registration is right they sit on the somata) and the correlation
for each rotation.

  python ca_day2_seeds.py --mouse 5941
  python ca_day2_seeds.py --selftest      (Day-1 mean image turned / scaled / shifted by known amounts)
"""
import argparse
import json
import os
import sys

import cv2
import h5py
import numpy as np
import pandas as pd
import scipy.io as sio

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ca_transfer_decisions import highpass  # noqa: E402

BASE = r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553"


def _ncc(a_hp, b_hp, margin=60, max_shift=120):
    """best translation of the central part of b inside a (both high-passed); (dy, dx, r):
    b pixel (y, x) = a pixel (y + dy, x + dx)"""
    tpl = b_hp[margin:-margin, margin:-margin]
    big = cv2.copyMakeBorder(a_hp, max_shift, max_shift, max_shift, max_shift, cv2.BORDER_CONSTANT, value=0)
    r = cv2.matchTemplate(big, tpl, cv2.TM_CCOEFF_NORMED)
    _, peak, _, loc = cv2.minMaxLoc(r)
    return int(loc[1] - max_shift - margin), int(loc[0] - max_shift - margin), float(peak)


def _warp(a_hp, ang, sc):
    H, W = a_hp.shape
    M = cv2.getRotationMatrix2D(((W - 1) / 2.0, (H - 1) / 2.0), ang, sc)
    return cv2.warpAffine(a_hp, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0), M


def bandpass(img):
    """5 x 5 median (the sensor's single-pixel pattern out), then a cell-scale band-pass 3 / 15 px, z-scored"""
    from scipy.ndimage import gaussian_filter, median_filter
    x = median_filter(np.asarray(img, np.float64), 5)
    x = gaussian_filter(x, 3) - gaussian_filter(x, 15)
    return ((x - x.mean()) / (x.std() + 1e-9)).astype(np.float32)


def register_similarity(mean1, mean2, rot_max=6.0, sc_max=0.05):
    """Day 1 -> Day 2: returns dict(angle, scale, M (2x3, Day-1 x/y -> warped x/y), dy, dx, r, r_translation, grid)"""
    a = bandpass(mean1)
    b = bandpass(mean2)
    best = None
    grid = []
    for ang in np.arange(-rot_max, rot_max + 1e-9, 0.5):
        for sc in np.arange(1 - sc_max, 1 + sc_max + 1e-9, 0.025):
            w, M = _warp(a, ang, sc)
            dy, dx, r = _ncc(w, b)
            grid.append((float(ang), float(sc), r))
            if best is None or r > best[2]:
                best = (float(ang), float(sc), r, dy, dx, M)
    a0, s0 = best[0], best[1]
    for ang in np.arange(a0 - 0.5, a0 + 0.5 + 1e-9, 0.1):
        for sc in np.arange(s0 - 0.02, s0 + 0.02 + 1e-9, 0.005):
            w, M = _warp(a, ang, sc)
            dy, dx, r = _ncc(w, b)
            if r > best[2]:
                best = (float(ang), float(sc), r, dy, dx, M)
    dy0, dx0, r0 = _ncc(a, b)
    out = dict(angle=best[0], scale=best[1], r=best[2], dy=best[3], dx=best[4], M=best[5],
               r_translation=r0, dy_translation=dy0, dx_translation=dx0, grid=grid)
    if best[2] < r0 + 0.01:                      # rotation / scale does not help clearly: translation only
        out.update(angle=0.0, scale=1.0, r=r0, dy=dy0, dx=dx0,
                   M=np.array([[1.0, 0, 0], [0, 1.0, 0]]), rot_used=False)
    else:
        out["rot_used"] = True
    return out


def act_bandpass(img):
    """activity projection (std dF/F): 3 x 3 median, band-pass 2 / 12 px"""
    from scipy.ndimage import gaussian_filter, median_filter
    x = median_filter(np.asarray(img, np.float64), 3)
    x = gaussian_filter(x, 2) - gaussian_filter(x, 12)
    return ((x - x.mean()) / (x.std() + 1e-9)).astype(np.float32)


def register_best(mean1, mean2, act1=None, act2=None, tol_px=6.0):
    """the mean-image registration, and - when both days' activity projections exist (std dF/F of the detection run) -
    the activity registration too. The activity one wins when they disagree by more than tol_px (median over a grid of
    the field of view): the mean image is dominated by vessels / out-of-focus tissue, which can move differently from
    the active cells (5939 Day 2: mean -2.0 deg (+29, +6) vs Cn / max / std 0 deg (+40, +21))."""
    rm = register_similarity(mean1, mean2)
    rm["source"] = "mean image"
    if act1 is None or act2 is None or np.shape(act1) != np.shape(mean1) or np.shape(act2) != np.shape(mean2):
        return rm
    a, b = act_bandpass(act1), act_bandpass(act2)
    best = None
    for ang in np.arange(-6, 6.01, 0.5):
        for sc in np.arange(0.95, 1.0501, 0.025):
            w, M = _warp(a, ang, sc)
            dy, dx, r = _ncc(w, b)
            if best is None or r > best[2]:
                best = (float(ang), float(sc), r, dy, dx, M)
    ra = dict(angle=best[0], scale=best[1], r=best[2], dy=best[3], dx=best[4], M=best[5], r_translation=np.nan,
              dy_translation=np.nan, dx_translation=np.nan, grid=[], rot_used=True, source="activity (std dF/F)")
    H, W = np.shape(mean1)
    gr, gc = np.meshgrid(np.linspace(0.2 * H, 0.8 * H, 5), np.linspace(0.2 * W, 0.8 * W, 5))
    q1 = np.c_[map_points(rm, gr.ravel(), gc.ravel())]
    q2 = np.c_[map_points(ra, gr.ravel(), gc.ravel())]
    d = float(np.median(np.hypot(q1[0] - q2[0], q1[1] - q2[1])))
    out = ra if d > tol_px else rm
    out = dict(out)
    out.update(disagreement_px=round(d, 1), mean_solution=dict(angle=rm["angle"], scale=rm["scale"], dy=rm["dy"], dx=rm["dx"], r=rm["r"]),
               activity_solution=dict(angle=ra["angle"], scale=ra["scale"], dy=ra["dy"], dx=ra["dx"], r=ra["r"]))
    return out


def map_points(reg, rows1, cols1):
    """1-based Day-1 (row, col) -> 1-based Day-2 (row, col)"""
    P = np.c_[np.asarray(cols1, float) - 1, np.asarray(rows1, float) - 1, np.ones(len(rows1))]
    Q = P @ reg["M"].T                                   # warped x, y (0-based)
    return Q[:, 1] - reg["dy"] + 1, Q[:, 0] - reg["dx"] + 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mouse")
    ap.add_argument("--phase", default="pre")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    m = a.mouse
    d1 = os.path.join(BASE, "mini1p_out", "20261019", f"{m}_d1")
    d2 = os.path.join(BASE, "mini1p_out", "20261021", f"{m}_d2")
    out = os.path.join(BASE, "analysis", "day2_seeds")
    os.makedirs(out, exist_ok=True)
    M1 = sio.loadmat(os.path.join(d1, "3_extract", f"{m}_d1_extract.mat"), variable_names=["spatial_weights", "mean_image"])
    S1 = np.asarray(M1["spatial_weights"], np.float32)
    mean1 = np.asarray(M1["mean_image"], np.float32)
    with h5py.File(os.path.join(d2, "1_movie", f"{m}_d2_{a.phase}_mc.h5"), "r") as h:
        mean2 = np.asarray(h["mean_image"], np.float32).T                # h5py reads MATLAB [H W] as [W H]
    R = pd.read_csv(os.path.join(d1, "4_cell_review", f"{m}_d1_cell_review.csv"))
    K = R[R.keep == 1]
    a1 = a2 = None
    pj1 = os.path.join(d1, "3_extract", f"{m}_d1_projections.mat")
    for pj2 in (os.path.join(d2, "3_extract", f"{m}_d2_projections.mat"),
                os.path.join(d2, "3_extract_v1_sensorreg", f"{m}_d2_projections.mat"),
                os.path.join(d2, "3_extract_v1_meanreg", f"{m}_d2_projections.mat")):
        if os.path.isfile(pj1) and os.path.isfile(pj2):
            a1 = np.asarray(sio.loadmat(pj1, variable_names=["std_dff"])["std_dff"], np.float32)
            a2 = np.asarray(sio.loadmat(pj2, variable_names=["std_dff"])["std_dff"], np.float32)
            break
    reg = register_best(mean1, mean2, a1, a2)
    print(f"{m}: registration from the {reg['source']}" + (f" (mean vs activity disagree by {reg['disagreement_px']} px)"
                                                         if "disagreement_px" in reg else ""))
    rows, cols = map_points(reg, K.row.values, K.col.values)
    ok = (rows >= 1) & (rows <= mean2.shape[0]) & (cols >= 1) & (cols <= mean2.shape[1])
    T = pd.DataFrame(dict(mark=np.arange(1, ok.sum() + 1), fov_row=np.round(rows[ok], 1), fov_col=np.round(cols[ok], 1),
                          day1_id=K.id.values[ok]))
    T.to_csv(os.path.join(out, f"marks_day2_{m}_d2.csv"), index=False)
    js = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in reg.items() if k != "grid"}
    js.update(mouse=m, n_day1_kept=int(len(K)), n_seeds=int(ok.sum()))
    json.dump(js, open(os.path.join(out, f"day2_registration_{m}.json"), "w"), indent=1)
    # QC: Day-1 outlines mapped onto the Day-2 mean image + correlation by rotation
    H2, W2 = mean2.shape
    outl = np.zeros(mean1.shape, np.float32)
    for j in np.flatnonzero(R.keep.values == 1):
        f = S1[:, :, j]
        if f.max() > 0:
            outl = np.maximum(outl, (f >= 0.3 * f.max()).astype(np.float32))
    Mfull = reg["M"].copy(); Mfull[0, 2] -= reg["dx"]; Mfull[1, 2] -= reg["dy"]
    outl2 = cv2.warpAffine(outl, Mfull, (W2, H2), flags=cv2.INTER_LINEAR)
    fig = plt.figure(figsize=(19, 7.5))
    g = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.45])
    for k, (img, msk, ttl) in enumerate(((mean1, outl, f"Day 1 mean image ({m}), {len(K)} confirmed neurons"),
                                         (mean2, outl2, "Day 2 mean image, the Day-1 outlines mapped"))):
        ax = fig.add_subplot(g[k])
        lo, hi = np.percentile(img, [1, 99.7])
        ax.imshow(np.clip((img - lo) / (hi - lo), 0, 1), cmap="gray"); ax.set_axis_off(); ax.set_title(ttl, fontsize=12)
        ax.contour(msk, levels=[0.5], colors=["#2ecc40"], linewidths=0.6)
    ax = fig.add_subplot(g[2])
    G = pd.DataFrame(reg["grid"], columns=["angle", "scale", "r"])
    best_by_angle = G.groupby("angle").r.max()
    ax.plot(best_by_angle.index, best_by_angle.values, "o-", color="#1f4e79")
    ax.axvline(reg["angle"], color="#c0392b", ls="--", lw=1)
    ax.set_xlabel("rotation Day 1 -> Day 2 (deg)"); ax.set_ylabel("image correlation r (best scale)")
    ax.set_title("which rotation fits", fontsize=12)
    fig.suptitle(f"{m}: Day 1 -> Day 2 registration: rotation {reg['angle']:.1f} deg, scale {reg['scale']:.3f}, "
                 f"shift ({-reg['dy']}, {-reg['dx']}) px, r = {reg['r']:.3f} (translation only r = {reg['r_translation']:.3f}); "
                 f"{ok.sum()} of {len(K)} Day-1 neurons inside the Day-2 frame -> forced seeds", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(out, f"day2_registration_{m}.png"), dpi=110); plt.close(fig)
    print(f"{m}: rotation {reg['angle']:.2f} deg, scale {reg['scale']:.3f}, dy {reg['dy']}, dx {reg['dx']}, r {reg['r']:.3f} "
          f"(translation only {reg['r_translation']:.3f}, rotation used {reg['rot_used']}); {ok.sum()} of {len(K)} "
          f"Day-1 neurons -> {out}")


def selftest():
    """turn / scale / shift the 5941 Day-1 mean image by known amounts (+ noise) and recover the seed positions"""
    rng = np.random.default_rng(1)
    d1 = os.path.join(BASE, "mini1p_out", "20261019", "5941_d1")
    mean1 = np.asarray(sio.loadmat(os.path.join(d1, "3_extract", "5941_d1_extract.mat"),
                                   variable_names=["mean_image"])["mean_image"], np.float32)
    R = pd.read_csv(os.path.join(d1, "4_cell_review", "5941_d1_cell_review.csv"))
    K = R[R.keep == 1]
    H, W = mean1.shape
    errs = []
    for t in range(6):
        ang, sc = rng.uniform(-4, 4), rng.uniform(0.97, 1.03)
        ty, tx = rng.uniform(-25, 25), rng.uniform(-25, 25)
        Mt = cv2.getRotationMatrix2D(((W - 1) / 2.0, (H - 1) / 2.0), ang, sc); Mt[0, 2] += tx; Mt[1, 2] += ty
        day2 = cv2.warpAffine(mean1, Mt, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        day2 = day2 + rng.normal(0, 0.05 * day2.std(), day2.shape).astype(np.float32)
        reg = register_similarity(mean1, day2)
        r2, c2 = map_points(reg, K.row.values, K.col.values)
        P = np.c_[K.col.values - 1, K.row.values - 1, np.ones(len(K))] @ Mt.T
        e = np.hypot(r2 - 1 - P[:, 1], c2 - 1 - P[:, 0])
        errs.append(np.median(e))
        print(f"  truth rot {ang:+.2f} scale {sc:.3f} shift ({ty:+.1f}, {tx:+.1f}) -> found rot {reg['angle']:+.2f} "
              f"scale {reg['scale']:.3f}; seed error median {np.median(e):.2f} px, max {e.max():.2f} px")
    ok = max(errs) < 2.0
    print(f"selftest {'ok' if ok else 'FAILED'}: worst median seed error {max(errs):.2f} px (cell radius 10 px)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
