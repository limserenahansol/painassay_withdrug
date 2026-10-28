"""ca_day2_recover.py - Day-1 confirmed neurons that were NOT FOUND on Day 2: put their Day-1 footprint on Day 2 (2026-10-22).

You (10-22): "the magenta x - I can see them on Day 2". On Day 2 a component was almost always 3-16 px away: the
Day-1 seed was removed or merged by EXTRACT's iterations (5939 179 -> 154 components), or it landed next to the cell
where the registration is locally off. So, for every Day-1 neuron not found on Day 2:
  1. place = the global Day 1 -> Day 2 transform (ca_day2_decisions) + the smooth local field (tiles + A pairs, used
     only when it predicts left-out tiles better than none), then +-3 px where the Day-1 local pattern (31 x 31 px Cn
     and std dF/F) matches the Day-2 one best (mean NCC); total correction <= 10 px. (A free +-10 px search picked
     noise maxima at its edge, 2026-10-22.)
  2. evidence for that place = the best NCC against the null of the SAME search started 20-40 px away (95th pct) -
     the search itself picks a maximum, so the plain patch null would be too lenient
  3. the footprint = the Day-1 footprint moved there (rotation / scale of the transform + the local offset), max 1
  -> analysis\\crossday\\<m>_recover_footprints.mat (add H x W x K, day1_id) for rescue_footprints.m (joint least squares
     of every Day-2 trace with these footprints fixed - EXTRACT cannot remove them) and <m>_recover.csv
Non-rigid check (seed independent): the Day-1 activity image (std dF/F, Cn; band-pass) moved by the global transform vs
the Day-2 one in 96 x 96 px tiles: the local shift of every tile with enough structure; plus the residual of the
class-A pairs (Day-2 centroid - transformed Day-1 centroid). -> <m>_local_registration.png / .csv
Your yellow marks (ca_yellow_marks.py, <m>_yellow_marks.csv) are matched to the recovered neurons (<= 20 px).

  python ca_day2_recover.py --mouse 5939
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.ndimage import gaussian_filter, median_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ca_day2_decisions import BASE, HALF_PATCH, fwd, inv, load_day, ncc, sample, window  # noqa: E402

SEARCH, LOCAL, HALF_FP, TILE, SIGMA = 10, 3, 30, 96, 120.0


def bp(img):
    x = median_filter(np.asarray(img, float), 3)
    return gaussian_filter(x, 2) - gaussian_filter(x, 12)


def score_at(W1, D2, rc):
    return float(np.mean([ncc(w1, window(D2[k], rc, HALF_PATCH)) for w1, k in zip(W1, ("Cn", "std"))]))


def search(W1, D2, rc0, r=LOCAL):
    best, bo = -9.0, (0, 0)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            s = score_at(W1, D2, rc0 + np.array([dy, dx]))
            if s > best:
                best, bo = s, (dy, dx)
    return best, bo


def tile_shifts(D1, D2, A):
    """seed-independent local registration: Day-1 band-passed std + Cn moved by A vs Day 2, per tile"""
    Ai = inv(A)
    H2, W2 = D2["std"].shape
    B2 = {k: bp(D2[k]) for k in ("std", "Cn")}
    B1 = {k: bp(D1[k]) for k in ("std", "Cn")}
    rows = []
    h = TILE // 2
    for r0 in range(h + 20, H2 - h - 20, 64):
        for c0 in range(h + 20, W2 - h - 20, 64):
            rc = np.array([r0, c0], float)
            W1 = [sample(B1[k], Ai, rc, h) for k in ("std", "Cn")]
            if min(np.std(w) for w in W1) <= 0:
                continue
            best, bo, sc0 = -9.0, (0, 0), None
            for dy in range(-8, 9):
                for dx in range(-8, 9):
                    s = np.mean([ncc(w1, window(B2[k], rc + np.array([dy, dx]), h)) for w1, k in zip(W1, ("std", "Cn"))])
                    if dy == 0 and dx == 0:
                        sc0 = s
                    if s > best:
                        best, bo = s, (dy, dx)
            rows.append(dict(row=r0, col=c0, dy=bo[0], dx=bo[1], ncc_best=round(best, 3), ncc_at_0=round(sc0, 3),
                             structure=round(float(np.std(W1[0])), 4)))
    T = pd.DataFrame(rows)
    if len(T):
        T["reliable"] = (T.ncc_best >= 0.3) & (T.ncc_best - T.ncc_at_0 >= 0.0)
    return T


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mouse", required=True)
    a = ap.parse_args(argv)
    m = a.mouse
    r1, r2 = os.path.join(BASE, "mini1p_out", "20261019"), os.path.join(BASE, "mini1p_out", "20261021")
    D1, D2 = load_day(r1, f"{m}_d1"), load_day(r2, f"{m}_d2")
    cd = os.path.join(BASE, "analysis", "crossday")
    J = json.load(open(os.path.join(cd, f"{m}_tracking.json")))
    A = np.array(J["transform"]["matrix_rowcol"]); Ai = inv(A)
    H2, W2 = D2["S"].shape[:2]
    # ---- non-rigid check
    TS = tile_shifts(D1, D2, A)
    TS.to_csv(os.path.join(cd, f"{m}_local_registration.csv"), index=False)
    PA = pd.read_csv(os.path.join(cd, f"{m}_pairs_all.csv"))
    PA = PA[PA.track_class == "A"]
    R1i = D1["R"].set_index("id"); R2i = D2["R"].set_index("id")
    P1 = R1i.loc[PA.day1_id.astype(int), ["row", "col"]].to_numpy(float)
    P2 = R2i.loc[PA.day2_id.astype(int), ["row", "col"]].to_numpy(float)
    res = P2 - fwd(A, P1) if len(PA) else np.zeros((0, 2))
    # smooth local field from the reliable tiles + the A pairs (Gaussian weights, sigma 120 px, shrunk to 0 where there is
    # little support); leave-one-out on the tiles tells whether the field predicts better than no local correction
    okt = TS[TS.reliable] if len(TS) else TS
    FP = np.r_[okt[["row", "col"]].to_numpy(float), P2] if len(okt) else P2
    FV = np.r_[okt[["dy", "dx"]].to_numpy(float), res] if len(okt) else res

    def field(q, skip=None):
        if not len(FP):
            return np.zeros(2)
        w = np.exp(-((FP - q) ** 2).sum(1) / (2 * SIGMA ** 2))
        if skip is not None:
            w[skip] = 0
        return (w[:, None] * FV).sum(0) / (w.sum() + 1.0)
    loo = [np.hypot(*(FV[k] - field(FP[k], k))) for k in range(len(FP))]
    zero = [np.hypot(*FV[k]) for k in range(len(FP))]
    use_field = bool(len(FP) >= 8 and np.median(loo) < 0.9 * np.median(zero))
    print(f"{m}: local field from {len(okt)} tiles + {len(res)} A pairs: leave-one-out error {np.median(loo) if loo else np.nan:.1f} px "
          f"vs {np.median(zero) if zero else np.nan:.1f} px without -> {'used' if use_field else 'NOT used (no coherent local shift)'}")
    # ---- recover the not-found neurons
    NF = pd.read_csv(os.path.join(cd, f"{m}_day1_not_found.csv"))
    rng = np.random.default_rng(1)
    S1, R1 = D1["S"], D1["R"]
    P2all = D2["R"][["row", "col"]].to_numpy(float)
    n2 = np.linalg.norm(D2["S"].reshape(-1, D2["S"].shape[2]), axis=0)
    add, rows, null = [], [], []
    for q in NF.itertuples():
        i = int(np.flatnonzero(R1.id.values == q.day1_id)[0])
        rc0 = np.array([q.day2_row, q.day2_col], float)
        if not (HALF_FP < rc0[0] < H2 - HALF_FP and HALF_FP < rc0[1] < W2 - HALF_FP):
            rows.append(dict(day1_id=int(q.day1_id), status="outside the Day-2 field of view")); continue
        W1 = [sample(D1[k], Ai, rc0, HALF_PATCH) for k in ("Cn", "std")]
        s0 = score_at(W1, D2, rc0)
        fld = np.round(field(rc0)).astype(int) if use_field else np.zeros(2, int)
        best, (dy, dx) = search(W1, D2, rc0 + fld)
        dy, dx = int(np.clip(dy + fld[0], -SEARCH, SEARCH)), int(np.clip(dx + fld[1], -SEARCH, SEARCH))
        for _ in range(3):                                  # the same search started 20-40 px away
            ang, rr = rng.uniform(0, 2 * np.pi), rng.uniform(20, 40)
            off = rc0 + rr * np.array([np.sin(ang), np.cos(ang)])
            if HALF_FP < off[0] < H2 - HALF_FP and HALF_FP < off[1] < W2 - HALF_FP:
                null.append(search(W1, D2, off)[0])
        c = np.round(rc0 + [dy, dx]).astype(int)           # 1-based Day-2 centre
        f = sample(S1[:, :, i], Ai, c - np.array([dy, dx], float), HALF_FP)   # Day-1 footprint seen at the new centre
        f[f < 0] = 0
        if f.max() <= 0:
            rows.append(dict(day1_id=int(q.day1_id), status="empty footprint")); continue
        F = np.zeros((H2, W2), np.float32)
        r_lo, c_lo = c[0] - 1 - HALF_FP, c[1] - 1 - HALF_FP
        F[r_lo:r_lo + f.shape[0], c_lo:c_lo + f.shape[1]] = f / f.max()
        # overlap with the existing Day-2 components (the one that may have taken its place)
        d = np.hypot(*(P2all - c).T)
        jn = int(np.argmin(d))
        cosn = float((F.ravel() @ D2["S"][:, :, jn].ravel()) / max(np.linalg.norm(F) * n2[jn], 1e-9))
        Rn = D2["R"].iloc[jn]
        add.append(F)
        rows.append(dict(day1_id=int(q.day1_id), status="recovered", day2_row_global=round(rc0[0], 1),
                         day2_col_global=round(rc0[1], 1), shift_dy=int(dy), shift_dx=int(dx), day2_row=int(c[0]),
                         day2_col=int(c[1]), field_dy=int(fld[0]), field_dx=int(fld[1]), pattern_at_global=round(s0, 3),
                         pattern_best=round(best, 3),
                         nearest_day2_id=int(Rn.id), nearest_px=round(float(d[jn]), 1), nearest_cos=round(cosn, 3),
                         nearest_keep=int(Rn.keep), nearest_day1_id=(int(Rn.day1_id) if pd.notna(Rn.get("day1_id")) else 0)))
    T = pd.DataFrame(rows)
    thr = float(np.percentile(null, 95)) if null else np.nan
    if len(T) and "pattern_best" in T:
        T["search_null95"] = round(thr, 3)
        T["pattern_ok"] = T.pattern_best >= thr
    # your yellow marks
    yp = os.path.join(cd, f"{m}_yellow_marks.csv")
    T["your_mark"] = 0
    if os.path.isfile(yp) and len(T):
        Y = pd.read_csv(yp)
        Y = Y[(Y.day2_row > 0) & (Y.day2_col > 0)]
        for y in Y.itertuples():
            ok = T.status == "recovered"
            if not ok.any():
                break
            dd = np.hypot(T.loc[ok, "day2_row_global"] - y.day2_row, T.loc[ok, "day2_col_global"] - y.day2_col)
            k = dd.idxmin()
            if dd[k] <= 20:
                T.at[k, "your_mark"] = int(y.mark)
    T.to_csv(os.path.join(cd, f"{m}_recover.csv"), index=False)
    rec = T[T.status == "recovered"] if len(T) else T
    if len(add):
        sio.savemat(os.path.join(cd, f"{m}_recover_footprints.mat"),
                    dict(add=np.stack(add, 2).astype(np.float32), day1_id=rec.day1_id.to_numpy(float),
                         shift=rec[["shift_dy", "shift_dx"]].to_numpy(float), pattern=rec.pattern_best.to_numpy(float)),
                    do_compression=True)
    # ---- figure: the non-rigid check
    fig, ax = plt.subplots(1, 3, figsize=(19, 6.4))
    lo, hi = np.percentile(D2["maxdff"], [1, 99.8])
    for a_ in ax[:2]:
        a_.imshow(np.clip((D2["maxdff"] - lo) / (hi - lo), 0, 1) ** 0.7, cmap="gray"); a_.set_axis_off()
    ok = TS[TS.reliable] if len(TS) else TS
    if len(ok):
        ax[0].quiver(ok.col, ok.row, ok.dx, -ok.dy, color="#ffdc00", angles="xy", scale_units="xy", scale=0.15, width=0.004)
        for r in ok.itertuples():
            ax[0].text(r.col + 4, r.row - 4, f"{np.hypot(r.dy, r.dx):.0f}", color="#ffdc00", fontsize=7)
    ax[0].set_title(f"A  local shift Day 1 -> Day 2 after the global transform, 96-px tiles\n(activity images, seed independent; "
                    f"arrows x 6.7; numbers = px); {len(ok)} reliable tiles", fontsize=10)
    if len(res):
        ax[1].quiver(P2[:, 1], P2[:, 0], res[:, 1], -res[:, 0], color="#2ecc40", angles="xy", scale_units="xy", scale=0.15, width=0.004)
    if len(rec):
        ax[1].quiver(rec.day2_col_global, rec.day2_row_global, rec.shift_dx, -rec.shift_dy, color="#f012be", angles="xy",
                     scale_units="xy", scale=0.15, width=0.004)
        for r in rec.itertuples():
            ax[1].text(r.day2_col + 5, r.day2_row + 9, str(r.day1_id), color="#f012be", fontsize=6.5)
    ax[1].set_title("B  green = class-A pairs (Day-2 centroid - transformed Day-1 centroid);\nmagenta = local correction "
                    "chosen for each recovered neuron (arrows x 6.7)", fontsize=10)
    mag_t = np.hypot(ok.dy, ok.dx) if len(ok) else np.array([])
    mag_a = np.hypot(res[:, 0], res[:, 1]) if len(res) else np.array([])
    mag_r = np.hypot(rec.shift_dy, rec.shift_dx) if len(rec) else np.array([])
    bins = np.arange(0, 12, 1)
    ax[2].hist([mag_t, mag_a, mag_r], bins=bins, label=[f"tiles (n={len(mag_t)})", f"A pairs (n={len(mag_a)})",
               f"recovered (n={len(mag_r)})"], color=["#d4ac0d", "#2ecc40", "#f012be"])
    ax[2].set_xlabel("local shift (px)"); ax[2].set_ylabel("count"); ax[2].legend(frameon=False, fontsize=9)
    ax[2].set_title("C  size of the local shifts", fontsize=10)
    for sp in ("top", "right"):
        ax[2].spines[sp].set_visible(False)
    fig.suptitle(f"{m}: local (non-rigid) registration check and the recovered Day-1 neurons (PROVISIONAL)   median tile shift "
                 f"{np.median(mag_t) if len(mag_t) else np.nan:.1f} px, A pairs {np.median(mag_a) if len(mag_a) else np.nan:.1f} px, "
                 f"recovered {np.median(mag_r) if len(mag_r) else np.nan:.1f} px", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(os.path.join(cd, f"{m}_local_registration.png"), dpi=100); plt.close(fig)
    n_ok = int(rec.pattern_ok.sum()) if len(rec) else 0
    print(f"{m}: {len(NF)} not found -> {len(rec)} recovered footprints ({n_ok} with the Day-1 pattern above the search null "
          f"{thr:.2f}), {int((T.your_mark > 0).sum())} at your yellow marks; local shift median: tiles "
          f"{np.median(mag_t) if len(mag_t) else np.nan:.1f} px, A pairs {np.median(mag_a) if len(mag_a) else np.nan:.1f} px, "
          f"recovered {np.median(mag_r) if len(mag_r) else np.nan:.1f} px")


if __name__ == "__main__":
    main()
