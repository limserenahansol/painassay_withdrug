"""ca_crossday_match.py - follow the same neurons from Day 1 to Day 2 (Soma-print-style constellation matching).

The Day-2 field of view can be shifted in x / y, slightly rotated or scaled, and moved a little in z (a neuron's
footprint then changes shape or it is fainter). Footprint overlap alone (ca_transfer_decisions: cosine >= 0.5 after
a translation) misses such neurons. Here, as in Soma-print (Schnitzer lab; our mini2p TRU-FACT work), a neuron is
also identified by its NEIGHBOURHOOD: the vectors from it to its nearest neighbours form a fingerprint that survives
a shift of the whole field.

1. global transform Day 1 -> Day 2: translation from the normalised cross-correlation of the two mean images, then
   a similarity (rotation, scale, shift) fitted by RANSAC on mutual-nearest centroid pairs (inlier <= 6 px), 3 rounds
2. per Day-1 neuron i and Day-2 candidate j within R px of T(i):
     d      distance between T(centroid i) and centroid j
     fp     fingerprint agreement: fraction of i's k = 8 neighbour vectors (rotated / scaled by T) that have a
            neighbour vector of j within 5 px (each used once)
     cos    footprint overlap after the shift (0 if a footprint is missing)
   cost = 0.4 d / R + 0.4 (1 - fp) + 0.2 (1 - cos); one-to-one assignment (Hungarian)
3. a pair is accepted when fp is above the null (95th percentile of fp for i against Day-2 cells 20-60 px away,
   i.e. not the same neuron) AND d <= R, or when d <= 6 px and cos >= 0.3 (same place, same shape)
Day-1 neurons with no accepted partner = not found on Day 2 (shown, not forced); Day-2 neurons with none = new.

  python ca_crossday_match.py --mouse 5941 [--day2-all]   (--day2-all: every Day-2 candidate, for the decision step)
Writes analysis\\crossday\\<mouse>_pairs.csv, <mouse>_unmatched.csv, <mouse>_match.json, <mouse>_crossday_map.png
  python ca_crossday_match.py --selftest   (Day-1 neurons, randomly shifted / rotated / scaled, 20 % dropped, 20 % added)
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
BASE = r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553"
R_MAX, K_NB, TOL_FP, TOL_RANSAC = 15.0, 8, 5.0, 6.0


# ------------------------------------------------------------------------------------------ geometry
def umeyama(A, B):
    """similarity B ~ s R A + t (2-D, no reflection)"""
    ma, mb = A.mean(0), B.mean(0)
    a, b = A - ma, B - mb
    U, S, Vt = np.linalg.svd(b.T @ a / len(A))
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1, d])
    Rm = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / max((a ** 2).sum() / len(A), 1e-12)
    t = mb - s * Rm @ ma
    return s, Rm, t


def apply(T, P):
    s, Rm, t = T
    return (s * (Rm @ P.T)).T + t


def fit_transform(P1, P2, t0, rng, n_iter=3000):
    """RANSAC first over every candidate pair within 60 px of the translated position (robust to a few degrees of
    rotation about any point), scored by how many Day-1 neurons then land within TOL_RANSAC of a Day-2 neuron;
    then 3 rounds of mutual-nearest refinement"""
    tr2 = cKDTree(P2)
    T = (1.0, np.eye(2), np.asarray(t0, float))
    Q0 = apply(T, P1)
    cand = [(i, j) for i in range(len(P1)) for j in tr2.query_ball_point(Q0[i], 60)]
    if len(cand) >= 2:
        cand = np.array(cand)
        best, best_n = T, int((tr2.query(Q0)[0] <= TOL_RANSAC).sum())
        for _ in range(n_iter):
            a, b = cand[rng.choice(len(cand), 2, replace=False)]
            if a[0] == b[0] or a[1] == b[1] or np.linalg.norm(P1[a[0]] - P1[b[0]]) < 40:
                continue
            Tc = umeyama(P1[[a[0], b[0]]], P2[[a[1], b[1]]])
            if not (0.9 < Tc[0] < 1.1) or abs(np.degrees(np.arctan2(Tc[1][1, 0], Tc[1][0, 0]))) > 10:
                continue
            n_in = int((tr2.query(apply(Tc, P1))[0] <= TOL_RANSAC).sum())
            if n_in > best_n:
                best, best_n = Tc, n_in
        T = best
    for _ in range(3):
        Q = apply(T, P1)
        tr2, tr1 = cKDTree(P2), cKDTree(Q)
        d12, j = tr2.query(Q)
        _, i_back = tr1.query(P2[j])
        mutual = (i_back == np.arange(len(Q))) & (d12 <= 25)
        A, Bm = P1[mutual], P2[j[mutual]]
        if len(A) < 3:
            break
        best, best_in = T, -1
        for _ in range(500):
            k = rng.choice(len(A), 2, replace=False)
            Tc = umeyama(A[k], Bm[k])
            if not (0.9 < Tc[0] < 1.1) or abs(np.degrees(np.arctan2(Tc[1][1, 0], Tc[1][0, 0]))) > 10:
                continue
            n_in = int((np.linalg.norm(apply(Tc, A) - Bm, axis=1) <= TOL_RANSAC).sum())
            if n_in > best_in:
                best, best_in = Tc, n_in
        inl = np.linalg.norm(apply(best, A) - Bm, axis=1) <= TOL_RANSAC
        T = umeyama(A[inl], Bm[inl]) if inl.sum() >= 3 else best
    return T


def neighbour_vectors(P, k=K_NB):
    tr = cKDTree(P)
    kk = min(k + 1, len(P))
    _, idx = tr.query(P, kk)
    return [P[idx[i, 1:]] - P[i] for i in range(len(P))]


def fp_score(v1, v2, tol=TOL_FP):
    """fraction of v1's neighbour vectors matched (one-to-one) by a v2 vector within tol"""
    if len(v1) == 0 or len(v2) == 0:
        return 0.0
    D = np.linalg.norm(v1[:, None, :] - v2[None, :, :], axis=2)
    ua, ub, n = set(), set(), 0
    for flat in np.argsort(D, axis=None):
        a, b = divmod(int(flat), D.shape[1])
        if D[a, b] > tol:
            break
        if a in ua or b in ub:
            continue
        ua.add(a); ub.add(b); n += 1
    return n / len(v1)


# ------------------------------------------------------------------------------------------ matching
def match(P1, P2, t0, cos_fn=None, seed=0):
    rng = np.random.default_rng(seed)
    T = fit_transform(P1, P2, t0, rng)
    Q = apply(T, P1)
    s, Rm, _ = T
    V1 = [(s * (Rm @ v.T)).T for v in neighbour_vectors(P1)]
    V2 = neighbour_vectors(P2)
    tr2 = cKDTree(P2)
    n1, n2 = len(P1), len(P2)
    cost = np.full((n1, n2), 9.0)
    FP = np.zeros((n1, n2)); DD = np.full((n1, n2), np.inf); CS = np.zeros((n1, n2))
    null = []
    for i in range(n1):
        for j in tr2.query_ball_point(Q[i], R_MAX):
            d = float(np.linalg.norm(Q[i] - P2[j]))
            f = fp_score(V1[i], V2[j])
            c = float(cos_fn(i, j)) if cos_fn else 0.0
            FP[i, j], DD[i, j], CS[i, j] = f, d, c
            cost[i, j] = 0.4 * d / R_MAX + 0.4 * (1 - f) + 0.2 * (1 - max(c, 0))
        far = [j for j in tr2.query_ball_point(Q[i], 60) if np.linalg.norm(Q[i] - P2[j]) >= 20]
        for j in rng.choice(far, min(3, len(far)), replace=False) if far else []:
            null.append(fp_score(V1[i], V2[j]))
    thr = float(np.percentile(null, 95)) if null else 0.5
    ra, rb = linear_sum_assignment(cost)
    rows = []
    for i, j in zip(ra, rb):
        if cost[i, j] >= 9:
            continue
        acc_fp = FP[i, j] > thr
        acc_shape = DD[i, j] <= 6 and CS[i, j] >= 0.3
        if acc_fp or acc_shape:
            rows.append(dict(i=int(i), j=int(j), dist_px=round(DD[i, j], 2), fingerprint=round(FP[i, j], 3),
                             cos=round(CS[i, j], 3), accepted_by="fingerprint" if acc_fp else "position + shape"))
    info = dict(scale=float(s), rotation_deg=float(np.degrees(np.arctan2(Rm[1, 0], Rm[0, 0]))), shift=[float(x) for x in T[2]],
                fp_null95=thr, n1=n1, n2=n2, n_pairs=len(rows))
    return pd.DataFrame(rows), info


# ------------------------------------------------------------------------------------------ data
def day_cells(root, tag, kept_only):
    R = pd.read_csv(os.path.join(root, tag, "4_cell_review", f"{tag}_cell_review.csv"))
    if kept_only:
        R = R[R.keep == 1]
    return R.reset_index(drop=True)


def run_mouse(m, day2_all, out):
    import scipy.io as sio
    from ca_transfer_decisions import register, footprint_info, cosine_shifted
    r1, r2 = os.path.join(BASE, "mini1p_out", "20261019"), os.path.join(BASE, "mini1p_out", "20261021")
    t1, t2 = f"{m}_d1", f"{m}_d2"
    C1, C2 = day_cells(r1, t1, True), day_cells(r2, t2, not day2_all)
    ld = lambda r, t: sio.loadmat(os.path.join(r, t, "3_extract", f"{t}_extract.mat"), variable_names=["spatial_weights", "mean_image"])
    M1, M2 = ld(r1, t1), ld(r2, t2)
    S1, S2 = np.asarray(M1["spatial_weights"], np.float32), np.asarray(M2["spatial_weights"], np.float32)
    dy, dx, peak = register(np.asarray(M1["mean_image"], np.float32), np.asarray(M2["mean_image"], np.float32))
    I1, I2 = footprint_info(S1), footprint_info(S2)
    P1 = C1[["row", "col"]].to_numpy(float); P2 = C2[["row", "col"]].to_numpy(float)

    def cos_fn(i, j):
        a, b = int(C1.id[i]) - 1, int(C2.id[j]) - 1
        if I1[a] is None or I2[b] is None:
            return 0.0
        return cosine_shifted(S1[:, :, a], S2[:, :, b], dy, dx, I1[a], I2[b])
    Pm, info = match(P1, P2, (-dy, -dx), cos_fn)
    info.update(mouse=m, ncc_translation=[int(-dy), int(-dx)], ncc_peak=peak, day2_set="all candidates" if day2_all else "kept")
    Pm["day1_id"] = C1.id.values[Pm.i]; Pm["day2_id"] = C2.id.values[Pm.j]
    Pm = Pm.drop(columns=["i", "j"])
    os.makedirs(out, exist_ok=True)
    sfx = "_all" if day2_all else ""
    Pm.to_csv(os.path.join(out, f"{m}_pairs{sfx}.csv"), index=False)
    un1 = sorted(set(C1.id) - set(Pm.day1_id)); un2 = sorted(set(C2.id) - set(Pm.day2_id))
    pd.DataFrame(dict(day=["day1"] * len(un1) + ["day2"] * len(un2), id=un1 + un2)).to_csv(
        os.path.join(out, f"{m}_unmatched{sfx}.csv"), index=False)
    json.dump(info, open(os.path.join(out, f"{m}_match{sfx}.json"), "w"), indent=1)
    plot_map(m, M1, M2, C1, C2, Pm, info, os.path.join(out, f"{m}_crossday_map{sfx}.png"))
    print(f"{m}: {len(Pm)} of {len(C1)} Day-1 neurons found on Day 2 ({(Pm.accepted_by == 'fingerprint').sum()} by fingerprint); "
          f"Day-1 only {len(un1)}, Day-2 only {len(un2)}; transform scale {info['scale']:.3f}, rotation {info['rotation_deg']:.2f} deg, "
          f"shift {info['shift'][0]:.1f}, {info['shift'][1]:.1f} px")
    return Pm, info


def plot_map(m, M1, M2, C1, C2, Pm, info, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(18, 8))
    for k, (M, C, ids_m, ttl) in enumerate(((M1, C1, set(Pm.day1_id), "Day 1"), (M2, C2, set(Pm.day2_id), "Day 2"))):
        img = np.asarray(M["mean_image"], float)
        lo, hi = np.percentile(img, [1, 99.7])
        ax[k].imshow(np.clip((img - lo) / (hi - lo), 0, 1), cmap="gray"); ax[k].set_axis_off()
        for r in C.itertuples():
            mt = r.id in ids_m
            ax[k].plot(r.col, r.row, "o", ms=6, mfc="none", mec="#2ecc40" if mt else "#ff4136", mew=1.3)
        ax[k].set_title(f"{ttl}: green = found on both days ({len(ids_m)}), red = this day only ({len(C) - len(ids_m)})")
    # matched pairs share a number (the Day-1 id) on both images
    for r in Pm.itertuples():
        a = C1[C1.id == r.day1_id].iloc[0]; b = C2[C2.id == r.day2_id].iloc[0]
        ax[0].text(a.col + 5, a.row - 5, str(r.day1_id), color="#2ecc40", fontsize=6)
        ax[1].text(b.col + 5, b.row - 5, str(r.day1_id), color="#2ecc40", fontsize=6)
    fig.suptitle(f"{m}: Day 1 -> Day 2 neuron tracking (constellation fingerprint + position + shape); transform scale "
                 f"{info['scale']:.3f}, rotation {info['rotation_deg']:.2f} deg, shift ({info['shift'][0]:.0f}, {info['shift'][1]:.0f}) px; "
                 f"numbers = Day-1 neuron id")
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def selftest():
    """Day-1 5941 neurons as truth: random similarity, jitter, 20 % dropped, 20 % new; precision / recall"""
    C = day_cells(os.path.join(BASE, "mini1p_out", "20261019"), "5941_d1", True)
    P1 = C[["row", "col"]].to_numpy(float)
    rng = np.random.default_rng(1)
    res = []
    for trial in range(10):
        ang = np.radians(rng.uniform(-4, 4)); sc = rng.uniform(0.97, 1.03); sh = rng.uniform(-40, 40, 2)
        Rm = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        keep = rng.random(len(P1)) > 0.2
        Q = (sc * (Rm @ P1[keep].T)).T + sh + rng.normal(0, 1.5, (keep.sum(), 2))
        new = rng.uniform(P1.min(0), P1.max(0), (int(0.2 * len(P1)), 2))
        P2 = np.vstack([Q, new])
        truth = {i: k for k, i in enumerate(np.flatnonzero(keep))}
        t0 = sh + rng.normal(0, 3, 2)                       # the NCC translation is only approximate
        Pm, info = match(P1, P2, t0, None, seed=trial)
        ok = sum(1 for r in Pm.itertuples() if truth.get(r.i) == r.j)
        res.append(dict(trial=trial, n_true=len(truth), n_pairs=len(Pm), correct=ok, precision=ok / max(len(Pm), 1),
                        recall=ok / len(truth), rot_true=np.degrees(ang), rot_fit=info["rotation_deg"], scale_true=sc,
                        scale_fit=info["scale"]))
    R = pd.DataFrame(res)
    print(R.round(3).to_string(index=False))
    print(f"mean precision {R.precision.mean():.3f}, recall {R.recall.mean():.3f}")
    return R


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mouse", default="")
    ap.add_argument("--day2-all", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        selftest(); return
    run_mouse(a.mouse, a.day2_all, os.path.join(BASE, "analysis", "crossday"))


if __name__ == "__main__":
    main()
