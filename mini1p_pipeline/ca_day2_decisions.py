"""ca_day2_decisions.py - Day-2 keep decisions and the Day 1 <-> Day 2 neuron tracking, decided on DAY-2 evidence.

Why a new step (2026-10-21 night): the Day-2 detection was seeded with every Day-1 confirmed neuron (forced seeds,
ca_day2_seeds.py), so a Day-2 candidate sits at every Day-1 position by construction. Position and the pattern of
neighbours (the Soma-print-style fingerprint) then only prove that the seed stayed where it was put - not that the
neuron is there on Day 2. The footprint shape does not prove it either: a round Gaussian seed left in place has
cosine 0.84-0.94 with the real Day-1 footprints (median, all 4 mice). Measured on Day 1 (pre vs post projections):
the local pattern of a 31 x 31 px window (Cn and std dF/F) tells the right place from the wrong neuron (AUC 0.94-0.98)
but an absent soma only weakly (0.56-0.79); soma contrast on the projections 0.66-0.88. So the evidence is
  patch   the local Day-2 pattern around the candidate = the Day-1 one (mean NCC of the Cn and std dF/F windows,
          31 x 31 px; null = the Day-2 window 20-40 px away, 95th percentile) -> right place, registration correct
  ev      the Day-2 movie proof: event-triggered image of the candidate's own Day-2 trace matches its footprint
          (event_image_corr) at least as well as the weakest 10 % of the Day-1 confirmed neurons
  verdict the Day-2 review rules (Ca-like transients, shape) keep it
  cos     reported (shape after the transform), only used to reject a clearly different shape (< 0.3)
and every pair stays PROVISIONAL until you look at the Day-1 | Day-2 windows (tracking_pairs pages).
Transform Day 1 -> Day 2: rotation + scale + shift from the two mean images (ca_day2_seeds.register_similarity),
refined twice on the confident pairs (similarity fit, Umeyama).

Classes (one-to-one assignment, cost = 1 - cos + 0.5 d / 10 px, d <= 10 px):
  A  same neuron            d <= 6 px, patch above null, ev >= floor, cos >= 0.3, Day-2 rules keep it
       Day-1 green  -> green (your Day-1 decision carried, decision_source user)
       Day-1 orange -> orange (kept; attention "Day-1 orange: re-check on Day 2", your rule of 10-21)
  B  probably the same      d <= 6 px (or <= 10 px with patch above null and cos >= 0.5), Day-2 rules keep it,
                            but the Day-2 movie proof / the local pattern / the shape is weak
       -> orange (kept; attention "Day-1 #k? <what is weak> - check")
  C  same place, Day 2 says not a cell   (d <= 6, the Day-2 rules drop it)
       -> NOT kept, shown (magenta) so you can add it back
  -  Day-1 neuron not found on Day 2 (none of the above) -> listed and marked, never forced (your rule)
  Day-1 deletions (yours) matched with cos >= 0.5 -> deleted again (your decision carried)
  Day-1 candidates left out by MY Day-1 rule (decision_source 'rule', not by you) that pass the Day-2 cell rules with
       the Day-2 movie proof and the local pattern -> NOT deleted: orange, class D1off "your call" (you, 2026-10-22:
       1191's brightest neuron was lost on both days this way)
  Forced pairs (2026-10-22): the Day-1 footprints of the not-found neurons put on Day 2 (ca_day2_recover.py +
       rescue_footprints.m; review column recovered_day1_id) are paired with their Day-1 neuron and judged on the
       Day-2 evidence like every pair (the local pattern against the null of the same search); your own pairs
       (analysis/crossday/<m>_pair_user.csv: day1_id, day2_id, note) are class A, your decision
  New Day-2 candidates (no Day-1 neuron there) -> kept only with a clear neuron shape (the strict bar of
  ca_transfer_decisions --strict-new, from the class-A cells) -> orange "new on Day 2"
Writes (backup of the review table first):
  <day2>/4_cell_review/<tag>_cell_review.csv  keep, decision_source, note, attention, day1_id, track_class, ...
  <day2>/5_analysis/decisions/<tag>_day2_decisions.csv / .json
  analysis/crossday/<m>_pairs.csv (kept pairs A + B, for the longitudinal analysis), <m>_pairs_all.csv (A, B, C),
  <m>_day1_not_found.csv, <m>_tracking_map.png, <m>_tracking_pairs_p<k>.png (Day-1 | Day-2 windows per neuron)

  python ca_day2_decisions.py --mouse 5941
  python ca_day2_decisions.py --selftest     (Day-1 5941 footprints moved by a known transform, 20 % removed)
"""
import argparse
import datetime
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.ndimage import map_coordinates
from scipy.optimize import linear_sum_assignment

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import safe_io  # noqa: E402,F401  (a figure save retries on a transient Windows write error)
from ca_day2_seeds import register_best, register_similarity  # noqa: E402
from ca_transfer_decisions import rule_keep  # noqa: E402

BASE = r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553"
R_MATCH, D_CLOSE, HALF_FP, HALF_PATCH = 10.0, 6.0, 24, 15
COS_A, COS_B = 0.5, 0.3


# ------------------------------------------------------------------------------------------ geometry
def affine_from_reg(reg):
    """2 x 3 matrix acting on 1-based (row, col, 1) Day 1 -> Day 2, from register_similarity"""
    M = reg["M"]                                         # on 0-based (x, y)
    A = np.zeros((2, 3))
    A[0, 0], A[0, 1] = M[1, 1], M[1, 0]
    A[1, 0], A[1, 1] = M[0, 1], M[0, 0]
    A[0, 2] = M[1, 2] - reg["dy"] + 1 - (A[0, 0] + A[0, 1])
    A[1, 2] = M[0, 2] - reg["dx"] + 1 - (A[1, 0] + A[1, 1])
    return A


def umeyama_affine(P, Q):
    """similarity Q ~ s R P + t as a 2 x 3 matrix"""
    mp, mq = P.mean(0), Q.mean(0)
    a, b = P - mp, Q - mq
    U, S, Vt = np.linalg.svd(b.T @ a / len(P))
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1, d])
    Rm = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / max((a ** 2).sum() / len(P), 1e-12)
    t = mq - s * Rm @ mp
    return np.c_[s * Rm, t]


def fwd(A, P):
    return np.c_[P, np.ones(len(P))] @ A.T


def inv(A):
    B = np.linalg.inv(np.r_[A, [[0, 0, 1]]])
    return B[:2]


def describe(A):
    s = np.sqrt(abs(np.linalg.det(A[:, :2])))
    ang = np.degrees(np.arctan2(A[1, 0], A[0, 0]))       # (row, col) frame
    return float(s), float(ang)


def sample(img, Ainv, rc, half):
    """window of img (Day-1 frame) seen in Day-2 coordinates: Day-2 window centred on rc (1-based) mapped back"""
    rr, cc = np.mgrid[-half:half + 1, -half:half + 1]
    P = np.c_[rr.ravel() + rc[0], cc.ravel() + rc[1]]
    Q = fwd(Ainv, P) - 1                                  # 0-based Day-1 pixel
    return map_coordinates(img, [Q[:, 0], Q[:, 1]], order=1, mode="constant", cval=0.0).reshape(rr.shape)


def window(img, rc, half):
    rr, cc = np.mgrid[-half:half + 1, -half:half + 1]
    return map_coordinates(img, [rr.ravel() + rc[0] - 1, cc.ravel() + rc[1] - 1], order=1, mode="constant",
                           cval=0.0).reshape(rr.shape)


def ncc(a, b):
    a = a - a.mean(); b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float((a * b).sum() / (na * nb)) if na > 0 and nb > 0 else 0.0


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float((a * b).sum() / (na * nb)) if na > 0 and nb > 0 else 0.0


# ------------------------------------------------------------------------------------------ data
def load_day(root, tag):
    xd = os.path.join(root, tag, "3_extract")
    M = sio.loadmat(os.path.join(xd, f"{tag}_extract.mat"), variable_names=["spatial_weights", "mean_image"])
    S = np.asarray(M["spatial_weights"], np.float32)
    if S.ndim == 2:
        S = S[:, :, None]
    P = sio.loadmat(os.path.join(xd, f"{tag}_projections.mat"), variable_names=["max_dff", "Cn", "std_dff"])
    R = pd.read_csv(os.path.join(root, tag, "4_cell_review", f"{tag}_cell_review.csv"))
    if len(R) != S.shape[2]:
        raise SystemExit(f"{tag}: review table {len(R)} rows, EXTRACT {S.shape[2]} candidates")
    return dict(S=S, mean=np.asarray(M["mean_image"], np.float32), maxdff=np.asarray(P["max_dff"], np.float32),
                Cn=np.asarray(P["Cn"], np.float32), std=np.asarray(P["std_dff"], np.float32), R=R)


def shape_numbers(S):
    from skimage.measure import label as sklabel, regionprops
    K = S.shape[2]
    mf, so = np.zeros(K), np.zeros(K)
    for j in range(K):
        f = S[:, :, j]
        if f.max() <= 0:
            continue
        L = sklabel(f >= 0.3 * f.max(), connectivity=2)
        pr = regionprops(L, intensity_image=f)
        mass = np.array([q.intensity_mean * q.area for q in pr])
        k0 = int(np.argmax(mass))
        mf[j], so[j] = mass[k0] / mass.sum(), pr[k0].solidity
    return mf, so


# ------------------------------------------------------------------------------------------ core
def track(D1, D2, A0, rng, forced=()):
    """returns the pair table (every Day-1 candidate vs its assigned Day-2 candidate), the refined transform, the
    patch null. forced = [(i, j), ...] row indices paired whatever the cost (recovered footprints, your pairs); they
    are left out of the transform refinement"""
    S1, S2, R1, R2 = D1["S"], D2["S"], D1["R"], D2["R"]
    forced = list(forced)
    fi, fj = {i for i, _ in forced}, {j for _, j in forced}
    P1 = R1[["row", "col"]].to_numpy(float)
    P2 = R2[["row", "col"]].to_numpy(float)
    n1 = np.linalg.norm(S1.reshape(-1, S1.shape[2]), axis=0)
    A = A0.copy()
    for it in range(3):
        Ai = inv(A)
        Q = fwd(A, P1)
        cost = np.full((len(P1), len(P2)), 9.0)
        CS = np.zeros_like(cost); DD = np.full_like(cost, np.inf)
        for i in range(len(P1)):
            if n1[i] <= 0:
                continue
            d = np.hypot(*(P2 - Q[i]).T)
            near = np.flatnonzero(d <= R_MATCH)
            if not len(near):
                continue
            f1 = sample(S1[:, :, i], Ai, Q[i], HALF_FP)
            for j in near:
                f2 = window(S2[:, :, j], Q[i], HALF_FP)
                c = cosine(f1, f2)
                CS[i, j], DD[i, j] = c, d[j]
                cost[i, j] = (1 - c) + 0.5 * d[j] / R_MATCH
        for i, j in forced:
            cost[i, :] = 9.0; cost[:, j] = 9.0
            CS[i, j] = cosine(sample(S1[:, :, i], Ai, Q[i], HALF_FP), window(S2[:, :, j], Q[i], HALF_FP))
            DD[i, j] = float(np.hypot(*(P2[j] - Q[i])))
            cost[i, j] = 0.0
        ra, rb = linear_sum_assignment(cost)
        good = [(i, j) for i, j in zip(ra, rb) if cost[i, j] < 9 and CS[i, j] >= 0.6 and DD[i, j] <= 4
                and i not in fi and j not in fj]
        if it < 2 and len(good) >= 6:
            I, J = np.array(good).T
            A = umeyama_affine(P1[I], P2[J])
    # patch evidence (max dF/F) for the assigned pairs and its null
    Ai = inv(A)
    rows, null = [], []
    for i, j in zip(ra, rb):
        if cost[i, j] >= 9:
            continue
        W1 = [sample(D1[k], Ai, P2[j], HALF_PATCH) for k in ("Cn", "std")]
        pt = np.mean([ncc(w1, window(D2[k], P2[j], HALF_PATCH)) for w1, k in zip(W1, ("Cn", "std"))])
        rows.append(dict(i=int(i), j=int(j), cos=round(CS[i, j], 3), dist_px=round(DD[i, j], 2), patch=round(pt, 3),
                         forced=bool((int(i), int(j)) in set(forced))))
        for _ in range(2):
            ang = rng.uniform(0, 2 * np.pi); rr = rng.uniform(20, 40)
            off = P2[j] + rr * np.array([np.sin(ang), np.cos(ang)])
            null.append(np.mean([ncc(w1, window(D2[k], off, HALF_PATCH)) for w1, k in zip(W1, ("Cn", "std"))]))
    return pd.DataFrame(rows), A, (float(np.percentile(null, 95)) if null else 0.5)


def decide(D1, D2, Pm, patch_thr, strict_q=10.0, ev_floor=None, fkind=None):
    """fkind: Day-2 row index -> dict(kind='recovered' | 'user', pattern_ok, note) for the forced pairs"""
    fkind = fkind or {}
    R1, R2 = D1["R"], D2["R"].copy()
    K2 = len(R2)
    num = lambda c: pd.to_numeric(R2[c], errors="coerce").values if c in R2 else np.full(K2, np.nan)
    ver = R2.verdict.astype(str).values
    rk = np.array([rule_keep(ver[k], num("area_ratio")[k], num("n_pieces")[k], num("axis_ratio")[k]) for k in range(K2)])
    src1 = R1.get("decision_source", pd.Series([""] * len(R1))).astype(str).values
    keep1 = R1.keep.values == 1
    rev1 = R1.get("reviewed", pd.Series([0] * len(R1))).fillna(0).astype(int).values
    ev2 = num("event_image_corr")
    if ev_floor is None:
        ev_floor = float(np.nanpercentile(pd.to_numeric(R1.event_image_corr, errors="coerce")[keep1], 10))
    out = dict(keep=np.zeros(K2, int), source=np.array(["rule"] * K2, object), note=np.array([""] * K2, object),
               attention=np.array([""] * K2, object), day1_id=np.full(K2, np.nan), cls=np.array([""] * K2, object),
               tcos=np.full(K2, np.nan), tpatch=np.full(K2, np.nan), tdist=np.full(K2, np.nan))
    assigned = set()
    cls_rows = []
    for p in Pm.itertuples():
        i, j = p.i, p.j
        did = int(R1.id.iloc[i])
        fk = fkind.get(j, {})
        pe = p.patch >= patch_thr if fk.get("kind") != "recovered" else bool(fk.get("pattern_ok"))
        close = p.dist_px <= D_CLOSE or fk.get("kind") == "recovered"      # placed there on purpose
        ok2 = ver[j] != "not_cell" and rk[j] == 1
        evok = bool(np.isfinite(ev2[j]) and ev2[j] >= ev_floor)
        c = ""
        if fk.get("kind") == "user":
            c = "A"
        elif keep1[i]:
            if close and ok2 and pe and (p.cos >= COS_B or fk) and evok:
                c = "A"
            elif ok2 and (close or (pe and p.cos >= COS_A)):
                c = "B"
            elif close:
                c = "C"
        elif (rev1[i] or src1[i] == "user") and p.cos >= COS_A:
            c = "D1off" if (src1[i] == "rule" and ok2 and evok and pe) else "deleted"
        if not c:
            continue
        assigned.add(j)
        out["day1_id"][j] = did; out["cls"][j] = c
        out["tcos"][j], out["tpatch"][j], out["tdist"][j] = p.cos, p.patch, p.dist_px
        ev = f"patch {p.patch:.2f}, event image {ev2[j]:.2f}, cos {p.cos:.2f}, {p.dist_px:.1f} px"
        if fk.get("kind") == "user":
            out["keep"][j] = 1; out["source"][j] = "user"
            out["note"][j] = f"{fk.get('note', 'your pair')} - Day-1 neuron #{did} ({ev})"
        elif c == "A":
            out["keep"][j] = 1
            if src1[i] == "user":
                out["source"][j] = "user"
                out["note"][j] = f"Day-1 neuron #{did} (your Day-1 keep), same footprint ({ev})"
            else:
                out["source"][j] = "rule"
                out["note"][j] = f"Day-1 neuron #{did} (Day-1 orange, kept by rule), same footprint ({ev})"
                out["attention"][j] = f"Day-1 orange #{did}: re-check on Day 2"
        elif c == "B":
            out["keep"][j] = 1
            weak = [w for w, bad in (("local pattern weak", not pe),
                                     ("Day-2 movie proof weak", not (np.isfinite(ev2[j]) and ev2[j] >= ev_floor)),
                                     ("shape differs", p.cos < COS_B), (f"{p.dist_px:.0f} px away", not close)) if bad]
            why = "; ".join(weak) if weak else "weak evidence"
            out["note"][j] = f"Day-1 neuron #{did}? ({why}; {ev})"
            out["attention"][j] = f"Day-1 #{did}? {why} - check"
        elif c == "C":
            out["note"][j] = f"Day-1 neuron #{did} at this place, but Day 2 says not a cell ({ver[j]}; {ev})"
            out["attention"][j] = f"Day-1 #{did} here, Day-2 rule deleted it - your call"
        elif c == "D1off":
            out["keep"][j] = 1
            out["note"][j] = (f"Day-1 #{did} was left out by my Day-1 rule (not by you); on Day 2 it passes the cell rules "
                              f"with the movie proof ({ev})")
            out["attention"][j] = f"Day-1 #{did} left out by my Day-1 rule, Day 2 a cell - your call (keep? add on Day 1 too?)"
        else:
            out["source"][j] = "user"
            out["note"][j] = f"your Day-1 delete #{did} carried ({ev})"
        if fk.get("kind") == "recovered":
            out["note"][j] = f"recovered (Day-1 footprint placed on Day 2): " + out["note"][j]
            out["attention"][j] = (f"recovered Day-1 #{did} ({fk.get('note', '')}) -> "
                                   + {"A": "same", "B": "check", "C": "Day-2 rules drop it"}.get(c, c))
        cls_rows.append(dict(day1_id=did, day2_id=int(R2.id.iloc[j]), track_class=c, cos=p.cos, patch=p.patch,
                             event_image=round(float(ev2[j]), 3), dist_px=p.dist_px, day1_source=src1[i], day2_verdict=ver[j],
                             forced=fk.get("kind", "")))
    # new Day-2 candidates: the strict neuron-shape bar from the class-A cells
    mf, so = shape_numbers(D2["S"])
    ar, ax_, ev = num("area_ratio"), num("axis_ratio"), num("event_image_corr")
    cl = np.nan_to_num(num("ca_like"), nan=0.0)
    ref = out["cls"] == "A"
    q = strict_q
    if ref.sum() >= 5:
        bar = dict(main_min=np.nanpercentile(mf[ref], q), sol_min=np.nanpercentile(so[ref], q),
                   area_lo=np.nanpercentile(ar[ref], q / 2), area_hi=np.nanpercentile(ar[ref], 100 - q / 2),
                   axis_max=np.nanpercentile(ax_[ref], 100 - q), ev_min=np.nanpercentile(ev[ref], q))
    else:
        bar = dict(main_min=0.88, sol_min=0.6, area_lo=0.35, area_hi=2.3, axis_max=2.3, ev_min=0.4)
    shape_ok = ((mf >= bar["main_min"]) & (so >= bar["sol_min"]) & (ar >= bar["area_lo"]) & (ar <= bar["area_hi"])
                & (ax_ <= bar["axis_max"]) & (ev >= bar["ev_min"]) & (cl == 1))
    # 'neuron-like body' (your Day-1 criterion on the max projection: one compact, solid, not too big or long body) -
    # used only to choose which dropped new candidates are worth your look
    out["body_ok"] = (mf >= 0.85) & (so >= bar["sol_min"]) & (ar <= bar["area_hi"]) & (ax_ <= bar["axis_max"])
    for j in range(K2):
        if j in assigned:
            continue
        out["cls"][j] = "new"
        if rk[j] == 1 and shape_ok[j]:
            out["keep"][j] = 1
            out["note"][j] = "new on Day 2 (no Day-1 neuron here): clear neuron shape (strict rule)"
            out["attention"][j] = "new on Day 2 - check"
        else:
            out["note"][j] = f"new on Day 2: not a clear neuron shape (strict rule, {ver[j]})"
    bar["event_image_floor_day1_p10"] = ev_floor
    return out, pd.DataFrame(cls_rows), bar


def forced_pairs(m, D1, D2):
    """(i, j) row-index pairs that are paired whatever the cost: recovered footprints (review column recovered_day1_id,
    ca_day2_recover.py) and your pairs (analysis/crossday/<m>_pair_user.csv)"""
    cd = os.path.join(BASE, "analysis", "crossday")
    id1 = {int(v): k for k, v in enumerate(D1["R"].id.values)}
    id2 = {int(v): k for k, v in enumerate(D2["R"].id.values)}
    forced, fkind = [], {}
    R2 = D2["R"]
    if "recovered_day1_id" in R2:
        rec = pd.read_csv(os.path.join(cd, f"{m}_recover.csv")) if os.path.isfile(os.path.join(cd, f"{m}_recover.csv")) else None
        for j in np.flatnonzero(pd.to_numeric(R2.recovered_day1_id, errors="coerce").notna().values):
            d1 = int(R2.recovered_day1_id.iloc[j])
            if d1 not in id1:
                continue
            info = rec[rec.day1_id == d1].iloc[0] if rec is not None and (rec.day1_id == d1).any() else None
            note = "" if info is None else (f"shift {int(info.shift_dy):+d}/{int(info.shift_dx):+d} px, pattern "
                                            f"{info.pattern_best:.2f} vs search null {info.search_null95:.2f}"
                                            + (f", your yellow mark {int(info.your_mark)}" if int(info.your_mark) else ""))
            forced.append((id1[d1], j))
            fkind[j] = dict(kind="recovered", pattern_ok=bool(info.pattern_ok) if info is not None else False, note=note,
                            your_mark=int(info.your_mark) if info is not None else 0)
    up = os.path.join(cd, f"{m}_pair_user.csv")
    if os.path.isfile(up):
        for r in pd.read_csv(up).itertuples():
            if int(r.day1_id) in id1 and int(r.day2_id) in id2:
                i, j = id1[int(r.day1_id)], id2[int(r.day2_id)]
                forced = [(a_, b_) for a_, b_ in forced if a_ != i and b_ != j] + [(i, j)]
                fkind[j] = dict(kind="user", note=str(r.note))
    return forced, fkind


# ------------------------------------------------------------------------------------------ figures
def norm_img(img, pr=(1, 99.8), g=0.7):
    lo, hi = np.percentile(img, pr)
    return np.clip((img - lo) / max(hi - lo, 1e-9), 0, 1) ** g


COL = dict(A="#2ecc40", Aorange="#ff9f1a", B="#ff9f1a", C="#f012be", new="#00d0ff", missing="#f012be", D1off="#ffdc00")


def plot_map(m, D1, D2, A, out, T, missing, path):
    S1, S2, R1, R2 = D1["S"], D2["S"], D1["R"], D2["R"]
    fig, ax = plt.subplots(1, 2, figsize=(22, 11 * max(S1.shape[0] / S1.shape[1], S2.shape[0] / S2.shape[1]) + 1.6))
    ax[0].imshow(norm_img(D1["maxdff"]), cmap="gray"); ax[1].imshow(norm_img(D2["maxdff"]), cmap="gray")
    for a_ in ax:
        a_.set_axis_off()
    by1 = {r.day1_id: r for r in T.itertuples()}
    fset = {int(r.day1_id) for r in T.itertuples() if getattr(r, "forced", "") == "recovered"}
    for i in np.flatnonzero(R1.keep.values == 1):
        did = int(R1.id.iloc[i]); f = S1[:, :, i]
        r = by1.get(did)
        if r is None or r.track_class in ("deleted", "D1off"):
            col = COL["missing"]
        elif r.track_class == "A":
            col = COL["A"] if r.day1_source == "user" else COL["Aorange"]
        else:
            col = COL[r.track_class]
        ax[0].contour((f >= 0.3 * f.max()).astype(float), levels=[0.5], colors=[col], linewidths=1.0,
                      linestyles="--" if r is None else "-")
        ax[0].text(R1.col.iloc[i] + 6, R1.row.iloc[i] - 6, str(did), color=col, fontsize=7)
    for r in T.itertuples():
        if r.track_class == "deleted":
            continue
        j = int(r.day2_id) - 1; f = S2[:, :, j]
        col = (COL["A"] if r.day1_source == "user" else COL["Aorange"]) if r.track_class == "A" else COL[r.track_class]
        rec_ = getattr(r, "forced", "") == "recovered"
        ax[1].contour((f >= 0.3 * f.max()).astype(float), levels=[0.5], colors=[col], linewidths=1.2 if rec_ else 1.0,
                      linestyles=":" if r.track_class == "C" else ("--" if rec_ else "-"))
        ax[1].text(R2.col.iloc[j] + 6, R2.row.iloc[j] - 6, ("r" if rec_ else "") + ("d" if r.track_class == "D1off" else "")
                   + str(int(r.day1_id)), color=col, fontsize=7)
    newk = np.flatnonzero((out["cls"] == "new") & (out["keep"] == 1))
    for j in newk:
        f = S2[:, :, j]
        ax[1].contour((f >= 0.3 * f.max()).astype(float), levels=[0.5], colors=[COL["new"]], linewidths=1.0)
        ax[1].text(R2.col.iloc[j] + 6, R2.row.iloc[j] - 6, f"n{int(R2.id.iloc[j])}", color=COL["new"], fontsize=7)
    for q in missing:
        ax[1].plot(q["day2_col"], q["day2_row"], "x", color=COL["missing"], ms=7, mew=1.6)
        ax[1].text(q["day2_col"] + 5, q["day2_row"] + 9, str(q["day1_id"]), color=COL["missing"], fontsize=6)
    nA = int(((T.track_class == "A") & (T.day1_source == "user")).sum()); nAo = int(((T.track_class == "A") & (T.day1_source != "user")).sum())
    nB, nC = int((T.track_class == "B").sum()), int((T.track_class == "C").sum())
    s, ang = describe(A)
    ax[0].set_title(f"Day 1 ({m}_d1) max dF/F: {int((R1.keep == 1).sum())} confirmed neurons, number = Day-1 id", fontsize=13)
    ax[1].set_title(f"Day 2 ({m}_d2) max dF/F: the same numbers = the same neurons; n<id> = new on Day 2 ({len(newk)})", fontsize=13)
    fig.suptitle(f"{m}: Day 1 -> Day 2 tracking (PROVISIONAL, for your check).  Every outline on Day 1 = a neuron you confirmed on Day 1; "
                 f"its colour = what happened to it on Day 2.\ngreen = same neuron, your Day-1 keep ({nA});  orange = same neuron, Day-1 "
                 f"orange ({nAo}) or to check ({nB});  magenta dotted = the Day-2 rules deleted it ({nC});  magenta x = not found ({len(missing)})"
                 f"\ncyan n<id> = new on Day 2;  dashed + r<id> = recovered ({len(fset)}: its Day-1 footprint placed on Day 2);  yellow d<id> "
                 f"= left out by my Day-1 rule, a cell on Day 2 ({int((T.track_class == 'D1off').sum())}).  "
                 f"Transform Day 1 -> Day 2: rotation {ang:.2f} deg, scale {s:.3f}.\nEvidence = local Cn / std pattern + Day-2 movie "
                 f"proof + Day-2 cell rules (position / shape alone prove nothing: the Day-1 neurons were Day-2 seeds).", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.89)); fig.savefig(path, dpi=110); plt.close(fig)


C3 = dict(both="#2ecc40", d1="#f012be", d2="#00d0ff")


def plot_map3(m, D1, D2, path, P, keep1, keep2):
    """the tracking map in your three categories (2026-10-22): green = the neuron exists on both days (the same number on
    both sides = the Day-1 id), magenta = Day 1 only, cyan = Day 2 only (n<id> = its Day-2 id)"""
    S1, S2, R1, R2 = D1["S"], D2["S"], D1["R"], D2["R"]
    b1 = {int(a): int(b) for a, b in zip(P.day1_id, P.day2_id)}
    b2 = {v: k for k, v in b1.items()}
    fig, ax = plt.subplots(1, 2, figsize=(22, 11 * max(S1.shape[0] / S1.shape[1], S2.shape[0] / S2.shape[1]) + 1.2))
    ax[0].imshow(norm_img(D1["maxdff"]), cmap="gray"); ax[1].imshow(norm_img(D2["maxdff"]), cmap="gray")
    for a_ in ax:
        a_.set_axis_off()
    n = dict(both=0, d1=0, d2=0)
    for i in np.flatnonzero(keep1 == 1):
        did = int(R1.id.iloc[i]); f = S1[:, :, i]
        k = "both" if did in b1 else "d1"
        n[k] += k == "d1"
        ax[0].contour((f >= 0.3 * f.max()).astype(float), levels=[0.5], colors=[C3[k]], linewidths=1.1)
        ax[0].text(R1.col.iloc[i] + 6, R1.row.iloc[i] - 6, str(did), color=C3[k], fontsize=7)
    for j in np.flatnonzero(keep2 == 1):
        d2 = int(R2.id.iloc[j]); f = S2[:, :, j]
        if f.max() <= 0:
            continue
        k = "both" if d2 in b2 else "d2"
        n[k] += 1
        lab = str(b2[d2]) if k == "both" else f"n{d2}"
        ax[1].contour((f >= 0.3 * f.max()).astype(float), levels=[0.5], colors=[C3[k]], linewidths=1.1)
        ax[1].text(R2.col.iloc[j] + 6, R2.row.iloc[j] - 6, lab, color=C3[k], fontsize=7)
    ax[0].set_title(f"Day 1 ({m}_d1): {int((keep1 == 1).sum())} neurons - green = also on Day 2 ({n['both']}), "
                    f"magenta = Day 1 only ({n['d1']})", fontsize=13)
    ax[1].set_title(f"Day 2 ({m}_d2): {int((keep2 == 1).sum())} neurons - green = also on Day 1 ({n['both']}), "
                    f"cyan = Day 2 only ({n['d2']})", fontsize=13)
    fig.suptitle(f"{m}: the same neurons on Day 1 and Day 2.  green = on both days (the same number on both sides),  magenta = Day 1 "
                 f"only,  cyan = Day 2 only (n<id>).  Background = max dF/F of each day.", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(path, dpi=110); plt.close(fig)
    return n


def plot_pairs(m, D1, D2, A, T, missing, out_dir, per_page=40):
    """Day-1 window | Day-2 window per Day-1 neuron (max dF/F, 41 px), the doubtful ones first"""
    Ai = inv(A)
    S1, S2, R1 = D1["S"], D2["S"], D1["R"]
    order = {"B": 0, "C": 1, "missing": 2, "A": 3}
    items = [dict(kind=r.track_class, day1_id=int(r.day1_id), day2_id=int(r.day2_id), cos=r.cos, patch=r.patch,
                  rc2=D2["R"][["row", "col"]].iloc[int(r.day2_id) - 1].to_numpy(float), src=r.day1_source)
             for r in T.itertuples() if r.track_class in ("A", "B", "C")]
    items += [dict(kind="missing", day1_id=q["day1_id"], day2_id=0, cos=np.nan, patch=q.get("pattern_there", np.nan),
                   rc2=np.array([q["day2_row"], q["day2_col"]]), src="") for q in missing]
    items.sort(key=lambda d: (order[d["kind"]], d["cos"] if np.isfinite(d["cos"]) else 0))
    files = []
    h = 20
    lo1, hi1 = np.percentile(D1["maxdff"], [1, 99.8]); lo2, hi2 = np.percentile(D2["maxdff"], [1, 99.8])
    for pg in range(int(np.ceil(len(items) / per_page))):
        chunk = items[pg * per_page:(pg + 1) * per_page]
        ncol = 8
        nrow = int(np.ceil(len(chunk) / (ncol / 2)))
        fig, axs = plt.subplots(nrow, ncol, figsize=(ncol * 2.1, nrow * 2.35))
        axs = np.atleast_2d(axs)
        for a_ in axs.ravel():
            a_.set_axis_off()
        for k, it in enumerate(chunk):
            r_, c_ = divmod(k, ncol // 2)
            a1, a2 = axs[r_, 2 * c_], axs[r_, 2 * c_ + 1]
            w1 = sample(D1["maxdff"], Ai, it["rc2"], h); w2 = window(D2["maxdff"], it["rc2"], h)
            a1.imshow(np.clip((w1 - lo1) / (hi1 - lo1), 0, 1) ** 0.7, cmap="gray")
            a2.imshow(np.clip((w2 - lo2) / (hi2 - lo2), 0, 1) ** 0.7, cmap="gray")
            i = int(np.flatnonzero(R1.id.values == it["day1_id"])[0])
            f1 = sample(S1[:, :, i], Ai, it["rc2"], h)
            col = {"A": COL["A"] if it["src"] == "user" else COL["Aorange"], "B": COL["B"], "C": COL["C"],
                   "missing": COL["missing"]}[it["kind"]]
            if f1.max() > 0:
                a1.contour(f1 >= 0.3 * f1.max(), levels=[0.5], colors=[col], linewidths=1.0)
                a2.contour(f1 >= 0.3 * f1.max(), levels=[0.5], colors=["white"], linewidths=0.6, linestyles=":")
            if it["day2_id"]:
                f2 = window(S2[:, :, it["day2_id"] - 1], it["rc2"], h)
                if f2.max() > 0:
                    a2.contour(f2 >= 0.3 * f2.max(), levels=[0.5], colors=[col], linewidths=1.0)
            lab = {"A": "same", "B": "check", "C": "Day-2 deleted", "missing": "not found"}[it["kind"]]
            a1.set_title(f"D1 #{it['day1_id']}", fontsize=8, color=col)
            a2.set_title(f"D2 {('#' + str(it['day2_id'])) if it['day2_id'] else '-'} {lab}"
                         + (f"\ncos {it['cos']:.2f} patch {it['patch']:.2f}" if np.isfinite(it["cos"]) else
                            (f"\npattern there {it['patch']:.2f}" if np.isfinite(it["patch"]) else "")),
                         fontsize=7, color=col)
        fig.suptitle(f"{m}: each Day-1 neuron (left, its outline) and the same place on Day 2 (right: Day-2 outline in "
                     f"colour, the Day-1 outline dotted white); max dF/F, 41 x 41 px; doubtful first - page {pg + 1}", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        p = os.path.join(out_dir, f"{m}_tracking_pairs_p{pg + 1}.png")
        fig.savefig(p, dpi=100); plt.close(fig); files.append(p)
    return files


def plot_new(m, D2, out, out_dir, per_page=48):
    """new Day-2 candidates (no Day-1 neuron there): the kept ones and the dropped ones that look like your Day-1 keeps,
    41 x 41 px of the Day-2 max dF/F with the footprint, most like your keeps first"""
    R2, S2 = D2["R"], D2["S"]
    idx = [j for j in range(len(R2)) if out["cls"][j] == "new" and (out["keep"][j] == 1 or "your call" in str(out["attention"][j]))]
    idx.sort(key=lambda j: -np.nan_to_num(out["p_like_you"][j]))
    lo, hi = np.percentile(D2["maxdff"], [1, 99.8])
    for pg in range(int(np.ceil(len(idx) / per_page))):
        chunk = idx[pg * per_page:(pg + 1) * per_page]
        ncol = 8; nrow = int(np.ceil(len(chunk) / ncol))
        fig, axs = plt.subplots(nrow, ncol, figsize=(ncol * 1.9, nrow * 2.1))
        axs = np.atleast_2d(axs)
        for a_ in axs.ravel():
            a_.set_axis_off()
        for k, j in enumerate(chunk):
            a_ = axs[k // ncol, k % ncol]
            rc = R2[["row", "col"]].iloc[j].to_numpy(float)
            w = window(D2["maxdff"], rc, 20)
            a_.imshow(np.clip((w - lo) / (hi - lo), 0, 1) ** 0.7, cmap="gray")
            f = window(S2[:, :, j], rc, 20)
            col = COL["new"] if out["keep"][j] == 1 else "#ff4136"
            if f.max() > 0:
                a_.contour(f >= 0.3 * f.max(), levels=[0.5], colors=[col], linewidths=1.0)
            a_.set_title(f"#{int(R2.id.iloc[j])} {'kept' if out['keep'][j] == 1 else 'dropped'}\np {out['p_like_you'][j]:.2f}",
                         fontsize=7.5, color=col)
        fig.suptitle(f"{m} Day 2: NEW candidates (no Day-1 neuron there) - cyan = kept by the strict rule, red = dropped "
                     f"but looks like your Day-1 keeps; p = similarity to your Day-1 decisions; page {pg + 1}", fontsize=9.5)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(os.path.join(out_dir, f"{m}_new_candidates_p{pg + 1}.png"), dpi=100); plt.close(fig)


# ------------------------------------------------------------------------------------------ run
def run_mouse(m, write=True):
    r1, r2 = os.path.join(BASE, "mini1p_out", "20261019"), os.path.join(BASE, "mini1p_out", "20261021")
    t1, t2 = f"{m}_d1", f"{m}_d2"
    D1, D2 = load_day(r1, t1), load_day(r2, t2)
    # who decided each Day-1 candidate, frozen at your Day-1 confirmation (analysis/crossday/<m>_day1_provenance.csv):
    # a re-run of the Day-1 post relabels every reviewed row 'user' ("your decision on the whole day"), which would hide
    # the candidates only MY rule left out (2026-10-22, 1191)
    pv = os.path.join(BASE, "analysis", "crossday", f"{m}_day1_provenance.csv")
    if os.path.isfile(pv):
        PV = pd.read_csv(pv).set_index("id")
        D1["R"]["decision_source"] = [PV.decision_source.get(int(i), s_) for i, s_ in
                                      zip(D1["R"].id, D1["R"].get("decision_source", pd.Series([""] * len(D1["R"]))))]
    reg = register_best(D1["mean"], D2["mean"], D1["std"], D2["std"])          # activity wins when they disagree
    A0 = affine_from_reg(reg)
    rng = np.random.default_rng(0)
    forced, fkind = forced_pairs(m, D1, D2)
    Pm, A, patch_thr = track(D1, D2, A0, rng, forced)
    out, T, bar = decide(D1, D2, Pm, patch_thr, fkind=fkind)
    # how much each Day-2 candidate looks like the cells YOU kept on Day 1 (ca_cell_classifier_day1.py; leave-one-mouse-
    # out AUC 0.75-0.86 - a ranking for your check, it does not decide): new candidates the strict rule dropped but that
    # look like your keeps are flagged for you
    out["p_like_you"] = np.full(len(D2["R"]), np.nan)
    try:
        import joblib
        from ca_cell_classifier_day1 import features as clf_features
        J = joblib.load(os.path.join(BASE, "analysis", "ca_selfcheck", "cell_classifier_day1.joblib"))
        F2, _ = clf_features(r2, t2)
        out["p_like_you"] = J["model"].predict_proba(F2[J["features"]])[:, 1]
        for j in range(len(D2["R"])):
            if out["cls"][j] != "new":
                continue
            pj = out["p_like_you"][j]
            if out["keep"][j] == 1:
                out["attention"][j] = f"new on Day 2 - check (looks like your Day-1 keeps: p {pj:.2f})"
            elif pj >= 0.6 and str(D2["R"].verdict.iloc[j]) != "not_cell" and bool(out["body_ok"][j]):
                out["attention"][j] = f"new on Day 2, dropped by the strict rule but looks like your Day-1 keeps (p {pj:.2f}) - your call"
    except Exception as e:                                                # noqa: BLE001
        print(f"{m}: classifier ranking not available ({e})")
    # Day-1 confirmed neurons not found
    P1 = D1["R"][["row", "col"]].to_numpy(float)
    Q = fwd(A, P1)
    found = set(T[T.track_class.isin(["A", "B", "C"])].day1_id)
    H2, W2 = D2["S"].shape[:2]
    missing = [dict(day1_id=int(D1["R"].id.iloc[i]), day2_row=float(Q[i, 0]), day2_col=float(Q[i, 1]),
                    inside=bool(1 <= Q[i, 0] <= H2 and 1 <= Q[i, 1] <= W2))
               for i in np.flatnonzero(D1["R"].keep.values == 1) if int(D1["R"].id.iloc[i]) not in found]
    # is the Day-1 pattern still there on Day 2 (no Day-2 component took it)? local Cn / std pattern at the mapped place
    Ai = inv(A)
    for q in missing:
        rc = np.array([q["day2_row"], q["day2_col"]])
        W1 = [sample(D1[k], Ai, rc, HALF_PATCH) for k in ("Cn", "std")]
        q["pattern_there"] = round(float(np.mean([ncc(w1, window(D2[k], rc, HALF_PATCH)) for w1, k in zip(W1, ("Cn", "std"))])), 3)
        q["visible_on_day2"] = bool(q["pattern_there"] >= patch_thr)
        q["nearest_day2_px"] = round(float(np.min(np.hypot(*(D2["R"][["row", "col"]].to_numpy(float) - rc).T))), 1)
    cd = os.path.join(BASE, "analysis", "crossday"); os.makedirs(cd, exist_ok=True)
    s, ang = describe(A)
    summary = dict(mouse=m, created=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), provisional=True,
                   transform=dict(rotation_deg=ang, scale=s, matrix_rowcol=A.tolist(), image_reg=dict(
                       angle=reg["angle"], scale=reg["scale"], r=reg["r"], r_translation=reg["r_translation"],
                       source=reg.get("source", "mean image"), disagreement_px=reg.get("disagreement_px"),
                       mean_solution=reg.get("mean_solution"), activity_solution=reg.get("activity_solution"))),
                   patch_null95=patch_thr, strict_bar={k: float(v) for k, v in bar.items()},
                   day1_confirmed=int((D1["R"].keep == 1).sum()), day2_candidates=int(len(D2["R"])),
                   A_same_green=int(((T.track_class == "A") & (T.day1_source == "user")).sum()),
                   A_same_day1_orange=int(((T.track_class == "A") & (T.day1_source != "user")).sum()),
                   B_check=int((T.track_class == "B").sum()), C_day2_deleted=int((T.track_class == "C").sum()),
                   day1_not_found=len(missing), day1_deletes_carried=int((T.track_class == "deleted").sum()),
                   recovered=int((T.get("forced", pd.Series([], dtype=str)) == "recovered").sum()),
                   recovered_kept=int(((T.get("forced", pd.Series([], dtype=str)) == "recovered") & T.track_class.isin(["A", "B"])).sum()),
                   day1_rule_off_day2_cell=int((T.track_class == "D1off").sum()),
                   new_kept=int(((out["cls"] == "new") & (out["keep"] == 1)).sum()),
                   new_maybe=int(sum(1 for a_ in out["attention"] if "your call" in str(a_) and "new on Day 2" in str(a_))),
                   day2_kept=int(out["keep"].sum()))
    if not write:
        return summary, out, T, missing, D1, D2, A
    T.to_csv(os.path.join(cd, f"{m}_pairs_all.csv"), index=False)
    T[T.track_class.isin(["A", "B"])][["day1_id", "day2_id", "track_class", "cos", "patch", "dist_px"]].to_csv(
        os.path.join(cd, f"{m}_pairs.csv"), index=False)
    pd.DataFrame(missing).to_csv(os.path.join(cd, f"{m}_day1_not_found.csv"), index=False)
    json.dump(summary, open(os.path.join(cd, f"{m}_tracking.json"), "w"), indent=1)
    # the Day-2 review table
    csv = os.path.join(r2, t2, "4_cell_review", f"{t2}_cell_review.csv")
    shutil.copy(csv, csv.replace(".csv", f"_before_day2_decisions_{datetime.datetime.now():%Y%m%d_%H%M}.csv"))
    R = D2["R"].copy()
    # your own Day-2 decisions (reviewed = 1: ca_exclude_rois.py --ids / --keep-ids) are never overwritten
    yours = R.get("reviewed", pd.Series([0] * len(R))).fillna(0).astype(int).values == 1
    old_keep = R.keep.values.copy()
    old_note = R.get("note", pd.Series([""] * len(R))).fillna("").astype(str).values
    R["keep"] = np.where(yours, old_keep, out["keep"])
    R["decision_source"] = np.where(yours, "user", out["source"])
    R["note"] = np.where(yours, old_note, out["note"])
    R["attention"] = np.where(yours, "", out["attention"])
    R["day1_id"] = out["day1_id"]; R["track_class"] = out["cls"]
    R["track_cos"] = out["tcos"]; R["track_patch"] = out["tpatch"]; R["track_dist_px"] = out["tdist"]
    R["p_like_you"] = np.round(out["p_like_you"], 3)
    R["reviewed"] = yours.astype(int)
    R.to_csv(csv, index=False)
    dd = os.path.join(r2, t2, "5_analysis", "decisions"); os.makedirs(dd, exist_ok=True)
    R.to_csv(os.path.join(dd, f"{t2}_day2_decisions.csv"), index=False)
    json.dump(summary, open(os.path.join(dd, f"{t2}_day2_decisions.json"), "w"), indent=1)
    # the detailed decision map (every class) for the QC sheet; <m>_tracking_map.png = your 3 colours, drawn by --replot
    plot_map(m, D1, D2, A, out, T, missing, os.path.join(cd, f"{m}_tracking_map_detail.png"))
    shutil.copy(os.path.join(cd, f"{m}_tracking_map_detail.png"), os.path.join(dd, f"{t2}_decisions_map.png"))   # QC sheet panel
    plot_pairs(m, D1, D2, A, T, missing, cd)
    try:
        plot_new(m, D2, out, cd)
    except Exception as e:                                                # noqa: BLE001
        print(f"{m}: new-candidate gallery failed ({e})")
    print(f"{m}: Day 1 confirmed {summary['day1_confirmed']} -> same neuron {summary['A_same_green']} green + "
          f"{summary['A_same_day1_orange']} Day-1 orange, check {summary['B_check']}, Day-2 deleted "
          f"{summary['C_day2_deleted']}, not found {summary['day1_not_found']} (recovered {summary['recovered']}, of them "
          f"{summary['recovered_kept']} same / check); Day-1 rule-off but a Day-2 cell {summary['day1_rule_off_day2_cell']}; "
          f"new kept {summary['new_kept']}; "
          f"Day-2 kept {summary['day2_kept']} of {summary['day2_candidates']}; rotation {ang:.2f} deg, scale {s:.3f}, "
          f"patch null {patch_thr:.2f}")
    return summary


def replot(m):
    """the tracking map + pair pages again from the FINAL Day-2 review table (after the duplicate check, the review flags
    and your exclusions), so the map you confirm on shows the decisions as they are now"""
    r1, r2 = os.path.join(BASE, "mini1p_out", "20261019"), os.path.join(BASE, "mini1p_out", "20261021")
    D1, D2 = load_day(r1, f"{m}_d1"), load_day(r2, f"{m}_d2")
    cd = os.path.join(BASE, "analysis", "crossday")
    J = json.load(open(os.path.join(cd, f"{m}_tracking.json")))
    A = np.array(J["transform"]["matrix_rowcol"])
    pv = os.path.join(cd, f"{m}_day1_provenance.csv")
    src = (pd.read_csv(pv).set_index("id").decision_source if os.path.isfile(pv)
           else D1["R"].set_index("id").decision_source).astype(str)
    R = D2["R"]
    tc = R.track_class.fillna("").astype(str).values
    rec = pd.to_numeric(R.get("recovered_day1_id", pd.Series([np.nan] * len(R))), errors="coerce").values
    rows = []
    for j in range(len(R)):
        if tc[j] in ("A", "B", "C", "D1off", "deleted") and pd.notna(R.day1_id.iloc[j]):
            d1 = int(R.day1_id.iloc[j])
            c = tc[j]
            if c in ("A", "B") and int(R.keep.iloc[j]) == 0:
                c = "C"                                     # deleted after the decisions (duplicate / you)
            yours = int(pd.to_numeric(R.get("reviewed", pd.Series([0] * len(R))), errors="coerce").fillna(0).iloc[j]) == 1
            if c == "C" and int(R.keep.iloc[j]) == 1 and yours:
                c = "A"                                     # you kept it: the same neuron (your decision)
            rows.append(dict(day1_id=d1, day2_id=int(R.id.iloc[j]), track_class=c, cos=R.track_cos.iloc[j],
                             patch=R.track_patch.iloc[j], dist_px=R.track_dist_px.iloc[j],
                             day1_source="user" if (c == "A" and yours) else src.get(d1, ""),
                             forced="recovered" if np.isfinite(rec[j]) else ""))
    T = pd.DataFrame(rows)
    nf = os.path.join(cd, f"{m}_day1_not_found.csv")
    missing = pd.read_csv(nf).to_dict("records") if os.path.isfile(nf) and os.path.getsize(nf) > 5 else []
    out = dict(cls=np.where(tc == "", "new", tc).astype(object), keep=R.keep.values.astype(int))
    pf = os.path.join(cd, f"{m}_pairs.csv")
    Pf = pd.read_csv(pf) if os.path.isfile(pf) else pd.DataFrame(columns=["day1_id", "day2_id"])
    plot_map3(m, D1, D2, os.path.join(cd, f"{m}_tracking_map.png"), Pf, D1["R"].keep.values.astype(int), R.keep.values.astype(int))
    dd = os.path.join(r2, f"{m}_d2", "5_analysis", "decisions")
    shutil.copy(os.path.join(cd, f"{m}_tracking_map.png"), os.path.join(dd, f"{m}_d2_decisions_map.png"))
    import glob as _glob
    for old in _glob.glob(os.path.join(cd, f"{m}_tracking_pairs_p*.png")):
        os.remove(old)
    plot_pairs(m, D1, D2, A, T, missing, cd)
    print(f"{m}: map + pair pages redrawn from the final review table ({int((R.keep == 1).sum())} kept)")


def selftest():
    """Day 2 = the Day-1 5941 candidates moved by a known rotation / scale / shift, 20 % of the confirmed neurons
    removed; the remaining must be found (class A) at the right ids, the removed must be 'not found'"""
    import cv2
    rng = np.random.default_rng(3)
    r1 = os.path.join(BASE, "mini1p_out", "20261019")
    D1 = load_day(r1, "5941_d1")
    H, W, K = D1["S"].shape
    ang, sc, ty, tx = 1.3, 1.015, -9.0, 14.0
    Mt = cv2.getRotationMatrix2D(((W - 1) / 2.0, (H - 1) / 2.0), ang, sc); Mt[0, 2] += tx; Mt[1, 2] += ty
    keep1 = np.flatnonzero(D1["R"].keep.values == 1)
    drop = set(rng.choice(keep1, int(0.2 * len(keep1)), replace=False))
    stay = [k for k in range(K) if k not in drop]
    S2 = np.stack([cv2.warpAffine(D1["S"][:, :, k], Mt, (W, H), flags=cv2.INTER_LINEAR) for k in stay], 2)
    warp = lambda im: cv2.warpAffine(im, Mt, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    R2 = D1["R"].iloc[stay].reset_index(drop=True).copy()
    P = np.c_[R2.col.values - 1, R2.row.values - 1, np.ones(len(R2))] @ Mt.T
    R2["row"], R2["col"] = P[:, 1] + 1, P[:, 0] + 1
    truth = {int(D1["R"].id.iloc[k]): n + 1 for n, k in enumerate(stay)}
    R2["id"] = np.arange(1, len(R2) + 1)
    D2 = dict(S=S2, mean=warp(D1["mean"]), maxdff=warp(D1["maxdff"]), Cn=warp(D1["Cn"]), std=warp(D1["std"]), R=R2)
    reg = register_similarity(D1["mean"], D2["mean"])
    Pm, A, thr = track(D1, D2, affine_from_reg(reg), rng)
    out, T, _ = decide(D1, D2, Pm, thr)
    TA = T[T.track_class.isin(["A", "B"])]
    correct = sum(truth.get(int(r.day1_id)) == int(r.day2_id) for r in TA.itertuples())
    kept_ids = set(int(D1["R"].id.iloc[k]) for k in keep1 if k not in drop)
    found = set(TA.day1_id.astype(int))
    s, a = describe(A)
    print(f"truth rotation {ang} deg, scale {sc}; fitted rotation {a:.2f} deg (row/col frame sign), scale {s:.3f}")
    print(f"confirmed neurons present {len(kept_ids)}: found {len(found & kept_ids)}, correct ids {correct} of {len(TA)}, "
          f"removed ones reported found {len(found - kept_ids)}")
    ok = correct == len(TA) and len(found & kept_ids) >= 0.95 * len(kept_ids) and not (found - kept_ids)
    print("selftest", "ok" if ok else "FAILED")
    return 0 if ok else 1


def fix_dedup(m):
    """after ca_duplicate_check --resolve: two Day-2 components followed to DIFFERENT Day-1 neurons are both kept again (you
    kept both on Day 1 = two neurons; the round forced seeds of neighbours 8-10 px apart overlap at cos ~0.57 by
    construction, so the overlap rule would merge them); and a tracked component is never dropped for an untracked one
    (the roles are swapped). The decisions before the duplicate check are in 5_analysis/decisions/<tag>_day2_decisions.csv."""
    import re as _re
    t2 = f"{m}_d2"
    csv = os.path.join(BASE, "mini1p_out", "20261021", t2, "4_cell_review", f"{t2}_cell_review.csv")
    R = pd.read_csv(csv).set_index("id", drop=False)
    Dd = pd.read_csv(os.path.join(BASE, "mini1p_out", "20261021", t2, "5_analysis", "decisions", f"{t2}_day2_decisions.csv")).set_index("id")
    for c in ("note", "attention", "decision_source"):
        R[c] = R[c].fillna("").astype(str)
    n_both, n_swap = 0, 0
    for a in R.index[R.decision_source == "auto-duplicate"]:
        mt = _re.search(r"duplicate of #(\d+)", R.at[a, "note"])
        if not mt or int(Dd.at[a, "keep"]) != 1:
            continue
        b = int(mt.group(1))
        da, db = R.at[a, "day1_id"], R.at[b, "day1_id"] if b in R.index else np.nan
        if pd.notna(da) and pd.notna(db) and int(da) != int(db):
            for c in ("keep", "decision_source", "note", "attention"):
                R.at[a, c] = Dd.at[a, c] if c != "keep" else 1
            R.at[a, "note"] = str(R.at[a, "note"]) + f"; kept although it overlaps #{b} (Day-1 #{int(db)}): you kept Day-1 #{int(da)} and #{int(db)} as two neurons"
            n_both += 1
        elif pd.notna(da) and pd.isna(db) and b in R.index and int(R.at[b, "keep"]) == 1:
            for c in ("keep", "decision_source", "note", "attention"):
                R.at[a, c] = Dd.at[a, c] if c != "keep" else 1
            R.at[b, "keep"] = 0; R.at[b, "decision_source"] = "auto-duplicate"
            R.at[b, "note"] = f"duplicate of #{a} (the Day-1 neuron #{int(da)}) - the tracked one kept"
            R.at[b, "attention"] = ""
            n_swap += 1
    R.to_csv(csv, index=False)
    print(f"{m}: duplicate check corrected - {n_both} neighbours that you kept as two Day-1 neurons kept again, "
          f"{n_swap} tracked components preferred over an untracked duplicate")


def flag_review(m):
    """attention marks only (keep is never changed) for two kinds of deleted candidates found on the 5941 Day-2 maps:
      SPLIT   >= 3 deleted candidates within 12 px on one bright compact body (EXTRACT cut one soma into pieces, each
              failing the shape / activity rules) - marked on the brightest piece: 'one neuron split into N?'
      REDO    your Day-1 delete carried to Day 2, but on Day 2 a bright compact body with a good movie proof
              (event image >= 0.7, footprint one solid piece) - 're-check: your Day-1 delete, looks like a neuron on Day 2'
    Writes the marks into the review table and decisions/<tag>_review_flags.csv"""
    from ca_cell_classifier_day1 import contrast
    t2 = f"{m}_d2"
    root = os.path.join(BASE, "mini1p_out", "20261021", t2)
    csv = os.path.join(root, "4_cell_review", f"{t2}_cell_review.csv")
    R = pd.read_csv(csv)
    S = np.asarray(sio.loadmat(os.path.join(root, "3_extract", f"{t2}_extract.mat"), variable_names=["spatial_weights"])["spatial_weights"], np.float32)
    mx = np.asarray(sio.loadmat(os.path.join(root, "3_extract", f"{t2}_projections.mat"), variable_names=["max_dff"])["max_dff"], float)
    R["attention"] = R.get("attention", pd.Series([""] * len(R))).fillna("").astype(str)
    cm = np.array([contrast(mx, S[:, :, j]) if S[:, :, j].max() > 0 else np.nan for j in range(S.shape[2])])
    mf, so = shape_numbers(S)
    dele = np.flatnonzero(R.keep.values == 0)
    P = R[["row", "col"]].to_numpy(float)
    rows, used = [], set()
    for j in sorted(dele, key=lambda q: -np.nan_to_num(cm[q])):
        if j in used or not np.isfinite(cm[j]) or cm[j] < 5:
            continue
        near = [q for q in dele if np.hypot(*(P[q] - P[j])) <= 12]
        kept_near = np.flatnonzero((R.keep.values == 1) & (np.hypot(*(P - P[j]).T) <= 8))
        if len(near) >= 3 and not len(kept_near):
            ids = [int(R.id.iloc[q]) for q in near]
            txt = f"bright body split into {len(near)} deleted candidates (#{', #'.join(map(str, ids))}) - one neuron? check"
            R.at[j, "attention"] = txt
            used.update(near)
            rows.append(dict(kind="SPLIT", id=int(R.id.iloc[j]), pieces=" ".join(map(str, ids)), contrast=round(float(cm[j]), 1)))
    ev = pd.to_numeric(R.event_image_corr, errors="coerce").values
    for j in dele:
        ax_j = float(pd.to_numeric(R.axis_ratio, errors="coerce").iloc[j])
        kept_close = np.any((R.keep.values == 1) & (np.hypot(*(P - P[j]).T) <= 10))   # a split piece of a kept cell
        if str(R.get("track_class", pd.Series([""] * len(R))).iloc[j]) == "deleted" and np.isfinite(ev[j]) and ev[j] >= 0.7 \
                and mf[j] >= 0.85 and so[j] >= 0.85 and ax_j <= 2.5 and not kept_close and np.isfinite(cm[j]) and cm[j] >= 4:
            did = R.day1_id.iloc[j]
            R.at[j, "attention"] = (f"re-check: your Day-1 delete #{int(did)}, but on Day 2 a bright compact body with a "
                                    f"good movie proof (event image {ev[j]:.2f})")
            rows.append(dict(kind="REDO", id=int(R.id.iloc[j]), pieces="", contrast=round(float(cm[j]), 1)))
    R.to_csv(csv, index=False)
    dd = os.path.join(root, "5_analysis", "decisions"); os.makedirs(dd, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(dd, f"{t2}_review_flags.csv"), index=False)
    print(f"{m}: review flags (keep unchanged) - {sum(r['kind'] == 'SPLIT' for r in rows)} bright bodies split into deleted "
          f"pieces, {sum(r['kind'] == 'REDO' for r in rows)} Day-1 deletes that look like neurons on Day 2")


def refresh_pairs(m, unpair=()):
    """rewrite analysis/crossday/<m>_pairs.csv from the CURRENT Day-2 review table (after the duplicate check or
    your exclusions): a pair counts when the Day-2 cell is kept and carries a Day-1 id; --unpair <Day-2 ids> = you
    said this Day-2 cell is NOT that Day-1 neuron (kept in <m>_unpair_user.csv)"""
    cd = os.path.join(BASE, "analysis", "crossday")
    t2 = f"{m}_d2"
    R = pd.read_csv(os.path.join(BASE, "mini1p_out", "20261021", t2, "4_cell_review", f"{t2}_cell_review.csv"))
    up = os.path.join(cd, f"{m}_unpair_user.csv")
    U = set(pd.read_csv(up).day2_id.astype(int)) if os.path.isfile(up) else set()
    if unpair:
        U |= set(int(x) for x in unpair)
        pd.DataFrame(dict(day2_id=sorted(U))).to_csv(up, index=False)
    rv = R.get("reviewed", pd.Series([0] * len(R))).fillna(0).astype(int) == 1
    # a pair = a kept Day-2 neuron followed to its Day-1 neuron (A, B), or one the Day-2 rules had deleted (C) that YOU kept
    k = (R.keep == 1) & R.day1_id.notna() & ~R.id.isin(U) & (R.track_class.isin(["A", "B"]) | (rv & (R.track_class == "C")))
    P = pd.DataFrame(dict(day1_id=R.day1_id[k].astype(int).values, day2_id=R.id[k].astype(int).values,
                          track_class=R.track_class[k].values, cos=R.track_cos[k].values, patch=R.track_patch[k].values,
                          dist_px=R.track_dist_px[k].values))
    P.to_csv(os.path.join(cd, f"{m}_pairs.csv"), index=False)
    print(f"{m}: {len(P)} Day-1 <-> Day-2 pairs with the Day-2 cell kept ({len(U)} unpaired by you) -> {cd}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mouse")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--refresh-pairs", action="store_true", help="only rewrite <m>_pairs.csv from the review table")
    ap.add_argument("--fix-dedup", action="store_true", help="after ca_duplicate_check --resolve (see fix_dedup)")
    ap.add_argument("--flag-review", action="store_true", help="attention marks for split bright bodies / doubtful carried deletes")
    ap.add_argument("--unpair", nargs="*", type=int, default=[], help="Day-2 ids that are NOT their Day-1 neuron")
    ap.add_argument("--replot", action="store_true", help="map + pair pages from the final review table")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.replot:
        replot(a.mouse)
        return 0
    if a.fix_dedup:
        fix_dedup(a.mouse)
        return 0
    if a.flag_review:
        flag_review(a.mouse)
        return 0
    if a.refresh_pairs or a.unpair:
        refresh_pairs(a.mouse, a.unpair)
        return 0
    run_mouse(a.mouse)
    return 0


if __name__ == "__main__":
    sys.exit(main())
