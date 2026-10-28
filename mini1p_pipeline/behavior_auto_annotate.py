"""Automatic behaviour annotation that mimics the user's manual scoring, written as a ScoringAB
file the scorer opens for correction (score_AB_dual_view_mini1p mode 4, REVIEW AUTO).

Built and checked on ONE session: 5941 SBI, 2026-10-19, scored by the user (gold standard).
Every frame of the AUTO file of that session is predicted by a model that did NOT see it
(5-fold cross-validation over interleaved 30-s chunks), so the agreement reported here is what
the method does on video it was not trained on - not a replay of the training labels. The
threshold of each fold is chosen inside its training chunks (nested), never on the test chunk.

What is annotated (camera frame n = isx_frame n; frames below are 0-based rows)
  delivery          frame-wise random forest: "is this the frame where the user tapped a
                    delivery?" from the motion, tool and grid features of behavior_features.py
                    and the thin-tool line features of behavior_tool_lines.py (--tool)
                    in 5 windows (-1.0 .. +1.5 s); peaks of its probability = taps. Only inside
                    the 4 stimulus blocks of the 16.5-min schedule (+10 s).
  stimulus type     from the block and the session's stimulus order (randomisation sheet)
  withdrawal/flinch frame-wise random forests, peaks = taps, within -0.2 .. +3 s of a delivery
  escape / rearing  side camera rule of behavior_auto.py: top of the mouse >= 45 px above its
                    session median AND an upright blob (height / width >= 1.3), >= 0.3 s
  general movement  NEW category (user, 2026-10-20: correct on 5941 SBI). Frames where the
                    silhouette is not the mouse alone (area jump, centroid jump) are bridged:
                    the mouse walks or turns - smoothed centroid speed > 0.8 px/frame or body
                    axis turning > 1.5 deg/frame; bouts >= 0.4 s (gaps < 0.3 s closed) with
                    >= 20 px of path or >= 25 deg of turning. A withdrawal jerk is shorter.
  grooming          NEW category (candidates, not annotated by the user yet): with the side
                    camera, the upper 45 % of the mouse (head, forepaws) moves > 2x its session
                    median and > 1.4x the lower part, while the floor position is still
                    (bottom-view centroid < 0.6 px/frame), >= 1 s (behavior_side_parts.py)
  NOT annotated     attending (the user marked 11 bouts of ~0.25 s), licking/biting and
                    guarding (0 marked): too few examples, and a silhouette has no snout or paw.

An automatic event matches a user tap when they are <= 0.3 s apart (one-to-one).

  python behavior_auto_annotate.py --features 5941_d1_post_bfeatures.csv \
      --grid 5941_d1_post_bgrid.npz --scoring ScoringAB_<user>.mat --out-dir OUT \
      --auto-mat ...\\20261019_auto\\ScoringAB_<video>.mat --stim "Pin prick,Heat,Mild touch,Light touch" \
      --meta "sessionNo=7,mouseID=5941,sexID=F,dayNo=1,phase=Post-treatment" --bot B --side S
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from behavior_auto import bouts  # noqa: E402
from behavior_supervised import BLOCKS, load_scoring  # noqa: E402

FPS = 30.0
TOL = 9                      # +-0.3 s match tolerance
LABEL_HALF = 3               # frames each side of a tap that are "positive" for training
MIN_GAP = {"delivery": 12, "withdrawal": 9, "flinch": 5}
CHUNK = 900                  # 30-s chunks = cross-validation groups
NFOLD = 5
REFLEX_WIN = (-6, 90)        # withdrawal / flinch are looked for -0.2 .. +3 s around a delivery
SKIP_START_S = 15.0          # no movement / grooming marks before this (room-light change)
V_MOVE, W_MOVE = 0.8, 1.5    # general movement: px/frame (24 px/s), deg/frame (45 deg/s)
MOTION = ["b_body_me", "b_edge_me", "b_out_me", "b_near_me", "b_shape_chg", "b_speed", "b_turn",
          "s_me", "b_tool_px", "tool_prox"]
SHAPE = ["b_area", "b_major", "b_minor", "s_top", "s_aspect"]
WINS = [("p2", -30, -16), ("p1", -15, -6), ("now", -5, 5), ("a1", 6, 15), ("a2", 16, 45)]
GWINS = [("p1", -15, -6), ("now", -5, 5), ("a1", 6, 15)]


# ---------------------------------------------------------------------------- features
def clean(x):
    x = pd.Series(np.asarray(x, float)).interpolate(limit_direction="both").values
    return np.where(np.isfinite(x), x, 0.0)


def rz(x):
    med = np.median(x)
    return (x - med) / (1.4826 * np.median(np.abs(x - med)) + 1e-6)


def win_mean(x, a, b):
    n = len(x)
    c = np.r_[0.0, np.cumsum(x)]
    t = np.arange(n)
    lo, hi = np.clip(t + a, 0, n), np.clip(t + b + 1, 0, n)
    return (c[hi] - c[lo]) / np.maximum(hi - lo, 1)


def win_max(x, a, b):
    n = len(x)
    r = pd.Series(x).rolling(b - a + 1, min_periods=1).max().values
    return r[np.clip(np.arange(n) + b, 0, n - 1)]


def frame_matrix(F, G, extra=()):
    F = F.copy()
    F["tool_prox"] = np.where(np.isfinite(F.b_tool_dist), np.clip(80 - F.b_tool_dist, 0, 80), 0)
    cols = {}
    for c in MOTION + list(extra):
        x = rz(clean(F[c]))
        base = win_mean(x, -45, -31)
        for nm, a, b in WINS:
            m = win_mean(x, a, b)
            cols[f"{c}_{nm}_mean"] = m
            cols[f"{c}_{nm}_max"] = win_max(x, a, b)
            cols[f"{c}_{nm}_d"] = m - base
    for c in SHAPE:
        x = rz(clean(F[c]))
        for nm, a, b in GWINS:
            cols[f"{c}_{nm}"] = win_mean(x, a, b)
        cols[f"{c}_chg"] = win_mean(x, 6, 15) - win_mean(x, -15, -6)
    for kind in ("fg", "me"):
        A = G[kind].astype(float)
        for j in range(A.shape[1]):
            x = rz(clean(A[:, j]))
            for nm, a, b in GWINS:
                cols[f"g{kind}{j}_{nm}"] = win_mean(x, a, b)
    X = pd.DataFrame(cols)
    return X.values.astype(np.float32), list(X.columns)


# ---------------------------------------------------------------------------- events
def frame_labels(n, ev, half=None):
    half = LABEL_HALF if half is None else half
    y = np.zeros(n, np.int8)
    for f in ev:
        y[max(0, f - half):f + half + 1] = 1
    return y


def near_mask(n, ev, lo, hi):
    m = np.zeros(n, bool)
    for f in ev:
        m[max(0, f + lo):max(0, min(n, f + hi + 1))] = True
    return m


def peaks(p, allowed, thr, gap):
    from scipy.signal import find_peaks
    q = np.convolve(p, np.ones(3) / 3, "same")
    q = np.where(allowed, q, 0.0)
    pk, _ = find_peaks(q, height=thr, distance=gap)
    return pk, q[pk]


def match(auto, user, tol=TOL):
    """one-to-one, closest pairs first; returns TP, FP, FN, [(i_auto, j_user)]"""
    auto, user = np.asarray(auto, int), np.asarray(user, int)
    pairs = []
    for i, a in enumerate(auto):
        for j in np.where(np.abs(user - a) <= tol)[0]:
            pairs.append((abs(int(user[j]) - int(a)), i, int(j)))
    pairs.sort()
    ua, uu, M = set(), set(), []
    for _, i, j in pairs:
        if i not in ua and j not in uu:
            ua.add(i); uu.add(j); M.append((i, j))
    tp = len(M)
    return tp, len(auto) - tp, len(user) - tp, M


def prf(tp, fp, fn):
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return dict(TP=tp, FP=fp, FN=fn, precision=round(p, 3), recall=round(r, 3),
                F1=round(2 * tp / max(2 * tp + fp + fn, 1), 3))


def best_thr(p, allowed, user, gap):
    best, bt = -1, 0.5
    for t in np.arange(0.10, 0.951, 0.025):
        ev, _ = peaks(p, allowed, t, gap)
        tp, fp, fn, _ = match(ev, user)
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        if f1 > best:
            best, bt = f1, float(t)
    return bt


# ---------------------------------------------------------------------------- model
def rf():
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=3, max_features="sqrt",
                                  class_weight="balanced_subsample", n_jobs=-1, random_state=0)


def train_rows(pool, hard, rng, frac=0.25):
    return pool[hard[pool] | (rng.random(len(pool)) < frac)]


def cv_detect(X, user_ev, pool_mask, allowed, gap, fold, rng, label):
    """Nested CV. Returns out-of-fold probabilities for every frame and per-fold thresholds."""
    n = len(X)
    y = frame_labels(n, user_ev)
    hard = near_mask(n, user_ev, -15, 15)
    P = np.zeros(n)
    thr = {}
    for k in range(NFOLD):
        te = fold == k
        Pin = np.zeros(n)
        for j in range(NFOLD):
            if j == k:
                continue
            ite = fold == j
            rows = train_rows(np.where(pool_mask & ~te & ~ite)[0], hard, rng)
            Pin[ite] = rf().fit(X[rows], y[rows]).predict_proba(X[ite])[:, 1]
        ue = user_ev[~te[user_ev]]
        thr[k] = best_thr(Pin, allowed & ~te, ue, gap)
        rows = train_rows(np.where(pool_mask & ~te)[0], hard, rng)
        P[te] = rf().fit(X[rows], y[rows]).predict_proba(X[te])[:, 1]
        print(f"  {label}: fold {k + 1}/{NFOLD}, threshold {thr[k]:.3f}", flush=True)
    return P, thr


def oof_events(P, allowed, thr, gap, fold):
    ev, pr, fo = [], [], []
    for k in range(NFOLD):
        e, q = peaks(P, allowed & (fold == k), thr[k], gap)
        ev += list(e); pr += list(q); fo += [k] * len(e)
    o = np.argsort(ev)
    return np.array(ev, int)[o], np.array(pr)[o], np.array(fo, int)[o]


# ---------------------------------------------------------------------------- rules
def rearing(F):
    top, asp = clean(F.s_top), clean(F.s_aspect)
    thr = np.median(top) - 45
    r, on, off = bouts((top < thr) & (asp >= 1.3), int(0.3 * FPS), int(0.3 * FPS))
    return r, on, off, float(thr)


def silhouette_artefacts(F):
    """frames where the bottom silhouette is not the mouse alone: its area jumps away from the
    5-s running median (< 0.65x: cut by the hand or a shadow; > 1.5x: merged with the hand /
    arm) or the centroid jumps > 15 px in one frame; widened by 3 frames"""
    a = np.asarray(F.b_area, float)
    rm = pd.Series(a).rolling(150, center=True, min_periods=15).median().values
    r = a / rm
    jump = np.r_[0, np.hypot(np.diff(np.asarray(F.b_cx, float)), np.diff(np.asarray(F.b_cy, float)))]
    bad = ~np.isfinite(r) | (r < 0.65) | (r > 1.5) | (jump > 15)
    return np.convolve(bad.astype(float), np.ones(7), "same") > 0


D_MIN_BODY = 0.2               # locomotion: net displacement >= 0.2 body lengths (~35 px, ~1.8 cm)


def locomotion(F, exclude=None):
    """LOCOMOTION, NOT USED (2026-10-21: proposed, then you kept the original movement definition). A to B only; turning in place,
    grooming and rearing are not movement). The centroid of the bottom silhouette (artefact frames bridged,
    7-frame mean) moves faster than V_MOVE px/frame; bouts >= 0.4 s, gaps < 0.3 s merged; a bout is kept only
    when its NET displacement (start -> end) is >= D_MIN_BODY x the session's median body length and < 30 % of
    it are artefact frames. Frames in `exclude` (grooming, rearing) are taken out before the bouts are formed.
    (Before 10-21 a body turn > 1.5 deg/frame also counted, and a bout needed only 20 px of PATH; that version
    is general_movement_v1.)"""
    bad = silhouette_artefacts(F)
    G = F[["b_cx", "b_cy", "b_theta"]].astype(float).copy()
    G[bad] = np.nan                     # bridged by interpolation in clean()
    cx = pd.Series(clean(G.b_cx)).rolling(7, center=True, min_periods=1).mean().values
    cy = pd.Series(clean(G.b_cy)).rolling(7, center=True, min_periods=1).mean().values
    th = np.degrees(np.unwrap(np.radians(2 * clean(G.b_theta)))) / 2
    th = pd.Series(th).rolling(7, center=True, min_periods=1).mean().values
    v = np.r_[0, np.hypot(np.diff(cx), np.diff(cy))]
    w = np.r_[0, np.abs(np.diff(th))]
    raw = v > V_MOVE
    if exclude is not None:
        ex = np.zeros(len(raw), bool)
        e = np.asarray(exclude, bool)[:len(raw)]
        ex[:len(e)] = e
        raw &= ~ex
    m, on, off = bouts(raw, int(0.4 * FPS), int(0.3 * FPS))
    body = float(np.nanmedian(np.asarray(F.b_major, float)))
    dmin = D_MIN_BODY * (body if np.isfinite(body) and body > 0 else 175.0)
    keep = []
    for a, b in zip(on, off):
        net = np.hypot(cx[b - 1] - cx[a], cy[b - 1] - cy[a])
        keep.append(net >= dmin and bad[a:b].mean() < 0.3)
    keep = np.array(keep, bool)
    out = np.zeros(len(m), bool)
    for a, b in zip(on[keep], off[keep]):
        out[a:b] = True
    if exclude is not None:
        out &= ~ex
    return out, on[keep], off[keep], v, w


def general_movement(F):
    """general movement (the definition of every result): centroid speed > V_MOVE OR body turn > W_MOVE,
    bouts >= 0.4 s (gaps < 0.3 s merged), kept with a path >= 20 px or a turn >= 25 deg and < 30 % artefact frames"""
    bad = silhouette_artefacts(F)
    G = F[["b_cx", "b_cy", "b_theta"]].astype(float).copy()
    G[bad] = np.nan
    cx = pd.Series(clean(G.b_cx)).rolling(7, center=True, min_periods=1).mean().values
    cy = pd.Series(clean(G.b_cy)).rolling(7, center=True, min_periods=1).mean().values
    th = np.degrees(np.unwrap(np.radians(2 * clean(G.b_theta)))) / 2
    th = pd.Series(th).rolling(7, center=True, min_periods=1).mean().values
    v = np.r_[0, np.hypot(np.diff(cx), np.diff(cy))]
    w = np.r_[0, np.abs(np.diff(th))]
    m, on, off = bouts((v > V_MOVE) | (w > W_MOVE), int(0.4 * FPS), int(0.3 * FPS))
    keep = np.array([(v[a:b].sum() >= 20 or abs(th[b - 1] - th[a]) >= 25) and bad[a:b].mean() < 0.3
                     for a, b in zip(on, off)], bool)
    out = np.zeros(len(m), bool)
    for a, b in zip(on[keep], off[keep]):
        out[a:b] = True
    return out, on[keep], off[keep], v, w


def grooming(F, SP, rear):
    """upper body moving, lower body and floor position still, >= 1 s; not while rearing"""
    sm = lambda x, w=15: pd.Series(x).rolling(w, center=True, min_periods=1).mean().values
    up, lo = sm(clean(SP["sp_up_me"])), sm(clean(SP["sp_low_me"]))
    spd = sm(clean(F.b_speed))
    raw = (up > 2.0 * np.median(up)) & (up > 1.4 * lo) & (spd < 0.6) & ~rear
    return bouts(raw, int(1.0 * FPS), int(0.3 * FPS))


def block_index(n):
    blk = np.zeros(n, int)
    for k, (_, s0, s1) in enumerate(BLOCKS[1:], 1):
        blk[int(s0 * FPS):min(n, int((s1 + 10) * FPS))] = k
    return blk


def response_table(dl, nxt_lim, ev):
    """per delivery: any event in [f-6, min(f+60, next delivery-1)]"""
    out = []
    for i, f in enumerate(dl):
        hi = min(f + 60, (dl[i + 1] - 1) if i + 1 < len(dl) else nxt_lim)
        out.append(int(np.any((ev >= f - 6) & (ev <= hi))))
    return np.array(out, int)


def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    po = np.mean(a == b)
    pe = np.mean(a) * np.mean(b) + (1 - np.mean(a)) * (1 - np.mean(b))
    return float((po - pe) / (1 - pe)) if pe < 1 else np.nan


# ---------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--grid", required=True)
    ap.add_argument("--scoring", required=True, help="the user's ScoringAB_*.mat (gold standard)")
    ap.add_argument("--stim", required=True, help="session stimulus order, comma separated")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--auto-mat", required=True)
    ap.add_argument("--meta", default="")
    ap.add_argument("--bot", default="")
    ap.add_argument("--side", default="")
    ap.add_argument("--label-half", type=int, default=LABEL_HALF)
    ap.add_argument("--tool", default="", help="<tag>_btool.npz of behavior_tool_lines.py (thin tools)")
    ap.add_argument("--side-parts", default="", help="<tag>_bside.npz of behavior_side_parts.py (grooming)")
    a = ap.parse_args(argv)
    globals()["LABEL_HALF"] = a.label_half
    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(a.auto_mat), exist_ok=True)
    t0 = time.time()
    rng = np.random.default_rng(0)
    stim = [s.strip() for s in a.stim.split(",")]
    meta = dict(kv.split("=", 1) for kv in a.meta.split(",") if "=" in kv)

    F = pd.read_csv(a.features)
    G = np.load(a.grid)
    S = load_scoring(a.scoring)
    n = min(len(F), S["nUsed"], len(G["fg"]))
    F = F.iloc[:n].reset_index(drop=True)
    G = {k: G[k][:n] for k in ("fg", "me")}
    u_del = np.sort(S["dF"].astype(int) - 1)
    u_typ = S["dT"][np.argsort(S["dF"])]
    re = S["re"]
    u_w = np.sort(re[re[:, 1] == 1, 0].astype(int) - 1)
    u_f = np.sort(re[re[:, 1] == 2, 0].astype(int) - 1)
    u_score = S["score"][:n]
    print(f"user: {len(u_del)} deliveries, {len(u_w)} withdrawal, {len(u_f)} flinch taps; "
          f"{n} frames", flush=True)

    extra = []
    if a.tool:
        Tz = np.load(a.tool)
        for j, nm in enumerate(Tz["names"]):
            F[str(nm)] = Tz["X"][:n, j]
            extra.append(str(nm))
    X, names = frame_matrix(F, G, extra)
    fold = (np.arange(n) // CHUNK) % NFOLD
    blk = block_index(n)
    in_blk = blk > 0
    print(f"features {X.shape} ({time.time() - t0:.0f} s)", flush=True)

    # ---- deliveries -------------------------------------------------------------------------
    Pd, thr_d = cv_detect(X, u_del, np.ones(n, bool), in_blk, MIN_GAP["delivery"], fold, rng, "delivery")
    a_del, a_dp, a_dfo = oof_events(Pd, in_blk, thr_d, MIN_GAP["delivery"], fold)
    tp, fp, fn, Md = match(a_del, u_del)
    res = dict(delivery=prf(tp, fp, fn))
    res["delivery_within_0p5s"] = prf(*match(a_del, u_del, 15)[:3])
    res["delivery_per_block"] = {}
    for k in range(1, 5):
        ae, ue = a_del[blk[a_del] == k], u_del[blk[u_del] == k]
        res["delivery_per_block"][f"block{k} {stim[k - 1]}"] = prf(*match(ae, ue)[:3])
    dt = [(int(a_del[i]) - int(u_del[j])) / FPS for i, j in Md]
    res["delivery_timing_auto_minus_user_s"] = dict(median=round(float(np.median(dt)), 3),
                                                    p10=round(float(np.percentile(dt, 10)), 3),
                                                    p90=round(float(np.percentile(dt, 90)), 3))
    a_typ = np.array([blk[f] for f in a_del], int)                   # key = block (1..4)
    # what the user's key was for the matched deliveries (type agreement)
    same_type = [int(a_typ[i] == u_typ[j]) for i, j in Md]
    res["delivery_type_same_key_as_user"] = round(float(np.mean(same_type)), 3) if same_type else np.nan

    # ---- withdrawal / flinch ----------------------------------------------------------------
    pool = near_mask(n, u_del, -30, 105)
    win_u = near_mask(n, u_del, *REFLEX_WIN)
    win_a = near_mask(n, a_del, *REFLEX_WIN)
    reflex = {}
    for nm, ue in (("withdrawal", u_w), ("flinch", u_f)):
        P, thr = cv_detect(X, ue, pool, win_u, MIN_GAP[nm], fold, rng, nm)
        ev_u, pr_u, _ = oof_events(P, win_u, thr, MIN_GAP[nm], fold)      # around YOUR deliveries
        ev_a, pr_a, _ = oof_events(P, win_a, thr, MIN_GAP[nm], fold)      # around AUTO deliveries
        reflex[nm] = dict(P=P, thr=thr, ev_u=ev_u, ev_a=ev_a, pr_a=pr_a)
        res[f"{nm}_taps_around_your_deliveries"] = prf(*match(ev_u, ue)[:3])
        res[f"{nm}_taps_fully_automatic"] = prf(*match(ev_a, ue)[:3])
        res[f"{nm}_taps_around_your_deliveries_within_0p5s"] = prf(*match(ev_u, ue, 15)[:3])
        uy = response_table(u_del, n - 1, ue)
        ay = response_table(u_del, n - 1, ev_u)
        res[f"{nm}_per_delivery"] = dict(agreement=round(float(np.mean(uy == ay)), 3),
                                         kappa=round(kappa(uy, ay), 3), user_pct=round(100 * uy.mean(), 1),
                                         auto_pct=round(100 * ay.mean(), 1))
        per = {}
        for k in range(1, 5):
            m = blk[u_del] == k
            per[f"block{k} {stim[k - 1]}"] = dict(n=int(m.sum()), user_pct=round(100 * uy[m].mean(), 1),
                                                  auto_pct=round(100 * ay[m].mean(), 1),
                                                  agreement=round(float(np.mean(uy[m] == ay[m])), 3))
        res[f"{nm}_per_delivery_by_block"] = per
        reflex[nm]["uy"], reflex[nm]["ay"] = uy, ay

    # ---- escape / rearing, general movement -------------------------------------------------
    rear, r_on, r_off, r_thr = rearing(F)
    u_esc = u_score == 4
    ub = np.diff(np.r_[0, u_esc.astype(int), 0])
    u_on, u_off = np.where(ub == 1)[0], np.where(ub == -1)[0]
    hit = [bool(rear[max(0, s - 15):e + 15].any()) for s, e in zip(u_on, u_off)]
    res["escape_rearing"] = dict(user_bouts=[[round(s / FPS, 2), round(e / FPS, 2)] for s, e in zip(u_on, u_off)],
                                 user_bouts_found_by_auto=int(sum(hit)), auto_bouts=int(len(r_on)),
                                 auto_time_s=round(float(rear.sum() / FPS), 1), side_top_threshold_y=r_thr)
    move, m_on, m_off, v, w = general_movement(F)
    # the first ~13 s are before the room light changes; the bottom silhouette is not reliable
    keep = m_on >= int(SKIP_START_S * FPS)
    move[:int(SKIP_START_S * FPS)] = False
    m_on, m_off = m_on[keep], m_off[keep]
    res["general_movement"] = dict(bouts=int(len(m_on)), time_s=round(float(move.sum() / FPS), 1),
                                   median_bout_s=round(float(np.median((m_off - m_on) / FPS)), 2) if len(m_on) else 0,
                                   rule=f"speed > {V_MOVE} px/frame or turning > {W_MOVE} deg/frame, "
                                        f">= 0.4 s, >= 20 px path or >= 25 deg")
    groom = np.zeros(n, bool); g_on = g_off = np.zeros(0, int)
    if a.side_parts:
        Zs = np.load(a.side_parts)
        SPd = {str(k): Zs["X"][:n, j] for j, k in enumerate(Zs["names"])}
        groom, g_on, g_off = grooming(F, SPd, rear)
        keep = g_on >= int(SKIP_START_S * FPS)
        groom[:int(SKIP_START_S * FPS)] = False
        g_on, g_off = g_on[keep], g_off[keep]
    res["grooming_candidates"] = dict(bouts=int(len(g_on)), time_s=round(float(groom.sum() / FPS), 1),
                                      rule="side upper-body motion > 2x median and > 1.4x lower body, "
                                           "floor speed < 0.6 px/frame, >= 1 s, not rearing",
                                      bouts_s=[[round(s / FPS, 1), round(e / FPS, 1)] for s, e in zip(g_on, g_off)])
    mv_after = response_table(u_del, n - 1, np.where(np.diff(np.r_[0, move.astype(int)]) == 1)[0])
    per = {}
    for k in range(1, 5):
        m = blk[u_del] == k
        per[f"block{k} {stim[k - 1]}"] = round(100 * mv_after[m].mean(), 1)
    res["general_movement_starts_after_your_deliveries_pct"] = per
    pct = {}
    for b, s0, s1 in BLOCKS:
        sl = slice(int(s0 * FPS), min(n, int(s1 * FPS)))
        pct[b] = round(100 * float(move[sl].mean()), 1)
    res["general_movement_pct_time_by_block"] = pct

    # ---- AUTO scoring file (out-of-fold predictions only) -------------------------------------
    w_ev, f_ev = reflex["withdrawal"]["ev_a"], reflex["flinch"]["ev_a"]
    refl = np.r_[np.c_[w_ev + 1, np.ones(len(w_ev))], np.c_[f_ev + 1, 2 * np.ones(len(f_ev))]]
    refl = refl[np.argsort(refl[:, 0])] if len(refl) else np.zeros((0, 2))
    score = np.zeros(n)
    score[rear] = 4
    # frames worth a look: weak automatic deliveries, and sub-threshold candidates (missed?)
    chk = []
    for k in range(NFOLD):
        e, q = peaks(Pd, in_blk & (fold == k), 0.6 * thr_d[k], MIN_GAP["delivery"])
        chk += [int(f) for f, qq in zip(e, q) if qq < thr_d[k] + 0.15]
    chk = np.unique(np.array(chk, int))
    info = dict(source="behavior_auto_annotate.py", created=time.strftime("%Y-%m-%d %H:%M"),
                trained_on=os.path.basename(a.scoring),
                cv="5-fold over interleaved 30-s chunks; every frame predicted by a model that did not see it",
                delivery_thresholds=",".join(f"{thr_d[k]:.3f}" for k in range(NFOLD)),
                match_tolerance_s=TOL / FPS,
                delivery_precision=res["delivery"]["precision"], delivery_recall=res["delivery"]["recall"],
                not_annotated="attending, licking/biting, guarding (too few examples)",
                stimulus_type="from the schedule block and the session order")
    mat = dict(score=score.reshape(-1, 1), uncert=np.zeros((n, 1), bool), timeAxis=(np.arange(n) / FPS).reshape(1, -1),
               affNames=np.array(["None", "Paw attending", "Licking or biting", "Sustained lifting / guarding",
                                  "Escape / rearing"], dtype=object),
               affKeys=np.array(["a", "s", "d", "f"], dtype=object),
               refNames=np.array(["Paw withdrawal", "Flinch / flick"], dtype=object),
               reflexEvents=refl.astype(float), dFrames=(a_del + 1).astype(float).reshape(-1, 1),
               dSec=(a_del / FPS).reshape(-1, 1), dTypes=a_typ.astype(float).reshape(-1, 1),
               stimNames=np.array(stim, dtype=object), frameRate=FPS, fpsTrue=FPS, hdrFps=FPS,
               nUsed=float(n), mode=4.0, lagMs=250.0, obsWindow=30.0, guardMin=2.0, syncOff=0.0,
               offFrames=0.0, botFile=a.bot, sidFile=a.side,
               moveVec=move.reshape(-1, 1), moveName="General movement (turn / steps)",
               groomVec=groom.reshape(-1, 1), groomName="Grooming (candidates)",
               checkFrames=(chk + 1).astype(float).reshape(-1, 1), autoInfo=info,
               autoDeliveryProb=Pd.astype(np.float32).reshape(-1, 1))
    for k in ("sessionNo", "mouseID", "sexID", "dayNo", "phase"):
        mat[k] = meta.get(k, "")
    sio.savemat(a.auto_mat, mat, do_compression=True)

    # ---- tables -----------------------------------------------------------------------------
    rows = []
    md = dict(Md)
    for i, f in enumerate(a_del):
        j = md.get(i)
        rows.append(dict(kind="delivery", frame=int(f) + 1, t_s=round(f / FPS, 3), block=int(blk[f]),
                         stim=stim[blk[f] - 1], prob=round(float(a_dp[i]), 3),
                         user_match_t_s=round(u_del[j] / FPS, 3) if j is not None else np.nan,
                         user_key=int(u_typ[j]) if j is not None else np.nan))
    for nm in ("withdrawal", "flinch"):
        for f, q in zip(reflex[nm]["ev_a"], reflex[nm]["pr_a"]):
            rows.append(dict(kind=nm, frame=int(f) + 1, t_s=round(f / FPS, 3), block=int(blk[f]),
                             stim=stim[blk[f] - 1] if blk[f] else "", prob=round(float(q), 3)))
    for s, e in zip(r_on, r_off):
        rows.append(dict(kind="escape/rearing", frame=int(s) + 1, t_s=round(s / FPS, 3), dur_s=round((e - s) / FPS, 2)))
    for s, e in zip(g_on, g_off):
        rows.append(dict(kind="grooming (candidate)", frame=int(s) + 1, t_s=round(s / FPS, 3), dur_s=round((e - s) / FPS, 2)))
    for s, e in zip(m_on, m_off):
        rows.append(dict(kind="general movement", frame=int(s) + 1, t_s=round(s / FPS, 3), dur_s=round((e - s) / FPS, 2),
                         block=int(blk[s])))
    pd.DataFrame(rows).sort_values(["kind", "frame"]).to_csv(os.path.join(a.out_dir, "auto_events.csv"), index=False)
    matched_u = {j for _, j in Md}
    miss = [dict(user_delivery_frame=int(f) + 1, t_s=round(f / FPS, 2), user_key=int(u_typ[j]), block=int(blk[f]),
                 max_prob_within_0p3s=round(float(Pd[max(0, f - TOL):f + TOL + 1].max()), 3))
            for j, f in enumerate(u_del) if j not in matched_u]
    pd.DataFrame(miss).to_csv(os.path.join(a.out_dir, "auto_missed_your_deliveries.csv"), index=False)
    extra = [dict(auto_delivery_frame=int(a_del[i]) + 1, t_s=round(a_del[i] / FPS, 2), block=int(blk[a_del[i]]),
                  prob=round(float(a_dp[i]), 3)) for i in range(len(a_del)) if i not in md]
    pd.DataFrame(extra).to_csv(os.path.join(a.out_dir, "auto_extra_deliveries_not_in_yours.csv"), index=False)
    D = pd.DataFrame(dict(delivery=np.arange(1, len(u_del) + 1), frame=u_del + 1, t_s=np.round(u_del / FPS, 3),
                          block=blk[u_del], your_key=u_typ, found_by_auto=[int(j in matched_u) for j in range(len(u_del))],
                          your_withdrawal=reflex["withdrawal"]["uy"], auto_withdrawal=reflex["withdrawal"]["ay"],
                          your_flinch=reflex["flinch"]["uy"], auto_flinch=reflex["flinch"]["ay"],
                          auto_movement_after=mv_after))
    D.to_csv(os.path.join(a.out_dir, "auto_vs_your_per_delivery.csv"), index=False)
    pd.DataFrame(dict(frame=np.arange(1, n + 1), p_delivery=Pd.round(4), p_withdrawal=reflex["withdrawal"]["P"].round(4),
                      p_flinch=reflex["flinch"]["P"].round(4), rearing=rear.astype(int), movement=move.astype(int),
                      grooming=groom.astype(int), speed_px_frame=v.round(3), turn_deg_frame=w.round(3))).to_csv(
        os.path.join(a.out_dir, "auto_frame_probabilities.csv"), index=False)

    # ---- final models for later sessions (trained on every frame of this session) -----------
    import joblib
    models = {}
    for nm, ue, pm in (("delivery", u_del, np.ones(n, bool)), ("withdrawal", u_w, pool), ("flinch", u_f, pool)):
        y = frame_labels(n, ue)
        rows_ = train_rows(np.where(pm)[0], near_mask(n, ue, -15, 15), rng)
        thr = thr_d if nm == "delivery" else reflex[nm]["thr"]
        models[nm] = dict(model=rf().fit(X[rows_], y[rows_]), threshold=float(np.median(list(thr.values()))),
                          min_gap=MIN_GAP[nm])
    joblib.dump(dict(models=models, feature_names=names, trained_on=os.path.basename(a.scoring),
                     created=time.strftime("%Y-%m-%d %H:%M")), os.path.join(a.out_dir, "auto_models.joblib"), compress=3)

    res["n_auto"] = dict(deliveries=int(len(a_del)), withdrawal=int(len(w_ev)), flinch=int(len(f_ev)),
                         rearing_bouts=int(len(r_on)), movement_bouts=int(len(m_on)), grooming_bouts=int(len(g_on)),
                         check_frames=int(len(chk)))
    res["files"] = dict(auto_mat=a.auto_mat, out_dir=a.out_dir)
    with open(os.path.join(a.out_dir, "auto_vs_your_annotation.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    np.savez_compressed(os.path.join(a.out_dir, "auto_annotation_arrays.npz"), a_del=a_del, a_typ=a_typ,
                        u_del=u_del, u_typ=u_typ, u_w=u_w, u_f=u_f, w_ev=w_ev, f_ev=f_ev,
                        w_ev_u=reflex["withdrawal"]["ev_u"], f_ev_u=reflex["flinch"]["ev_u"], rear=rear, move=move,
                        u_esc=u_esc, groom=groom, Pd=Pd, Pw=reflex["withdrawal"]["P"], Pf=reflex["flinch"]["P"], blk=blk)
    print(json.dumps(res, indent=1, default=float))
    print(f"done in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
