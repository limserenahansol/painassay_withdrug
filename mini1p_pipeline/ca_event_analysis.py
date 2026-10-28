"""ca_event_analysis.py - the basic event analyses you asked for (2026-10-22). All 16 sessions (4 mice x Day 1 / Day 2 x
baseline / injection); neurons pooled over the mice (the mouse kept as a label). z = dF/F / robust SD (MAD) of the session.

  window    which response window finds the responsive neurons best? A grid of windows - baseline 0.5 / 1 / 2 / 3 / 5 / 10 s
            long, ending at the event or 1 s before it (gap), response 0-0.5 ... 0-10 s, delayed (0.25-1, 0.5-2, 1-3, 2-5 s ...)
            and early (-0.5-1 s) - each scored without circularity:
              replication = % of neurons responsive (Wilcoxon over trials, p < 0.05) on the ODD trials AND on the EVEN trials
                            with the same sign
              two shams   = the same with fake events: (1) JITTER - every event moved 2-5 s earlier or later (tests that the
                            response is locked to the event itself at the scale of seconds); (2) BLOCK - the event train
                            shifted inside its block (keeps number, spacing and block; tests any event-related change,
                            including slow ones)
              quality     = replication (real) - replication (jitter sham)  -> the window choice (event-locked responses)
              slow change = replication (real) - replication (block sham)   -> reported
              contamination = % of trials with ANOTHER delivery inside the window
            the choice is made among the CLEAN windows (baseline <= 3 s, response ending <= 3 s) because a delivery comes
            every ~3 s: a 10-s window holds ~3 deliveries and the movement / arousal they cause (it finds more neurons, but
            not the response to one stimulus - shown separately). Checked leave-one-mouse-out: a window chosen on 3 mice is
            scored on the 4th.
  classify  every neuron x event x session with the chosen window: within-block time-shift test (1000 shuffles, as before),
            BH q <= 0.20 within the session -> activated / inhibited / non-responsive; AUC (z*s)
  overlap   stimulus, withdrawal and flinch come within ~0.1-0.5 s: (a) latencies; (b) the same stimulus with vs without a
            withdrawal; (c) encoding model (ridge regression, 0-3 s kernels (0.5-s steps) of every stimulus type, withdrawal, flinch,
            movement / escape onset + movement and grooming states, block nuisance; cross-validated): the variance each
            family explains ALONE, tested against the family shifted inside its blocks -> stimulus / reflex / both /
            cannot-separate / movement-only neurons
  plots     your basic plots 1-7
Writes analysis\\event_analysis\\*.png / *.csv and summary.json.

  python ca_event_analysis.py --step all            (window, classify, overlap, plots)
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import binomtest, fisher_exact, mannwhitneyu, norm, rankdata, spearmanr, wilcoxon

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import safe_io  # noqa: E402,F401
import ca_day1_analysis as CA  # noqa: E402
import pairstars as PS  # noqa: E402  (user rule 10-22: every star shows the pair it compares)
import figfoot  # noqa: E402  (10-24: the methods text under a figure goes into the slide notes, <png>.notes.txt)
import winwords as WW  # noqa: E402  (10-24: each event's response window in the P3a panel titles)

warnings.filterwarnings("ignore", category=RuntimeWarning)
BASE = r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553"
OD = os.path.join(BASE, "analysis", os.environ.get("CA_EVENT_OUT", "event_analysis"))   # 10-23: CA_EVENT_OUT = a sensitivity run
MO = ["5941", "5939", "5937", "1191"]
ARMS = {1: {"5941": "SBI-553", "5939": "SBI-553", "5937": "vehicle", "1191": "vehicle"},
        2: {"5941": "vehicle", "5939": "vehicle", "5937": "SBI-553", "1191": "SBI-553"}}
COND = ["baseline", "vehicle", "SBI-553"]
CC = {"baseline": "#7f8c8d", "vehicle": "#2471a3", "SBI-553": "#c0392b"}
MC = {"5941": "#e67e22", "5939": "#8e44ad", "5937": "#16a085", "1191": "#34495e"}
MK = {"5941": "o", "5939": "s", "5937": "^", "1191": "D"}
C3 = {"both": "#2ecc40", "d1": "#f012be", "d2": "#00d0ff"}
CATC = {"activated": "#e74c3c", "inhibited": "#2e86c1", "non-responsive": "#c8cdd2"}
STIM = ["Light touch", "Mild touch", "Pin prick", "Heat"]
REFL = ["withdrawal", "flinch"]
MOVE = ["movement onset", "escape/rearing onset", "grooming onset"]
AFF = ["attending onset"]                                       # added 10-22 (your scoring code 1; behaviour window)
EVENTS = STIM + REFL + MOVE + AFF
SEARCH_EV = STIM + REFL + ["movement onset"]
SHORT = {"Light touch": "light", "Mild touch": "mild", "Pin prick": "pin", "Heat": "heat", "withdrawal": "withdrawal",
         "flinch": "flinch", "movement onset": "movement", "escape/rearing onset": "escape", "grooming onset": "grooming",
         "attending onset": "attending"}
SCOL = {"Light touch": "#5dade2", "Mild touch": "#f5b041", "Pin prick": "#e74c3c", "Heat": "#8e44ad"}
FPS = 30.0
SEG_S = [CA.SKIP_S, 180, 360, 390, 570, 600, 780, 810, 990]     # schedule segments: no-stimulus start, blocks, rests, end
BLOCK_S = [(180, 360), (390, 570), (600, 780), (810, 990)]
ROOT = {1: os.path.join(BASE, "mini1p_out", "20261019"), 2: os.path.join(BASE, "mini1p_out", "20261021")}
# 10-23: the timing-corrected copies of the final scoring (scoring_timefix.py: withdrawal / flinch -0.40 s Day 1, -0.37 s
# Day 2; deliveries moved only where their own reflex came first; 3 Day-2 clicker fixes). SCORING_SUFFIX="" = the originals.
SCORING_SFX = os.environ.get("SCORING_SUFFIX", "_timefix")
SCORE = {1: os.path.join(BASE, "scoring_mini1p", "20261019_final" + SCORING_SFX),
         2: os.path.join(BASE, "scoring_mini1p", "20261021_final" + SCORING_SFX)}
PRES = [0.5, 1, 2, 3, 5, 10]
GAPS = [0.0, 1.0]
POSTS = [(-0.5, 1), (-0.5, 2), (0, 0.5), (0, 1), (0, 1.5), (0, 2), (0, 3), (0, 5), (0, 10), (0.25, 1), (0.25, 2), (0.5, 1.5),
         (0.5, 2), (0.5, 3), (1, 3), (1, 5), (2, 5)]
DEFAULT = (2.0, 0.0, 0.0, 2.0)                                 # the window used so far: [-2, 0] s vs [0, 2] s
# a window = (pre, gap, a, b): baseline [a0 - gap - pre, a0 - gap) with a0 = min(a, 0), response [a, b), seconds from the event
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "xtick.labelsize": 11,
                     "ytick.labelsize": 11, "legend.fontsize": 10.5, "axes.spines.top": False, "axes.spines.right": False})


def cond_of(m, d, ph):
    return "baseline" if ph == "pre" else ARMS[int(d)][str(m)]


def drug_of(m, d):
    return ARMS[int(d)][str(m)]


EV_SUB, B6_SUB, FN_SUF = None, None, ""          # split parts (10-22): events / behaviours of one part, file suffix
KEY_PAIRS = "brackets = the 2 conditions compared: base-veh and base-SBI = the same neurons (mice), baseline of that day vs after the injection; veh-SBI = after vehicle vs after SBI-553.  * p < 0.05, ** p < 0.01, *** p < 0.001, ns = not significant"


def star(p):
    if p is None or not np.isfinite(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


bh = CA.bh


# ------------------------------------------------------------------------------------------------ data
BGREG = False                      # --bg-regress: the field (background outside every footprint) regressed out of each neuron


def regress_background(s, m, d, ph):
    """z of every neuron minus beta * z of the field brightness (mean of the pixels outside every footprint,
    <rid>_background.npz); beta per neuron by least squares over the session. Removes what is shared with the field
    (focus / posture changes, EXTRACT background over- or under-correction, diffuse neuropil)"""
    t = f"{m}_d{d}"
    f = os.path.join(ROOT[d], t, "5_analysis", CA.TRACES, f"{t}_{ph}_background.npz")
    if not os.path.isfile(f):
        print(f"  (no background file for {t} {ph})", flush=True)
        return None
    Bg = np.load(f)
    b = pd.Series(np.asarray(Bg["b"], float), index=np.asarray(Bg["frame"], int)).reindex(s["T"]["frame"].astype(int)).values
    med = np.nanmedian(b)
    bz = (b - med) / (1.4826 * np.nanmedian(np.abs(b - med)))
    Z = np.asarray(s["T"]["z"], float)
    ok = np.isfinite(bz)[:, None] & np.isfinite(Z)
    x = np.where(ok, bz[:, None], 0.0)
    y = np.where(ok, Z, 0.0)
    n = ok.sum(0)
    mx, my = x.sum(0) / n, y.sum(0) / n
    beta = ((x * y).sum(0) / n - mx * my) / np.maximum((x * x).sum(0) / n - mx * mx, 1e-9)
    Zc = Z - beta[None, :] * np.where(np.isfinite(bz), bz, 0.0)[:, None]
    Zc[~np.isfinite(Z)] = np.nan
    s["T"]["z"] = Zc.astype(np.float32)
    return beta


def load_all():
    S = {}
    for m in MO:
        for d in (1, 2):
            for ph in ("pre", "post"):
                s = CA.load_session(ROOT[d], m, ph, SCORE[d], day=d)
                if s is None or s["B"] is None:
                    continue
                if BGREG:
                    beta = regress_background(s, m, d, ph)
                    if beta is not None:
                        s["bg_beta"] = beta
                Sm, _ = CA.scoring_for(SCORE[d], m, ph)
                sc = np.atleast_1d(Sm["score"]).astype(int)
                s["held"] = {nm: np.r_[False, sc == code] for nm, code in (("attending", 1), ("lick/bite", 2), ("guarding", 3),
                                                                           ("escape/rearing", 4))}
                s["B"]["ev"]["attending onset"] = CA.onsets(sc == 1) + 1      # as ca_cell_event_matrix (bout after >= 2 s without)
                Z = np.asarray(s["T"]["z"], float)
                ok = np.isfinite(Z)
                s["Zc"] = np.vstack([np.zeros((1, Z.shape[1])), np.cumsum(np.where(ok, Z, 0), 0)])
                s["Nc"] = np.vstack([np.zeros((1, Z.shape[1]), np.int32), np.cumsum(ok, 0, dtype=np.int32)])
                s["fr"] = s["T"]["frame"].astype(int)
                s["ids"] = s["T"]["cell_ids"].astype(int)
                s["day"] = d
                S[(m, d, ph)] = s
    return S


def deliveries(s):
    ev = s["B"]["ev"]
    return np.sort(np.concatenate([np.asarray(ev.get(k, []), int) for k in STIM]))


def ev_rows(s, frames):
    fr = s["fr"]
    f = np.asarray(frames, int)
    i = np.searchsorted(fr, f)
    ok = i < len(fr)
    ok[ok] &= fr[i[ok]] == f[ok]
    return i[ok], f[ok]


def win_rows(pre, gap, a, b):
    """rows of the baseline [a0 - gap - pre, a0 - gap) (a0 = min(a, 0)) and of the response [a, b) relative to the event"""
    a0 = min(a, 0.0) - gap
    return int(round((a0 - pre) * FPS)), int(round(a0 * FPS)), int(round(a * FPS)), int(round(b * FPS))


def wtxt(w):
    pre, gap, a, b = w
    a0 = min(a, 0) - gap
    return f"baseline {a0 - pre:g}..{a0:g} s, response {a:g}..{b:g} s"


def win_mean(s, idx, k0, k1):
    """mean z over rows idx + k0 ... idx + k1 - 1 (NaN frames ignored; NaN if < 60 % valid)"""
    n = len(s["fr"])
    a, b = idx + k0, idx + k1
    out = np.full((len(idx), s["Zc"].shape[1]), np.nan)
    v = (a >= 0) & (b <= n) & (b > a)
    if v.any():
        sm = s["Zc"][b[v]] - s["Zc"][a[v]]
        nn = (s["Nc"][b[v]] - s["Nc"][a[v]]).astype(float)
        out[v] = sm / np.where(nn >= 0.6 * (k1 - k0), nn, np.nan)
    return out


def responses(s, frames, pre, gap, a, b):
    """trials x cells: mean z in the response window minus mean z in the baseline window (baseline after the first 15 s)"""
    p0, p1, k0, k1 = win_rows(pre, gap, a, b)
    idx, f = ev_rows(s, frames)
    keep = (f + p0) / FPS >= CA.SKIP_S
    idx, f = idx[keep], f[keep]
    R = win_mean(s, idx, k0, k1) - win_mean(s, idx, p0, p1)
    good = np.isfinite(R).mean(1) > 0.5 if len(R) else np.zeros(0, bool)
    return R[good], f[good]


def wsr_vec(R):
    """Wilcoxon signed-rank test of every column (normal approximation; NaN and zeros dropped): two-sided p, sign"""
    X = np.where(np.isfinite(R), R, 0.0)
    nz = X != 0
    n = nz.sum(0).astype(float)
    rk = rankdata(np.abs(X), axis=0) - (~nz).sum(0)[None, :]
    Tp = np.where(X > 0, rk, 0).sum(0)
    var = n * (n + 1) * (2 * n + 1) / 24
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (Tp - n * (n + 1) / 4) / np.sqrt(var)
    p = 2 * norm.sf(np.abs(z))
    p[n < 5] = np.nan
    return p, np.sign(z)


def split_rep(R):
    """% of cells responsive (p < 0.05) with the same sign on the odd AND on the even trials"""
    if len(R) < 10:
        return np.nan
    p1, s1 = wsr_vec(R[0::2])
    p2, s2 = wsr_vec(R[1::2])
    return float(np.mean((p1 < 0.05) & (p2 < 0.05) & (s1 == s2)))


def shift_train(s, frames, rng, min_s=15.0):
    """the event train shifted circularly inside each schedule segment: same number, spacing and block, timing destroyed"""
    f = np.sort(np.asarray(frames, int))
    bnd = np.array([int(x * FPS) for x in SEG_S] + [int(s["fr"][-1]) + 1])
    out = f.copy()
    seg = np.searchsorted(bnd, f, side="right") - 1
    for k in np.unique(seg):
        if k < 0 or k + 1 >= len(bnd):
            continue
        a0, a1 = bnd[k], bnd[k + 1]
        L = a1 - a0
        m = int(min(min_s * FPS, L // 2))
        if L < 2 * m or m < 1:
            continue
        off = int(rng.integers(m, L - m + 1))
        msk = seg == k
        out[msk] = a0 + (f[msk] - a0 + off) % L
    return np.sort(out)


def jitter_train(frames, rng, lo_s=2.0, hi_s=5.0):
    """every event moved by 2-5 s, earlier or later at random: keeps the block, the state and the stimulus density,
    destroys the timing at the scale of seconds"""
    f = np.asarray(frames, int)
    j = rng.uniform(lo_s, hi_s, len(f)) * rng.choice([-1.0, 1.0], len(f))
    return np.sort(f + np.round(j * FPS).astype(int))


def contamination(s, frames, pre, gap, a, b):
    """% of trials with ANOTHER stimulus delivery inside [baseline start, response end]"""
    dl = deliveries(s)
    f = np.asarray(frames, int)
    if not len(f) or not len(dl):
        return np.nan
    p0, _, _, k1 = win_rows(pre, gap, a, b)
    c = 0
    for x in f:
        o = dl[(dl >= x + p0) & (dl <= x + k1)]
        c += np.any(np.abs(o - x) > 3)
    return c / len(f)


# ------------------------------------------------------------------------------------------------ step: window
def step_window(S):
    rng = np.random.default_rng(7)
    rows = []
    t0 = time.time()
    for (m, d, ph), s in S.items():
        for ev in SEARCH_EV:
            frames = np.asarray(s["B"]["ev"].get(ev, []), int)
            if len(frames) < 10:
                continue
            sh_blk = [shift_train(s, frames, rng) for _ in range(5)]
            sh_jit = [jitter_train(frames, rng) for _ in range(5)]
            for pre in PRES:
                for gap in GAPS:
                    for a, b in POSTS:
                        R, f = responses(s, frames, pre, gap, a, b)
                        if len(R) < 10:
                            continue
                        rr = split_rep(R)
                        rb = [split_rep(responses(s, sh, pre, gap, a, b)[0]) for sh in sh_blk]
                        rj = [split_rep(responses(s, sh, pre, gap, a, b)[0]) for sh in sh_jit]
                        p, _ = wsr_vec(R)
                        rows.append(dict(mouse=m, day=d, phase=ph, condition=cond_of(m, d, ph), event=ev,
                                         family="stim" if ev in STIM else "beh", pre=float(pre), gap=float(gap), a=float(a),
                                         b=float(b), n_trials=len(R), rep_real=rr, rep_jitter=np.nanmean(rj),
                                         rep_block=np.nanmean(rb), quality=rr - np.nanmean(rj), slow=rr - np.nanmean(rb),
                                         resp_bh=float(np.mean(bh(p) <= 0.2)), contamination=contamination(s, f, pre, gap, a, b)))
        print(f"  window search {m} d{d} {ph}: {time.time() - t0:.0f} s", flush=True)
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(OD, "window_search.csv"), index=False)
    return T


STRATS = ["default (2 s / 0-2 s)", "one per family", "one per event"]
CLEAN_PRE, CLEAN_END = 3.0, 3.0


def is_clean(T):
    return (T.pre <= CLEAN_PRE) & (T.b <= CLEAN_END)


def choose_windows(T):
    """leave-one-mouse-out check of three ways to choose the window among the CLEAN windows; the winner chooses the final
    windows on all mice (the same window on both days). The same check allowing every window (10-s ones too) is reported."""
    key = ["pre", "gap", "a", "b"]

    def best(G):
        g = G.groupby(key).quality.mean()
        return tuple(float(x) for x in g.idxmax())

    def pick(tr, strat, ev, fa):
        if strat == STRATS[0]:
            return DEFAULT
        if strat == STRATS[1] or ev is None:
            return best(tr[tr.family == fa])
        return best(tr[tr.event == ev])
    res = []
    for wname, F in (("clean", T[is_clean(T)]), ("all windows", T)):
        for held in MO:
            tr = F[F.mouse != held]
            for st in STRATS:
                q = []
                for ev, g in T[T.mouse == held].groupby("event"):
                    w = pick(tr, st, ev, g.family.iloc[0])
                    r = g[(g.pre == w[0]) & (g.gap == w[1]) & (g.a == w[2]) & (g.b == w[3])]
                    q += list(r.quality)
                res.append(dict(windows=wname, held_out=held, strategy=st, quality=float(np.nanmean(q)) if q else np.nan))
    L = pd.DataFrame(res)
    L.to_csv(os.path.join(OD, "window_leave_one_mouse_out.csv"), index=False)
    Lc = L[L.windows == "clean"]
    win = Lc.groupby("strategy").quality.mean().idxmax()
    Tc = T[is_clean(T)]
    W = {}
    for ev in EVENTS:
        fa = "stim" if ev in STIM else "beh"
        w = pick(Tc, win, ev if ev in SEARCH_EV else None, fa)
        W[ev] = {1: w, 2: w}
    ws = os.environ.get("CA_WIN_SET", "")                     # 10-24: a window per event, JSON {event: [pre, gap, a, b]}
    if ws:
        for ev, w in json.loads(ws).items():
            if ev in W:
                W[ev] = {1: tuple(float(x) for x in w), 2: tuple(float(x) for x in w)}
        win = win + " (windows set per event: CA_WIN_SET)"
    fs = os.environ.get("CA_STIM_WIN", "")                    # 10-23 sensitivity: force the stimulus window, e.g. "0.5,0,0,1"
    if fs:
        w = tuple(float(x) for x in fs.split(","))
        for ev in STIM:
            W[ev] = {1: w, 2: w}
        win = win + f" (stimuli forced to {w})"
    return W, L, win


def fig_window(T, W, L, win, col="quality", fn="W1_window_search.png"):
    G = T.groupby(["event", "pre", "gap", "a", "b"]).agg(q=(col, "mean")).reset_index()
    ROWS = [(0.0, p_) for p_ in PRES]                          # baseline ending at the event (the gap-1 s rows are in the csv)
    fig = plt.figure(figsize=(18, 8.6))
    gs = fig.add_gridspec(2, 4, hspace=0.62, wspace=0.3)
    lab = [f"{a:g}..{b:g}" for a, b in POSTS]
    for k, ev in enumerate(SEARCH_EV):
        a = fig.add_subplot(gs[k // 4, k % 4])
        g = G[G.event == ev]
        M = np.full((len(ROWS), len(POSTS)), np.nan)
        for i, (g_, pre) in enumerate(ROWS):
            for j, (x0, x1) in enumerate(POSTS):
                q = g[(g.pre == pre) & (g.gap == g_) & (g.a == x0) & (g.b == x1)]
                if len(q):
                    M[i, j] = 100 * q.q.iloc[0]
        im = a.imshow(M, cmap="viridis", aspect="auto", vmin=0, vmax=max(1, np.nanmax(M)))
        for i, (g_, pre) in enumerate(ROWS):
            for j, (x0, x1) in enumerate(POSTS):
                if pre > CLEAN_PRE or x1 > CLEAN_END:
                    a.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fc="w", ec="none", alpha=0.55))
        a.set_xticks(range(len(POSTS)), lab, rotation=90, fontsize=9.5)
        a.set_yticks(range(len(ROWS)), [f"{p_:g} s" for g_, p_ in ROWS], fontsize=10.5)
        if k % 4 == 0:
            a.set_ylabel("baseline length (s)")
        a.set_xlabel("response window (s from the event)", fontsize=10)
        if col == "quality":
            for d, colr, ls in ((1, "#ff2d2d", "-"), (2, "#ffffff", "--")):
                w = W[ev][d]
                if (w[1], w[0]) in ROWS and (w[2], w[3]) in POSTS:
                    a.add_patch(plt.Rectangle((POSTS.index((w[2], w[3])) - 0.5, ROWS.index((w[1], w[0])) - 0.5), 1, 1, fill=False,
                                              ec=colr, lw=2.6, ls=ls))
        a.plot(POSTS.index((0, 2)), ROWS.index((0.0, 2)), "x", color="w", ms=8, mew=2)
        a.set_title(SHORT[ev], fontsize=13)
        cb = fig.colorbar(im, ax=a, shrink=0.8, pad=0.02)
        cb.ax.tick_params(labelsize=9.5)
        if k % 4 == 3:                                         # 10-24: say which sham and the unit
            cb.set_label("% called: real minus\n" + ("jitter" if col == "quality" else "block") + " sham (points)", fontsize=10)
    if col != "quality":
        fig.savefig(os.path.join(OD, fn), dpi=125, bbox_inches="tight")
        plt.close(fig)
        return
    a = fig.add_subplot(gs[1, 3])
    Lc = L[L.windows == "clean"]
    La = L[(L.windows == "all windows") & (L.strategy == STRATS[2])]
    mu = [100 * Lc[Lc.strategy == s_].quality.mean() for s_ in STRATS] + [100 * La.quality.mean()]
    a.bar(range(4), mu, color=["#2ecc40" if s_ == win else "0.8" for s_ in STRATS] + ["#f5b041"], width=0.62, zorder=0)
    for m in MO:
        y = [100 * Lc[(Lc.held_out == m) & (Lc.strategy == s_)].quality.iloc[0] for s_ in STRATS] + [100 * La[La.held_out == m].quality.iloc[0]]
        a.plot(range(4), y, "-", marker=MK[m], color=MC[m], lw=1.2, label=m)
    a.set_xticks(range(4), ["old\n2 / 0-2", "one per\nfamily", "one per\nevent", "10-s windows\nallowed"], fontsize=9.5)
    a.set_ylabel("quality in the held-out mouse (%)")
    a.set_title("chosen on 3 mice, tested on the 4th", fontsize=12)
    a.legend(frameon=False, fontsize=8.5, ncol=2)
    fig.savefig(os.path.join(OD, fn), dpi=125, bbox_inches="tight")
    plt.close(fig)


def fig_named_windows(T):
    """the windows you named: symmetric 1/1, 2/2, 3/3, 5/5, 10/10 s - quality and contamination"""
    sym = [(1, 0, 1), (2, 0, 2), (3, 0, 3), (5, 0, 5), (10, 0, 10)]
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.9))
    out = {}
    ecol = {"withdrawal": "#2c3e50", "flinch": "#7f8c8d", "movement onset": "#16a085"}
    for ev in SEARCH_EV:
        g = T[(T.event == ev) & (T.gap == 0)]
        sel = lambda p, x, y: g[(g.pre == p) & (g.a == x) & (g.b == y)]
        q = [100 * sel(*w).quality.mean() for w in sym]
        sl = [100 * sel(*w).slow.mean() for w in sym]
        c = [100 * sel(*w).contamination.mean() for w in sym]
        col = SCOL.get(ev, ecol.get(ev))
        ax[0].plot(range(5), q, "-o", color=col, lw=2, label=SHORT[ev])
        ax[1].plot(range(5), sl, "-o", color=col, lw=2)
        ax[2].plot(range(5), c, "-o", color=col, lw=2)
        out[ev] = dict(locked=q, slow=sl, contamination=c)
    # 10-24 figure review: 'real - jitter sham (%)' was cryptic -> what is subtracted and its unit; panel B gets its own legend
    # (the slide shows A alone and B + C together); A and B get head room so the legend sits above the lines
    for a, yl, t in ((ax[0], "% neurons called: real minus\njitter sham (percentage points)", "A  locked to the event (the choice)"),
                     (ax[1], "% neurons called: real minus\nblock sham (percentage points)", "B  any change, incl. slow ones"),
                     (ax[2], "% trials with another delivery\ninside the window", "C  the next stimulus falls inside")):
        a.set_xticks(range(5), ["1/1", "2/2", "3/3", "5/5", "10/10"])
        a.set_xlabel("window: s before / s after the event")
        a.set_ylabel(yl)
        a.set_title(t, loc="left")
    for a in ax[:2]:
        lo_, hi_ = a.get_ylim()
        a.set_ylim(lo_, hi_ + 0.5 * (hi_ - lo_))
        a.legend(*ax[0].get_legend_handles_labels(), frameon=False, fontsize=10.5, ncol=2, loc="upper right",
                 columnspacing=1.0, handlelength=1.6)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(os.path.join(OD, "W2_named_windows.png"), dpi=140)
    plt.close(fig)
    return out


def fig_timing(S, W):
    """when do the responses happen? Day 1 delivery marks = the hand-marked key taps, Day 2 = the pressure logger. Mean trace of the
    neurons activated on the ODD trials (the chosen window, p < 0.01), plotted on the EVEN trials (no selection bias)"""
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.8))
    # 10-23 (figure review): deliveries that the timing fix moved to 2 frames before their own reflex have that latency BY
    # CONSTRUCTION; panel A shows only the measured ones (deliveries left where they were), the moved ones are counted apart
    moved = set()
    cf = os.path.join(BASE, "scoring_mini1p", "timefix_changes.csv")
    if SCORING_SFX and os.path.isfile(cf):
        C_ = pd.read_csv(cf)
        C_ = C_[C_.change.str.startswith("moved")]
        moved = {(str(r.mouse), int(r.day), str(r.phase), int(r.new_frame)) for r in C_.itertuples()}
    lat, nmov = {1: [], 2: []}, {1: 0, 2: 0}
    for (m, d, ph), s in S.items():
        wd = np.asarray(s["B"]["ev"]["withdrawal"], int)
        for x in deliveries(s):
            w = wd[(wd >= x) & (wd <= x + 2 * FPS)]
            if not len(w):
                continue
            if (m, d, ph, int(x)) in moved:
                nmov[d] += 1
            else:
                lat[d].append((w[0] - x) / FPS)
    a = ax[0]
    for d, col in ((1, "#e67e22"), (2, "#2471a3")):
        v = np.array(lat[d])
        a.hist(v, bins=np.arange(0, 2.01, 1 / 15), histtype="step", lw=2.2, color=col,
               label=f"Day {d} ({'key marks' if d == 1 else 'clicker'}): median {np.median(v):.2f} s, n = {len(v)}"
                     f"\n(+{nmov[d]} set to 0.07 s by the 10-23 fix, not shown)")
    a.set_xlabel("withdrawal after the delivery mark (s)")
    a.set_ylabel("deliveries (measured)")
    a.set_title("A  delivery -> withdrawal (measured)", loc="left")
    a.set_ylim(0, a.get_ylim()[1] * 1.45)
    a.legend(frameon=True, framealpha=0.9, edgecolor="none", fontsize=8.5, loc="upper right")
    out = dict(wd_latency_median={d: float(np.median(lat[d])) for d in (1, 2)},
               wd_within_05={d: float(np.mean(np.array(lat[d]) <= 0.5)) for d in (1, 2)}, n_lat={d: len(lat[d]) for d in (1, 2)},
               n_set_by_fix={d: nmov[d] for d in (1, 2)})
    for k, (evs, ttl, nm) in enumerate(((STIM, "B  any stimulus delivery", "stim"), (["withdrawal"], "C  withdrawal", "wd"),
                                        (["movement onset"], "D  movement onset", "move"))):
        a = ax[k + 1]
        for d, col in ((1, "#e67e22"), (2, "#2471a3")):
            Y = []
            tt = None
            for (m, dd, ph), s in S.items():
                if dd != d:
                    continue
                for ev in evs:
                    X, tt_, used = CA.peri(s, s["B"]["ev"].get(ev, []))
                    if X is None or len(used) < 10:
                        continue
                    tt = tt_
                    pre_, gap_, a_, b_ = W[ev][d]
                    a0 = min(a_, 0) - gap_
                    pre = (tt >= a0 - pre_) & (tt < a0)
                    Xs = X - np.nanmean(X[:, pre, :], 1, keepdims=True)
                    R = np.nanmean(Xs[:, (tt >= a_) & (tt < b_), :], 1)
                    p, sg = wsr_vec(R[0::2])
                    sel = (p < 0.01) & (sg > 0)
                    Xb = X - np.nanmean(X[:, (tt >= -3) & (tt < 0), :], 1, keepdims=True)
                    if sel.any():
                        Y.append(np.nanmean(Xb[1::2][:, :, sel], 0).T)
            if not Y:
                continue
            Y = np.vstack(Y)
            mu = CA.sm(np.nanmean(Y, 0)[:, None], 4)[:, 0]
            se = CA.sm((np.nanstd(Y, 0) / np.sqrt(len(Y)))[:, None], 4)[:, 0]
            a.plot(tt, mu, color=col, lw=2.2, label=f"Day {d} (n = {len(Y)})")
            a.fill_between(tt, mu - se, mu + se, color=col, alpha=0.2, lw=0)
            r = (tt >= -1) & (tt <= 4)
            tpk = float(tt[r][np.argmax(mu[r])])
            half = mu[r].max() / 2
            above = np.flatnonzero((mu[r] >= half) & (tt[r] >= -1))
            out[f"peak_{nm}_d{d}"] = tpk
            out[f"halfrise_{nm}_d{d}"] = float(tt[r][above[0]]) if len(above) else np.nan
            if nm == "move":
                a.axvline(tpk, color=col, lw=0.9, ls=":", label=f"peak Day {d} ({tpk:.2f} s)")
        a.axvline(0, color="k", lw=0.8, ls="--")
        a.axhline(0, color="0.6", lw=0.6)
        a.set_xlim(-3, 6)
        a.set_xlabel("time from the event mark (s)")
        a.set_title(ttl, loc="left")
        a.legend(frameon=False, fontsize=9.5)
        a.set_ylabel("z (baseline -3..0 s), mean +- SEM\nactivated neurons, held-out trials" if k == 0 else "z (baseline -3..0 s), mean +- SEM")
    fig.tight_layout()
    fig.savefig(os.path.join(OD, "W3_timing.png"), dpi=140)
    plt.close(fig)
    return out


# ------------------------------------------------------------------------------------------------ step: classify
def shift_p(s, frames, pre, gap, a, b, n_shuf=1000, seed=0):
    """within-segment time-shift test (as CA.shift_null) with a free window: observed mean response and two-sided p"""
    fr = s["fr"]
    Zc, Nc = s["Zc"], s["Nc"]
    n = len(fr)
    p0, p1, k0, k1 = win_rows(pre, gap, a, b)
    lo = int(np.searchsorted(fr, int(CA.SKIP_S * FPS))) - p0
    idx, _ = ev_rows(s, frames)
    idx = idx[(idx >= lo) & (idx + k1 < n)]
    if len(idx) < CA.MIN_TRIALS:
        return None, None, 0

    def resp(ii):
        ii = ii[(ii >= lo) & (ii + k1 < n)]
        x = (Zc[ii + k1] - Zc[ii + k0]) / np.maximum(Nc[ii + k1] - Nc[ii + k0], 1)
        y = (Zc[ii + p1] - Zc[ii + p0]) / np.maximum(Nc[ii + p1] - Nc[ii + p0], 1)
        return np.nanmean(x - y, 0)
    obs = resp(idx)
    bnd = np.unique(np.searchsorted(fr, np.array([int(x * FPS) for x in SEG_S] + [int(fr[-1]) + 1])))
    seg = np.searchsorted(bnd, idx, side="right") - 1
    rng = np.random.default_rng(seed)
    cnt = np.zeros(Zc.shape[1])
    w = max(k1, -p0)
    for _ in range(n_shuf):
        ii = idx.copy()
        for k in np.unique(seg):
            if k < 0 or k + 1 >= len(bnd):
                continue
            A0, A1 = bnd[k], bnd[k + 1]
            L = A1 - A0
            if L <= 2 * w:
                continue
            msk = seg == k
            ii[msk] = A0 + (idx[msk] - A0 + int(rng.integers(w, L - w))) % L
        cnt += np.abs(resp(np.sort(ii))) >= np.abs(obs)
    return obs, (1 + cnt) / (1 + n_shuf), len(idx)


def step_classify(S, W):
    rows = []
    for (m, d, ph), s in S.items():
        for ev in EVENTS:
            pre, gap, a, b = W[ev][d]
            obs, p, nt = shift_p(s, s["B"]["ev"].get(ev, []), pre, gap, a, b)
            if obs is None:
                continue
            q = bh(p)
            cls = np.where(q <= CA.Q_RESP, np.where(obs > 0, "activated", "inhibited"), "non-responsive")
            for j, c in enumerate(s["ids"]):
                rows.append(dict(mouse=m, day=d, phase=ph, condition=cond_of(m, d, ph), event=ev, cell=int(c), n_trials=nt,
                                 pre=pre, gap=gap, a=a, b=b, resp=float(obs[j]), auc=float(obs[j] * (b - a)), p=float(p[j]),
                                 q=float(q[j]), cls=cls[j]))
        print(f"  classified {m} d{d} {ph}", flush=True)
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(OD, "cells_classes.csv"), index=False)
    return T


# ------------------------------------------------------------------------------------------------ step: overlap
def boxcar(nb, on, n_box=6, w=5):
    """0.5-s boxcars 0-3 s after each onset bin (10-Hz bins); 6 weights per event type (15-70 events per session)"""
    X = np.zeros((nb, n_box))
    for k in range(n_box):
        for j in range(w):
            ii = on + k * w + j
            ii = ii[(ii >= 0) & (ii < nb)]
            np.add.at(X[:, k], ii, 1.0)
    return X


def seg_roll(Xc, seg, rng, min_bins=50):
    Y = Xc.copy()
    for k in np.unique(seg):
        r = np.flatnonzero(seg == k)
        L = len(r)
        if L < 2 * min_bins + 1:
            continue
        Y[r] = np.roll(Xc[r], int(rng.integers(min_bins, L - min_bins)), axis=0)
    return Y


FAMILIES = {"stimulus": STIM, "reflex": ["withdrawal", "flinch"],
            "movement": ["movement onset", "escape onset", "movement state", "grooming state"]}
FAMILIES["stimulus or reflex"] = FAMILIES["stimulus"] + FAMILIES["reflex"]


def glm_session(s, n_null=100, seed=0):
    ev = s["B"]["ev"]
    Z = np.asarray(s["T"]["z"], float)
    fr = s["fr"]
    t0 = int(np.searchsorted(fr, int(CA.SKIP_S * FPS)))
    Z, frs = Z[t0:], fr[t0:]
    nb = len(Z) // 3
    Zb = np.nanmean(Z[:nb * 3].reshape(nb, 3, -1), 1)
    fb = frs[:nb * 3:3]

    def bins(frames):
        f = np.asarray(frames, int)
        f = f[(f >= fb[0]) & (f <= fb[-1] + 2)]
        return np.searchsorted(fb, f, side="right") - 1
    G = {k: boxcar(nb, bins(ev.get(k, []))) for k in STIM}
    G["withdrawal"] = boxcar(nb, bins(ev["withdrawal"]))
    G["flinch"] = boxcar(nb, bins(ev["flinch"]))
    G["movement onset"] = boxcar(nb, bins(ev["movement onset"]))
    G["escape onset"] = boxcar(nb, bins(ev["escape/rearing onset"]))
    mv = np.asarray(s["B"]["st"]["move"], float)
    gr = np.asarray(s["B"]["st"]["groom"], float)
    G["movement state"] = mv[np.clip(fb, 0, len(mv) - 1)][:, None]
    G["grooming state"] = gr[np.clip(fb, 0, len(gr) - 1)][:, None]
    tb = fb / FPS
    sg = np.searchsorted(np.array(SEG_S + [1e9]), tb, side="right") - 1
    Nu = []
    for k in np.unique(sg):
        Nu += [(sg == k).astype(float), np.where(sg == k, tb - tb[sg == k].min(), 0.0)]
    G["_nuisance"] = np.column_stack(Nu)
    names = list(G)
    X = np.hstack([G[g] for g in names])
    gid = np.concatenate([[i] * G[g].shape[1] for i, g in enumerate(names)])
    good = np.isfinite(Zb).mean(1) > 0.9
    X, Y, sgr = X[good], Zb[good], sg[good]
    cm = np.nanmean(Y, 0)
    Y = np.where(np.isfinite(Y), Y, cm[None, :])
    Y = Y - Y.mean(0)
    sd = X.std(0)
    sd[sd == 0] = 1
    X = (X - X.mean(0)) / sd
    fold = (np.arange(len(Y)) // 200) % 3
    yy = np.maximum((Y ** 2).sum(0), 1e-12)

    def cv_r2(Xm, lam):
        pred = np.zeros_like(Y)
        for k in range(3):
            tr, te = fold != k, fold == k
            A = Xm[tr]
            Wt = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ Y[tr])
            pred[te] = Xm[te] @ Wt
        return 1 - ((Y - pred) ** 2).sum(0) / yy
    lams = [300.0, 3000.0, 30000.0]
    r_l = [np.median(cv_r2(X, l_)) for l_ in lams]
    lam = lams[int(np.argmax(r_l))]
    r_full = cv_r2(X, lam)
    rng = np.random.default_rng(seed)
    out = dict(r2_full=r_full, lam=lam)
    for fam, mem in FAMILIES.items():
        cols = np.isin(gid, [names.index(g) for g in mem])
        r_red = cv_r2(X[:, ~cols], lam)
        uv = r_full - r_red
        null = np.zeros((n_null, Y.shape[1]))
        for i in range(n_null):
            Xr = X.copy()
            Xr[:, cols] = seg_roll(X[:, cols], sgr, rng)
            null[i] = cv_r2(Xr, lam) - r_red
        out[f"uv_{fam}"] = uv
        out[f"p_{fam}"] = (1 + (null >= uv[None, :]).sum(0)) / (1 + n_null)
    return out


TYPES = ["stimulus", "reflex", "stimulus + reflex", "stimulus or reflex (cannot separate)", "movement only", "none"]
TYPC = {"stimulus": "#8e44ad", "reflex": "#e67e22", "stimulus + reflex": "#c0392b",
        "stimulus or reflex (cannot separate)": "#f1c40f", "movement only": "#16a085", "none": "#dfe3e6"}


def stack_barh(ax, cnt, types, colors, fs=11, thin=6.0, h=0.62):
    """10-24 (figure review: thin segments carried no value): horizontal stacked bars, one per row of cnt (counts; columns =
    types), in % of the row (x axis 0-100). The value is written inside every segment >= thin %; a thinner non-zero segment
    gets its value just ABOVE its own bar, spread sideways when neighbours are close, with a leader line in the segment's
    colour. Sets the y limits with row 0 on top (the axis comes out inverted: do not invert it again)."""
    tot = cnt.sum(1).values.astype(float)
    left = np.zeros(len(cnt))
    thin_ = [[] for _ in range(len(cnt))]                   # per bar: (segment centre, text, colour)
    for t in types:
        if t not in cnt:
            continue
        v = 100 * cnt[t].values / np.where(tot > 0, tot, np.nan)
        v = np.nan_to_num(v)
        ax.barh(range(len(cnt)), v, left=left, height=h, color=colors[t], edgecolor="w")
        for i, (l_, w_) in enumerate(zip(left, v)):
            if w_ >= thin:
                ax.text(l_ + w_ / 2, i, f"{w_:.0f}", ha="center", va="center", fontsize=fs)
            elif w_ > 0:
                c_ = matplotlib.colors.to_rgb(colors[t])
                c_ = tuple(0.65 * x for x in c_) if sum(c_) > 2.4 else c_            # a very light segment: darker line
                thin_[i].append((l_ + w_ / 2, f"{w_:.1f}" if w_ < 1 else f"{w_:.0f}", c_))
        left += v
    fig = ax.figure
    ax_w = max(ax.get_position().width * fig.get_figwidth(), 1.0)                # axes width (in), approximate
    hw = lambda s: 100 * len(s) * 0.62 * fs / 72 / ax_w / 2                      # half a label's width (x units)
    for i, L in enumerate(thin_):
        L.sort(key=lambda r: r[0])
        pos = []
        for k, (xc, s, _) in enumerate(L):
            pos.append(xc if not pos else max(xc, pos[-1] + hw(L[k - 1][1]) + hw(s) + 1.0))
        if pos and pos[-1] + hw(L[-1][1]) > 101:                                 # keep the row inside the axis
            sh = pos[-1] + hw(L[-1][1]) - 101
            pos = [p - sh for p in pos]
        for (xc, s, c_), xp in zip(L, pos):
            e0 = i - h / 2                                                       # top edge of the bar (inverted axis)
            ax.plot([xc, xc, xp], [e0, e0 - 0.05, e0 - 0.1], color=c_, lw=1.4, solid_capstyle="butt", clip_on=False)
            ax.text(xp, e0 - 0.11, s, ha="center", va="bottom", fontsize=fs, clip_on=False)
    ax.set_ylim(len(cnt) - 1 + h / 2 + 0.12, -h / 2 - 0.36)


def neuron_type(r):
    S_, R_, SR_, M_ =(r["sig_stimulus"], r["sig_reflex"], r["sig_stimulus or reflex"], r["sig_movement"])
    if S_ and R_:
        return "stimulus + reflex"
    if S_:
        return "stimulus"
    if R_:
        return "reflex"
    if SR_:
        return "stimulus or reflex (cannot separate)"
    if M_:
        return "movement only"
    return "none"


def step_overlap(S, T, W):
    rows_split, rows_glm = [], []
    t0 = time.time()
    for (m, d, ph), s in S.items():
        ev = s["B"]["ev"]
        wd = np.asarray(ev["withdrawal"], int)
        X = T[(T.mouse == m) & (T.day == d) & (T.phase == ph)]
        dl = deliveries(s)
        for k in STIM:
            f = np.asarray(ev.get(k, []), int)
            if len(f) < 10:
                continue
            has = []
            for x in f:
                nx = dl[dl > x + 3]
                hi = min(x + int(2 * FPS), nx[0] - 1) if len(nx) else x + int(2 * FPS)
                has.append(bool(np.any((wd >= x) & (wd <= hi))))
            has = np.array(has)
            if has.sum() < 4 or (~has).sum() < 4:
                continue
            pre, gap, a, b = W[k][d]
            Rw, _ = responses(s, f[has], pre, gap, a, b)
            Rn, _ = responses(s, f[~has], pre, gap, a, b)
            if len(Rw) < 4 or len(Rn) < 4:
                continue
            cl = X[X.event == k].set_index("cell").cls
            for j, c in enumerate(s["ids"]):
                rows_split.append(dict(mouse=m, day=d, phase=ph, condition=cond_of(m, d, ph), stim=k, cell=int(c),
                                       cls=cl.get(int(c), "n.a."), with_wd=float(np.nanmean(Rw[:, j])),
                                       without_wd=float(np.nanmean(Rn[:, j])), n_with=int(len(Rw)), n_without=int(len(Rn))))
        g = glm_session(s)
        qs = {fam: bh(g[f"p_{fam}"]) for fam in FAMILIES}
        for j, c in enumerate(s["ids"]):
            r = dict(mouse=m, day=d, phase=ph, condition=cond_of(m, d, ph), cell=int(c), r2_full=float(g["r2_full"][j]))
            for fam in FAMILIES:
                r[f"uv_{fam}"] = float(g[f"uv_{fam}"][j])
                r[f"p_{fam}"] = float(g[f"p_{fam}"][j])
                r[f"sig_{fam}"] = bool(qs[fam][j] <= CA.Q_RESP and g[f"uv_{fam}"][j] > 0)
            rows_glm.append(r)
        print(f"  overlap {m} d{d} {ph}: {time.time() - t0:.0f} s (ridge lambda {g['lam']:g})", flush=True)
    Sp = pd.DataFrame(rows_split)
    Gm = pd.DataFrame(rows_glm)
    Gm["type"] = Gm.apply(neuron_type, axis=1)
    Gm["also_movement"] = Gm.sig_movement & Gm.type.isin(TYPES[:4])
    Sp.to_csv(os.path.join(OD, "O2_with_without_withdrawal.csv"), index=False)
    Gm.to_csv(os.path.join(OD, "O3_encoding_model.csv"), index=False)
    return Sp, Gm


def fig_overlap(Sp, Gm):
    """O1: the 18-in figure (A-D) and, for slides (10-22: fonts were at 84 %), two parts: _p1 = A + B, _p2 = C + D + legend"""
    out = {}

    def panel_a(a, fs=10, lfs=9):
        # 10-24 figure review: the result text and the line labels sat on the data -> the result goes above the axes, both
        # reference lines go into the legend, the legend goes under the axes
        X = Sp[Sp.cls == "activated"]
        for d, col in ((1, "#e67e22"), (2, "#2471a3")):
            Y = X[X.day == d]
            a.scatter(Y.without_wd, Y.with_wd, s=13, color=col, alpha=0.55, label=f"Day {d} (n = {len(Y)} pairs)")
        sub = ""
        if len(X) > 10:
            v = np.r_[X.with_wd, X.without_wd]
            mx, mn = np.nanpercentile(v, 99), min(0, np.nanpercentile(v, 1))
            a.plot([mn, mx], [mn, mx], "k--", lw=1, label="equal (with = without)")
            a.plot([mn / 2, mx / 2], [mn, mx], color="#c0392b", ls=":", lw=1.8, label="without = half of with")
            a.set_xlim(mn, mx)
            a.set_ylim(mn, mx)
            p = wilcoxon(X.with_wd, X.without_wd).pvalue
            keep = float(np.mean(X.without_wd > 0.5 * X.with_wd))
            out["split"] = dict(n=int(len(X)), p=float(p), frac_keep_half=keep, median_with=float(X.with_wd.median()),
                                median_without=float(X.without_wd.median()),
                                by_day={d: dict(n=int((X.day == d).sum()), keep=float(np.mean(X[X.day == d].without_wd > 0.5 * X[X.day == d].with_wd)))
                                        for d in (1, 2) if (X.day == d).sum()})
            sub = (f"{100 * keep:.0f}% of pairs keep >= half (right of the red line)\n"
                   f"with vs without a withdrawal: {star(p) or 'ns'} (Wilcoxon)")
            a.text(0, 1.02, sub, transform=a.transAxes, va="bottom", ha="left", fontsize=fs, linespacing=1.15)
        a.set_xlabel("response when NO withdrawal follows (z)")
        a.set_ylabel("response when a withdrawal follows (z)")
        a.set_title("A  stimulus-activated neuron x stimulus pairs", loc="left", pad=2.9 * fs + 6 if sub else 6)
        a.legend(frameon=False, fontsize=lfs, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, columnspacing=1.2,
                 handletextpad=0.4)

    def panel_b(b, lfs=9):
        # 10-23 review: the axes ran to -10 % and squeezed the cloud; now 1st-99.5th percentile, the rest on the frame
        xs_, ys_ = 100 * Gm.uv_stimulus.values, 100 * Gm.uv_reflex.values
        x0_, x1_ = min(-1.0, np.nanpercentile(xs_, 1) * 1.1), max(1.0, np.nanpercentile(xs_, 99.5) * 1.1)
        y0_, y1_ = min(-1.0, np.nanpercentile(ys_, 1) * 1.1), max(1.0, np.nanpercentile(ys_, 99.5) * 1.1)
        sig_ = (Gm.sig_stimulus.astype(bool) | Gm.sig_reflex.astype(bool)).values
        hb = []
        for d, col in ((1, "#e67e22"), (2, "#2471a3")):
            k_ = (Gm.day == d).values
            xc, yc = np.clip(xs_[k_], x0_, x1_), np.clip(ys_[k_], y0_, y1_)
            s_ = sig_[k_]
            b.scatter(xc[~s_], yc[~s_], s=9, facecolors="none", edgecolors=col, alpha=0.5, lw=0.6)
            b.scatter(xc[s_], yc[s_], s=16, color=col, alpha=0.85)
            hb.append(Line2D([], [], ls="", marker="o", ms=6, color=col, label=f"Day {d} (n = {int(k_.sum())})"))
        out_ = int(((xs_ < x0_) | (xs_ > x1_) | (ys_ < y0_) | (ys_ > y1_)).sum())
        hb += [Line2D([], [], ls="", marker="o", ms=6, color="0.4", label="filled = significant"),
               Line2D([], [], ls="", marker="o", ms=6, mfc="none", mec="0.4", label="open = not significant")]
        b.axhline(0, color="0.6", lw=0.6)
        b.axvline(0, color="0.6", lw=0.6)
        b.set_xlim(x0_, x1_)
        b.set_ylim(y0_, y1_)
        b.set_xlabel("variance ONLY the stimuli explain (%)")
        b.set_ylabel("variance ONLY withdrawal + flinch explain (%)")
        b.set_title("B  encoding model, every neuron", loc="left")
        # 10-24 figure review: the legend and the 'beyond the axes' note sat on the data -> both under the axes
        b.legend(handles=hb, frameon=False, fontsize=lfs, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2,
                 columnspacing=1.2, handletextpad=0.3, title=f"{out_} neurons beyond the axes are drawn on the frame",
                 title_fontsize=lfs - 1)

    def panel_cd(c, k, fs=10.5):
        grp, ttl, keys = (("day", "C  what each neuron follows", [1, 2]), ("condition", "D  same classes, per condition", COND))[k]
        cnt = Gm.groupby([grp, "type"]).size().unstack(fill_value=0).reindex(keys).fillna(0)
        stack_barh(c, cnt, TYPES, TYPC, fs=fs)                 # 10-24: thin segments get their value outside the bar
        nn_ = cnt.sum(1).astype(int).values
        # 10-25 figure review: Day 2 marks come from the hand clicker (the deck's word); one row of Gm = one neuron in one
        # session (a neuron recorded at baseline and after the injection counts twice) -> '% of neuron-sessions'
        c.set_yticks(range(len(cnt)), [(f"Day {x} ({'taps' if x == 1 else 'clicker'})" if grp == "day" else str(x)) + f"\nn = {nn_[i]}"
                                       for i, x in enumerate(cnt.index)])
        c.set_xlabel("% of neuron-sessions")
        c.set_xlim(0, 100)
        c.set_title(ttl, loc="left")
        out[f"types_by_{grp}"] = {str(i): {t: float(100 * cnt.loc[i, t] / cnt.loc[i].sum()) for t in cnt.columns} for i in cnt.index}

    fig = plt.figure(figsize=(18, 5.8))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1.2, 1.2], wspace=0.45, top=0.8)
    panel_a(fig.add_subplot(gs[0])); panel_b(fig.add_subplot(gs[1]))
    for k in (0, 1):
        panel_cd(fig.add_subplot(gs[2 + k]), k)
    fig.legend(handles=[Patch(color=TYPC[t], label=t) for t in TYPES], loc="upper center", ncol=3, frameon=False, fontsize=10,
               bbox_to_anchor=(0.74, 0.0))                 # under C + D (the legends of A and B sit under their own axes)
    fig.savefig(os.path.join(OD, "O1_near_simultaneous.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    with plt.rc_context({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13, "xtick.labelsize": 12, "ytick.labelsize": 12}):
        fig, ax = plt.subplots(1, 2, figsize=(12.6, 6.6))
        panel_a(ax[0], fs=12, lfs=12); panel_b(ax[1], lfs=12)
        fig.tight_layout(w_pad=2.5)
        fig.savefig(os.path.join(OD, "O1_near_simultaneous_p1.png"), dpi=140, bbox_inches="tight")
        plt.close(fig)
        fig, ax = plt.subplots(1, 2, figsize=(12.6, 5.6))
        for k in (0, 1):
            panel_cd(ax[k], k, fs=12.5)
        fig.legend(handles=[Patch(color=TYPC[t], label=t) for t in TYPES], loc="lower center", ncol=3, frameon=False, fontsize=12)
        fig.tight_layout(rect=(0, 0.14, 1, 1))
        fig.savefig(os.path.join(OD, "O1_near_simultaneous_p2.png"), dpi=140, bbox_inches="tight")
        plt.close(fig)
    return out


# ------------------------------------------------------------------------------------------------ plots
def pairs():
    P = {}
    for m in MO:
        f = os.path.join(BASE, "analysis", "crossday", f"{m}_pairs.csv")
        P[m] = pd.read_csv(f) if os.path.isfile(f) else pd.DataFrame(columns=["day1_id", "day2_id"])
    return P


def binned(s, bin_s=1.0):
    Z = np.asarray(s["T"]["z"], float)
    k = (s["fr"] / FPS // bin_s).astype(int)
    nb = k.max() + 1
    ok = np.isfinite(Z)
    sm = np.zeros((nb, Z.shape[1]))
    nn = np.zeros((nb, Z.shape[1]))
    np.add.at(sm, k, np.where(ok, Z, 0))
    np.add.at(nn, k, ok)
    return sm / np.where(nn > 0, nn, np.nan), np.arange(nb) * bin_s


def spans(v):
    v = np.asarray(v, bool).astype(int)
    return np.flatnonzero(np.diff(np.r_[0, v]) == 1), np.flatnonzero(np.diff(np.r_[v, 0]) == -1)


EVROWS = [("light", "Light touch"), ("mild", "Mild touch"), ("pin", "Pin prick"), ("heat", "Heat"), ("withdrawal", "withdrawal"),
          ("flinch", "flinch"), ("movement", "move"), ("grooming", "groom"), ("attending", "attending"),
          ("lick/bite", "lick/bite"), ("guarding", "guarding"), ("escape", "escape/rearing")]
EVROWC = {"light": "#5dade2", "mild": "#f5b041", "pin": "#e74c3c", "heat": "#8e44ad", "withdrawal": "#2c3e50",
          "flinch": "#7f8c8d", "movement": "#16a085", "grooming": "#d35400", "attending": "#c0392b", "lick/bite": "#e84393",
          "guarding": "#6c3483", "escape": "#2980b9"}


def event_panel(e, s, tmax, labels=True):
    ev, st, held = s["B"]["ev"], s["B"]["st"], s["held"]
    for yy, (lab, key) in enumerate(EVROWS):
        col = EVROWC[lab]
        if key in ev:
            e.vlines(np.asarray(ev[key]) / FPS / 60, yy + 0.1, yy + 0.9, color=col, lw=0.7)
        else:
            on, off = spans(st[key] if key in st else held[key])
            e.broken_barh(list(zip(on / FPS / 60, (off - on + 1) / FPS / 60)), (yy + 0.1, 0.8), color=col)
    e.set_ylim(len(EVROWS), 0)
    e.set_xlim(0, tmax)
    e.set_yticks(np.arange(len(EVROWS)) + 0.5, [l_ for l_, _ in EVROWS] if labels else [""] * len(EVROWS), fontsize=8.5)
    for b0, b1 in BLOCK_S:
        e.axvspan(b0 / 60, b1 / 60, color="#f5b041", alpha=0.08, lw=0)


def plot1_whole(S, P):
    """basic plot 1: every neuron over the four sessions of a mouse (one row = one neuron; grey = not there that day)"""
    files = []
    for m in MO:
        keys = [(m, 1, "pre"), (m, 1, "post"), (m, 2, "pre"), (m, 2, "post")]
        if not all(k in S for k in keys):
            continue
        Bn = {k: binned(S[k]) for k in keys}
        pr = P[m]
        id1, id2 = S[keys[0]]["ids"], S[keys[2]]["ids"]
        t1, t2 = set(pr.day1_id.astype(int)), set(pr.day2_id.astype(int))
        rows = [("both", int(a), int(b)) for a, b in zip(pr.day1_id, pr.day2_id)]
        rows += [("d1", int(c), None) for c in id1 if int(c) not in t1]
        rows += [("d2", None, int(c)) for c in id2 if int(c) not in t2]

        def col(k, cid):
            if cid is None:
                return np.full(len(Bn[k][1]), np.nan)
            i = np.flatnonzero(S[k]["ids"] == cid)
            return Bn[k][0][:, i[0]] if len(i) else np.full(len(Bn[k][1]), np.nan)
        M = {k: np.array([col(k, r[1] if k[1] == 1 else r[2]) for r in rows]) for k in keys}
        key = np.array([np.nanmean(np.r_[M[keys[0]][i], M[keys[2]][i]]) for i in range(len(rows))])
        cat = np.array([{"both": 0, "d1": 1, "d2": 2}[r[0]] for r in rows])
        o = np.lexsort((-np.nan_to_num(key, nan=-9), cat))
        def draw(ks, suf, figsize):
            fig = plt.figure(figsize=figsize)
            small = len(ks) == 2
            gs = fig.add_gridspec(2, len(ks) + 1, height_ratios=[4.0, 1.7], width_ratios=[0.05] + [1] * len(ks), hspace=0.06, wspace=0.05,
                                  left=0.07, right=0.88 if small else 0.93, top=0.84 if small else 0.9, bottom=0.08)
            cm = plt.get_cmap("magma").copy()
            cm.set_bad("#e3e3e3")
            a0 = fig.add_subplot(gs[0, 0])
            a0.imshow(np.array([[matplotlib.colors.to_rgb(C3[rows[i][0]])] for i in o]), aspect="auto", interpolation="nearest")
            a0.set_xticks([])
            a0.set_yticks([])
            a0.set_ylabel(f"{len(rows)} neurons (sorted by mean activity)", fontsize=12)
            im = None
            for c_, k in enumerate(ks):
                a = fig.add_subplot(gs[0, c_ + 1])
                tmax = Bn[k][1][-1] / 60
                im = a.imshow(np.ma.masked_invalid(M[k][o]), aspect="auto", cmap=cm, vmin=-0.5, vmax=3, interpolation="nearest",
                              extent=[0, tmax, len(rows), 0])
                cnd = cond_of(*k)
                a.set_title(f"Day {k[1]} {'baseline' if k[2] == 'pre' else 'after ' + cnd}", color=CC[cnd], fontsize=14, fontweight="bold")
                a.set_yticks([])
                a.set_xticks([])
                for b0, _ in BLOCK_S:
                    a.axvline(b0 / 60, color="w", lw=0.6, ls=":")
                e = fig.add_subplot(gs[1, c_ + 1])
                event_panel(e, S[k], tmax, labels=c_ == 0)
                e.set_xlabel("time in session (min)")
            cax = fig.add_axes([0.895 if small else 0.94, 0.45, 0.009, 0.36])
            fig.colorbar(im, cax=cax).set_label("z (dF/F / robust SD), 1-s bins")
            fig.legend(handles=[Patch(color=C3["both"], label="on both days"), Patch(color=C3["d1"], label="Day 1 only"),
                                Patch(color=C3["d2"], label="Day 2 only"), Patch(color="#e3e3e3", label="not in that session"),
                                Patch(color="#f5b041", alpha=0.3, label="stimulus blocks (white dotted lines)")],
                       loc="upper left", ncol=5, frameon=False, fontsize=10 if small else 11, bbox_to_anchor=(0.05, 0.935 if small else 0.955))
            fig.suptitle(f"{m}: every neuron in {'all four sessions' if len(ks) == 4 else 'the two Day-' + str(ks[0][1]) + ' sessions'}; "
                         "every scored event below", fontsize=15, x=0.6, y=0.985)
            fn = os.path.join(OD, f"P1_whole_{m}{suf}.png")
            fig.savefig(fn, dpi=110, bbox_inches="tight", pad_inches=0.1)
            plt.close(fig)
            return fn
        files.append(draw(keys, "", (19, 10.5)))
        draw(keys[:2], "_d1", (12.4, 7.4))                       # the two parts used on the slides (10-22, legibility)
        draw(keys[2:], "_d2", (12.4, 7.4))
        for k in keys:
            Zb, tb = Bn[k]
            pm = np.nanmean(Zb, 1)
            nost = (tb >= CA.SKIP_S) & (tb < 180)
            blk = np.zeros(len(tb), bool)
            for b0, b1 in BLOCK_S:
                blk |= (tb >= b0) & (tb < b1)
            STATS1.setdefault(m, {})[f"d{k[1]}_{k[2]}"] = dict(no_stimulus=float(np.nanmean(pm[nost])), blocks=float(np.nanmean(pm[blk])),
                                                               n=int(np.isfinite(Zb).any(0).sum()))
        STATS1[m]["n_rows"] = len(rows)
        STATS1[m]["n_both"] = int((cat == 0).sum())
    return files


STATS1 = {}


def plot2_average(S):
    """basic plot 2: population mean z over time (5-s bins) - top: one row per mouse, bottom: mean +- SEM over mice"""
    fig = plt.figure(figsize=(17, 7.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.5], hspace=0.12, wspace=0.08)
    out = {}
    ax_b = []
    for c_, c in enumerate(COND):
        per = {}
        for m in MO:
            Y = [np.nanmean(binned(s, 5.0)[0], 1)[:200] for (mm, d, ph), s in S.items() if mm == m and cond_of(mm, d, ph) == c]
            if Y:
                L_ = min(len(y) for y in Y)
                per[m] = np.nanmean(np.vstack([y[:L_] for y in Y]), 0)
        L_ = min(len(y) for y in per.values())
        ms = [m for m in MO if m in per]
        H = np.vstack([per[m][:L_] for m in ms])
        x = np.arange(L_) * 5 / 60
        a = fig.add_subplot(gs[0, c_])
        a.imshow(H, aspect="auto", cmap="magma", vmin=-0.2, vmax=1.0, extent=[0, x[-1] + 5 / 60, len(H), 0], interpolation="nearest")
        a.set_yticks(np.arange(len(H)) + 0.5, ms if c_ == 0 else [""] * len(H))
        a.set_xticks([])
        a.set_title(c, color=CC[c], fontsize=14, fontweight="bold")
        b = fig.add_subplot(gs[1, c_], sharey=ax_b[0] if ax_b else None)
        ax_b.append(b)
        for m in ms:
            b.plot(x, per[m][:L_], color=MC[m], lw=0.8, alpha=0.6)
        mu, se = H.mean(0), H.std(0) / np.sqrt(len(H))
        b.plot(x, mu, color=CC[c], lw=2.6)
        b.fill_between(x, mu - se, mu + se, color=CC[c], alpha=0.25, lw=0)
        for b0, b1 in BLOCK_S:
            b.axvspan(b0 / 60, b1 / 60, color="#f5b041", alpha=0.12, lw=0)
        b.set_xlabel("time in session (min)")
        if c_ == 0:
            b.set_ylabel("population mean z\n(thick = mean of the mice)")
        else:
            plt.setp(b.get_yticklabels(), visible=False)
        out[c] = dict(no_stimulus=float(mu[(x > 0.25) & (x < 3)].mean()), blocks=float(mu[x >= 3].mean()))
    fig.legend(handles=[Line2D([], [], color=MC[m], lw=1.5, label=m) for m in MO]
               + [Patch(color="#f5b041", alpha=0.3, label="stimulus blocks")],
               loc="lower center", ncol=5, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.02))
    fig.savefig(os.path.join(OD, "P2_average_over_mice.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def mcnemar_p(b, c):
    n = b + c
    return binomtest(int(b), int(n), 0.5).pvalue if n > 0 else np.nan


def is_resp(v):
    return np.asarray(pd.Series(v).isin(["activated", "inhibited"]))


def within_day(X, drug):
    """the same neurons of the same day: baseline row and injection row of every neuron (drug days only)"""
    pre = X[(X.phase == "pre") & np.array([drug_of(m, d) == drug for m, d in zip(X.mouse, X.day)], bool)]
    post = X[(X.phase == "post") & (X.condition == drug)]
    return pre.merge(post, on=["mouse", "day", "cell"], suffixes=("_b", "_i"))



# 10-22 night: the 4 groups of the bar plots - each injection next to the baseline of its own day
G4 = ["base_veh", "vehicle", "base_sbi", "SBI-553"]
L4 = {"base_veh": "base\n(veh day)", "vehicle": "veh", "base_sbi": "base\n(SBI day)", "SBI-553": "SBI"}
C4 = {"base_veh": "#7f8c8d", "vehicle": "#2471a3", "base_sbi": "#7f8c8d", "SBI-553": "#c0392b"}


def sub4(X, g):
    if g in ("vehicle", "SBI-553"):
        return X[X.condition == g]
    drug = "vehicle" if g == "base_veh" else "SBI-553"
    return X[(X.phase == "pre") & np.array([ARMS[int(d)][str(m)] == drug for m, d in zip(X.mouse, X.day)], bool)]


KEY4 = ("base = the baseline session of that day. Blue / red bracket = that day's baseline vs after the injection (the same "
        "neurons / mice); black = after vehicle vs after SBI-553.  * p < 0.05, ** p < 0.01, *** p < 0.001, ns = not significant")


def plot3_categories(T):
    """basic plot 3a: % responsive (activated + inhibited) per event and condition, with the tests"""
    res = []
    TOPS3 = []
    EVS_ = [e for e in EVENTS if EV_SUB is None or e in EV_SUB]
    fig, ax = plt.subplots(1, len(EVS_), figsize=(19, 5.4) if EV_SUB is None else (2.6 * len(EVS_) + 1.4, 4.6), sharey=True,
                           squeeze=False)
    ax = ax[0]
    for k, ev in enumerate(EVS_):
        a = ax[k]
        X = T[T.event == ev]
        tops = []
        for c in COND:                                          # the table (baseline = both days, as before)
            Y = X[X.condition == c]
            n = len(Y)
            na, ni = int((Y.cls == "activated").sum()), int((Y.cls == "inhibited").sum())
            res.append(dict(event=ev, condition=c, n=n, activated=na, inhibited=ni, pct_resp=100 * (na + ni) / max(n, 1),
                            pct_act=100 * na / max(n, 1), pct_inh=100 * ni / max(n, 1)))
        XP = {"base_veh": 0.0, "vehicle": 1.0, "base_sbi": 2.35, "SBI-553": 3.35}
        for g in G4:                                            # the plot: each injection next to its own day's baseline
            Y = sub4(X, g)
            n = len(Y)
            if not n:
                a.text(XP[g], 1.5, "no\nevents", ha="center", va="bottom", fontsize=9.5, color="0.4")
                continue
            na, ni = int((Y.cls == "activated").sum()), int((Y.cls == "inhibited").sum())
            a.bar(XP[g], 100 * na / n, color=C4[g], width=0.72)
            a.bar(XP[g], 100 * ni / n, bottom=100 * na / n, color=C4[g], alpha=0.45, hatch="//", edgecolor="w", width=0.72)
            pm = [100 * is_resp(Y[Y.mouse == m].cls).mean() for m in MO if (Y.mouse == m).any()]
            for j, m in enumerate(MO):
                Z = Y[Y.mouse == m]
                if len(Z):                                     # 10-24: a mouse at 0 % stays visible on the x axis
                    a.scatter(XP[g] + (j - 1.5) * 0.13, 100 * is_resp(Z.cls).mean(), marker=MK[m], color=MC[m], s=24, zorder=4,
                              edgecolor="w", lw=0.5, clip_on=False)
            tops.append(max([100 * (na + ni) / n] + pm))
        pmc = {}
        for drug, x in (("vehicle", 1), ("SBI-553", 2)):
            M_ = within_day(X, drug)
            rb, ri = is_resp(M_.cls_b), is_resp(M_.cls_i)
            p = mcnemar_p(int((rb & ~ri).sum()), int((~rb & ri).sum()))
            res.append(dict(event=ev, condition=f"{drug}: baseline -> injection", p_mcnemar=p, lost=int((rb & ~ri).sum()),
                            gained=int((~rb & ri).sum()), base_pct=100 * rb.mean() if len(rb) else np.nan,
                            inj_pct=100 * ri.mean() if len(ri) else np.nan, n_pairs=int(len(M_))))
            pmc[drug] = p
        Rv = {c: is_resp(X[X.condition == c].cls) for c in ("vehicle", "SBI-553")}
        pf = np.nan
        if len(Rv["vehicle"]) and len(Rv["SBI-553"]):
            pf = fisher_exact([[Rv["SBI-553"].sum(), (~Rv["SBI-553"]).sum()], [Rv["vehicle"].sum(), (~Rv["vehicle"]).sum()]])[1]
            res.append(dict(event=ev, condition="SBI-553 vs vehicle", p_fisher=pf))
        # every star with its pair: base-veh / base-SBI = the same neurons on that day (McNemar); veh-SBI = Fisher
        y1 = max(tops + [0]) + 4
        TOPS3.append(max(tops + [0]))
        PS.bracket(a, XP["base_veh"], XP["vehicle"], y1, pmc.get("vehicle", np.nan), CC["vehicle"], fs=12, h=1.5)
        PS.bracket(a, XP["base_sbi"], XP["SBI-553"], y1, pmc.get("SBI-553", np.nan), CC["SBI-553"], fs=12, h=1.5)
        PS.bracket(a, XP["vehicle"], XP["SBI-553"], y1 + 11, pf, "k", fs=12, h=1.5)
        a.set_xticks([XP[g] for g in G4], [L4[g] for g in G4], fontsize=10)
        a.set_xlim(-0.6, 3.95)
        w_ = WW.of(ev)                                         # 10-24: the full event name and its response window
        a.set_title(f"{ev}\nresponse {w_[2]:g}-{w_[3]:g} s", fontsize=13 if EV_SUB is not None else 10.5, linespacing=1.15)
    ax[0].set_ylabel("% of neurons responsive\n(solid = activated, hatched = inhibited)")
    ymax = max(r["pct_resp"] for r in res if "pct_resp" in r)
    ax[0].set_ylim(0, max(TOPS3 + [ymax]) + 24)
    # 10-24 figure review: the key showed only grey solid / hatched -> the condition colours, UP / DOWN and the mice
    # (handles ordered for matplotlib's column-wise filling: row 1 = colours + UP / DOWN, row 2 = the mice)
    hc = [Patch(color=CC["baseline"], label="baseline"), Patch(color=CC["vehicle"], label="after vehicle"),
          Patch(color=CC["SBI-553"], label="after SBI-553"), Patch(color="0.45", label="solid = UP (activated)"),
          Patch(facecolor="0.8", hatch="//", edgecolor="w", label="hatched = DOWN (inhibited)")]
    hm = [Line2D([], [], marker=MK[m], ls="", color=MC[m], ms=7, label=f"{m} (dot = mouse)" if j == 0 else m)
          for j, m in enumerate(MO)]
    hh = [hc[0], hm[0], hc[1], hm[1], hc[2], hm[2], hc[3], hm[3], hc[4]]
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.legend(handles=hh, loc="upper center", ncol=5, frameon=False, fontsize=10.5, bbox_to_anchor=(0.5, 0.035),
               columnspacing=1.4, handlelength=1.8)
    # 10-24: the methods key goes into the slide notes; the figure keeps the one-line star key (under the legend)
    figfoot.note(fig, KEY4 + "  (blue / red: McNemar; black: Fisher exact)", "en", y=0.035 - 0.8 / fig.get_figheight())
    fig.savefig(os.path.join(OD, f"P3a_responsive_pct{FN_SUF}.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    R3 = pd.DataFrame(res)
    if not FN_SUF:
        R3.to_csv(os.path.join(OD, "P3a_responsive_pct.csv"), index=False)
    return R3


def plot3_sessions(T):
    """basic plot 3e: % responsive across the four sessions in their real order, per mouse"""
    EVS_ = [e for e in EVENTS if EV_SUB is None or e in EV_SUB]
    fig, ax = plt.subplots(1, len(EVS_), figsize=(19, 4.4) if EV_SUB is None else (2.7 * len(EVS_) + 1.2, 4.4), sharey=True,
                           squeeze=False)
    ax = ax[0]
    xs = [(1, "pre"), (1, "post"), (2, "pre"), (2, "post")]
    for k, ev in enumerate(EVS_):
        a = ax[k]
        X = T[T.event == ev]
        for m in MO:
            y, cs = [], []
            for d, ph in xs:
                Z = X[(X.mouse == m) & (X.day == d) & (X.phase == ph)]
                y.append(100 * is_resp(Z.cls).mean() if len(Z) else np.nan)
                cs.append(CC[cond_of(m, d, ph)])
            a.plot(range(4), y, "-", color=MC[m], lw=1.3, alpha=0.8)
            a.scatter(range(4), y, c=cs, marker=MK[m], s=42, zorder=3, edgecolor=MC[m], lw=1.3)
        a.set_xticks(range(4), ["D1\nbase", "D1\ninj", "D2\nbase", "D2\ninj"], fontsize=9.5)
        a.set_title(SHORT[ev], fontsize=13)
    ax[0].set_ylabel("% responsive")
    fig.legend(handles=[Line2D([], [], marker=MK[m], color=MC[m], label=m) for m in MO] + [Patch(color=CC[c], label=c) for c in COND],
               loc="lower center", ncol=7, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(OD, f"P3e_sessions{FN_SUF}.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot3_transitions(T):
    """basic plot 3b: the same neurons, baseline -> injection of the same day, per drug (all events pooled)"""
    a_ = T[T.phase == "pre"][["mouse", "day", "cell", "event", "cls"]].rename(columns={"cls": "base"})
    b_ = T[T.phase == "post"][["mouse", "day", "cell", "event", "cls"]].rename(columns={"cls": "inj"})
    TT = a_.merge(b_, on=["mouse", "day", "cell", "event"])
    TT["drug"] = [drug_of(m, d) for m, d in zip(TT.mouse, TT.day)]
    cats = ["activated", "inhibited", "non-responsive"]
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.6), sharey=True)
    trans = []
    for k, drug in enumerate(("vehicle", "SBI-553")):
        a = ax[k]
        Z = TT[TT.drug == drug]
        for i, cb in enumerate(cats):
            Wd = Z[Z.base == cb]
            left = 0
            for ci in cats:
                v = 100 * (Wd.inj == ci).mean() if len(Wd) else 0
                a.barh(i, v, left=left, color=CATC[ci], edgecolor="w")
                if v > 7:
                    a.text(left + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=11)
                left += v
                trans.append(dict(drug=drug, base=cb, inj=ci, n=len(Wd), pct=v))
            a.text(101, i, f"n = {len(Wd)}", va="center", fontsize=10)
        a.set_yticks(range(3), [f"{c}\nat baseline" for c in cats])
        a.set_xlim(0, 114)
        a.invert_yaxis()
        a.set_xlabel("% of these neuron x event pairs after the injection")
        a.set_title(f"{drug} day", color=CC[drug], loc="left", fontsize=14, fontweight="bold")
    fig.legend(handles=[Patch(color=CATC[c], label=f"{c} after the injection") for c in cats], loc="lower center", ncol=3,
               frameon=False, fontsize=11, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(os.path.join(OD, "P3b_transitions.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(trans).to_csv(os.path.join(OD, "P3b_transitions.csv"), index=False)
    st = {}
    for drug in ("vehicle", "SBI-553"):
        Z = TT[(TT.drug == drug) & is_resp(TT.base)]
        st[drug] = (int(is_resp(Z.inj).sum()), int(len(Z)))
        Zn = TT[(TT.drug == drug) & ~is_resp(TT.base)]
        st[drug + "_new"] = (int(is_resp(Zn.inj).sum()), int(len(Zn)))
    p = fisher_exact([[st["SBI-553"][0], st["SBI-553"][1] - st["SBI-553"][0]],
                      [st["vehicle"][0], st["vehicle"][1] - st["vehicle"][0]]])[1]
    pn = fisher_exact([[st["SBI-553_new"][0], st["SBI-553_new"][1] - st["SBI-553_new"][0]],
                       [st["vehicle_new"][0], st["vehicle_new"][1] - st["vehicle_new"][0]]])[1]
    return dict(stay=st, p_stay=float(p), p_new=float(pn))


def plot3_stability_overlap(T, P):
    """basic plot 3c: Day-1 vs Day-2 baseline of the tracked neurons; overlap of the responsive neurons between events"""
    rows = []
    for m in MO:
        pr = P[m][["day1_id", "day2_id"]].astype(int).rename(columns={"day1_id": "c1", "day2_id": "c2"})
        A = T[(T.mouse == m) & (T.day == 1) & (T.phase == "pre")][["cell", "event", "cls"]].rename(columns={"cell": "c1", "cls": "d1"})
        B_ = T[(T.mouse == m) & (T.day == 2) & (T.phase == "pre")][["cell", "event", "cls"]].rename(columns={"cell": "c2", "cls": "d2"})
        M_ = pr.merge(A, on="c1").merge(B_, on=["c2", "event"])
        M_["mouse"] = m
        rows.append(M_)
    St = pd.concat(rows, ignore_index=True)
    # 10-24 figure review: the n labels were ~6-7 pt, empty events had no mark, the heatmap diagonal (100 %, clipped at 60 %)
    # was solid yellow without a value and the cell numbers ~7 pt -> n over each pair at 11 pt, 'n/a' where nothing can be
    # counted, the diagonal grey (an event with itself), cell numbers 11 pt
    fig, ax = plt.subplots(1, 2, figsize=(14.5, 5.2), gridspec_kw=dict(width_ratios=[1.25, 1]))
    a = ax[0]
    stab = []
    for i, ev in enumerate(EVENTS):
        Z = St[St.event == ev]
        if not len(Z):
            a.text(i, 1, "n/a", ha="center", va="bottom", fontsize=11, color="0.4")
            continue
        r1 = is_resp(Z.d1)
        same = (Z.d1.values == Z.d2.values) & r1
        p2 = Z.d2.value_counts(normalize=True)
        chance = 100 * sum((Z.d1 == c).sum() * p2.get(c, 0) for c in ("activated", "inhibited")) / max(r1.sum(), 1)
        frac = 100 * same.sum() / max(r1.sum(), 1)
        k_, n_ = int(same.sum()), int(r1.sum())
        pb = binomtest(k_, n_, min(max(chance / 100, 1e-6), 1 - 1e-6), alternative="greater").pvalue if n_ else np.nan
        stab.append(dict(event=ev, n_tracked=len(Z), d1_resp=n_, same_on_d2=k_, pct=frac, chance=chance, p=pb))
        a.bar(i - 0.2, frac, width=0.38, color=C3["both"])
        a.bar(i + 0.2, chance, width=0.38, color="0.78")
        if n_:
            a.text(i, max(frac, chance) + 1.2, f"{k_}/{n_}", ha="center", va="bottom", fontsize=11)
        else:
            a.text(i, 1, "n/a", ha="center", va="bottom", fontsize=11, color="0.4")
        PS.bracket(a, i - 0.2, i + 0.2, max(frac, chance) + 8.5, pb, "k", fs=12.5, h=1.2, inset=0)   # green vs grey (chance)
    a.set_xticks(range(len(EVENTS)), [SHORT[e] for e in EVENTS], rotation=35, fontsize=11, ha="right", rotation_mode="anchor")
    a.tick_params(axis="y", labelsize=11)
    a.set_ylim(0, max([max(r["pct"], r["chance"]) for r in stab] + [10]) + 22)
    a.set_ylabel("% of the Day-1 baseline responsive neurons")
    a.set_title("A  same category again on Day-2 baseline (green; grey = chance)\nbracket = green vs grey (binomial test); "
                "number = same / Day-1 responsive", loc="left", fontsize=12)
    b = ax[1]
    X = T[T.phase == "pre"]
    K = len(EVENTS)
    Jm = np.full((K, K), np.nan)
    sets = {e: X[X.event == e].set_index(["mouse", "day", "cell"]).cls.isin(["activated", "inhibited"]) for e in EVENTS}
    for i, e1 in enumerate(EVENTS):
        for j, e2 in enumerate(EVENTS):
            ix = sets[e1].index.intersection(sets[e2].index)
            u = (sets[e1][ix] | sets[e2][ix]).sum()
            Jm[i, j] = 100 * (sets[e1][ix] & sets[e2][ix]).sum() / u if u else np.nan
    Jd = Jm.copy()
    np.fill_diagonal(Jd, np.nan)                               # display only: an event with itself (100 %) is grey
    cmv = plt.get_cmap("viridis").copy()
    cmv.set_bad("0.85")
    im = b.imshow(np.ma.masked_invalid(Jd), cmap=cmv, vmin=0, vmax=60)
    for i in range(K):
        for j in range(K):
            if np.isfinite(Jm[i, j]) and i != j:
                b.text(j, i, f"{Jm[i, j]:.0f}", ha="center", va="center", fontsize=11, color="w" if Jm[i, j] < 35 else "k")
    lab = [SHORT[e] for e in EVENTS]
    b.set_xticks(range(K), lab, rotation=45, ha="right", fontsize=11)
    b.set_yticks(range(K), lab, fontsize=11)
    b.set_title("B  do the same neurons respond to several events?\n(baseline; both / either, %; grey = the event itself)",
                loc="left", fontsize=12)
    cb_ = fig.colorbar(im, ax=b, shrink=0.8, ticks=[0, 20, 40, 60])
    cb_.set_label("overlap: responsive to both /\nresponsive to either (%)", fontsize=11)
    cb_.ax.tick_params(labelsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OD, "P3c_stability_overlap.png"), dpi=140)
    plt.close(fig)
    Sb = pd.DataFrame(stab)
    Sb.to_csv(os.path.join(OD, "P3c_stability.csv"), index=False)
    return Sb, pd.DataFrame(Jm, index=EVENTS, columns=EVENTS)


def plot3_event_heatmaps(S, T, W):
    """basic plot 3d, per event: every neuron's mean response (sorted activated / inhibited / non-responsive) and the mean
    trace of each category, in the three conditions"""
    files = []
    for ev in EVENTS:
        fig = plt.figure(figsize=(16.5, 8.6))
        gs = fig.add_gridspec(2, 3, height_ratios=[2.2, 1], hspace=0.32, wspace=0.45)
        ok = False
        ax_b, ax_t, im = [], [], None
        for c_, c in enumerate(COND):
            tr = {"activated": [], "inhibited": [], "non-responsive": []}
            tt = None
            for (m, d, ph), s in S.items():
                if cond_of(m, d, ph) != c:
                    continue
                X, tt_, used = CA.peri(s, s["B"]["ev"].get(ev, []))
                if X is None or len(used) < CA.MIN_TRIALS:
                    continue
                tt = tt_
                pre = (tt >= -3) & (tt < 0)                    # display baseline (the classes use the chosen window)
                Xb = np.nanmean(X - np.nanmean(X[:, pre, :], 1, keepdims=True), 0)
                cl = T[(T.mouse == m) & (T.day == d) & (T.phase == ph) & (T.event == ev)].set_index("cell").cls
                for j, cid in enumerate(s["ids"]):
                    tr[cl.get(int(cid), "non-responsive")].append(Xb[:, j])
            if tt is None or not any(tr.values()):
                a = fig.add_subplot(gs[0, c_]); a.set_axis_off()
                a.text(0.5, 0.5, f"no {SHORT[ev]} analysis\nafter {c}\n(< {CA.MIN_TRIALS} events in every session)" if c != "baseline"
                       else f"no {SHORT[ev]} analysis\n(< {CA.MIN_TRIALS} events)", ha="center", va="center", fontsize=12, color="0.4",
                       transform=a.transAxes)
                continue
            ok = True
            a = fig.add_subplot(gs[0, c_])
            ax_t.append(a)
            w = W[ev][2]
            rw = (tt >= w[2]) & (tt < w[3])
            rows, labs = [], []
            for g in ("activated", "inhibited", "non-responsive"):
                if tr[g]:
                    Y = np.array(tr[g])
                    v = np.nanmean(Y[:, rw], 1)
                    rows.append(Y[np.argsort(v if g == "inhibited" else -v)])
                    labs.append((g, len(Y)))
            Y = np.vstack(rows)
            im = a.imshow(CA.sm(Y.T, 6).T, aspect="auto", cmap="RdBu_r", vmin=-1.2, vmax=1.2, extent=[tt[0], tt[-1], len(Y), 0],
                          interpolation="nearest")
            y0, ylab = 0, -1e9
            for g, n_ in labs:
                a.axhline(y0, color="k", lw=0.8)
                yl = max(y0 + n_ / 2, ylab + 0.11 * len(Y))            # 10-23: labels at least 11 % of the rows apart
                ylab = min(yl, len(Y) * 0.97)
                a.text(tt[-1] + 0.25, ylab, f"{'non-resp.' if g == 'non-responsive' else g}\n{n_} ({100 * n_ / len(Y):.0f}%)",
                       va="center", fontsize=10, color={"activated": "#c0392b", "inhibited": "#1f618d", "non-responsive": "0.35"}[g])
                y0 += n_
            a.set_xlabel(f"time from {SHORT[ev]} (s)", fontsize=11)
            a.axvline(0, color="k", lw=0.8, ls="--")
            a.set_xlim(-5, 8)
            a.set_yticks([])
            a.set_title(f"{c} ({len(Y)} neuron-sessions)", color=CC[c], fontsize=13.5, fontweight="bold")
            b = fig.add_subplot(gs[1, c_], sharey=ax_b[0] if ax_b else None)
            ax_b.append(b)
            for g in ("activated", "inhibited", "non-responsive"):
                if tr[g]:
                    Yg = np.array(tr[g]).T
                    mu = CA.sm(np.nanmean(Yg, 1)[:, None], 6)[:, 0]
                    se = CA.sm((np.nanstd(Yg, 1) / np.sqrt(Yg.shape[1]))[:, None], 6)[:, 0]
                    b.plot(tt, mu, color=CATC[g] if g != "non-responsive" else "#95a5a6", lw=2)
                    b.fill_between(tt, mu - se, mu + se, color=CATC[g], alpha=0.2, lw=0)
            b.axvspan(w[2], w[3], color="0.9", zorder=0)
            b.axvline(0, color="0.4", lw=0.8, ls="--")
            b.axhline(0, color="0.6", lw=0.6)
            b.set_xlim(-5, 8)
            b.set_xlabel(f"time from {SHORT[ev]} (s)")
            if c_ == 0:
                b.set_ylabel("z (baseline -3..0 s)\nmean +- SEM")
        if not ok:
            plt.close(fig)
            continue
        if im is not None:                                     # 10-23: every heatmap has a colour bar with its unit
            cb = fig.colorbar(im, ax=ax_t, shrink=0.75, pad=0.07, fraction=0.025)
            cb.set_label("z (baseline -3..0 s), trial mean")
        w = W[ev][2]
        fig.legend(handles=[Line2D([], [], color=CATC["activated"], lw=2, label="activated (UP)"),
                            Line2D([], [], color=CATC["inhibited"], lw=2, label="inhibited (DOWN)"),
                            Line2D([], [], color="#95a5a6", lw=2, label="non-responsive"),
                            Patch(color="0.9", label=f"response window {w[2]:g}-{w[3]:g} s"),
                            Patch(color=CATC["activated"], alpha=0.2, label="shading = SEM over neuron-sessions")],
                   loc="lower center", ncol=5, frameon=False, fontsize=11, bbox_to_anchor=(0.5, -0.02))
        fig.suptitle(f"{ev}: every neuron-session sorted by its class (time-shift test, BH q <= 0.20), baseline / after vehicle / "
                     f"after SBI-553", fontsize=13.5, x=0.01, ha="left")
        fn = os.path.join(OD, f"P3d_{SHORT[ev].replace('/', '_')}.png")
        fig.savefig(fn, dpi=120, bbox_inches="tight")
        plt.close(fig)
        files.append(fn)
    return files


def centroids():
    cen = {}
    for m in MO:
        for d in (1, 2):
            t = f"{m}_d{d}"
            R = pd.read_csv(os.path.join(ROOT[d], t, "4_cell_review", f"{t}_cell_review.csv"))
            cen[(m, d)] = R.set_index("id")[["row", "col"]]
    return cen


def plot4_topography(T):
    """basic plot 4: are the responsive neurons clustered in the field of view? mean nearest-neighbour distance of the
    responsive neurons vs the same number drawn at random from that session's neurons (500 draws)"""
    rng = np.random.default_rng(3)
    cen = centroids()
    rows = []

    def nnd(P_):
        D = np.hypot(P_[:, None, 0] - P_[None, :, 0], P_[:, None, 1] - P_[None, :, 1])
        np.fill_diagonal(D, np.inf)
        return float(np.mean(D.min(1)))
    for (m, d, ph, ev), X in T.groupby(["mouse", "day", "phase", "event"]):
        C_ = cen[(m, int(d))].reindex(X.cell.values).to_numpy(float)
        okc = np.isfinite(C_).all(1)
        for grp, sel in (("responsive", is_resp(X.cls)), ("activated", (X.cls == "activated").values),
                         ("inhibited", (X.cls == "inhibited").values)):
            r = sel & okc
            if r.sum() < 4 or r.sum() > okc.sum() - 3:
                continue
            obs = nnd(C_[r])
            null = np.array([nnd(C_[okc][rng.choice(okc.sum(), r.sum(), replace=False)]) for _ in range(500)])
            rows.append(dict(mouse=m, day=d, phase=ph, condition=cond_of(m, d, ph), event=ev, group=grp, n=int(r.sum()),
                             nnd=obs, nnd_null=float(null.mean()), z=(null.mean() - obs) / max(null.std(), 1e-9),
                             p=(1 + np.sum(null <= obs)) / 501))
    Tp = pd.DataFrame(rows)
    Tp.to_csv(os.path.join(OD, "P4_topography.csv"), index=False)
    # 10-24 figure review: the dashed line and the ringed dots were unexplained, the legend sat on the dashed line, the last
    # category had no data and no mark -> legend (with both keys) outside on the right, 'no data' written where nothing is
    fig, ax = plt.subplots(figsize=(13.5, 4.6))
    Z0 = Tp[Tp.group == "responsive"]
    for i, ev in enumerate(EVENTS):
        if not (Z0.event == ev).any():
            ax.text(i, 0, "no\ndata", ha="center", va="center", fontsize=10.5, color="0.4",
                    bbox=dict(fc="w", ec="none", pad=1.5), zorder=4)
        for j, c in enumerate(COND):
            Z = Z0[(Z0.event == ev) & (Z0.condition == c)]
            x = i + (j - 1) * 0.25
            ax.scatter(x + np.random.default_rng(i * 3 + j).uniform(-0.06, 0.06, len(Z)), Z.z, color=CC[c], s=32,
                       edgecolor=["k" if p_ < 0.05 else "none" for p_ in Z.p], lw=1.3, label=c if i == 0 else None)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.axhline(1.645, color="0.6", lw=0.8, ls="--")
    ax.set_xticks(range(len(EVENTS)), [SHORT[e] for e in EVENTS])
    ax.set_xlim(-0.6, len(EVENTS) - 0.4)                       # the last category off the axis end
    ax.set_ylabel("clustering z\n(> 0 = closer together than random)")
    hs = [Line2D([], [], ls="", marker="o", ms=7, color=CC[c], label=c) for c in COND]
    hs += [Line2D([], [], color="0.6", lw=1, ls="--", label="dashed: z = 1.645\n(p = 0.05, one-sided)"),
           Line2D([], [], ls="", marker="o", ms=7, mfc="0.8", mec="k", mew=1.3, label="black ring: clustered,\np < 0.05 (500 random draws)")]
    ax.legend(handles=hs, frameon=False, fontsize=11, loc="upper left", bbox_to_anchor=(1.01, 1.0), labelspacing=0.7)
    fig.tight_layout()
    fig.savefig(os.path.join(OD, "P4_topography.png"), dpi=140)
    plt.close(fig)
    return Tp


def plot4_maps(T):
    """basic plot 4b: per mouse, where the activated / inhibited neurons sit (rows = sessions, columns = events)"""
    cen = centroids()
    files = []
    for m in MO:
        keys = [(1, "pre"), (1, "post"), (2, "pre"), (2, "post")]
        fig, ax = plt.subplots(4, len(EVENTS), figsize=(19, 9.6))
        for i, (d, ph) in enumerate(keys):
            C_ = cen[(m, d)]
            for k, ev in enumerate(EVENTS):
                a = ax[i, k]
                X = T[(T.mouse == m) & (T.day == d) & (T.phase == ph) & (T.event == ev)]
                P_ = C_.reindex(X.cell.values)
                for g in ("non-responsive", "inhibited", "activated"):
                    s_ = (X.cls == g).values
                    a.scatter(P_.col.values[s_], P_.row.values[s_], s=15 if g != "non-responsive" else 7, color=CATC[g],
                              edgecolor="k" if g != "non-responsive" else "none", lw=0.3)
                a.set_xlim(C_.col.min() - 30, C_.col.max() + 30)
                a.set_ylim(C_.row.max() + 30, C_.row.min() - 30)
                a.set_aspect("equal")
                a.set_xticks([])
                a.set_yticks([])
                for sp in a.spines.values():
                    sp.set_visible(True)
                    sp.set_color("0.8")
                if i == 0:
                    a.set_title(SHORT[ev], fontsize=12)
                if k == 0:
                    cnd = cond_of(m, d, ph)
                    a.set_ylabel(f"Day {d}\n{'baseline' if ph == 'pre' else cnd}", color=CC[cnd], fontsize=11.5, fontweight="bold")
        fig.legend(handles=[Patch(color=CATC[g], label=g) for g in ("activated", "inhibited", "non-responsive")],
                   loc="lower center", ncol=3, frameon=False, fontsize=11)
        fig.suptitle(f"{m}: where the responsive neurons are (each dot = one neuron at its place in the field of view)", fontsize=14)
        fig.tight_layout(rect=(0, 0.04, 1, 0.96))
        fn = os.path.join(OD, f"P4b_maps_{m}.png")
        fig.savefig(fn, dpi=110)
        plt.close(fig)
        files.append(fn)
    return files


def plot5_fate(S, W, F=None):
    """basic plot 5: neurons activated / inhibited at the baseline of a day - chosen on the ODD baseline trials so regression
    to the mean cannot fake a change - their response on the EVEN baseline trials vs after the injection"""
    if F is None:                                          # F given = redraw from the saved P5_fate.csv (--step p5)
        rows = []
        for (m, d, ph), s in S.items():
            if ph != "pre" or (m, d, "post") not in S:
                continue
            s2 = S[(m, d, "post")]
            for ev in EVENTS:
                pre, gap, a, b = W[ev][d]
                R1, _ = responses(s, s["B"]["ev"].get(ev, []), pre, gap, a, b)
                R2, _ = responses(s2, s2["B"]["ev"].get(ev, []), pre, gap, a, b)
                if len(R1) < 10 or len(R2) < 5:
                    continue
                p, sg = wsr_vec(R1[0::2])
                for j, c in enumerate(s["ids"]):
                    if not np.isfinite(p[j]) or p[j] >= 0.05:
                        continue
                    k = np.flatnonzero(s2["ids"] == c)
                    if not len(k):
                        continue
                    rows.append(dict(mouse=m, day=d, drug=drug_of(m, d), event=ev, cell=int(c),
                                     group="activated" if sg[j] > 0 else "inhibited",
                                     base_heldout=float(np.nanmean(R1[1::2, j])), inj=float(np.nanmean(R2[:, k[0]]))))
        F = pd.DataFrame(rows)
        if not FN_SUF:
            F.to_csv(os.path.join(OD, "P5_fate.csv"), index=False)
    res = []
    evs = [ev for ev in EVENTS if len(F) and ((F.event == ev).groupby(F.drug).sum() >= 3).all() and (F.event == ev).sum() >= 6
           and (EV_SUB is None or ev in EV_SUB)]
    if not evs:
        return F, pd.DataFrame()
    # 10-24: the parts were 6.4 in tall (shown at 81 %: n labels and 'ns' < 9 pt); without the footer they are 5.6 in tall
    fig, ax = plt.subplots(2, len(evs), figsize=(2.15 * len(evs) + 2, 8.2) if EV_SUB is None else (2.9 * len(evs) + 1.6, 6.0),
                           squeeze=False)
    big = EV_SUB is not None                                   # the split parts (on the slides) get the big fonts
    fs_t, fs_n, fs_y = (12, 11, 11) if big else (9.5, 8.5, 9)
    fs_b = 12.5 if big else 11                                 # bracket stars ('ns' is drawn 1.5 pt smaller)
    for i, g in enumerate(("activated", "inhibited")):
        for k, ev in enumerate(evs):
            a = ax[i, k]
            ch = {}
            marks_ = []
            for j, drug in enumerate(("vehicle", "SBI-553")):
                Z = F[(F.event == ev) & (F.group == g) & (F.drug == drug)].dropna(subset=["base_heldout", "inj"])
                x0 = j * 2.3
                if len(Z) < 3:                                 # an empty pair says why
                    a.annotate(f"n={len(Z)}: not shown", xy=(x0 + 0.5, 0), xycoords=("data", "axes fraction"), xytext=(0, -22 if big else -17),
                               textcoords="offset points", ha="center", va="top", fontsize=fs_n, color="0.15")
                    continue
                mu = [Z.base_heldout.mean(), Z.inj.mean()]
                se = [Z.base_heldout.std() / np.sqrt(len(Z)), Z.inj.std() / np.sqrt(len(Z))]
                a.errorbar([x0, x0 + 1], mu, yerr=se, color=CC[drug], lw=2.2, marker="o", capsize=3)
                p = wilcoxon(Z.base_heldout, Z.inj).pvalue if len(Z) >= 6 else np.nan
                res.append(dict(event=ev, group=g, drug=drug, n=len(Z), base=mu[0], inj=mu[1], p=p))
                ch[drug] = (Z.inj - Z.base_heldout).values
                yt = max(mu[0] + se[0], mu[1] + se[1]) if g == "activated" else min(mu[0] - se[0], mu[1] - se[1])
                marks_.append((x0, yt, p, CC[drug]))
                # n under the tick labels (inside the axes it sat on the points near the bottom)
                a.annotate(f"n={len(Z)}" + (" cells" if big else "") + ("" if np.isfinite(p) else ("\nno test" if big else ": no test")),
                           xy=(x0 + 0.5, 0),
                           xycoords=("data", "axes fraction"), xytext=(0, -22 if big else -17), textcoords="offset points",
                           ha="center", va="top", fontsize=fs_n, color="0.15")
            # every star with its pair: coloured bracket = held-out baseline vs injection of the same neurons (Wilcoxon);
            # black bracket at the top = the change after vehicle vs the change after SBI-553 (Mann-Whitney)
            lo_, hi_ = a.get_ylim(); sp_ = hi_ - lo_
            if g == "activated":
                a.set_ylim(lo_, hi_ + 0.5 * sp_)
            else:
                a.set_ylim(lo_ - 0.32 * sp_, hi_ + 0.3 * sp_)
            for x0_, yt_, pp_, cc_ in marks_:
                if not np.isfinite(pp_):                       # fewer than 6 neurons: said in the n label
                    continue
                if g == "activated":
                    PS.bracket(a, x0_, x0_ + 1, yt_ + 0.06 * sp_, pp_, cc_, fs=fs_b, inset=0.05)
                else:
                    PS.bracket(a, x0_, x0_ + 1, yt_ - 0.06 * sp_, pp_, cc_, fs=fs_b, inset=0.05, below=True)
            if len(ch) == 2 and min(len(ch["vehicle"]), len(ch["SBI-553"])) >= 6:   # 10-23: same rule as the coloured brackets
                p2 = mannwhitneyu(ch["vehicle"], ch["SBI-553"]).pvalue
                res.append(dict(event=ev, group=g, drug="SBI vs vehicle change", p=p2))
                PS.bracket(a, 0.5, 2.8, 0.9, p2, "k", fs=fs_b, axes_y=True, inset=0, label="change")   # 10-24: says what it compares
            a.axhline(0, color="0.6", lw=0.6)
            a.set_xticks([0, 1, 2.3, 3.3], ["base", "veh", "base", "SBI"], fontsize=fs_t)
            a.tick_params(axis="y", labelsize=fs_y)
            a.set_xlim(-0.5, 3.8)
            if i == 0:
                a.set_title(SHORT[ev], fontsize=14 if big else 12.5)
            if k == 0:
                a.set_ylabel(f"{g} at baseline\nresponse (z), mean +- SEM", fontsize=12 if big else 10)
    # 10-24 figure review: 'base' appeared twice on the x axis and the bracket key was only in the footer -> the tick labels
    # take the colour of their day and a short key on top says which day each colour is and what the black bracket compares
    for a in ax.ravel():
        for tl, cc_ in zip(a.get_xticklabels(), (CC["vehicle"], CC["vehicle"], CC["SBI-553"], CC["SBI-553"])):
            tl.set_color(cc_)
    fig.legend(handles=[Line2D([], [], color=CC["vehicle"], lw=2.2, marker="o", label="vehicle day (base → veh)"),
                        Line2D([], [], color=CC["SBI-553"], lw=2.2, marker="o", label="SBI-553 day (base → SBI)"),
                        Line2D([], [], color="k", lw=1.2, label="'change' = veh-day change vs SBI-day change")],
               loc="upper center", ncol=3, frameon=False, fontsize=11.5 if big else 10, handlelength=1.8, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0.03 if EV_SUB is None else 0.045, 1, 0.95 if EV_SUB is None else 0.925))
    # 10-24: the methods text goes into the slide notes; the figure keeps the one-line star key
    figfoot.note(fig, "coloured bracket = the same neurons, held-out baseline vs after the injection (Wilcoxon); black bracket "
                      "('change') = the change after vehicle vs the change after SBI-553 (Mann-Whitney, both groups >= 6); no test = "
                      "fewer than 6 neurons; n = neurons pooled over the mice.  " + PS.KEY, "en", y=0.004)
    fig.savefig(os.path.join(OD, f"P5_fate{FN_SUF}.png"), dpi=140)
    plt.close(fig)
    Rf = pd.DataFrame(res)
    if not FN_SUF:
        Rf.to_csv(os.path.join(OD, "P5_fate_tests.csv"), index=False)
    return F, Rf


def cmh_p(tables):
    """Cochran-Mantel-Haenszel chi-square (continuity-corrected) over 2 x 2 tables [[a, b], [c, d]], one per mouse;
    a mouse without information (no event in either session, or one session empty) is left out; NaN if none is left"""
    from scipy.stats import chi2
    num, var = 0.0, 0.0
    for t in tables:
        (a, b), (c, d) = np.asarray(t, float)
        n1, n2, m1, m2 = a + b, c + d, a + c, b + d
        N = n1 + n2
        if N <= 1 or n1 == 0 or n2 == 0 or m1 == 0 or m2 == 0:
            continue
        num += a - n1 * m1 / N
        var += n1 * n2 * m1 * m2 / (N * N * (N - 1))
    if var <= 0:
        return np.nan
    return float(chi2.sf(max(abs(num) - 0.5, 0.0) ** 2 / var, 1))


def per_delivery(S):
    """your rule (behavior_day_analysis): reflex = from the delivery to 2 s later or the next delivery; held behaviours =
    a bout starting 0-3 s after the delivery"""
    rows = []
    for (m, d, ph), s in S.items():
        ev = s["B"]["ev"]
        wd, fl = np.asarray(ev["withdrawal"], int), np.asarray(ev["flinch"], int)
        dl = sorted((int(x), k) for k in STIM for x in np.asarray(ev.get(k, []), int))
        held = dict(s["held"])
        held["movement"] = s["B"]["st"]["move"]
        held["grooming"] = s["B"]["st"]["groom"]
        ons = {nm: spans(v)[0] for nm, v in held.items()}
        for i, (x, k) in enumerate(dl):
            nxt = dl[i + 1][0] if i + 1 < len(dl) else 10 ** 9
            hi = min(x + int(2 * FPS), nxt - 1)
            r = dict(mouse=m, day=d, phase=ph, condition=cond_of(m, d, ph), stim=k,
                     withdrawal=bool(np.any((wd >= x) & (wd <= hi))), flinch=bool(np.any((fl >= x) & (fl <= hi))))
            for nm, on in ons.items():
                r[nm] = bool(np.any((on >= x) & (on <= x + 3 * FPS)))
                v = np.asarray(held[nm], bool)
                r[f"{nm}_s"] = float(v[x:min(len(v), x + int(3 * FPS))].sum() / FPS)
            rows.append(r)
    return pd.DataFrame(rows)


BEH6 = ["withdrawal", "flinch", "attending", "lick/bite", "guarding", "escape/rearing", "movement", "grooming"]


def plot6_behaviour(S, D=None):
    """basic plot 6: % of the deliveries of each stimulus followed by each behaviour; baseline / vehicle / SBI-553, all mice.
    D = the saved P6_per_delivery.csv (redraw only: --step p6)"""
    D = per_delivery(S) if D is None else D
    if not FN_SUF:
        D.to_csv(os.path.join(OD, "P6_per_delivery.csv"), index=False)
    B6_ = [b for b in BEH6 if B6_SUB is None or b in B6_SUB]
    fig, ax = (plt.subplots(2, 4, figsize=(18, 8.2), sharex=True) if B6_SUB is None else
               plt.subplots(1, len(B6_), figsize=(3.6 * len(B6_) + 0.8, 4.8), sharex=True, squeeze=False))
    res = []
    for k, bname in enumerate(B6_):
        a = ax.ravel()[k]
        ymax = 5
        OFF = {"base_veh": -0.36, "vehicle": -0.13, "base_sbi": 0.13, "SBI-553": 0.36}
        for i, st in enumerate(STIM):
            vals = {}
            for c in COND:                                      # the numbers for the table (baseline = both days)
                vals[c] = D[(D.stim == st) & (D.condition == c)].groupby("mouse")[bname].mean() * 100
            v4 = {}
            for g in G4:                                        # the plot: each injection next to its own day's baseline
                Z = sub4(D[D.stim == st], g)
                pm = Z.groupby("mouse")[bname].mean() * 100
                x = i + OFF[g]
                a.bar(x, pm.mean() if len(pm) else 0, width=0.21, color=C4[g],
                      yerr=pm.std() / np.sqrt(len(pm)) if len(pm) > 1 else 0, capsize=2, error_kw=dict(lw=0.8),
                      label=({"base_veh": "baseline (that day)", "vehicle": "vehicle", "SBI-553": "SBI-553"}.get(g)
                             if (k == 0 and i == 0) else None))
                a.scatter(np.full(len(pm), x), pm.values, s=12, color="k", zorder=4, clip_on=False, edgecolor="w", lw=0.4)
                v4[g] = pm
                ymax = max(ymax, pm.max() if len(pm) else 0)
            pst = {}
            for drug, x in (("vehicle", i), ("SBI-553", i + 0.27)):
                tabs = []
                for m in MO:
                    dd = [d for d in (1, 2) if ARMS[d][m] == drug][0]
                    B0 = D[(D.mouse == m) & (D.day == dd) & (D.stim == st) & (D.phase == "pre")][bname]
                    B1 = D[(D.mouse == m) & (D.day == dd) & (D.stim == st) & (D.phase == "post")][bname]
                    tabs.append([[B1.sum(), (~B1).sum()], [B0.sum(), (~B0).sum()]])
                p = cmh_p(tabs)
                res.append(dict(behaviour=bname, stim=st, test=f"{drug} vs its baseline (CMH by mouse)", p=p))
                pst[drug] = p
            tabs = []
            for m in MO:
                V = D[(D.mouse == m) & (D.stim == st) & (D.condition == "vehicle")][bname]
                Sx = D[(D.mouse == m) & (D.stim == st) & (D.condition == "SBI-553")][bname]
                tabs.append([[Sx.sum(), (~Sx).sum()], [V.sum(), (~V).sum()]])
            p = cmh_p(tabs)
            res.append(dict(behaviour=bname, stim=st, test="SBI-553 vs vehicle (CMH by mouse)", p=p,
                            base_pct=float(vals["baseline"].mean()) if len(vals["baseline"]) else np.nan,
                            veh_pct=float(vals["vehicle"].mean()) if len(vals["vehicle"]) else np.nan,
                            sbi_pct=float(vals["SBI-553"].mean()) if len(vals["SBI-553"]) else np.nan))
            # every star with its pair, only p < 0.05 drawn: base-veh / base-SBI = that day's baseline vs the injection,
            # veh-SBI = the two injection sessions (all CMH by mouse)
            yb = max([float(v.max()) for v in v4.values() if len(v)] + [0.0]) + 4
            PS.bracket(a, i + OFF["base_veh"], i + OFF["vehicle"], yb, pst.get("vehicle", np.nan), CC["vehicle"], fs=10, inset=0.005, h=1.5)
            PS.bracket(a, i + OFF["base_sbi"], i + OFF["SBI-553"], yb, pst.get("SBI-553", np.nan), CC["SBI-553"], fs=10, inset=0.005, h=1.5)
            PS.bracket(a, i + OFF["vehicle"], i + OFF["SBI-553"], yb + 10, p, "k", fs=10, inset=0.005, h=1.5)
        a.set_xticks(range(4), ["light", "mild", "pin", "heat"])
        a.set_title(bname, fontsize=13)
        a.set_ylim(0, min(140, max(ymax * 1.3 + 14, ymax + 28)))
        if a.get_ylim()[1] > 100:                              # 10-24: a % of deliveries stops at 100; the room above
            a.set_yticks(np.arange(0, 101, 20))                # is only for the brackets (unlabelled)
        if k % 4 == 0:
            a.set_ylabel("% of deliveries\n(bar = mean of mice, dot = mouse)")
    ax.ravel()[0].legend(frameon=False, fontsize=9.5, loc="upper left")
    fig.tight_layout(rect=(0, 0.03 if B6_SUB is None else 0.055, 1, 1))
    # 10-24: the methods text goes into the slide notes; the figure keeps the one-line star key
    figfoot.note(fig, KEY4 + "; Cochran-Mantel-Haenszel over the deliveries, each mouse its own stratum.", "en", y=0.004)
    fig.savefig(os.path.join(OD, f"P6_behaviour_per_stimulus{FN_SUF}.png"), dpi=140)
    plt.close(fig)
    Rb = pd.DataFrame(res)
    if FN_SUF:
        return D, Rb
    Rb.to_csv(os.path.join(OD, "P6_behaviour_tests.csv"), index=False)
    def p6b(behs, fn, figsize):
        fig, ax = plt.subplots(1, len(behs), figsize=figsize, sharex=True, squeeze=False)
        for k, bname in enumerate(behs):
            a = ax[0, k]
            for i, st in enumerate(STIM):
                for j, c in enumerate(COND):
                    pm = D[(D.stim == st) & (D.condition == c)].groupby("mouse")[f"{bname}_s"].mean()
                    x = i + (j - 1) * 0.27
                    a.bar(x, pm.mean() if len(pm) else 0, width=0.25, color=CC[c])
                    a.scatter(np.full(len(pm), x), pm.values, s=12, color="k", zorder=4, clip_on=False, edgecolor="w", lw=0.4)
            a.set_xticks(range(4), ["light", "mild", "pin", "heat"])
            a.set_title(bname, fontsize=13)
            a.set_ylabel("s of the behaviour in the\n3 s after a delivery" if k == 0 else "s in the 3 s after a delivery")
            a.set_ylim(bottom=0)                               # 10-24: seconds cannot be negative
            if not (D[f"{bname}_s"].fillna(0) > 0).any():      # e.g. guarding: nothing to show -> say so, axis 0..0.05 s
                a.set_ylim(0, 0.05)
                a.text(0.5, 0.55, f"no {bname} after any delivery\n(0 s in every session)", transform=a.transAxes,
                       ha="center", va="center", fontsize=12, color="0.3")
        fig.legend(handles=[Patch(color=CC[c], label={"baseline": "baseline (both days)", "vehicle": "after vehicle",
                                                       "SBI-553": "after SBI-553"}[c]) for c in COND]
                   + [Line2D([], [], marker="o", ls="", color="k", ms=4, label="dot = one mouse (n = 4)")],
                   loc="lower center", ncol=4, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 0.0))
        fig.tight_layout(rect=(0, 0.1, 1, 1))
        fig.savefig(os.path.join(OD, fn), dpi=140, bbox_inches="tight")
        plt.close(fig)
    # 10-23: whole figure + two native parts (the automatic split cut the legend); bars = mean of the mice, descriptive
    p6b(["attending", "lick/bite", "guarding", "escape/rearing", "movement", "grooming"], "P6b_behaviour_seconds.png", (18, 4.4))
    p6b(["attending", "lick/bite", "guarding"], "P6b_behaviour_seconds_p1.png", (13, 4.8))
    p6b(["escape/rearing", "movement", "grooming"], "P6b_behaviour_seconds_p2.png", (13, 4.8))
    return D, Rb


def plot7_brain_behaviour(T, D, M=None):
    """basic plot 7: per session (16): behaviour per delivery vs % activated / % inhibited neurons (any stimulus).
    M = the saved P7_sessions.csv (redraw only: --step p7)"""
    behs = ["withdrawal", "flinch", "attending", "movement", "escape/rearing"]
    if M is None:
        X = T[T.event.isin(STIM)]

        def any_cls(v):
            v = set(v)
            return "activated" if "activated" in v else ("inhibited" if "inhibited" in v else "non-responsive")
        per = X.groupby(["mouse", "day", "phase", "cell"]).cls.agg(any_cls).reset_index()
        N = per.groupby(["mouse", "day", "phase"]).cls.value_counts(normalize=True).unstack(fill_value=0).reset_index()
        Bh = D.groupby(["mouse", "day", "phase"])[behs].mean().reset_index()
        M = N.merge(Bh, on=["mouse", "day", "phase"])
        M["condition"] = [cond_of(m, d, p) for m, d, p in zip(M.mouse, M.day, M.phase)]
        M.to_csv(os.path.join(OD, "P7_sessions.csv"), index=False)
    # 10-24 figure review: the stats titles ran into each other (the last was cut) -> two short lines; the parts (crop_split
    # rows: activated / inhibited, the legend band repeated) left ~30 % of the slide empty -> taller panels (each part ~13 x 5.3
    # in, the slide box is 12.4 x 5.2); the same y axis in every panel of a row; fill colour = condition named in the legend
    fig, ax = plt.subplots(2, len(behs), figsize=(13, 10.4), sharey="row")
    res = []
    for i, g in enumerate(("activated", "inhibited")):
        for k, bname in enumerate(behs):
            a = ax[i, k]
            if g not in M:
                continue
            for xv, yv, mm, cc in zip(M[bname], M[g], M.mouse, M.condition):
                a.scatter(100 * xv, 100 * yv, marker=MK[mm], s=60, color=CC[cc], edgecolor=MC[mm], lw=1.8, zorder=3, clip_on=False)
            rho, p = spearmanr(M[bname], M[g])
            dm = M.copy()
            for col in (bname, g):
                dm[col] = dm[col] - dm.groupby("mouse")[col].transform("mean")
            rw, pw = spearmanr(dm[bname], dm[g])
            res.append(dict(group=g, behaviour=bname, rho=rho, p=p, rho_within_mouse=rw, p_within=pw))
            pf = lambda q: "p < 0.001" if q < 0.001 else f"p = {q:.3f}" if q < 0.01 else f"p = {q:.2f}"
            a.set_title(f"all: rho {rho:+.2f}, {pf(p)}\nwithin mouse: {rw:+.2f}, {pf(pw)}", fontsize=10.5, linespacing=1.2)
            a.set_xlabel(f"{bname}\nper delivery (%)", fontsize=12)
            a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
            a.tick_params(labelsize=11)
            if k == 0:
                a.set_ylabel(f"% {g} neurons\n(any stimulus)", fontsize=12)
    hd = lambda t: Line2D([], [], ls="", marker="", label=t)          # a text-only legend entry (header)
    fig.legend(handles=[hd("mouse (shape, edge):")]
               + [Line2D([], [], marker=MK[m], ls="", mec=MC[m], mfc="w", mew=1.8, ms=8, label=m) for m in MO]
               + [hd("   fill = condition:")]
               + [Line2D([], [], marker="o", ls="", mfc=CC[c], mec=CC[c], ms=9, label=c) for c in COND],
               loc="upper center", ncol=9, frameon=False, fontsize=11.5, handletextpad=0.3, columnspacing=1.0, handlelength=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=3.5, w_pad=1.2)
    fig.savefig(os.path.join(OD, "P7_brain_behaviour.png"), dpi=140)
    plt.close(fig)
    Rr = pd.DataFrame(res)
    Rr.to_csv(os.path.join(OD, "P7_correlations.csv"), index=False)
    return Rr


def jdump(J, path):
    def conv(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(J, f, indent=1, default=conv)


# ------------------------------------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="all", choices=["all", "window", "rest", "classify", "overlap", "plots", "p7", "p6", "p5"])
    ap.add_argument("--bg-regress", action="store_true", help="regress the field brightness out of every neuron; writes "
                    "analysis\\event_analysis_bgreg")
    a = ap.parse_args(argv)
    global OD, BGREG
    if a.bg_regress:
        BGREG = True
        OD = os.path.join(BASE, "analysis", "event_analysis_bgreg")
    steps = {"all": {"window", "classify", "overlap", "plots"}, "rest": {"classify", "overlap", "plots"}}.get(a.step, {a.step})
    os.makedirs(OD, exist_ok=True)
    if a.step == "p7":                                         # redraw P7 from its table (no session loading)
        M = pd.read_csv(os.path.join(OD, "P7_sessions.csv")); M["mouse"] = M.mouse.astype(str)
        plot7_brain_behaviour(None, None, M)
        print("-> P7 redrawn"); return
    if a.step == "p5":                                         # redraw P5 and its parts from P5_fate.csv
        F = pd.read_csv(os.path.join(OD, "P5_fate.csv")); F["mouse"] = F.mouse.astype(str)
        for suf, sub in (("", None), ("_stim", STIM), ("_beh1", ["withdrawal", "flinch", "movement onset"]),
                         ("_beh2", ["escape/rearing onset", "grooming onset", "attending onset"])):
            globals()["EV_SUB"], globals()["FN_SUF"] = sub, suf
            plot5_fate(None, None, F)
        globals()["EV_SUB"], globals()["FN_SUF"] = None, ""
        print("-> P5 redrawn"); return
    if a.step == "p6":                                         # redraw the P6 parts from the per-delivery table
        Dd = pd.read_csv(os.path.join(OD, "P6_per_delivery.csv")); Dd["mouse"] = Dd.mouse.astype(str)
        for suf, sub in (("_w", ["withdrawal", "flinch"]), ("_x", ["attending", "lick/bite"])):
            globals()["B6_SUB"], globals()["FN_SUF"] = sub, suf
            plot6_behaviour(None, Dd)
        globals()["B6_SUB"], globals()["FN_SUF"] = None, ""
        print("-> P6 parts redrawn"); return
    js = os.path.join(OD, "summary.json")
    J = json.load(open(js, encoding="utf-8")) if os.path.isfile(js) else {}
    t0 = time.time()
    S = load_all()
    print(f"{len(S)} sessions loaded ({time.time() - t0:.0f} s)" + (" - field regressed out" if BGREG else ""), flush=True)
    if BGREG:
        J["bg_beta"] = {f"{m}_d{d}_{ph}": dict(median=float(np.nanmedian(s_["bg_beta"])), q90=float(np.nanpercentile(s_["bg_beta"], 90)))
                        for (m, d, ph), s_ in S.items() if "bg_beta" in s_}
    wf = os.path.join(OD, "window_search.csv")
    if "window" in steps or not os.path.isfile(wf):
        Tw = step_window(S)
    else:
        Tw = pd.read_csv(wf)
        Tw["mouse"] = Tw.mouse.astype(str)
    W, Lw, win = choose_windows(Tw)
    J["windows"] = {ev: {str(d): list(W[ev][d]) for d in (1, 2)} for ev in EVENTS}
    J["window_strategy"] = win
    J["window_lomo"] = {f"{k[0]} | {k[1]}": v for k, v in Lw.groupby(["windows", "strategy"]).quality.mean().to_dict().items()}
    if True:                                                   # the window figures are cheap: always redrawn
        fig_window(Tw, W, Lw, win)
        fig_window(Tw, W, Lw, win, col="slow", fn="W1b_window_search_slow.png")
        J["named_windows"] = fig_named_windows(Tw)
        J["timing"] = fig_timing(S, W)
        G = Tw.groupby(["event", "pre", "gap", "a", "b"]).quality.mean()
        Gs = Tw.groupby(["event", "pre", "gap", "a", "b"]).slow.mean()
        J["window_best_slow"] = {ev: dict(window=list(Gs.loc[ev].idxmax()), slow=float(Gs.loc[ev].max()),
                                          locked_there=float(G.loc[ev].loc[Gs.loc[ev].idxmax()])) for ev in SEARCH_EV}
        J["window_quality"] = {ev: {"chosen": float(G.get((ev,) + tuple(W[ev][1]), np.nan)),
                                    "default": float(G.get((ev,) + DEFAULT, np.nan)),
                                    "best_any": float(G.loc[ev].max()), "best_any_window": list(G.loc[ev].idxmax())} for ev in SEARCH_EV}
        Cn = Tw.groupby(["event", "pre", "gap", "a", "b"]).contamination.mean()
        J["window_contamination"] = {ev: {"chosen": float(Cn.get((ev,) + tuple(W[ev][1]), np.nan)),
                                          "best_any": float(Cn.loc[ev].loc[G.loc[ev].idxmax()])} for ev in SEARCH_EV}
        jdump(J, js)
        print("windows:", {SHORT[ev]: (W[ev][1], W[ev][2]) for ev in EVENTS}, "| strategy:", win, flush=True)
    cf = os.path.join(OD, "cells_classes.csv")
    if "classify" in steps or not os.path.isfile(cf):
        step_classify(S, W)
    T = pd.read_csv(cf)
    T["mouse"] = T.mouse.astype(str)
    if "overlap" in steps:
        Sp, Gm = step_overlap(S, T, W)
        J["overlap"] = fig_overlap(Sp, Gm)
        jdump(J, js)
    elif "plots" in steps and os.path.isfile(os.path.join(OD, "O3_encoding_model.csv")):
        Sp = pd.read_csv(os.path.join(OD, "O2_with_without_withdrawal.csv"))
        Gm = pd.read_csv(os.path.join(OD, "O3_encoding_model.csv"))
        J["overlap"] = fig_overlap(Sp, Gm)
    if "plots" in steps:
        P = pairs()
        J["plot1"] = plot1_whole(S, P)
        J["plot1_stats"] = STATS1
        J["plot2"] = plot2_average(S)
        R3 = plot3_categories(T)
        plot3_sessions(T)
        J["plot3"] = dict(pct=R3[R3.pct_resp.notna()][["event", "condition", "pct_resp", "pct_act", "pct_inh", "n"]].to_dict("records"),
                          mcnemar=R3[R3.p_mcnemar.notna()][["event", "condition", "p_mcnemar", "lost", "gained", "base_pct",
                                                            "inj_pct", "n_pairs"]].to_dict("records"),
                          fisher=R3[R3.p_fisher.notna()][["event", "p_fisher"]].to_dict("records"))
        J["plot3_transitions"] = plot3_transitions(T)
        Sb, Jm = plot3_stability_overlap(T, P)
        J["plot3_stability"] = Sb.to_dict("records")
        J["plot3_overlap"] = Jm.round(1).to_dict()
        J["plot3_events"] = plot3_event_heatmaps(S, T, W)
        Tp = plot4_topography(T)
        R0 = Tp[Tp.group == "responsive"]
        J["plot4"] = dict(n_tests=int(len(R0)), n_sig=int((R0.p < 0.05).sum()), expected=0.05 * len(R0),
                          by_event={ev: int((R0[R0.event == ev].p < 0.05).sum()) for ev in EVENTS},
                          n_by_event={ev: int((R0.event == ev).sum()) for ev in EVENTS})
        J["plot4_maps"] = plot4_maps(T)
        F, Rf = plot5_fate(S, W)
        J["plot5"] = Rf.to_dict("records")
        Dd, Rb = plot6_behaviour(S)
        J["plot6"] = Rb.to_dict("records")
        global EV_SUB, B6_SUB, FN_SUF                          # the split parts of the wide figures (10-22, legibility)
        for suf, sub in (("_stim", STIM), ("_beh1", ["withdrawal", "flinch", "movement onset"]),
                         ("_beh2", ["escape/rearing onset", "grooming onset", "attending onset"])):
            EV_SUB, FN_SUF = sub, suf
            plot3_categories(T); plot3_sessions(T); plot5_fate(S, W)
        EV_SUB = None
        for suf, sub in (("_a", ["withdrawal", "flinch", "attending", "lick/bite"]), ("_b", ["escape/rearing", "movement", "grooming"]),
                         ("_w", ["withdrawal", "flinch"]), ("_x", ["attending", "lick/bite"])):
            B6_SUB, FN_SUF = sub, suf
            plot6_behaviour(S)
        B6_SUB, FN_SUF = None, ""
        Rr = plot7_brain_behaviour(T, Dd)
        J["plot7"] = Rr.to_dict("records")
        jdump(J, js)
    print(f"-> {OD} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
