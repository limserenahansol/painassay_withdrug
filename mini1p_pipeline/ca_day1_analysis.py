"""Day-1 Ca analysis of the SBI-553 mini1p cohort: baseline session vs injection session, per mouse.

Inputs per mouse-day (made by ca_post_mouse.sh): 5_analysis/traces/<rid>_traces.npz + _events.csv
(ca_trace_outputs.py: local dF/F, robust z, OASIS events) and the final scoring files (your manual
deliveries / reflexes / affective behaviour + automatic movement and grooming). Pre and post share
one EXTRACT run, so cell k is the same neuron in both sessions (paired tests).
Join rule (K = 0): scoring / camera frame n (1-based) was exposed during isx_frame n (0-based).

A  WHOLE SESSION (16 vs 16 min; first 15 s left out: LED switch-on decay)
   per cell: transient rate (events/min), deconvolved activity (OASIS s per min), mean local dF/F;
   also per schedule block (the 3-min no-stimulus start, each stimulus block) and split by the
   mouse's state (moving vs still frames, automatic movement), because SBI-553 lowered movement
   and activity could follow movement rather than the drug.
   Test: pre vs post per cell, Wilcoxon signed-rank (two-sided) within each mouse.
B  EVENT-LOCKED (z = local dF/F / robust SD)
   events: deliveries of each stimulus, withdrawal, flinch, movement onset, grooming onset,
   escape/rearing onset (movement / grooming / escape onsets need 2 s without that state before).
   Per cell and event type: trial response r = mean z [0, +2] s - mean z [-2, 0] s (mini2p
   deck windows); Wilcoxon signed-rank of r over trials (>= 5 trials), Benjamini-Hochberg over the
   cells of that mouse, session and event; q <= 0.20 = responsive (exploratory, as the mini2p
   deck), UP or DOWN by the sign of the median r; q <= 0.05 counted as well.
   ROBUST = also significant against 1000 time-shifted copies of the same event train (circular
   shift inside each schedule segment, one offset per segment, so the events keep their spacing
   and their block; BH q <= 0.20) - drift, slow dynamics and the block state cannot pass it.
   Population (deck slide 14): AUC of the trial-baselined z over [-2,0] vs [0,2], Wilcoxon
   across cells. Response change pre -> post: per cell, Wilcoxon across cells.
C  ARMS: SBI-553 (5941, 5939) vs vehicle (5937, 1191); n = 2 mice per arm, so the arm comparison
   is shown per mouse, never tested.

Outputs (--out): CSV tables and the figures listed in FIGURES.md written next to them.
  python ca_day1_analysis.py --out-root <mini1p_out/20261019> --scoring-dir <final> --out <analysis/ca_day1>
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import wilcoxon

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import safe_io  # noqa: E402,F401  (a figure save retries on a transient Windows write error)
import pairstars as PS  # noqa: E402  (user rule 10-22: every star shows the pair it compares)

FPS = 30.0
SKIP_S = 15.0
# Day switch (2026-10-21): CA_DAY=2 runs every analysis on Day 2 (crossover: 5937 / 1191 SBI-553, 5941 / 5939
# vehicle); paths <mouse>_d<DAY>, output files ca_d<DAY>_*. Default 1 = Day 1 exactly as before.
DAY = int(os.environ.get("CA_DAY", "1"))
_SEX = {"5941": "F", "5939": "M", "5937": "M", "1191": "F"}
_ARMS = {1: {"5941": "SBI-553", "5939": "SBI-553", "5937": "vehicle", "1191": "vehicle"},
         2: {"5937": "SBI-553", "1191": "SBI-553", "5941": "vehicle", "5939": "vehicle"}}
MICE = [(m, a, _SEX[m]) for a_ in ("SBI-553", "vehicle") for m, a in _ARMS[DAY].items() if a == a_]
ARMC = {"SBI-553": "#c0392b", "vehicle": "#2471a3"}
PHC = {"pre": "#7f8c8d", "post": "#c0392b"}
STIM = ["Light touch", "Mild touch", "Pin prick", "Heat"]
BLOCKS = [("no stimulus", 0, 180), ("stim1", 180, 360), ("stim2", 390, 570), ("stim3", 600, 780), ("stim4", 810, 990)]
EVENTS = STIM + ["withdrawal", "flinch", "movement onset", "grooming onset", "escape/rearing onset"]
EVC = {"Light touch": "#5dade2", "Mild touch": "#f5b041", "Pin prick": "#e74c3c", "Heat": "#8e44ad",
       "withdrawal": "#6c3483", "flinch": "#1e8449", "movement onset": "#17a2b8", "grooming onset": "#d6338a",
       "escape/rearing onset": "#21618c"}
PRE_W, RESP_W, WIN = (-2.0, 0.0), (0.0, 2.0), (-5.0, 8.0)
# CA_WIN=best (2026-10-22): the windows chosen by the window search of ca_event_analysis.py (W1 / W2, one per family):
# stimuli 0-1 s after vs the 0.5 s before, behaviours 0-0.5 s after vs the 1 s before. Unset = the old 2-s windows above.
WIN_MODE = os.environ.get("CA_WIN", "old")
BEST_W = {"stim": ((-0.5, 0.0), (0.0, 1.0)), "beh": ((-1.0, 0.0), (0.0, 0.5))}


def windows(evn):
    """(pre window, response window) of an event, s from the event"""
    if WIN_MODE == "best":
        return BEST_W["stim"] if evn in STIM else BEST_W["beh"]
    return PRE_W, RESP_W


def wtxt(evn):
    """the response rule of an event as text, e.g. 'z[0, 1] - z[-0.5, 0] s'"""
    (a0, a1), (b0, b1) = windows(evn)
    g = lambda v: f"{v:g}"
    return f"z[{g(b0)}, {g(b1)}] - z[{g(a0)}, {g(a1)}] s"
Q_RESP = 0.20
MIN_TRIALS = 5


# ------------------------------------------------------------------------------------------ io
def scoring_for(scoring_dir, mouse, phase):
    for p in glob.glob(os.path.join(scoring_dir, "ScoringAB_*.mat")):
        S = sio.loadmat(p, squeeze_me=True)
        ph = "pre" if str(S["phase"]).lower().startswith("base") else "post"
        if str(S["mouseID"]).strip() == mouse and ph == phase:
            return S, p
    return None, None


def onsets(v, gap_s=2.0):
    """first frame index of each bout that follows >= gap_s without the state"""
    v = np.asarray(v, bool)
    d = np.diff(np.r_[0, v.astype(int)])
    on = np.where(d == 1)[0]
    off = np.where(np.diff(np.r_[v.astype(int), 0]) == -1)[0]
    keep = []
    for o in on:
        prev_off = off[off < o]
        if len(prev_off) == 0 or o - prev_off[-1] >= gap_s * FPS:
            keep.append(o)
    return np.array(keep, int)


def behaviour(S):
    """event times and per-frame states in isx frames (camera frame n = isx frame n)"""
    stim = [str(x) for x in np.atleast_1d(S["stimNames"])]
    dF = np.atleast_1d(S["dFrames"]).astype(int)
    dT = np.atleast_1d(S["dTypes"]).astype(int)
    re = np.atleast_2d(S["reflexEvents"]).astype(int).reshape(-1, 2)
    sc = np.atleast_1d(S["score"]).astype(int)
    n = len(sc)
    move = np.asarray(S["moveVec"]).astype(bool).ravel()[:n] if "moveVec" in S else np.zeros(n, bool)
    groom = np.asarray(S["groomVec"]).astype(bool).ravel()[:n] if "groomVec" in S else np.zeros(n, bool)
    ev = {}
    for nm in STIM:
        k = stim.index(nm) + 1 if nm in stim else -1
        ev[nm] = np.sort(dF[dT == k])
    ev["withdrawal"] = np.sort(re[re[:, 1] == 1, 0])
    ev["flinch"] = np.sort(re[re[:, 1] == 2, 0])
    ev["movement onset"] = onsets(move) + 1          # vector index i = camera frame i + 1
    ev["grooming onset"] = onsets(groom) + 1
    ev["escape/rearing onset"] = onsets(sc == 4) + 1
    # per isx frame (index = isx frame): state vectors shifted by one
    st = {"move": np.r_[False, move], "groom": np.r_[False, groom], "n_cam": n}
    return dict(ev=ev, st=st, stim_order=stim)


TRACES = "traces"          # 5_analysis/<TRACES>: 'traces' or 'traces_bgcorr' (background-corrected version)
# z of the trace files = local dF/F / robust SD (MAD) of the SAME session. The MAD contains the cell's
# activity: when a session is less active its MAD is smaller and the same dF/F response is larger
# in z (5941 post: MAD -25 %, noise unchanged). Tests inside one session do not care (scale-free),
# but pre-vs-post SIZES do. Z_NORM = 'noise': z = dF/F / the cell's noise SD (PSD estimate),
# pooled over the day's two sessions - one fixed scale per cell, so pre and post are comparable.
Z_NORM = "mad"


def renorm_z(T, td, tag, phase):
    sn = np.asarray(T["sn"], float)
    other = {"pre": "post", "post": "pre"}.get(phase)
    fo = os.path.join(td, f"{tag}_{other}_traces.npz") if other else ""
    if fo and os.path.isfile(fo):
        O = np.load(fo)
        if np.array_equal(O["cell_ids"], T["cell_ids"]):
            sn = np.sqrt(np.nanmean(np.vstack([sn ** 2, np.asarray(O["sn"], float) ** 2]), 0))
    with np.errstate(invalid="ignore", divide="ignore"):
        T["z"] = (np.asarray(T["dff"], float) / np.where(sn > 1e-6, sn, np.nan)).astype(np.float32)
    return T


def load_session(out_root, mouse, phase, scoring_dir, day=None):
    day = DAY if day is None else day
    tag = f"{mouse}_d{day}"
    rid = f"{tag}_{phase}"
    td = os.path.join(out_root, tag, "5_analysis", TRACES)
    f = os.path.join(td, f"{rid}_traces.npz")
    if not os.path.isfile(f):
        return None
    T = dict(np.load(f))
    if Z_NORM == "noise":
        T = renorm_z(T, td, tag, phase)
    E = pd.read_csv(os.path.join(td, f"{rid}_events.csv"))
    S, sp = scoring_for(scoring_dir, mouse, phase)
    B = behaviour(S) if S is not None else None
    # registration quality per frame (motion_correct_h5 shifts table), aligned to the trace rows:
    # |shift| and the frame's correlation with the template - residual brain motion (z / non-rigid)
    # lowers the correlation and can put artefacts into the traces
    mot = None
    fsh = os.path.join(out_root, tag, "1_movie", f"{rid}_mc_shifts.csv")
    if os.path.isfile(fsh):
        Sh = pd.read_csv(fsh)
        fr = T["frame"].astype(int)
        mag = pd.Series(np.hypot(Sh.x_shift_px, Sh.y_shift_px).values, index=Sh.isx_frame.values)
        cor = pd.Series(Sh.corr_after.values, index=Sh.isx_frame.values)
        mot = dict(shift=mag.reindex(fr).values.astype(float), corr=cor.reindex(fr).values.astype(float))
        c = mot["corr"]
        med = np.nanmedian(c)
        mot["corr_thr"] = med - 5 * 1.4826 * np.nanmedian(np.abs(c - med))
    return dict(rid=rid, mouse=mouse, phase=phase, T=T, E=E, B=B, scoring=sp, motion=mot)


def sm(A, k=6):
    """0.2-s running mean along time (axis 0), display only"""
    return pd.DataFrame(A).rolling(k, center=True, min_periods=1).mean().values


# ------------------------------------------------------------------------------------ stats
def bh(p):
    p = np.asarray(p, float)
    q = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return q
    ps = p[ok]
    o = np.argsort(ps)
    m = len(ps)
    qq = ps[o] * m / np.arange(1, m + 1)
    qq = np.minimum.accumulate(qq[::-1])[::-1]
    out = np.empty(m)
    out[o] = np.minimum(qq, 1)
    q[ok] = out
    return q


def wsr(x, y=None):
    x = np.asarray(x, float)
    if y is not None:
        x = x - np.asarray(y, float)
    x = x[np.isfinite(x)]
    if len(x) < 3 or np.allclose(x, 0):
        return np.nan
    try:
        return float(wilcoxon(x).pvalue)
    except ValueError:
        return np.nan


def pstr(p):
    if not np.isfinite(p):
        return "n.a."
    return f"p = {p:.3g}" if p >= 1e-4 else "p < 0.0001"


# ------------------------------------------------------------------------- A: whole session
def whole_session(s):
    T, E, B = s["T"], s["E"], s["B"]
    fr, t = T["frame"].astype(int), T["t_s"]
    ok = T["valid"].astype(bool) & (t >= SKIP_S)
    ids = T["cell_ids"].astype(int)
    dur_min = ok.sum() / FPS / 60
    blocks = []
    for bn, s0, s1 in BLOCKS:
        nm = bn
        if B is not None and bn.startswith("stim"):
            nm = B["stim_order"][int(bn[-1]) - 1]
        blocks.append((nm, max(s0, SKIP_S), s1))
    mv = np.zeros(len(fr), bool)
    if B is not None:
        st = B["st"]["move"]
        inr = fr < len(st)
        mv[inr] = st[fr[inr]]
    rows = []
    for j, c in enumerate(ids):
        e = E[E.cell == c]
        on = e.onset_frame.values.astype(int)
        r = dict(mouse=s["mouse"], phase=s["phase"], cell=int(c),
                 rate=len(e) / dur_min, deconv=np.nansum(T["s"][ok, j]) / dur_min,
                 mean_dff=np.nanmean(T["dff"][ok, j]), amp=e.amplitude_dff.mean() if len(e) else np.nan)
        for nm, s0, s1 in blocks:
            m = ok & (t >= s0) & (t < s1)
            r[f"rate | {nm}"] = ((e.onset_s >= s0) & (e.onset_s < s1)).sum() / max(m.sum() / FPS / 60, 1e-9)
            ms = m & ~mv                                   # the same block, still frames only
            mins = ms.sum() / FPS / 60
            r[f"still rate | {nm}"] = np.isin(on, fr[ms]).sum() / mins if mins > 0.5 else np.nan
            r[f"still min | {nm}"] = mins
        nost = (t >= SKIP_S) & (t < BLOCKS[0][2])
        for lab, m in (("moving", ok & mv), ("still", ok & ~mv), ("no stimulus, still", ok & ~mv & nost)):
            mins = m.sum() / FPS / 60
            rows_on = np.isin(on, fr[m])
            r[f"rate | {lab}"] = rows_on.sum() / mins if mins > 0.5 else np.nan
            r[f"min | {lab}"] = mins
        rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------- B: event-locked
def peri(s, ev_frames, pre0=None):
    """trials x time x cells of z around isx frames ev_frames (rows looked up by isx frame); pre0 = the start of the event's
    pre window (an event is used when that start is after SKIP_S; default the old -2 s)"""
    T = s["T"]
    fr = T["frame"].astype(int)
    Z = T["z"]
    k0, k1 = int(WIN[0] * FPS), int(WIN[1] * FPS)
    out, used = [], []
    for f in ev_frames:
        i = np.searchsorted(fr, f)
        if i >= len(fr) or fr[i] != f:
            continue
        if fr[i] / FPS + (PRE_W[0] if pre0 is None else pre0) < SKIP_S or i + k0 < 0 or i + k1 >= len(fr):
            continue
        w = Z[i + k0:i + k1 + 1]
        if np.mean(np.isnan(w[:, 0])) > 0.2:
            continue
        out.append(w)
        used.append(f)
    tt = np.arange(k0, k1 + 1) / FPS
    if not out:
        return None, tt, []
    return np.stack(out), tt, used


def shift_null(s, used, n_shuf=1000, seed=0, pre=None, rsp=None):
    """two-sided p of each cell's mean response against the SAME event train shifted in time
    WITHIN its schedule segment (the 3-min no-stimulus start, each stimulus block, each rest): one
    random circular offset per segment and shuffle, so the events keep their spacing AND their
    block context. A shift over the whole session (the first version) moved deliveries out of the
    stimulus blocks, where activity is in a different state (A7: it ramps up through a block), so
    the block state itself could pass as a time-locked response."""
    T = s["T"]
    fr = T["frame"].astype(int)
    Z = np.asarray(T["z"], float)
    ok = np.isfinite(Z)
    Zc = np.vstack([np.zeros((1, Z.shape[1])), np.cumsum(np.where(ok, Z, 0), 0)])
    Nc = np.vstack([np.zeros((1, Z.shape[1])), np.cumsum(ok, 0)])
    n = len(fr)
    pre, rsp = (PRE_W if pre is None else pre), (RESP_W if rsp is None else rsp)     # both start / end at the event
    wa, wb = int(round((rsp[1] - rsp[0]) * FPS)), int(round((pre[1] - pre[0]) * FPS))   # response / pre window (frames)
    w = max(wa, wb)
    st = int(np.searchsorted(fr, int(SKIP_S * FPS))) + wb
    idx = np.searchsorted(fr, np.asarray(used, int))

    def resp(ii):
        ii = ii[(ii >= st) & (ii + wa < n)]
        a = (Zc[ii + wa] - Zc[ii]) / np.maximum(Nc[ii + wa] - Nc[ii], 1)
        b = (Zc[ii] - Zc[ii - wb]) / np.maximum(Nc[ii] - Nc[ii - wb], 1)
        return np.nanmean(a - b, 0)
    obs = resp(idx)
    # segment bounds (rows): start, the blocks and the rests between them, the end
    bounds_s = sorted({SKIP_S + w / FPS} | {b_ for _, b0, b1 in BLOCKS for b_ in (b0, b1)} | {fr[-1] / FPS})
    bounds = np.unique(np.searchsorted(fr, (np.array(bounds_s) * FPS).astype(int)))
    seg = np.searchsorted(bounds, idx, side="right") - 1
    rng = np.random.default_rng(seed)
    cnt = np.zeros(Z.shape[1])
    for _ in range(n_shuf):
        ii = idx.copy()
        for k in np.unique(seg):
            if k < 0 or k + 1 >= len(bounds):
                continue
            a0, a1 = bounds[k], bounds[k + 1]
            L = a1 - a0
            if L <= 2 * w:
                continue
            msk = seg == k
            ii[msk] = a0 + (idx[msk] - a0 + int(rng.integers(w, L - w))) % L
        cnt += np.abs(resp(np.sort(ii))) >= np.abs(obs)
    return (1 + cnt) / (1 + n_shuf)


def event_tests(s, null=True):
    rows, psth = [], {}
    if s["B"] is None:
        return pd.DataFrame(), psth
    ids = s["T"]["cell_ids"].astype(int)
    for evn in EVENTS:
        pre_w, rsp_w = windows(evn)
        X, tt, used = peri(s, s["B"]["ev"].get(evn, []), pre0=pre_w[0])
        if X is None:
            continue
        pre = (tt >= pre_w[0]) & (tt < pre_w[1])
        rsp = (tt >= rsp_w[0]) & (tt < rsp_w[1])
        base = np.nanmean(X[:, pre, :], 1, keepdims=True)
        Xb = X - base                                    # trial-baselined z
        r = np.nanmean(Xb[:, rsp, :], 1)                 # trials x cells
        # speed of the response: the fraction of it reached in the first 100 ms. GCaMP rises over
        # ~0.1-0.2 s and decays with tau ~1.3 s, so a change that is mostly there 1-3 frames after
        # the event is faster than the indicator - an optical / focus step, not calcium (5941 cell
        # #27: a 'DOWN' dip mostly within 2 frames of each withdrawal)
        mtr = np.nanmean(Xb, 0)                           # time x cells (trial mean, baselined)
        early = np.nanmean(mtr[(tt >= 0) & (tt < 0.1)], 0)
        seg = sm(mtr[rsp], 3)                             # 0.1-s smoothing for the extremum only
        ext = seg[np.nanargmax(np.abs(seg), 0), np.arange(seg.shape[1])]      # signed peak in the response window
        step_frac = np.where(np.abs(ext) > 1e-6, early / np.where(np.abs(ext) > 1e-6, ext, 1), np.nan)
        dt = 1 / FPS
        auc_pre = np.trapezoid(np.nan_to_num(Xb[:, pre, :]), dx=dt, axis=1).mean(0)
        auc_post = np.trapezoid(np.nan_to_num(Xb[:, rsp, :]), dx=dt, axis=1).mean(0)
        p = np.array([wsr(r[:, j]) if len(used) >= MIN_TRIALS else np.nan for j in range(len(ids))])
        q = bh(p)
        med = np.nanmedian(r, 0)
        cls = np.where(q <= Q_RESP, np.where(med > 0, "UP", "DOWN"), "no response")
        ps_ = shift_null(s, used, pre=pre_w, rsp=rsp_w) if (null and len(used) >= MIN_TRIALS) else np.full(len(ids), np.nan)
        qs_ = bh(ps_)
        psth[evn] = dict(mean=sm(np.nanmean(Xb, 0)), tt=tt, n=len(used), cls=cls,   # display: 0.2-s mean
                         r_trials=r, pre_trials=np.nanmean(X[:, pre, :], 1), frames=np.asarray(used),
                         # sham response: z[pre] - z[the same length before it] (old: z[-2,0] - z[-4,-2]), both
                         # BEFORE the delivery - the same drift, but no response to this delivery (decoding control)
                         sham_trials=np.nanmean(X[:, pre, :], 1) - np.nanmean(
                             X[:, (tt >= 2 * pre_w[0] - pre_w[1]) & (tt < pre_w[0]), :], 1),
                         Xb=Xb.astype(np.float32))
        for j, c in enumerate(ids):
            rows.append(dict(mouse=s["mouse"], phase=s["phase"], event=evn, cell=int(c), n_trials=len(used),
                             resp_mean=float(np.nanmean(r[:, j])), resp_median=float(med[j]), p=p[j], q=q[j],
                             cls=cls[j], strict=bool(q[j] <= 0.05) if np.isfinite(q[j]) else False,
                             p_shift=ps_[j], q_shift=qs_[j],
                             cls_shift=("UP" if np.nanmean(r[:, j]) > 0 else "DOWN") if (np.isfinite(qs_[j]) and qs_[j] <= Q_RESP)
                             else "no response",
                             robust=bool(cls[j] != "no response" and np.isfinite(qs_[j]) and qs_[j] <= Q_RESP),
                             auc_pre=auc_pre[j], auc_post=auc_post[j], step_frac=float(step_frac[j])))
    return pd.DataFrame(rows), psth


# ----------------------------------------------------------------------------- figures
def fig_whole(W, out):
    mice = [m for m in MICE if m[0] in set(W.mouse)]
    fig, ax = plt.subplots(1, len(mice) + 1, figsize=(4.2 * (len(mice) + 1), 4.6))
    ax = np.atleast_1d(ax)
    summ = []
    for k, (m, arm, sex) in enumerate(mice):
        P = W[W.mouse == m].pivot(index="cell", columns="phase", values="rate").dropna()
        if not {"pre", "post"} <= set(P.columns):
            continue
        for _, r in P.iterrows():
            ax[k].plot([0, 1], [r.pre, r.post], color="0.75", lw=0.6)
        ax[k].plot([0, 1], [P.pre.median(), P.post.median()], color=ARMC[arm], lw=3, marker="o", label="median")
        p = wsr(P.post, P.pre)
        ax[k].set_xticks([0, 1], ["baseline", "injection"])
        ax[k].set_xlim(-0.3, 1.3)
        ax[k].set_ylabel("Ca transients per min")
        ax[k].set_title(f"{m} ({arm}, {sex})\nn = {len(P)} cells, Wilcoxon {pstr(p)}", fontsize=10, color=ARMC[arm])
        ax[k].legend(fontsize=8, frameon=False)
        lr = np.log2((P.post + 0.1) / (P.pre + 0.1))
        summ.append(dict(mouse=m, arm=arm, n_cells=len(P), pre_median=P.pre.median(), post_median=P.post.median(),
                         log2_ratio_median=lr.median(), p_wilcoxon=p))
    S = pd.DataFrame(summ)
    a = ax[-1]
    for i, arm in enumerate(["vehicle", "SBI-553"]):
        v = S[S.arm == arm]
        a.scatter(np.full(len(v), i) + np.linspace(-0.08, 0.08, max(len(v), 1))[:len(v)], v.log2_ratio_median,
                  color=ARMC[arm], s=60, zorder=3)
        for _, r in v.iterrows():
            a.annotate(r.mouse, (i, r.log2_ratio_median), xytext=(8, 0), textcoords="offset points", fontsize=8)
    a.axhline(0, color="k", lw=0.8)
    a.set_xticks([0, 1], ["vehicle", "SBI-553"]); a.set_xlim(-0.5, 1.5)
    a.set_ylabel("median over cells of log2(post / pre)")
    a.set_title("per mouse (n = 2 per arm: shown, not tested)", fontsize=10)
    fig.suptitle("Whole session: Ca transient rate per cell, baseline vs injection (first 15 s excluded)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"ca_d{DAY}_A1_whole_session_rate.png"), dpi=120); plt.close(fig)
    return S


def fig_block_aligned(sessions, out, bin_s=10.0, pre_s=60.0):
    """A7: activity aligned to the START of each stimulus block (from pre_s before to the end of the
    3-min block, 10-s bins): transients per cell per minute, all frames (solid) and still frames
    only (dashed), baseline vs injection. Shows whether the stimulation raises activity in a
    sustained way (a state during the block) rather than as time-locked responses."""
    mice = [m for m in MICE if any(s["mouse"] == m[0] for s in sessions)]
    fig, ax = plt.subplots(len(mice), len(STIM), figsize=(3.6 * len(STIM), 2.8 * len(mice)), squeeze=False)
    rows = []
    edges = np.arange(-pre_s, 180 + bin_s / 2, bin_s)
    ctr = (edges[:-1] + edges[1:]) / 2
    for i, (m, arm, sex) in enumerate(mice):
        for s in [x for x in sessions if x["mouse"] == m]:
            if s["B"] is None:
                continue
            T, E = s["T"], s["E"]
            t = T["t_s"]; fr = T["frame"].astype(int)
            mv = s["B"]["st"]["move"][fr.clip(0, len(s["B"]["st"]["move"]) - 1)]
            nc = len(T["cell_ids"])
            on = E.onset_s.values
            onf = E.onset_frame.values.astype(int)
            still_on = ~mv[np.searchsorted(fr, onf).clip(0, len(fr) - 1)]
            for k, (bn, b0, b1) in enumerate(BLOCKS[1:]):
                st = s["B"]["stim_order"][k]
                j = STIM.index(st)
                a = ax[i, j]
                rr, rs = [], []
                for e0, e1 in zip(edges[:-1], edges[1:]):
                    w0, w1 = b0 + e0, b0 + e1
                    msk = (t >= max(w0, SKIP_S)) & (t < w1)
                    mins = msk.sum() / FPS / 60
                    mins_s = (msk & ~mv).sum() / FPS / 60
                    n_all = ((on >= w0) & (on < w1)).sum()
                    n_st = ((on >= w0) & (on < w1) & still_on).sum()
                    rr.append(n_all / nc / mins if mins > 0.05 else np.nan)
                    rs.append(n_st / nc / mins_s if mins_s > 0.05 else np.nan)
                    rows.append(dict(mouse=m, phase=s["phase"], stimulus=st, t_from_block_start=(e0 + e1) / 2,
                                     rate_all=rr[-1], rate_still=rs[-1], still_fraction=mins_s / mins if mins else np.nan))
                a.plot(ctr, rr, "-", color=PHC[s["phase"]], lw=1.4, label=("baseline" if s["phase"] == "pre" else "injection"))
                a.plot(ctr, rs, "--", color=PHC[s["phase"]], lw=0.9)
                a.axvspan(0, 180, color="0.95", lw=0, zorder=0)
                a.axvline(0, color="k", lw=0.6)
                if i == 0:
                    a.set_title(st, fontsize=9, color=EVC[st])
                if j == 0:
                    a.set_ylabel(f"{m} ({arm})\ntransients / cell / min", fontsize=8, color=ARMC[arm])
                if i == len(mice) - 1:
                    a.set_xlabel("s from block start", fontsize=8)
                a.tick_params(labelsize=7)
        ax[i, 0].legend(fontsize=6, frameon=False)
    fig.suptitle("A7 activity through each stimulus block (grey = the 3-min block, 10-s bins; solid = all frames, dashed = "
                 "still frames only), baseline vs injection", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(out, f"ca_d{DAY}_A7_block_aligned.png"), dpi=115); plt.close(fig)
    return pd.DataFrame(rows)


def fig_blocks_still(W, out):
    """A6: the activity of each schedule block from frames WITHOUT movement only. In the baseline
    session of 5941 the stimulus blocks were 2-3x more active than the start, but the mouse also
    moved more during them; this separates the block (stimulation / state) from the movement"""
    mice = [m for m in MICE if m[0] in set(W.mouse.astype(str))]
    names = ["no stimulus"] + STIM
    fig, ax = plt.subplots(1, len(mice), figsize=(4.6 * len(mice), 4.6), sharey=False, squeeze=False)
    rows = []
    for k, (m, arm, sex) in enumerate(mice):
        a = ax[0, k]
        x = np.arange(len(names))
        for dx, ph in ((-0.18, "pre"), (0.18, "post")):
            V = W[(W.mouse.astype(str) == m) & (W.phase == ph)]
            mu = [V[f"still rate | {n}"].mean() if f"still rate | {n}" in V else np.nan for n in names]
            se = [V[f"still rate | {n}"].std() / np.sqrt(max(V[f"still rate | {n}"].notna().sum(), 1))
                  if f"still rate | {n}" in V else np.nan for n in names]
            mins = [V[f"still min | {n}"].iloc[0] if f"still min | {n}" in V and len(V) else np.nan for n in names]
            a.bar(x + dx, mu, 0.36, yerr=se, color=PHC[ph], capsize=2, label="baseline" if ph == "pre" else "injection")
            for xi, mn in zip(x, mins):
                a.text(xi + dx, 0.1, f"{mn:.1f}'", ha="center", fontsize=6, color="w", rotation=90)
        for i, n in enumerate(names):
            c = f"still rate | {n}"
            P = W[W.mouse.astype(str) == m].pivot(index="cell", columns="phase", values=c).dropna() if c in W else pd.DataFrame()
            if {"pre", "post"} <= set(P.columns) and len(P) >= 3:
                p = wsr(P.post, P.pre)
                rows.append(dict(mouse=m, arm=arm, block=n, n_cells=len(P), pre_mean=P.pre.mean(), post_mean=P.post.mean(), p=p))
                if np.isfinite(p) and p < 0.05:                   # bracket = the baseline bar vs the injection bar
                    PS.bracket(a, i - 0.18, i + 0.18, max(P.pre.mean(), P.post.mean()) * 1.1 + 0.1, p, "k", fs=11, inset=0)
        a.set_xticks(x, names, rotation=30, ha="right", fontsize=8)
        a.set_ylabel("transients per min, still frames only\n(mean +- SEM over cells; minutes of still time in the bars)",
                     fontsize=8)
        a.set_title(f"{m} ({arm})", color=ARMC[arm], fontsize=10)
        a.legend(fontsize=7, frameon=False)
    fig.suptitle("A6 block activity without movement: each schedule block, frames where the mouse was still "
                 "(bracket = baseline vs injection bars of that block: * p < 0.05, ** p < 0.01, *** p < 0.001, Wilcoxon over cells)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(out, f"ca_d{DAY}_A6_blocks_still_only.png"), dpi=120); plt.close(fig)
    return pd.DataFrame(rows)


def fig_spontaneous(W, out):
    """WT deck slide 13 idea, in Ca: activity before any stimulus (first 3 min, 15 s left out),
    all frames and still frames only (movement removed), baseline vs injection session"""
    mice = [m for m in MICE if m[0] in set(W.mouse)]
    fig, ax = plt.subplots(2, len(mice), figsize=(4.2 * len(mice), 8), squeeze=False)
    rows = []
    for k, (m, arm, sex) in enumerate(mice):
        for i, c in enumerate(("rate | no stimulus", "rate | no stimulus, still")):
            a = ax[i, k]
            P = W[W.mouse == m].pivot(index="cell", columns="phase", values=c).dropna() if c in W else pd.DataFrame()
            if not {"pre", "post"} <= set(P.columns):
                a.axis("off"); continue
            for _, r in P.iterrows():
                a.plot([0, 1], [r.pre, r.post], color="0.78", lw=0.6)
            a.plot([0, 1], [P.pre.median(), P.post.median()], color=ARMC[arm], lw=3, marker="o", label="median")
            p = wsr(P.post, P.pre)
            mins = W[(W.mouse == m)].groupby("phase")["min | no stimulus, still"].first()
            a.set_xticks([0, 1], ["baseline", "injection"]); a.set_xlim(-0.3, 1.3)
            a.set_ylabel("Ca transients per min")
            extra = "" if i == 0 else (f"\nstill time: base {mins.get('pre', np.nan):.1f} min, inj {mins.get('post', np.nan):.1f} min")
            a.set_title(f"{m} ({arm}) - {'all frames' if i == 0 else 'still frames only'}\nn = {len(P)} cells, Wilcoxon "
                        f"{pstr(p)}{extra}", fontsize=9, color=ARMC[arm])
            a.legend(fontsize=7, frameon=False)
            rows.append(dict(mouse=m, arm=arm, condition=c.split("| ")[1], n_cells=len(P), pre_median=P.pre.median(),
                             post_median=P.post.median(), p=p))
    fig.suptitle("Before any stimulus (first 3 min of each session): spontaneous Ca activity, baseline vs injection; "
                 "bottom row = frames without movement only, so lower movement after SBI-553 cannot explain a change",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(out, f"ca_d{DAY}_A3_spontaneous.png"), dpi=120); plt.close(fig)
    return pd.DataFrame(rows)


def fig_timecourse(sessions, out):
    """population transient rate per minute through each session (block colours behind), and the
    fraction of time moving: does the change build up over the injection session (drug onset), and
    does it follow the movement?"""
    mice = [m for m in MICE if any(s["mouse"] == m[0] for s in sessions)]
    fig, ax = plt.subplots(len(mice), 1, figsize=(14, 2.8 * len(mice)), squeeze=False)
    rows = []
    for k, (m, arm, sex) in enumerate(mice):
        a = ax[k, 0]
        a2 = a.twinx()
        for s in [x for x in sessions if x["mouse"] == m]:
            T, E = s["T"], s["E"]
            t = T["t_s"]
            nmin = int(t[-1] // 60)
            nc = len(T["cell_ids"])
            rate = [((E.onset_s >= 60 * i) & (E.onset_s < 60 * (i + 1))).sum() / nc for i in range(nmin)]
            if nmin:
                rate[0] = rate[0] * 60 / max(60 - SKIP_S, 1)          # first minute: 15 s left out
            x = np.arange(nmin) + 0.5
            a.plot(x, rate, "o-", color=PHC[s["phase"]], ms=3, label=("baseline" if s["phase"] == "pre" else "injection"))
            if s["B"] is not None:
                mv = s["B"]["st"]["move"]
                mvm = [mv[int(60 * i * FPS):int(60 * (i + 1) * FPS)].mean() if len(mv) > 60 * i * FPS else np.nan
                       for i in range(nmin)]
                a2.plot(x, 100 * np.array(mvm), ":", color=PHC[s["phase"]], lw=1)
                if s["phase"] == "pre":
                    for j, (bn, s0, s1) in enumerate(BLOCKS[1:]):
                        a.axvspan(s0 / 60, s1 / 60, color="0.94", lw=0, zorder=0)
            for i, r in enumerate(rate):
                rows.append(dict(mouse=m, arm=arm, phase=s["phase"], minute=i + 1, transients_per_cell=r))
        a.set_ylabel("transients per cell\nper minute")
        a2.set_ylabel("% time moving (dotted)", fontsize=8)
        a.set_title(f"{m} ({arm}): through the session (grey = stimulus blocks; block order differs between sessions)",
                    fontsize=9, color=ARMC[arm])
        a.legend(fontsize=7, frameon=False, loc="upper left")
    ax[-1, 0].set_xlabel("minute of the session")
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"ca_d{DAY}_A4_timecourse.png"), dpi=115); plt.close(fig)
    return pd.DataFrame(rows)


def fig_effect_map(W, out_root, out):
    """where in the field of view activity went up or down: each cell filled with log2(post / pre)
    of its transient rate"""
    from skimage import measure
    mice = [m for m in MICE if m[0] in set(W.mouse.astype(str))]
    fig, ax = plt.subplots(1, len(mice), figsize=(4.6 * len(mice), 4.8), squeeze=False)
    for k, (m, arm, sex) in enumerate(mice):
        a = ax[0, k]
        f = os.path.join(out_root, f"{m}_d{DAY}", "3_extract", f"{m}_d{DAY}_extract.mat")
        if not os.path.isfile(f):
            a.axis("off"); continue
        M = sio.loadmat(f, squeeze_me=True, variable_names=["spatial_weights", "mean_image"])
        Sp = np.asarray(M["spatial_weights"], np.float32)
        mi = np.asarray(M["mean_image"], float)
        P = W[W.mouse.astype(str) == m].pivot(index="cell", columns="phase", values="rate").dropna()
        lr = np.log2((P.post + 0.1) / (P.pre + 0.1))
        lo, hi = np.percentile(mi, [1, 99.7])
        a.imshow(mi, cmap="gray", vmin=lo, vmax=hi)
        cmap = plt.get_cmap("RdBu_r")
        allxy = []
        for c, v in lr.items():
            fp = Sp[:, :, int(c) - 1]
            cs = measure.find_contours(fp / max(fp.max(), 1e-9), 0.5)
            if not cs:
                continue
            xy = max(cs, key=len)
            allxy.append(xy)
            a.fill(xy[:, 1], xy[:, 0], color=cmap(np.clip((v + 2) / 4, 0, 1)), lw=0)
        if allxy:
            A = np.vstack(allxy)
            a.set_xlim(A[:, 1].min() - 30, A[:, 1].max() + 30); a.set_ylim(A[:, 0].max() + 30, A[:, 0].min() - 30)
        a.set_xticks([]); a.set_yticks([])
        a.set_title(f"{m} ({arm}): {int((lr > 0).sum())} up, {int((lr < 0).sum())} down\nmedian log2 ratio {lr.median():.2f}",
                    fontsize=9, color=ARMC[arm])
    sm_ = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(-2, 2))
    fig.colorbar(sm_, ax=ax.ravel().tolist(), fraction=0.02, label="log2(injection / baseline) transient rate")
    fig.suptitle("Where activity changed after the injection: each cell coloured by its rate change (red up, blue down)",
                 fontsize=10)
    fig.savefig(os.path.join(out, f"ca_d{DAY}_A5_effect_map.png"), dpi=115); plt.close(fig)


def fig_blocks(W, out):
    mice = [m for m in MICE if m[0] in set(W.mouse)]
    names = ["no stimulus"] + STIM + ["still", "moving"]
    fig, ax = plt.subplots(1, len(mice), figsize=(5.2 * len(mice), 4.6), sharey=True)
    ax = np.atleast_1d(ax)
    rows = []
    for k, (m, arm, sex) in enumerate(mice):
        x = np.arange(len(names))
        for dx, ph in ((-0.18, "pre"), (0.18, "post")):
            V = W[(W.mouse == m) & (W.phase == ph)]
            mu = [V[f"rate | {n}"].mean() if f"rate | {n}" in V else np.nan for n in names]
            se = [V[f"rate | {n}"].std() / np.sqrt(V[f"rate | {n}"].notna().sum()) if f"rate | {n}" in V else np.nan for n in names]
            ax[k].bar(x + dx, mu, 0.36, yerr=se, color=PHC[ph], label="baseline" if ph == "pre" else "injection",
                      capsize=2, error_kw=dict(lw=0.8))
        for i, n in enumerate(names):
            c = f"rate | {n}"
            P = W[W.mouse == m].pivot(index="cell", columns="phase", values=c).dropna() if c in W else pd.DataFrame()
            if {"pre", "post"} <= set(P.columns) and len(P) >= 3:
                p = wsr(P.post, P.pre)
                rows.append(dict(mouse=m, arm=arm, condition=n, n_cells=len(P), pre_mean=P.pre.mean(), post_mean=P.post.mean(), p=p))
                if np.isfinite(p) and p < 0.05:                   # bracket = the baseline bar vs the injection bar
                    PS.bracket(ax[k], i - 0.18, i + 0.18, max(P.pre.mean(), P.post.mean()) * 1.12 + 0.05, p, "k", fs=11, inset=0)
        ax[k].set_xticks(x, names, rotation=35, ha="right", fontsize=8)
        ax[k].set_title(f"{m} ({arm})", color=ARMC[arm], fontsize=10)
        ax[k].set_ylabel("Ca transients per min (mean +- SEM over cells)")
        ax[k].legend(fontsize=8, frameon=False)
    fig.suptitle("Transient rate by schedule block and by the mouse's state (automatic movement); "
                 "bracket = baseline vs injection bars: * p < 0.05, ** p < 0.01, *** p < 0.001, Wilcoxon signed-rank over cells",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"ca_d{DAY}_A2_blocks_and_movement.png"), dpi=120); plt.close(fig)
    return pd.DataFrame(rows)


def fig_responsive(R, out, suffix="", rule="Wilcoxon over trials"):
    mice = [m for m in MICE if m[0] in set(R.mouse)]
    evs = [e for e in EVENTS if e in set(R.event)]
    fig, ax = plt.subplots(len(mice), 1, figsize=(15, 3.3 * len(mice)), squeeze=False)
    rows = []
    for k, (m, arm, sex) in enumerate(mice):
        a = ax[k, 0]
        x = np.arange(len(evs))
        for dx, ph in ((-0.2, "pre"), (0.2, "post")):
            up, dn, lab = [], [], []
            for e in evs:
                V = R[(R.mouse == m) & (R.phase == ph) & (R.event == e)]
                n = len(V)
                u = (V.cls == "UP").sum(); d = (V.cls == "DOWN").sum(); rb = int(V.robust.sum())
                up.append(100 * u / max(n, 1)); dn.append(100 * d / max(n, 1))
                nt = int(V.n_trials.iloc[0]) if n else 0
                lab.append(f"{u}+{d}/{n}\n({rb} robust)\n{nt} tr")
                rows.append(dict(mouse=m, arm=arm, phase=ph, event=e, n_cells=n, n_trials=nt, n_up=u, n_down=d,
                                 pct_up=100 * u / max(n, 1), pct_down=100 * d / max(n, 1),
                                 n_strict=int(V.strict.sum()), n_robust=rb))
            a.bar(x + dx, up, 0.38, color=PHC[ph], label=("baseline" if ph == "pre" else "injection") + " UP")
            a.bar(x + dx, [-v for v in dn], 0.38, color=PHC[ph], alpha=0.45, hatch="//",
                  label=("baseline" if ph == "pre" else "injection") + " DOWN")
            for i, l in enumerate(lab):
                a.text(i + dx, max(up[i], 2) + 1, l, ha="center", fontsize=6)
        a.axhline(0, color="k", lw=0.8)
        lim = max(10, a.get_ylim()[1] + 8, -a.get_ylim()[0] + 2)
        a.set_ylim(-lim, lim)
        a.set_xticks(x, evs, fontsize=8)
        a.set_ylabel("% of cells\n(UP above, DOWN below 0)")
        a.set_title(f"{m} ({arm}): responsive cells per event, q <= {Q_RESP:.2f} (BH over cells), "
                    f"r = z[0,2] - z[-2,0] s, {rule}; labels = UP+DOWN/cells (robust = significant by both rules), "
                    f"trials",
                    fontsize=9, color=ARMC[arm])
        a.legend(fontsize=7, frameon=False, ncol=4)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"ca_d{DAY}_B1{suffix}_responsive_cells.png"), dpi=120); plt.close(fig)
    return pd.DataFrame(rows)


def fig_responsive_matched(PS, out, n_rep=50):
    """% responsive with the SAME number of deliveries in both sessions: the count of
    'significant' cells grows with the number of trials (the synthetic check: a vehicle mouse
    with identical responses went from 13 to 26 Heat cells with 16 vs 19 trials), so pre and
    post are compared on n = min(n_pre, n_post) trials drawn at random, 50 times (mean +- SD)"""
    rows = []
    rng = np.random.default_rng(5)
    for m, arm, sex in MICE:
        for e in EVENTS:
            a_, b_ = PS.get((m, "pre", e)), PS.get((m, "post", e))
            if a_ is None or b_ is None:
                continue
            n = min(a_["n"], b_["n"])
            if n < MIN_TRIALS:
                continue
            res = {"pre": [], "post": []}
            for ph, d in (("pre", a_), ("post", b_)):
                r = d["r_trials"]
                for _ in range(n_rep if d["n"] > n else 1):
                    idx = rng.choice(d["n"], n, replace=False) if d["n"] > n else np.arange(n)
                    p = np.array([wsr(r[idx, j]) for j in range(r.shape[1])])
                    q = bh(p)
                    med = np.nanmedian(r[idx], 0)
                    res[ph].append((np.sum((q <= Q_RESP) & (med > 0)), np.sum((q <= Q_RESP) & (med < 0)), r.shape[1]))
            for ph in ("pre", "post"):
                A = np.array(res[ph], float)
                rows.append(dict(mouse=m, arm=arm, event=e, phase=ph, n_trials_matched=n, n_cells=int(A[0, 2]),
                                 pct_up=100 * A[:, 0].mean() / A[0, 2], pct_up_sd=100 * A[:, 0].std() / A[0, 2],
                                 pct_down=100 * A[:, 1].mean() / A[0, 2], pct_down_sd=100 * A[:, 1].std() / A[0, 2]))
    D = pd.DataFrame(rows)
    if not len(D):
        return D
    mice = [m for m in MICE if m[0] in set(D.mouse)]
    evs = [e for e in EVENTS if e in set(D.event)]
    fig, ax = plt.subplots(len(mice), 1, figsize=(15, 3.1 * len(mice)), squeeze=False)
    for k, (m, arm, sex) in enumerate(mice):
        a = ax[k, 0]
        x = np.arange(len(evs))
        for dx, ph in ((-0.2, "pre"), (0.2, "post")):
            V = D[(D.mouse == m) & (D.phase == ph)].set_index("event").reindex(evs)
            a.bar(x + dx, V.pct_up, 0.38, yerr=V.pct_up_sd, color=PHC[ph], capsize=2,
                  label=("baseline" if ph == "pre" else "injection") + " UP")
            a.bar(x + dx, -V.pct_down, 0.38, yerr=V.pct_down_sd, color=PHC[ph], alpha=0.45, hatch="//", capsize=2,
                  label=("baseline" if ph == "pre" else "injection") + " DOWN")
        V = D[(D.mouse == m) & (D.phase == "pre")].set_index("event").reindex(evs)
        for i, n in enumerate(V.n_trials_matched):
            if np.isfinite(n):
                a.text(i, a.get_ylim()[1] * 0.85 if a.get_ylim()[1] > 0 else 1, f"{int(n)} tr", ha="center", fontsize=7)
        a.axhline(0, color="k", lw=0.8)
        a.set_xticks(x, evs, fontsize=8)
        a.set_ylabel("% of cells, matched trials\n(UP above, DOWN below)")
        a.set_title(f"{m} ({arm}): responsive cells with the same number of deliveries in both sessions "
                    f"(n tr = min of the two; {n_rep} random draws, mean +- SD)", fontsize=9, color=ARMC[arm])
        a.legend(fontsize=7, frameon=False, ncol=4)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"ca_d{DAY}_B1b_responsive_matched_trials.png"), dpi=120); plt.close(fig)
    return D


def fig_psth(PS, R, out):
    """per mouse: population mean +- SEM of the trial-baselined z, baseline vs injection, per event"""
    for m, arm, sex in MICE:
        evs = [e for e in EVENTS if (m, "pre", e) in PS or (m, "post", e) in PS]
        if not evs:
            continue
        fig, ax = plt.subplots(2, len(evs), figsize=(2.9 * len(evs), 6.6), squeeze=False)
        for k, e in enumerate(evs):
            a = ax[0, k]
            for ph in ("pre", "post"):
                d = PS.get((m, ph, e))
                if d is None:
                    continue
                mu = np.nanmean(d["mean"], 1)
                se = np.nanstd(d["mean"], 1) / np.sqrt(d["mean"].shape[1])
                a.plot(d["tt"], mu, color=PHC[ph], lw=1.3, label=f"{'baseline' if ph == 'pre' else 'injection'} ({d['n']} tr)")
                a.fill_between(d["tt"], mu - se, mu + se, color=PHC[ph], alpha=0.25, lw=0)
            a.axvline(0, color="k", lw=0.8); a.axhline(0, color="0.5", lw=0.5)
            a.axvspan(*windows(e)[1], color="0.92", lw=0, zorder=0)
            a.set_title(e, fontsize=9, color=EVC[e])
            a.set_xlabel("time from event (s)", fontsize=8)
            if k == 0:
                a.set_ylabel("z, baselined [-2,0] s\nmean +- SEM over cells")
            a.legend(fontsize=6, frameon=False)
            # pre vs post response per cell
            b = ax[1, k]
            P = R[(R.mouse == m) & (R.event == e)].pivot(index="cell", columns="phase", values="resp_mean").dropna()
            if {"pre", "post"} <= set(P.columns):
                for _, r in P.iterrows():
                    b.plot([0, 1], [r.pre, r.post], color="0.8", lw=0.5)
                b.plot([0, 1], [P.pre.median(), P.post.median()], color=ARMC[arm], lw=2.5, marker="o")
                b.set_title(f"n = {len(P)} cells, {pstr(wsr(P.post, P.pre))}", fontsize=8)
            b.axhline(0, color="k", lw=0.5)
            b.set_xticks([0, 1], ["base", "inj"]); b.set_xlim(-0.3, 1.3)
            if k == 0:
                b.set_ylabel("response per neuron, delta z\n(mean z 0-2 s - mean z -2-0 s)" if WIN_MODE != "best" else
                             "response per neuron, delta z\n(best window: response - pre)")
        fig.suptitle(f"{m} ({arm}): event-locked population activity, baseline vs injection session "
                     f"(top: mean of all cells; bottom: each cell, median bold, Wilcoxon signed-rank)", fontsize=11,
                     color=ARMC[arm])
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"ca_d{DAY}_B2_psth_{m}.png"), dpi=110); plt.close(fig)


def fig_heatmaps(PS, out):
    for m, arm, sex in MICE:
        evs = [e for e in EVENTS if (m, "pre", e) in PS and (m, "post", e) in PS]
        if not evs:
            continue
        fig, ax = plt.subplots(1, 2 * len(evs), figsize=(1.9 * 2 * len(evs), 7.5), squeeze=False)
        for k, e in enumerate(evs):
            dp, dq = PS[(m, "pre", e)], PS[(m, "post", e)]
            tt = dp["tt"]
            rsp = (tt >= windows(e)[1][0]) & (tt < windows(e)[1][1])
            order = np.argsort(-np.nanmean(dp["mean"][rsp], 0))
            for i, (d, ph) in enumerate(((dp, "baseline"), (dq, "injection"))):
                a = ax[0, 2 * k + i]
                im = a.imshow(d["mean"][:, order].T, aspect="auto", cmap="RdBu_r", vmin=-1.5, vmax=1.5,
                              extent=[tt[0], tt[-1], len(order) - 0.5, -0.5], interpolation="nearest")
                a.axvline(0, color="k", lw=0.8)
                a.set_title(f"{e}\n{ph} ({d['n']} tr)", fontsize=8, color=EVC[e])
                a.set_xlabel("s", fontsize=7)
                a.tick_params(labelsize=6)
                if 2 * k + i == 0:
                    a.set_ylabel("cell (sorted by the baseline response, same order in both)")
        fig.colorbar(im, ax=ax.ravel().tolist(), fraction=0.01, label="trial-mean z, baselined [-2,0] s")
        fig.suptitle(f"{m} ({arm}): every cell's mean response, baseline and injection side by side", fontsize=11,
                     color=ARMC[arm])
        fig.savefig(os.path.join(out, f"ca_d{DAY}_B3_heatmaps_{m}.png"), dpi=110); plt.close(fig)


def fig_auc(R, out):
    """mini2p deck slide 14: population AUC [-2,0] vs [0,2], Wilcoxon across cells, per session"""
    mice = [m for m in MICE if m[0] in set(R.mouse)]
    evs = [e for e in EVENTS if e in set(R.event)]
    fig, ax = plt.subplots(len(mice), len(evs), figsize=(2.3 * len(evs), 2.8 * len(mice)), squeeze=False)
    rows = []
    for i, (m, arm, sex) in enumerate(mice):
        for k, e in enumerate(evs):
            a = ax[i, k]
            for j, ph in enumerate(("pre", "post")):
                V = R[(R.mouse == m) & (R.phase == ph) & (R.event == e)]
                if not len(V):
                    continue
                mu = [V.auc_pre.mean(), V.auc_post.mean()]
                se = [V.auc_pre.std() / np.sqrt(len(V)), V.auc_post.std() / np.sqrt(len(V))]
                a.bar(np.array([0, 1]) + 2.6 * j, mu, 0.9, yerr=se, color=["0.75", PHC[ph]], capsize=2)
                p = wsr(V.auc_post, V.auc_pre)
                a.text(0.5 + 2.6 * j, max(mu) + max(se) * 1.2, pstr(p).replace("p = ", "p="), ha="center", fontsize=6)
                rows.append(dict(mouse=m, arm=arm, phase=ph, event=e, n_cells=len(V), n_trials=int(V.n_trials.iloc[0]),
                                 auc_pre_mean=mu[0], auc_post_mean=mu[1], p=p))
            a.axhline(0, color="k", lw=0.5)
            a.set_xticks([0.5, 3.1], ["base", "inj"], fontsize=7)
            a.tick_params(labelsize=6)
            if i == 0:
                a.set_title(e, fontsize=8, color=EVC[e])
            if k == 0:
                a.set_ylabel(f"{m} ({arm})\nAUC z*s (mean +- SEM)", fontsize=7, color=ARMC[arm])
    fig.suptitle("Population AUC of the trial-baselined z: [-2,0] s (grey) vs [0,2] s (colour), per session; "
                 "Wilcoxon signed-rank across cells", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"ca_d{DAY}_B4_population_auc.png"), dpi=120); plt.close(fig)
    return pd.DataFrame(rows)


def fig_groups(PS, out):
    """mini2p deck slide 12: UP / DOWN / no-response group averages per event and session"""
    for m, arm, sex in MICE:
        evs = [e for e in EVENTS if (m, "pre", e) in PS or (m, "post", e) in PS]
        if not evs:
            continue
        fig, ax = plt.subplots(2, len(evs), figsize=(2.9 * len(evs), 6.2), squeeze=False)
        for r, ph in enumerate(("pre", "post")):
            for k, e in enumerate(evs):
                a = ax[r, k]
                d = PS.get((m, ph, e))
                if d is None:
                    a.axis("off"); continue
                for cl, col in (("UP", "#c0392b"), ("DOWN", "#1f618d"), ("no response", "0.7")):
                    sel = d["cls"] == cl
                    if sel.sum() == 0:
                        continue
                    mu = np.nanmean(d["mean"][:, sel], 1)
                    se = np.nanstd(d["mean"][:, sel], 1) / np.sqrt(sel.sum())
                    a.plot(d["tt"], mu, color=col, lw=1.2, label=f"{cl} (n={sel.sum()})")
                    a.fill_between(d["tt"], mu - se, mu + se, color=col, alpha=0.2, lw=0)
                a.axvline(0, color="k", lw=0.8); a.axhline(0, color="0.5", lw=0.5)
                a.axvspan(*windows(e)[1], color="0.93", lw=0, zorder=0)
                a.set_title(f"{e} - {'baseline' if ph == 'pre' else 'injection'} ({d['n']} tr)", fontsize=8, color=EVC[e])
                a.legend(fontsize=6, frameon=False)
                a.tick_params(labelsize=6)
                if k == 0:
                    a.set_ylabel("z, baselined [-2,0] s\nmean +- SEM over cells", fontsize=7)
                if r == 1:
                    a.set_xlabel("time from event (s)", fontsize=7)
        fig.suptitle(f"{m} ({arm}): UP, DOWN and non-responsive cells (q <= {Q_RESP:.2f}), each session classified on "
                     f"its own; grey band = the [0,2] s test window", fontsize=10, color=ARMC[arm])
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"ca_d{DAY}_B5_groups_{m}.png"), dpi=110); plt.close(fig)


def fig_maps(R, out_root, out, suffix="", rule="Wilcoxon"):
    """mini2p deck slide 9: where the responsive cells sit in the field of view (background = the
    max dF/F projection, where somata show; responsive cells filled and numbered)"""
    from skimage import measure
    for m, arm, sex in MICE:
        f = os.path.join(out_root, f"{m}_d{DAY}", "3_extract", f"{m}_d{DAY}_extract.mat")
        V = R[R.mouse == m]
        if not os.path.isfile(f) or not len(V):
            continue
        M = sio.loadmat(f, squeeze_me=True, variable_names=["spatial_weights", "mean_image"])
        Sp = np.asarray(M["spatial_weights"], np.float32)
        bg = np.asarray(M["mean_image"], float)
        pf = os.path.join(out_root, f"{m}_d{DAY}", "3_extract", f"{m}_d{DAY}_projections.mat")
        if os.path.isfile(pf):
            P = sio.loadmat(pf, squeeze_me=True)
            for k in ("max_dff", "max_bgsub", "CnPNR"):
                if k in P and np.asarray(P[k]).shape == bg.shape:
                    bg = np.asarray(P[k], float); break
        evs = [e for e in EVENTS if e in set(V.event)]
        fig, ax = plt.subplots(2, len(evs), figsize=(2.9 * len(evs), 6.2), squeeze=False)
        lo, hi = np.nanpercentile(bg, [1, 99.7])
        cont = {}
        for c in V.cell.unique():
            fp = Sp[:, :, int(c) - 1]
            cs = measure.find_contours(fp / max(fp.max(), 1e-9), 0.5)
            if cs:
                cont[int(c)] = max(cs, key=len)
        for r, ph in enumerate(("pre", "post")):
            for k, e in enumerate(evs):
                a = ax[r, k]
                a.imshow(bg, cmap="gray", vmin=lo, vmax=hi)
                W = V[(V.phase == ph) & (V.event == e)]
                for _, row in W.iterrows():
                    xy = cont.get(int(row.cell))
                    if xy is None:
                        continue
                    if row.cls == "no response":
                        a.plot(xy[:, 1], xy[:, 0], color="#d0d0d0", lw=0.5, alpha=0.8)
                        continue
                    col = "#e8302a" if row.cls == "UP" else "#2c9be0"
                    a.fill(xy[:, 1], xy[:, 0], color=col, alpha=0.95, lw=0)
                    a.text(xy[:, 1].mean() + 9, xy[:, 0].mean(), str(int(row.cell)), color="w", fontsize=5.5,
                           fontweight="bold", bbox=dict(boxstyle="round,pad=0.15", fc=col, ec="none", alpha=0.9))
                a.text(0.02, 0.02, f"{int((W.cls == 'UP').sum())} up, {int((W.cls == 'DOWN').sum())} down"
                       f" / {len(W)}", transform=a.transAxes, color="w", fontsize=7, fontweight="bold")
                if r == 0:
                    a.set_title(e, fontsize=9, color=EVC[e], fontweight="bold")
                if k == 0:
                    a.text(-0.04, 0.5, "baseline" if ph == "pre" else "injection", transform=a.transAxes,
                           rotation=90, ha="right", va="center", fontsize=9)
                a.set_xticks([]); a.set_yticks([])
                if cont:                                     # zoom to the cells (+ 30 px)
                    allxy = np.vstack(list(cont.values()))
                    a.set_xlim(max(allxy[:, 1].min() - 30, 0), min(allxy[:, 1].max() + 30, bg.shape[1]))
                    a.set_ylim(min(allxy[:, 0].max() + 30, bg.shape[0]), max(allxy[:, 0].min() - 30, 0))
        fig.suptitle(f"{m} ({arm}): where the responsive cells are, one map per event (same cells and numbers in both "
                     f"sessions)\nred = UP, blue = DOWN, grey outline = not significant ({rule}, q <= {Q_RESP:.2f}, BH over cells)",
                     fontsize=10, color=ARMC[arm])
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"ca_d{DAY}_B6{suffix}_maps_{m}.png"), dpi=120); plt.close(fig)


def decode(PS, out, n_perm=200):
    """which stimulus was delivered, from the population response of one session
    (multinomial logistic regression, L2, standardised cells, stratified 5-fold CV), against a
    label-permutation null. CONTROL: the same decoder on a SHAM response, z[-2,0] - z[-4,-2], both
    windows before the delivery. The stimuli come in separate 3-min blocks, so the raw pre-event
    activity 'decodes' them through slow drift alone (the synthetic check: 0.73-0.89 with no
    stimulus information at all); a difference of two adjacent windows cancels the drift, so the
    sham decoder is the fair chance level for the response decoder."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rows = []
    for m, arm, sex in MICE:
        for ph in ("pre", "post"):
            Xs, ys, Ps = [], [], []
            for k, st in enumerate(STIM):
                d = PS.get((m, ph, st))
                if d is None or d["n"] < 8:
                    continue
                Xs.append(d["r_trials"]); Ps.append(d["sham_trials"]); ys += [k] * d["n"]
            if len(Xs) < 2:
                continue
            X, P0, y = np.nan_to_num(np.vstack(Xs)), np.nan_to_num(np.vstack(Ps)), np.array(ys)
            nf = min(5, np.bincount(y)[np.bincount(y) > 0].min())
            cv = StratifiedKFold(nf, shuffle=True, random_state=0)
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
            acc = cross_val_score(clf, X, y, cv=cv).mean()
            acc0 = cross_val_score(clf, P0, y, cv=cv).mean()
            rng = np.random.default_rng(1)
            null = [cross_val_score(clf, X, rng.permutation(y), cv=cv).mean() for _ in range(n_perm)]
            rows.append(dict(mouse=m, arm=arm, phase=ph, n_stimuli=len(Xs), n_trials=len(y), n_cells=X.shape[1],
                             acc_response=acc, acc_pre_window=acc0, chance=1 / len(Xs),
                             null_p95=float(np.percentile(null, 95)),
                             p_perm=(1 + np.sum(np.array(null) >= acc)) / (1 + n_perm)))
    D = pd.DataFrame(rows)
    if not len(D):
        return D
    mice = [m for m in MICE if m[0] in set(D.mouse)]
    fig, ax = plt.subplots(figsize=(3.2 * len(mice) + 1, 4.6))
    for i, (m, arm, sex) in enumerate(mice):
        for j, ph in enumerate(("pre", "post")):
            V = D[(D.mouse == m) & (D.phase == ph)]
            if not len(V):
                continue
            r = V.iloc[0]
            x = i * 3 + j
            ax.bar(x, r.acc_response, 0.8, color=PHC[ph], label=("baseline" if ph == "pre" else "injection") if i == 0 else None)
            ax.plot([x - 0.4, x + 0.4], [r.acc_pre_window] * 2, color="k", lw=2,
                    label="sham response before the delivery (control)" if i == 0 and j == 0 else None)
            ax.plot([x - 0.4, x + 0.4], [r.null_p95] * 2, color="k", lw=1, ls=":",
                    label="95th pct of 200 label shuffles" if i == 0 and j == 0 else None)
            ax.text(x, r.acc_response + 0.02, f"{r.acc_response:.2f}\n{d_pstr(r.p_perm)}", ha="center", fontsize=7)
    ax.axhline(0.25, color="0.5", lw=0.8)
    ax.set_xticks([i * 3 + 0.5 for i in range(len(mice))], [f"{m}\n({arm})" for m, arm, _ in mice])
    ax.set_ylabel("cross-validated accuracy\n(which of the 4 stimuli)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    ax.set_title("Population decoding of the stimulus from the [0,2] s response (logistic regression, 5-fold CV); "
                 "grey line = chance 0.25", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"ca_d{DAY}_B7_stimulus_decoding.png"), dpi=120); plt.close(fig)
    return D


def outcome_split(sessions, PS, out):
    """same stimulus, different behaviour: deliveries followed by a withdrawal (tap from the
    delivery to 2 s later or the next delivery) vs deliveries without one. Per cell: mean
    response difference, Mann-Whitney U over trials, BH over cells; >= 4 trials of each."""
    from scipy.stats import mannwhitneyu
    rows, curves = [], {}
    for s in sessions:
        if s["B"] is None:
            continue
        m, ph = s["mouse"], s["phase"]
        allD = np.sort(np.concatenate([s["B"]["ev"][st] for st in STIM]))
        W = s["B"]["ev"]["withdrawal"]
        ids = s["T"]["cell_ids"].astype(int)
        for st in STIM:
            d = PS.get((m, ph, st))
            if d is None or d["Xb"] is None:
                continue
            wd = []
            for f in d["frames"]:
                nxt = allD[allD > f]
                hi = min(f + int(2 * FPS), (nxt[0] - 1) if len(nxt) else f + int(2 * FPS))
                wd.append(bool(np.any((W >= f) & (W <= hi))))
            wd = np.array(wd)
            if wd.sum() < 4 or (~wd).sum() < 4:
                continue
            r = d["r_trials"]
            pv = []
            for j in range(len(ids)):
                a_, b_ = r[wd, j], r[~wd, j]
                a_, b_ = a_[np.isfinite(a_)], b_[np.isfinite(b_)]
                pv.append(mannwhitneyu(a_, b_).pvalue if len(a_) >= 4 and len(b_) >= 4 else np.nan)
            q = bh(pv)
            for j, c in enumerate(ids):
                rows.append(dict(mouse=m, phase=ph, stimulus=st, cell=int(c), n_with=int(wd.sum()), n_without=int((~wd).sum()),
                                 resp_with=float(np.nanmean(r[wd, j])), resp_without=float(np.nanmean(r[~wd, j])),
                                 p=pv[j], q=q[j]))
            curves[(m, ph, st)] = (np.nanmean(d["Xb"][wd], (0, 2)), np.nanmean(d["Xb"][~wd], (0, 2)), d["tt"],
                                   int(wd.sum()), int((~wd).sum()), int(np.sum(np.asarray(q) <= Q_RESP)))
    D = pd.DataFrame(rows)
    if not curves:
        return D
    keys = sorted(curves, key=lambda k: ([x[0] for x in MICE].index(k[0]), k[1] != "pre", STIM.index(k[2])))
    nc = min(len(keys), 6)
    nr = int(np.ceil(len(keys) / nc))
    fig, ax = plt.subplots(nr, nc, figsize=(3.1 * nc, 2.9 * nr), squeeze=False)
    for i, k in enumerate(keys):
        a = ax[i // nc, i % nc]
        cw, cn, tt, nw, nn, nsig = curves[k]
        a.plot(tt, sm(cw[:, None])[:, 0], color="#6c3483", lw=1.3, label=f"withdrawal ({nw})")
        a.plot(tt, sm(cn[:, None])[:, 0], color="0.5", lw=1.3, label=f"no withdrawal ({nn})")
        a.axvline(0, color="k", lw=0.7); a.axhline(0, color="0.6", lw=0.4)
        a.axvspan(*windows(k[2])[1], color="0.93", lw=0, zorder=0)
        arm = dict((x[0], x[1]) for x in MICE)[k[0]]
        a.set_title(f"{k[0]} {'baseline' if k[1] == 'pre' else 'injection'}: {k[2]}\n{nsig} cells differ (q <= {Q_RESP})",
                    fontsize=8, color=ARMC[arm])
        a.legend(fontsize=6, frameon=False)
        a.tick_params(labelsize=6)
        a.set_xlabel("time from delivery (s)", fontsize=7)
        if i % nc == 0:
            a.set_ylabel("mean z of all cells\n(baselined [-2,0] s)", fontsize=7)
    for j in range(len(keys), nr * nc):
        ax[j // nc, j % nc].axis("off")
    fig.suptitle("Same stimulus, different behaviour: population response on deliveries WITH vs WITHOUT a withdrawal "
                 "(cells: Mann-Whitney over trials, BH)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"ca_d{DAY}_B8_withdrawal_vs_not.png"), dpi=115); plt.close(fig)
    return D


def fig_exemplars(R, PS, out_root, out, n_per=3):
    """mini2p deck slide 10 style: the clearest cells, every trial visible. Per mouse and stimulus
    (heat, pin, withdrawal): the n_per strongest UP and n_per strongest DOWN cells of the baseline
    (time-shift rule), single-trial heatmaps of baseline and injection and the mean +- SEM, so each
    claim can be checked by eye"""
    from skimage import measure
    evs = ["Heat", "Pin prick", "withdrawal"]
    for m, arm, sex in MICE:
        V = R[(R.mouse == m) & (R.phase == "pre")]
        if not len(V):
            continue
        f = os.path.join(out_root, f"{m}_d{DAY}", "3_extract", f"{m}_d{DAY}_extract.mat")
        Sp = None
        if os.path.isfile(f):
            M = sio.loadmat(f, squeeze_me=True, variable_names=["spatial_weights", "mean_image"])
            Sp, mi = np.asarray(M["spatial_weights"], np.float32), np.asarray(M["mean_image"], float)
        picks = []
        for e in evs:
            W = V[(V.event == e) & (V.cls_shift != "no response")]
            for lab, asc in (("UP", False), ("DOWN", True)):
                Q = W[W.cls_shift == lab].sort_values("resp_mean", ascending=asc).head(n_per)
                picks += [(e, lab, int(c)) for c in Q.cell]
        if not picks:
            continue
        fig, ax = plt.subplots(len(picks), 4, figsize=(12, 1.9 * len(picks)), squeeze=False,
                               gridspec_kw=dict(width_ratios=[0.6, 1, 1, 1.2]))
        for i, (e, lab, c) in enumerate(picks):
            for j, ph in enumerate(("pre", "post")):
                d = PS.get((m, ph, e))
                a = ax[i, 1 + j]
                if d is None or d["Xb"] is None:
                    a.axis("off"); continue
                ids = list(R[(R.mouse == m) & (R.phase == ph) & (R.event == e)].cell)
                col = ids.index(c) if c in ids else None
                if col is None:
                    a.axis("off"); continue
                tt = d["tt"]
                X = d["Xb"][:, :, col]
                a.imshow(sm(X.T).T, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3,
                         extent=[tt[0], tt[-1], X.shape[0] + 0.5, 0.5], interpolation="nearest")
                a.axvline(0, color="k", lw=0.6)
                a.set_title(f"{'baseline' if ph == 'pre' else 'injection'}: {X.shape[0]} trials", fontsize=7)
                a.tick_params(labelsize=6)
                b = ax[i, 3]
                mu = sm(np.nanmean(X, 0)[:, None])[:, 0]
                se = np.nanstd(X, 0) / np.sqrt(X.shape[0])
                b.plot(tt, mu, color=PHC[ph], lw=1.2, label="baseline" if ph == "pre" else "injection")
                b.fill_between(tt, mu - se, mu + se, color=PHC[ph], alpha=0.2, lw=0)
            b = ax[i, 3]
            b.axvline(0, color="k", lw=0.6); b.axhline(0, color="0.6", lw=0.4)
            b.axvspan(*windows(e)[1], color="0.93", lw=0, zorder=0)
            b.tick_params(labelsize=6)
            if i == 0:
                b.legend(fontsize=6, frameon=False)
            a0 = ax[i, 0]
            a0.axis("off")
            if Sp is not None:
                fp = Sp[:, :, c - 1]
                yy, xx = np.unravel_index(np.argmax(fp), fp.shape)
                r0, r1, c0, c1 = max(yy - 40, 0), yy + 40, max(xx - 40, 0), xx + 40
                crop = mi[r0:r1, c0:c1]
                a0.imshow(crop, cmap="gray", vmin=np.percentile(crop, 1), vmax=np.percentile(crop, 99.5))
                cs = measure.find_contours(fp[r0:r1, c0:c1] / max(fp.max(), 1e-9), 0.5)
                for q in cs:
                    a0.plot(q[:, 1], q[:, 0], color="#2ecc71", lw=0.8)
            a0.set_title(f"#{c} {e} {lab}", fontsize=7, color=EVC[e])
        fig.suptitle(f"{m} ({arm}): the clearest responsive cells (baseline, time-shift rule), every trial; z baselined "
                     f"[-2,0] s; left = the cell on the mean image", fontsize=10, color=ARMC[arm])
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(os.path.join(out, f"ca_d{DAY}_B9_exemplar_cells_{m}.png"), dpi=100); plt.close(fig)


def fig_motion_control(sessions, PS, R, out):
    """registration artefacts around the events. (1) mean |shift| and frame-template correlation
    around each event type - if heat deliveries make the brain move, a 'response' could be motion;
    (2) the responsive cells again using only CLEAN trials (no frame in [-2,+2] s below the
    session's correlation threshold, median - 5 MAD, and the 0.5-s median |shift| moving < 3 px),
    Wilcoxon rule"""
    rows, curves = [], {}
    k0, k1 = int(WIN[0] * FPS), int(WIN[1] * FPS)
    tt = np.arange(k0, k1 + 1) / FPS
    for s in sessions:
        if s.get("motion") is None or s["B"] is None:
            continue
        m, ph = s["mouse"], s["phase"]
        fr = s["T"]["frame"].astype(int)
        sh, co, thr = s["motion"]["shift"], s["motion"]["corr"], s["motion"]["corr_thr"]
        shs = pd.Series(sh).rolling(15, center=True, min_periods=1).median().values   # 0.5-s median: estimation jitter out
        for e in EVENTS:
            d = PS.get((m, ph, e))
            if d is None:
                continue
            idx = np.searchsorted(fr, d["frames"])
            Wsh = np.stack([shs[i + k0:i + k1 + 1] for i in idx])
            Wco = np.stack([co[i + k0:i + k1 + 1] for i in idx])
            curves[(m, ph, e)] = (np.nanmean(Wsh - np.nanmean(Wsh[:, (tt >= -2) & (tt < 0)], 1, keepdims=True), 0),
                                  np.nanmean(Wco, 0))
            w = (tt >= -2) & (tt <= 2)
            bad = (np.nanmin(Wco[:, w], 1) < thr) | ((np.nanmax(Wsh[:, w], 1) - np.nanmin(Wsh[:, w], 1)) > 3.0)
            r = d["r_trials"]
            n_clean = int((~bad).sum())
            up = dn = up_all = dn_all = np.nan
            if n_clean >= MIN_TRIALS:
                p = np.array([wsr(r[~bad, j]) for j in range(r.shape[1])]); q = bh(p)
                med = np.nanmedian(r[~bad], 0)
                up, dn = int(np.sum((q <= Q_RESP) & (med > 0))), int(np.sum((q <= Q_RESP) & (med < 0)))
            V = R[(R.mouse == m) & (R.phase == ph) & (R.event == e)]
            up_all, dn_all = int((V.cls == "UP").sum()), int((V.cls == "DOWN").sum())
            rows.append(dict(mouse=m, phase=ph, event=e, n_trials=len(bad), n_bad_motion=int(bad.sum()),
                             up_all=up_all, down_all=dn_all, up_clean=up, down_clean=dn, n_cells=r.shape[1]))
    D = pd.DataFrame(rows)
    if not len(D):
        return D
    mice = [m for m in MICE if m[0] in set(D.mouse)]
    evs = [e for e in STIM + ["withdrawal", "movement onset"] if e in set(D.event)]
    fig, ax = plt.subplots(2 * len(mice), len(evs), figsize=(2.5 * len(evs), 3.4 * len(mice)), squeeze=False)
    for i, (m, arm, sex) in enumerate(mice):
        for k, e in enumerate(evs):
            for row, idx_, lab in ((2 * i, 0, "|shift| change (px)"), (2 * i + 1, 1, "frame-template r")):
                a = ax[row, k]
                for ph in ("pre", "post"):
                    c = curves.get((m, ph, e))
                    if c is not None:
                        a.plot(tt, sm(c[idx_][:, None])[:, 0], color=PHC[ph], lw=1,
                               label=("baseline" if ph == "pre" else "injection"))
                a.axvline(0, color="k", lw=0.5)
                a.tick_params(labelsize=6)
                if row == 2 * i:
                    V = D[(D.mouse == m) & (D.event == e)]
                    t_ = "; ".join(f"{'b' if r_.phase == 'pre' else 'i'}: {r_.n_bad_motion}/{r_.n_trials} bad, UP {r_.up_all}->{r_.up_clean}"
                                   for r_ in V.itertuples())
                    a.set_title(f"{m} {e}\n{t_}", fontsize=6, color=EVC[e])
                if k == 0:
                    a.set_ylabel(lab, fontsize=7)
                if i == 0 and k == 0 and row == 0:
                    a.legend(fontsize=6, frameon=False)
    fig.suptitle("B10 motion control: registration around each event (|shift| change from [-2,0] s, and the frame's correlation "
                 "with the template)\ntitles: trials with a motion problem in [-2,+2] s, and UP cells with all trials -> clean "
                 "trials only (Wilcoxon rule)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(out, f"ca_d{DAY}_B10_motion_control.png"), dpi=110); plt.close(fig)
    return D


def d_pstr(p):
    return "p=" + (f"{p:.3f}" if p >= 0.001 else "<0.001")


def fig_arm_blocks(W, out):
    """C2: the main comparison in one picture - per mouse, the change (log2 injection / baseline,
    median over cells) of the activity WITHOUT movement, before any stimulus and during the
    stimulus blocks; SBI-553 mice next to vehicle mice (n = 2 per arm: shown, not tested)"""
    conds = [("no stimulus", "still rate | no stimulus")] + [(st, f"still rate | {st}") for st in STIM]
    rows = []
    for m, arm, sex in MICE:
        V = W[W.mouse.astype(str) == m]
        for lab, c in conds:
            if c not in V:
                continue
            P = V.pivot(index="cell", columns="phase", values=c).dropna()
            if {"pre", "post"} <= set(P.columns) and len(P) >= 3:
                lr = np.log2((P.post + 0.1) / (P.pre + 0.1))
                rows.append(dict(mouse=m, arm=arm, sex=sex, condition=lab, n_cells=len(P), log2_ratio_median=lr.median(),
                                 p=wsr(P.post, P.pre)))
    D = pd.DataFrame(rows)
    if not len(D):
        return D
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(conds))
    offs = {"5941": -0.24, "5939": -0.08, "5937": 0.08, "1191": 0.24}
    for m, arm, sex in MICE:
        V = D[D.mouse == m].set_index("condition").reindex([c for c, _ in conds])
        if not V.log2_ratio_median.notna().any():
            continue
        ax.plot(x + offs[m], V.log2_ratio_median, marker="o" if sex == "F" else "s", ls=":", lw=0.7, ms=7,
                color=ARMC[arm], label=f"{m} ({arm}, {sex})")
        for xi, (_, r) in zip(x, V.iterrows()):
            if np.isfinite(r.p) and r.p < 0.05:
                ax.text(xi + offs[m], r.log2_ratio_median, "*", ha="center", va="bottom", fontsize=12, color=ARMC[arm])
    ax.axhline(0, color="k", lw=0.7)
    ax.axvspan(-0.5, 0.5, color="0.94", lw=0, zorder=0)
    ax.set_xticks(x, [c for c, _ in conds])
    ax.set_xlim(-0.5, len(conds) - 0.5)
    ax.set_ylabel("activity change, injection vs baseline\nmedian over cells of log2(post/pre), still frames")
    ax.set_title("C2 SBI-553 (red) vs vehicle (blue): activity without movement before any stimulus (grey) and in each "
                 "stimulus block;\n* next to a point = that mouse, injection vs its baseline (Wilcoxon over its cells, p < 0.05); "
                 "n = 2 mice per arm - compare the arms by eye",
                 fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"ca_d{DAY}_C2_arm_blocks_still.png"), dpi=120); plt.close(fig)
    return D


def fig_arm_summary(RS, WS, out):
    """per mouse change in % responsive (UP + DOWN) and in transient rate, by arm"""
    evs = [e for e in EVENTS if e in set(RS.event)]
    fig, ax = plt.subplots(1, 2, figsize=(16, 4.8), gridspec_kw=dict(width_ratios=[3.2, 1]))
    a = ax[0]
    x = np.arange(len(evs))
    for m, arm, sex in MICE:
        V = RS[RS.mouse == m]
        if not len(V):
            continue
        d = []
        for e in evs:
            pre = V[(V.event == e) & (V.phase == "pre")]
            post = V[(V.event == e) & (V.phase == "post")]
            d.append((post.pct_up.sum() + post.pct_down.sum()) - (pre.pct_up.sum() + pre.pct_down.sum())
                     if len(pre) and len(post) else np.nan)
        off = {"5941": -0.2, "5939": -0.07, "5937": 0.07, "1191": 0.2}[m]
        a.scatter(x + off, d, color=ARMC[arm], marker="o" if sex == "F" else "s", s=45, label=f"{m} ({arm}, {sex})")
    a.axhline(0, color="k", lw=0.8)
    a.set_xticks(x, evs, fontsize=8)
    a.set_ylabel("change in % responsive cells\n(injection - baseline, UP + DOWN)")
    a.set_title("Responsive cells, injection minus baseline session, per mouse (n = 2 per arm, not tested)", fontsize=10)
    a.legend(fontsize=8, frameon=False)
    b = ax[1]
    for i, arm in enumerate(["vehicle", "SBI-553"]):
        v = WS[WS.arm == arm]
        b.scatter(np.full(len(v), i), v.log2_ratio_median, color=ARMC[arm], s=60)
        for _, r in v.iterrows():
            b.annotate(r.mouse, (i, r.log2_ratio_median), xytext=(8, 0), textcoords="offset points", fontsize=8)
    b.axhline(0, color="k", lw=0.8)
    b.set_xticks([0, 1], ["vehicle", "SBI-553"]); b.set_xlim(-0.5, 1.5)
    b.set_ylabel("transient rate, median log2(post/pre)")
    b.set_title("whole-session activity", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"ca_d{DAY}_C1_arm_summary.png"), dpi=120); plt.close(fig)


# -------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--scoring-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--traces", default="traces", help="traces (default) or traces_bgcorr")
    ap.add_argument("--z-norm", default="mad", choices=["mad", "noise"],
                    help="mad = z of the trace files (per-session MAD); noise = dF/F / the cell's noise SD over both sessions")
    a = ap.parse_args(argv)
    global TRACES, Z_NORM
    TRACES = a.traces
    Z_NORM = a.z_norm
    os.makedirs(a.out, exist_ok=True)
    W, R, PS, info, SESS = [], [], {}, [], []
    for m, arm, sex in MICE:
        for ph in ("pre", "post"):
            s = load_session(a.out_root, m, ph, a.scoring_dir)
            if s is None:
                print(f"{m} {ph}: no traces yet")
                continue
            if s["B"] is None:
                print(f"{m} {ph}: no scoring file")
            n_isx = len(s["T"]["frame"])
            n_cam = s["B"]["st"]["n_cam"] if s["B"] else np.nan
            info.append(dict(mouse=m, arm=arm, phase=ph, n_cells=len(s["T"]["cell_ids"]), n_isx_rows=n_isx,
                             n_cam_frames=n_cam, scoring=s["scoring"],
                             **{f"n {e}": len(s["B"]["ev"][e]) for e in EVENTS} if s["B"] else {}))
            w = whole_session(s); w["arm"] = arm; W.append(w)
            SESS.append(s)
            r, ps = event_tests(s)
            if len(r):
                r["arm"] = arm; R.append(r)
            for e, d in ps.items():
                PS[(m, ph, e)] = d
            print(f"{s['rid']}: {len(w)} cells, events " +
                  ", ".join(f"{e} {d['n']}" for e, d in ps.items()))
    if not W:
        print("nothing to analyse yet"); return 1
    W = pd.concat(W); R = pd.concat(R) if R else pd.DataFrame()
    pd.DataFrame(info).to_csv(os.path.join(a.out, f"ca_d{DAY}_sessions.csv"), index=False)
    W.round(5).to_csv(os.path.join(a.out, f"ca_d{DAY}_cells_whole_session.csv"), index=False)
    WS = fig_whole(W, a.out)
    WS.round(5).to_csv(os.path.join(a.out, f"ca_d{DAY}_A1_whole_session_tests.csv"), index=False)
    # the same comparison on the other activity measures (a rate change should show in all of them)
    rows = []
    for m, arm, sex in MICE:
        for c, lab in (("rate", "transients per min"), ("deconv", "OASIS deconvolved activity per min"),
                       ("mean_dff", "mean local dF/F"), ("amp", "mean transient amplitude (dF/F)")):
            P = W[W.mouse.astype(str) == m].pivot(index="cell", columns="phase", values=c).dropna()
            if {"pre", "post"} <= set(P.columns) and len(P) >= 3:
                rows.append(dict(mouse=m, arm=arm, measure=lab, n_cells=len(P), pre_median=P.pre.median(),
                                 post_median=P.post.median(), p_wilcoxon=wsr(P.post, P.pre)))
    pd.DataFrame(rows).round(6).to_csv(os.path.join(a.out, f"ca_d{DAY}_A1b_all_activity_measures.csv"), index=False)
    fig_blocks(W, a.out).round(5).to_csv(os.path.join(a.out, f"ca_d{DAY}_A2_blocks_tests.csv"), index=False)
    fig_spontaneous(W, a.out).round(5).to_csv(os.path.join(a.out, f"ca_d{DAY}_A3_spontaneous_tests.csv"), index=False)
    fig_timecourse(SESS, a.out).round(4).to_csv(os.path.join(a.out, f"ca_d{DAY}_A4_timecourse.csv"), index=False)
    fig_effect_map(W, a.out_root, a.out)
    fig_blocks_still(W, a.out).round(5).to_csv(os.path.join(a.out, f"ca_d{DAY}_A6_blocks_still_tests.csv"), index=False)
    fig_block_aligned(SESS, a.out).round(4).to_csv(os.path.join(a.out, f"ca_d{DAY}_A7_block_aligned.csv"), index=False)
    fig_arm_blocks(W, a.out).round(5).to_csv(os.path.join(a.out, f"ca_d{DAY}_C2_arm_blocks_still.csv"), index=False)
    if len(R):
        R.round(5).to_csv(os.path.join(a.out, f"ca_d{DAY}_cells_event_tests.csv"), index=False)
        RS = fig_responsive(R, a.out)
        RS.to_csv(os.path.join(a.out, f"ca_d{DAY}_B1_responsive_counts.csv"), index=False)
        fig_responsive_matched(PS, a.out).round(3).to_csv(os.path.join(a.out, f"ca_d{DAY}_B1b_responsive_matched.csv"), index=False)
        fig_psth(PS, R, a.out)
        fig_heatmaps(PS, a.out)
        fig_auc(R, a.out).round(5).to_csv(os.path.join(a.out, f"ca_d{DAY}_B4_population_auc.csv"), index=False)
        fig_groups(PS, a.out)
        fig_maps(R, a.out_root, a.out)
        # the same with the time-shift rule (validated on synthetic data: finds 4x more inhibited
        # cells than the per-trial Wilcoxon at the same false-positive level)
        R2 = R.copy(); R2["cls"] = R2["cls_shift"]
        fig_responsive(R2, a.out, suffix="c", rule="1000 time-shifted copies of the event train").to_csv(
            os.path.join(a.out, f"ca_d{DAY}_B1c_responsive_counts_shift_rule.csv"), index=False)
        fig_maps(R2, a.out_root, a.out, suffix="b", rule="time-shift null")
        fig_arm_summary(RS, WS, a.out)
        decode(PS, a.out).round(4).to_csv(os.path.join(a.out, f"ca_d{DAY}_B7_stimulus_decoding.csv"), index=False)
        outcome_split(SESS, PS, a.out).round(5).to_csv(os.path.join(a.out, f"ca_d{DAY}_B8_withdrawal_vs_not.csv"), index=False)
        fig_exemplars(R, PS, a.out_root, a.out)
        fig_motion_control(SESS, PS, R, a.out).to_csv(os.path.join(a.out, f"ca_d{DAY}_B10_motion_control.csv"), index=False)
        # response change pre -> post, per mouse and event
        rows = []
        for (m, e), V in R.groupby(["mouse", "event"]):
            P = V.pivot(index="cell", columns="phase", values="resp_mean").dropna()
            if {"pre", "post"} <= set(P.columns) and len(P) >= 3:
                rows.append(dict(mouse=m, arm=V.arm.iloc[0], event=e, n_cells=len(P), pre_mean=P.pre.mean(),
                                 post_mean=P.post.mean(), p_wilcoxon=wsr(P.post, P.pre)))
        pd.DataFrame(rows).round(5).to_csv(os.path.join(a.out, f"ca_d{DAY}_B2_response_change_tests.csv"), index=False)
    print("ca_day1_analysis done ->", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
