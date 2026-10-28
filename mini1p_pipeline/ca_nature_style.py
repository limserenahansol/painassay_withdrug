"""ca_nature_style.py - new calcium / behaviour figures modelled on Figs 2-4 of Pennington et al., Nature 2026 ("An amygdala to
anterior hypothalamic circuit gates stress sensitivity", s41586-026-11075-5), applied to our SBI-553 / vehicle crossover
(2026-10-23, PI: "figure 2 and 3, 4 you can refer to, to make some new plots for our calcium and behavior analysis").

What and why (the paper panel each figure mirrors; analysis\\pi_extra\\nature_style\\paper_figures.md has the panel list):
  NS1_pca_trajectories    paper Fig 2j / 4h (PCA of the trial-averaged population around each stimulus; PC1 = intensity).
                          The 185 tracked neurons (same cells in all four sessions of a mouse) pooled over the 4 mice; trial-mean
                          z (each trial minus its own 0.5 s before = the stimulus baseline window of WW.of), -1 to 2.5 s, 100-ms bins;
                          one PCA fitted on all 16 averages (4 sessions x 4 stimuli). A: baseline trajectories in PC1-PC2,
                          B: PC1 over time per stimulus and session, C: PC1 response in the best window (0-0.5 s) with the same
                          projection computed from each mouse's neurons alone. Descriptive (no test); split-half check.
  NS2_noxious_neurons     paper Fig 2h-i / 4f-g (responsive neurons scale with stimulus intensity: time course + each neuron's
                          mean) and Fig 2n (responses of the shock-responsive subpopulation to every stimulus). Neurons activated
                          by Pin prick or Heat on half of the same-day baseline trials (Wilcoxon signed-rank over trials,
                          p < 0.05, mean > 0; every neuron of each day; two folds: select on odd / measure on even trials and
                          the reverse), baseline measured only on the held-out half (no selection bias), after the injection on
                          all trials: responses to all 4 stimuli.
                          Baseline vs injection = the same neurons -> Wilcoxon signed-rank paired by neuron; Friedman across the
                          4 stimuli (the analogue of the paper's RM-ANOVA over amplitudes).
  NS3_overlap             paper Fig 2c / 2g / 4e / 2m (% of neurons responsive to one stimulus, the other, or both). Every neuron
                          of each day (cells_classes.csv, cls = activated): touch (Light / Mild) vs noxious (Pin / Heat), and
                          stimulus (any of 4) vs reflex (withdrawal / flinch); the overlap expected if independent is marked.
                          Overlap within a day: McNemar (same neurons); after vehicle vs after SBI-553: Fisher exact (pooled).
  NS4_tuning_crossover    paper Fig 2h-i / 2n (tuning across stimuli) in the SAME tracked neurons across the crossover (user
                          rule: cross-day z comparisons only in tracked neurons, each neuron with itself, baseline-normalized).
                          Tuning = response (cells_classes resp, best window) to the 4 stimuli; noxious preference index NPI =
                          mean(Pin, Heat) - mean(Light, Mild). Wilcoxon signed-rank paired by neuron; per-mouse medians.
  (numbers only)          4-way stimulus decoding and noxious-vs-touch decoding per session (LDA, 5-fold CV, label-shuffle null)
                          - reported in summary.json as the reason no decoding-over-time figure is shown.
Fig 3 of the paper (optogenetic / chemogenetic behaviour) has no counterpart in our data (no manipulation of activity); the
behaviour side is covered by NS3 B (stimulus vs reflex neurons).

4 mice, drug order confounded with day (5941 and 5939 got SBI-553 on Day 1, 5937 and 1191 on Day 2): every test is
exploratory, neurons pooled over mice; per-mouse values are shown (no test across 4 mice can reach p < 0.05).

Inputs  analysis\\event_analysis\\cells_classes.csv, analysis\\crossday\\<m>_pairs.csv, single trials via ca_event_analysis.load_all()
        (timing-corrected scoring, SCORING_SUFFIX=_timefix), windows via winwords.of().
Outputs analysis\\pi_extra\\nature_style\\: NS1..NS4 .png + _kr.png, CSV tables (ns1_pca.csv, ns2_neurons.csv,
        ns3_overlap.csv, ns4_npi.csv, ns_decoding.csv), summary.json.
Run     python -I -X utf8 ca_nature_style.py          (~1-2 min; loads all 16 sessions once, ~2 GB)
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, friedmanchisquare, wilcoxon, binomtest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import safe_io  # noqa: E402,F401
import pairstars as PS  # noqa: E402  (every star shows the pair it compares)
import figfoot  # noqa: E402  (10-24: methods text -> slide notes, <png>.notes.txt)
import winwords as WW  # noqa: E402
import ca_event_analysis as E  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
B = r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553"
EA = os.path.join(B, "analysis", "event_analysis")
CD = os.path.join(B, "analysis", "crossday")
OUT = os.path.join(B, "analysis", "pi_extra", "nature_style")
MO = ["5941", "5939", "5937", "1191"]
SBI_DAY = {"5941": 1, "5939": 1, "5937": 2, "1191": 2}
CC = {"baseline": "#7f8c8d", "vehicle": "#2471a3", "SBI-553": "#c0392b"}
MC = {"5941": "#e67e22", "5939": "#8e44ad", "5937": "#16a085", "1191": "#34495e"}
MK = {"5941": "o", "5939": "s", "5937": "^", "1191": "D"}
STIM = ["Light touch", "Mild touch", "Pin prick", "Heat"]
# 10-24 audit: Mild touch was pale orange (low contrast as a title) -> darker; Light touch a little darker too
SCOL = {"Light touch": "#3498db", "Mild touch": "#c27c0e", "Pin prick": "#e74c3c", "Heat": "#8e44ad"}
SESS = ["vb", "vi", "sb", "si"]
# 10-24 audit: the two baselines (grey dashed vs grey solid) looked the same -> dark grey dashed vs light grey solid
SCOLS = {"vb": "#3b4444", "vi": CC["vehicle"], "sb": "#99a3a4", "si": CC["SBI-553"]}
SLS = {"vb": (0, (4, 1.6)), "vi": "-", "sb": "-", "si": "-"}
FPS = 30.0
T0, T1, BIN = -1.0, 2.5, 3                         # PCA / trace span (s) and bin (frames = 100 ms)
SEED = 20261023
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11,
                     "legend.fontsize": 11, "axes.spines.top": False, "axes.spines.right": False})

TXT = {
    "vb": ("vehicle day: baseline", "vehicle 날: 기준"), "vi": ("after vehicle", "vehicle 후"),
    "sb": ("SBI-553 day: baseline", "SBI-553 날: 기준"), "si": ("after SBI-553", "SBI-553 후"),
    "vb2": ("veh-day\nbase", "veh 날\n기준"), "vi2": ("after\nveh", "veh\n후"), "sb2": ("SBI-day\nbase", "SBI 날\n기준"),
    "si2": ("after\nSBI", "SBI\n후"),
    "vb3": ("base", "기준"), "vi3": ("veh", "veh"), "sb3": ("base", "기준"), "si3": ("SBI", "SBI"),
    "time": ("time from stimulus (s)", "자극부터 시간 (s)"),
    # 10-25: KR figures use the deck's stimulus names (Light touch / Mild touch / Pin prick / Heat), not Korean words
    "Light touch": ("Light touch", "Light touch"), "Mild touch": ("Mild touch", "Mild touch"),
    "Pin prick": ("Pin prick", "Pin prick"), "Heat": ("Heat", "Heat"),
}


def T_(k, lg):
    return TXT[k][1 if lg == "kr" else 0]


def L(lg, en, kr):
    return kr if lg == "kr" else en


def font(lg):
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"] if lg == "kr" else ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def foot(fig, txt, fs=10, y=0.006):
    """10-24 (user: details in the notes, not on the slide): the methods text goes to <png>.notes.txt (figfoot -> slide
    notes); the figure keeps only the one-line star key when the text has one. Returns the number of lines drawn (1)."""
    figfoot.note(fig, txt, None, fs=11.5)          # language from the text (Hangul -> Korean key)
    return 1


def win_of(ev):
    """'0-1 s': the response window of one event (winwords.of)"""
    a, b = WW.of(ev)[2:]
    return f"{a:g}-{b:g} s"


def stim_words():
    """the response windows of the 4 stimuli in words: '0-0.5 s', or with WIN_MODE=mix e.g. 'light, mild, heat 0-1 s;
    pin 0-0.5 s' (each stimulus is measured in its own window, winwords.of)"""
    ws = {ev: WW.of(ev)[2:] for ev in STIM}
    if len(set(ws.values())) == 1:
        a, b = next(iter(ws.values()))
        return f"{a:g}-{b:g} s"
    g = {}
    for ev, w in ws.items():
        g.setdefault(w, []).append(WW.SHORT[ev])
    return "; ".join(f"{', '.join(v)} {a:g}-{b:g} s" for (a, b), v in g.items())


def ptxt(p):
    if p is None or not np.isfinite(p):
        return "n.a."
    if p < 0.001:
        return "p < 0.001"
    t = f"{p:.3f}" if p < 0.01 else f"{p:.2f}"
    if p < 0.05 <= float(t):                     # 0.0495 must not print as 0.05
        t = f"{p:.4f}"
    return f"p = {t}"


def wil(x, y=None):
    x = np.asarray(x, float)
    d = x if y is None else x - np.asarray(y, float)
    d = d[np.isfinite(d)]
    if len(d) < 6 or np.all(d == 0):
        return np.nan
    return float(wilcoxon(d, zero_method="wilcox").pvalue)


def boot_ci(v, n=4000):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) < 3:
        return np.nan, np.nan
    rng = np.random.default_rng(SEED)
    m = np.median(v[rng.integers(0, len(v), (n, len(v)))], 1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def mcnemar_p(a, b):
    """exact McNemar: a, b = paired booleans (same neurons)"""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    return float(binomtest(n01, n01 + n10, 0.5).pvalue) if n01 + n10 else 1.0


def sess_key(m, c):
    sd = SBI_DAY[m]
    d = 3 - sd if c in ("vb", "vi") else sd
    return (m, d, "pre" if c in ("vb", "sb") else "post")


def jconv(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


# ------------------------------------------------------------------------------------------------ trials
def stim_trials(s, ev):
    """single trials of one stimulus in one session, the SAME trials for both outputs:
    R  trials x cells   best-window response (response-window mean z minus baseline-window mean z; ca_event_analysis)
    X  trials x time x cells  z from T0 to T1 s, each trial minus its own baseline window (the 0.5 s before)
    in delivery order (odd / even = 1st, 3rd, ... / 2nd, 4th, ...)"""
    pre, gap, a, b = WW.of(ev)
    R, f = E.responses(s, s["B"]["ev"][ev], pre, gap, a, b)
    Xp, tt, used = E.CA.peri(s, f, pre0=-(pre + gap))
    used = np.asarray(used, int)
    keep = np.isin(f, used)
    R, f = R[keep], f[keep]
    Xp = Xp[np.searchsorted(used, f)] if len(used) else Xp
    bl = (tt >= -(pre + gap)) & (tt < -gap)
    Xp = Xp - np.nanmean(Xp[:, bl, :], 1, keepdims=True)
    k = (tt >= T0 - 1e-9) & (tt <= T1 + 1e-9)
    return R, Xp[:, k, :].astype(np.float32), tt[k], f


def build_trials(S):
    """dict (mouse, day, phase, stim) -> (R, X, tt, frames)"""
    D = {}
    for key, s in S.items():
        for ev in STIM:
            D[key + (ev,)] = stim_trials(s, ev)
    return D


def tracked_cols(S, m):
    """per session c: column indices of the tracked neurons (pair order) in that session"""
    P = pd.read_csv(os.path.join(CD, f"{m}_pairs.csv"))
    cols = {}
    for c in SESS:
        k = sess_key(m, c)
        ids = S[k]["ids"]
        idc = P[f"day{k[1]}_id"].astype(int).to_numpy()
        cols[c] = np.array([int(np.flatnonzero(ids == i)[0]) for i in idc])
    return P, cols


# ------------------------------------------------------------------------------------------------ NS1 PCA
def binm(x, k=BIN):
    n = x.shape[0] // k
    return x[:n * k].reshape((n, k) + x.shape[1:]).mean(1)


def ns1_pca(S, D):
    mice, Mfull, Modd, Mev, ntr = [], {}, {}, {}, {}
    per_m = {}
    for m in MO:
        P, cols = tracked_cols(S, m)
        per_m[m] = len(P)
        mice += [m] * len(P)
        for c in SESS:
            for ev in STIM:
                R, X, tt, f = D[sess_key(m, c) + (ev,)]
                Xc = X[:, :, cols[c]]
                Mfull.setdefault((c, ev), []).append(np.nanmean(Xc, 0))
                Modd.setdefault((c, ev), []).append(np.nanmean(Xc[0::2], 0))
                Mev.setdefault((c, ev), []).append(np.nanmean(Xc[1::2], 0))
                ntr.setdefault((c, ev), {})[m] = int(len(Xc))
    for Mx in (Mfull, Modd, Mev):
        for k in Mx:
            Mx[k] = np.nan_to_num(np.concatenate(Mx[k], 1))
    mice = np.array(mice)
    tb = binm(tt[:, None])[:, 0]
    keys = [(c, ev) for c in SESS for ev in STIM]
    Xall = np.concatenate([binm(Mfull[k]) for k in keys], 0)
    mu = Xall.mean(0)
    U, Sv, Vt = np.linalg.svd(Xall - mu, full_matrices=False)
    var = Sv ** 2 / (Sv ** 2).sum()
    W = Vt[:3].T.copy()
    a, b = WW.of("Pin prick")[2:]
    kw = (tt >= a - 1e-9) & (tt < b - 1e-9)
    # sign: PC1 response (best window) positive on average; PC2: its loadings sum positive
    r1 = np.mean([Mfull[k][kw].mean(0) @ W[:, 0] for k in keys])
    if r1 < 0:
        W[:, 0] *= -1
    if W[:, 1].sum() < 0:
        W[:, 1] *= -1
    proj = {k: (binm(Mfull[k]) - mu) @ W for k in keys}
    # PC1 response in the best window: (window mean - baseline window mean (= 0 by construction)) . w
    resp = {k: float(Mfull[k][kw].mean(0) @ W[:, 0]) for k in keys}
    odd = {k: float(Modd[k][kw].mean(0) @ W[:, 0]) for k in keys}
    even = {k: float(Mev[k][kw].mean(0) @ W[:, 0]) for k in keys}
    r_split = float(np.corrcoef([odd[k] for k in keys], [even[k] for k in keys])[0, 1])
    # the same projection from each mouse's neurons alone (unit-length sub-vector of the PC1 loadings)
    mresp = {}
    for m in MO:
        j = mice == m
        wm = W[j, 0] / np.linalg.norm(W[j, 0])
        mresp[m] = {k: float(Mfull[k][kw].mean(0)[j] @ wm) for k in keys}
    load_share = {m: float((W[mice == m, 0] ** 2).sum()) for m in MO}
    rows = [dict(session=c, stim=ev, pc1_resp=resp[(c, ev)], pc1_odd=odd[(c, ev)], pc1_even=even[(c, ev)],
                 **{f"pc1_{m}": mresp[m][(c, ev)] for m in MO}, **{f"n_trials_{m}": ntr[(c, ev)][m] for m in MO})
            for c, ev in keys]
    return dict(tb=tb, proj=proj, var=var, resp=resp, odd=odd, even=even, r_split=r_split, mresp=mresp, load_share=load_share,
                n=len(mice), per_m=per_m, table=pd.DataFrame(rows), keys=keys, ntr=ntr,
                frac_neurons={m: per_m[m] / len(mice) for m in MO})


# ------------------------------------------------------------------------------------------------ NS2 noxious-responsive
def ns2_noxious(S, D):
    """every neuron of each day; two-fold cross-validated selection on the same-day baseline: fold 1 selects on the odd
    trials (activated by Pin prick or Heat: Wilcoxon signed-rank over trials p < 0.05, mean > 0) and measures on the even
    trials, fold 2 the reverse; a neuron selected in either fold is kept and its baseline is the mean of the held-out halves
    of the folds that selected it (never the trials that selected it). After the injection: all trials."""
    rows, traces = [], {}
    for m in MO:
        for drug, (cb, ci) in (("vehicle", ("vb", "vi")), ("SBI-553", ("sb", "si"))):
            kb, ki = sess_key(m, cb), sess_key(m, ci)
            nc = len(S[kb]["ids"])
            fsel = np.zeros((2, nc), bool)                   # fold 0: selected on odd trials, fold 1: on even trials
            for ev in ("Pin prick", "Heat"):
                Rb = D[kb + (ev,)][0]
                for fo, part in ((0, Rb[0::2]), (1, Rb[1::2])):
                    p = np.array([wil(part[:, j]) for j in range(nc)])
                    fsel[fo] |= (p < 0.05) & (np.nanmean(part, 0) > 0)
            sel = fsel.any(0)
            wts = fsel / np.maximum(fsel.sum(0), 1)          # weight of each fold's held-out half

            def held(A):
                """held-out baseline: fold 0 measures on the even half, fold 1 on the odd half; A = trials x ... x cells"""
                ev_m, od_m = np.nanmean(A[1::2], 0), np.nanmean(A[0::2], 0)
                return ev_m * wts[0] + od_m * wts[1]
            for j in range(nc):
                r = dict(mouse=m, day=kb[1], drug=drug, cell=int(S[kb]["ids"][j]), selected=bool(sel[j]),
                         sel_odd=bool(fsel[0, j]), sel_even=bool(fsel[1, j]))
                rows.append(r)
            for ev in STIM:
                Rb, Ri = D[kb + (ev,)][0], D[ki + (ev,)][0]
                hb = held(Rb)
                ia = np.nanmean(Ri, 0)
                for j in range(nc):
                    rows[-nc + j][f"base_held|{ev}"] = float(hb[j]) if sel[j] else np.nan
                    rows[-nc + j][f"base_all|{ev}"] = float(np.nanmean(Rb[:, j]))
                    rows[-nc + j][f"inj|{ev}"] = float(ia[j])
                Xb, Xi = D[kb + (ev,)][1], D[ki + (ev,)][1]
                tt = D[kb + (ev,)][2]
                traces.setdefault((cb, ev), []).append(held(Xb)[:, sel])
                traces.setdefault((ci, ev), []).append(np.nanmean(Xi[:, :, sel], 0))
    N = pd.DataFrame(rows)
    for k in traces:
        traces[k] = np.concatenate(traces[k], 1)
    Z = N[N.selected]
    st = {}
    for drug, (cb, ci) in (("vehicle", ("vb", "vi")), ("SBI-553", ("sb", "si"))):
        q = Z[Z.drug == drug]
        st[drug] = dict(n=len(q), n_all=int((N.drug == drug).sum()), per_mouse={m: int((q.mouse == m).sum()) for m in MO},
                        per_mouse_all={m: int(((N.drug == drug) & (N.mouse == m)).sum()) for m in MO})
        for ev in STIM:
            b, i = q[f"base_held|{ev}"], q[f"inj|{ev}"]
            st[drug][ev] = dict(med_base=float(b.median()), med_inj=float(i.median()), mean_base=float(b.mean()),
                                mean_inj=float(i.mean()), p_base_vs_0=wil(b), p_inj_vs_0=wil(i), p_base_inj=wil(i, b),
                                med_chg=float((i - b).median()),
                                mice_chg={m: float((i - b)[q.mouse == m].median()) if (q.mouse == m).any() else None for m in MO})
        for tag, col in (("base", "base_held"), ("inj", "inj")):
            M = q[[f"{col}|{ev}" for ev in STIM]].dropna().to_numpy()
            st[drug][f"friedman_{tag}"] = float(friedmanchisquare(*M.T).pvalue) if len(M) >= 5 else np.nan
    return dict(N=N, traces=traces, tt=tt, st=st)


# ------------------------------------------------------------------------------------------------ NS3 overlap
def ns3_overlap():
    T = pd.read_csv(os.path.join(EA, "cells_classes.csv")); T["mouse"] = T.mouse.astype(str)
    up = T.assign(up=T.cls == "activated").pivot_table(index=["mouse", "day", "phase", "cell"], columns="event", values="up",
                                                       aggfunc="first")
    up = up.reset_index()
    for c in ["withdrawal", "flinch"] + STIM:
        up[c] = up[c].astype("boolean").fillna(False).astype(bool)
    up["touch"] = up["Light touch"] | up["Mild touch"]
    up["noxious"] = up["Pin prick"] | up["Heat"]
    up["stimulus"] = up["touch"] | up["noxious"]
    up["reflex"] = up["withdrawal"] | up["flinch"]
    up["session"] = [next(c for c in SESS if sess_key(m, c) == (m, d, ph)) for m, d, ph in zip(up.mouse, up.day, up.phase)]
    res = {}
    for pair, (x, y) in (("touch_noxious", ("touch", "noxious")), ("stimulus_reflex", ("stimulus", "reflex"))):
        r = {}
        for c in SESS:
            q = up[up.session == c]
            bx, by = q[x].to_numpy(), q[y].to_numpy()
            r[c] = dict(n=len(q), x_only=100 * float((bx & ~by).mean()), both=100 * float((bx & by).mean()),
                        y_only=100 * float((~bx & by).mean()), any=100 * float((bx | by).mean()),
                        expected_both=100 * float(bx.mean() * by.mean()), x=100 * float(bx.mean()), y=100 * float(by.mean()),
                        p_assoc=float(fisher_exact([[int((bx & by).sum()), int((bx & ~by).sum())],
                                                    [int((~bx & by).sum()), int((~bx & ~by).sum())]])[1]),
                        mice={m: dict(both=100 * float((bx & by)[q.mouse.to_numpy() == m].mean()),
                                      any=100 * float((bx | by)[q.mouse.to_numpy() == m].mean()),
                                      n=int((q.mouse == m).sum())) for m in MO})
        # tests on 'both' (overlap): within day McNemar (same neurons), after vehicle vs after SBI-553 Fisher
        for cb, ci in (("vb", "vi"), ("sb", "si")):
            a = up[up.session == cb].set_index(["mouse", "cell"])
            b = up[up.session == ci].set_index(["mouse", "cell"])
            idx = a.index.intersection(b.index)
            r[f"p_{cb}_{ci}"] = mcnemar_p((a.loc[idx, x] & a.loc[idx, y]).to_numpy(), (b.loc[idx, x] & b.loc[idx, y]).to_numpy())
        qv, qs = up[up.session == "vi"], up[up.session == "si"]
        bv, bs = int((qv[x] & qv[y]).sum()), int((qs[x] & qs[y]).sum())
        r["p_vi_si"] = float(fisher_exact([[bv, len(qv) - bv], [bs, len(qs) - bs]])[1])
        res[pair] = r
    return up, res


# ------------------------------------------------------------------------------------------------ NS4 tuning crossover
def ns4_npi():
    T = pd.read_csv(os.path.join(EA, "cells_classes.csv")); T["mouse"] = T.mouse.astype(str)
    R = T[T.event.isin(STIM)].set_index(["mouse", "day", "phase", "event", "cell"]).resp
    rows = []
    for m in MO:
        P = pd.read_csv(os.path.join(CD, f"{m}_pairs.csv"))
        for p in P.itertuples():
            ids = {1: int(p.day1_id), 2: int(p.day2_id)}
            r = dict(mouse=m, track_class=p.track_class, day1_id=ids[1], day2_id=ids[2])
            for c in SESS:
                k = sess_key(m, c)
                for ev in STIM:
                    r[f"{c}|{ev}"] = float(R.get((m, k[1], k[2], ev, ids[k[1]]), np.nan))
                r[f"npi_{c}"] = (np.mean([r[f"{c}|Pin prick"], r[f"{c}|Heat"]]) -
                                 np.mean([r[f"{c}|Light touch"], r[f"{c}|Mild touch"]]))
            rows.append(r)
    X = pd.DataFrame(rows)
    X["chg_veh"] = X.npi_vi - X.npi_vb
    X["chg_sbi"] = X.npi_si - X.npi_sb
    X["d_norm"] = X.chg_sbi - X.chg_veh
    st = dict(n=len(X), per_mouse={m: int((X.mouse == m).sum()) for m in MO},
              med={c: float(X[f"npi_{c}"].median()) for c in SESS},
              p_vb_vi=wil(X.npi_vi, X.npi_vb), p_sb_si=wil(X.npi_si, X.npi_sb), p_vi_si=wil(X.npi_si, X.npi_vi),
              p_vb_sb=wil(X.npi_sb, X.npi_vb), p_vb_0=wil(X.npi_vb), p_sb_0=wil(X.npi_sb),
              med_chg_veh=float(X.chg_veh.median()), med_chg_sbi=float(X.chg_sbi.median()),
              med_norm=float(X.d_norm.median()), ci_norm=boot_ci(X.d_norm), p_norm=wil(X.d_norm),
              mice_chg={m: dict(veh=float(X.chg_veh[X.mouse == m].median()), sbi=float(X.chg_sbi[X.mouse == m].median()))
                        for m in MO})
    st["mice_sbi_lower"] = int(sum(st["mice_chg"][m]["sbi"] < st["mice_chg"][m]["veh"] for m in MO))
    XA = X[X.track_class == "A"]
    st["A_only"] = dict(n=len(XA), med_norm=float(XA.d_norm.median()), p_norm=wil(XA.d_norm))
    # tuning stability across the injection: Spearman over neurons, per stimulus, baseline vs injection, each day
    stab = {}
    for ev in STIM:
        stab[ev] = dict(veh=float(X[f"vb|{ev}"].corr(X[f"vi|{ev}"], method="spearman")),
                        sbi=float(X[f"sb|{ev}"].corr(X[f"si|{ev}"], method="spearman")),
                        days=float(X[f"vb|{ev}"].corr(X[f"sb|{ev}"], method="spearman")))
    st["stability"] = stab
    return X, st


# ------------------------------------------------------------------------------------------------ decoding (numbers only)
def decoding(S, D, n_shuf=200, n_rep=10):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    rng = np.random.default_rng(SEED)
    rows = []
    for key in sorted(S):
        m, d, ph = key
        Rs, y = [], []
        for i, ev in enumerate(STIM):
            Rs.append(D[key + (ev,)][0]); y += [i] * len(D[key + (ev,)][0])
        Xr = np.nan_to_num(np.concatenate(Rs)); y = np.array(y)
        r = dict(mouse=m, day=d, phase=ph, condition=E.cond_of(m, d, ph), n_trials=len(y), n_cells=Xr.shape[1])
        for task, yy, chance in (("stim4", y, 0.25), ("noxious_vs_touch", (y >= 2).astype(int), None)):
            def acc(lab, seed):
                cv = StratifiedKFold(5, shuffle=True, random_state=seed)
                return float((cross_val_predict(LDA(solver="lsqr", shrinkage="auto"), Xr, lab, cv=cv) == lab).mean())
            real = float(np.mean([acc(yy, s_) for s_ in range(n_rep)]))
            null = np.array([acc(rng.permutation(yy), 1000 + s_) for s_ in range(n_shuf)])
            r[f"{task}_acc"] = real
            r[f"{task}_null_mean"] = float(null.mean())
            r[f"{task}_null95"] = float(np.percentile(null, 95))
            r[f"{task}_p"] = float((1 + (null >= real).sum()) / (1 + n_shuf))
        rows.append(r)
    return pd.DataFrame(rows)



# ------------------------------------------------------------------------------------------------ figure helpers
def smooth(x, w=3):
    """moving average over w bins along axis 0 (display only)"""
    k = np.ones(w) / w
    pad = np.pad(x, ((w // 2, w // 2),) + ((0, 0),) * (x.ndim - 1), mode="edge")
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), 0, pad)


def sess_handles(lg, lw=2.2):
    return [Line2D([], [], color=SCOLS[c], ls=SLS[c], lw=lw, label=T_(c, lg)) for c in SESS]


def mouse_handles(s=7):
    return [Line2D([], [], marker=MK[m], ls="", color=MC[m], mec="k", mew=0.4, ms=s, label=m) for m in MO]


def title_left(fig, txt, y=0.985, fs=13):
    fig.text(0.01, y, txt, fontsize=fs, ha="left", va="top", fontweight="bold")


def wspan():
    a, b = WW.of("Pin prick")[2:]
    return f"{a:g}-{b:g} s"


def permouse(d_, key="per_mouse"):
    return ", ".join(f"{m} {d_[key][m]}" for m in MO)


# ------------------------------------------------------------------------------------------------ NS1 figure
def fig_ns1(P, lg, fn):
    font(lg)
    fig = plt.figure(figsize=(15, 6.3))
    tb, proj, var = P["tb"], P["proj"], P["var"]
    w0 = WW.of("Pin prick")
    # 10-25: no in-figure title (it repeated the slide title); the citation is in the notes (foot text below)
    # A: baseline trajectories
    a = fig.add_axes([0.058, 0.115, 0.237, 0.71])            # (footer + title went to the notes: taller axes)
    k = (tb >= -0.5) & (tb <= 2.0)
    for ev in STIM:
        p = smooth((proj[("vb", ev)] + proj[("sb", ev)]) / 2)[k]
        t = tb[k]
        a.plot(p[:, 0], p[:, 1], color=SCOL[ev], lw=2.2)
        i0, i1 = np.argmin(np.abs(t - 0)), np.argmin(np.abs(t - w0[3]))
        a.plot(p[i0, 0], p[i0, 1], "o", mfc="white", mec=SCOL[ev], mew=2, ms=8, zorder=4)
        a.plot(p[i1, 0], p[i1, 1], "o", color=SCOL[ev], ms=8, zorder=4)
        a.annotate("", xy=(p[-1, 0], p[-1, 1]), xytext=(p[-3, 0], p[-3, 1]),
                   arrowprops=dict(arrowstyle="-|>", color=SCOL[ev], lw=2, mutation_scale=14))
    a.set_xlabel(L(lg, f"PC1 ({100 * var[0]:.0f}% of variance)", f"PC1 (분산의 {100 * var[0]:.0f}%)"))
    a.set_ylabel(L(lg, f"PC2 ({100 * var[1]:.0f}% of variance)", f"PC2 (분산의 {100 * var[1]:.0f}%)"))
    a.set_title(L(lg, "A  baseline (mean of the two baseline\n    sessions), -0.5 to 2 s", "A  기준 (두 기준 세션 평균),\n    -0.5 ~ 2 s"),
                loc="left", fontsize=12.5)
    h = [Line2D([], [], color=SCOL[ev], lw=2.2, label=T_(ev, lg)) for ev in STIM]
    h += [Line2D([], [], marker="o", ls="", mfc="white", mec="k", mew=1.5, ms=7, label=L(lg, "stimulus (0 s)", "자극 (0 s)")),
          Line2D([], [], marker="o", ls="", color="k", ms=7, label=L(lg, f"end of window ({w0[3]:g} s)", f"반응 창 끝 ({w0[3]:g} s)"))]
    yl = a.get_ylim(); a.set_ylim(yl[0], yl[1] + 0.42 * (yl[1] - yl[0]))
    a.legend(handles=h, frameon=False, fontsize=11, loc="upper left", ncol=2, handlelength=1.4, borderaxespad=0.1,
             columnspacing=0.8)
    # B: PC1 over time, 2 x 2
    ys = np.concatenate([smooth(proj[kk])[:, 0] for kk in proj])
    lo, hi = np.min(ys), np.max(ys)
    pad = 0.08 * (hi - lo)
    pos = [(0.375, 0.50), (0.555, 0.50), (0.375, 0.115), (0.555, 0.115)]
    for i, ev in enumerate(STIM):
        b = fig.add_axes([pos[i][0], pos[i][1], 0.155, 0.325])
        b.axvspan(w0[2], w0[3], color="#f3e9c6", lw=0)
        b.axvline(0, color="0.4", lw=0.8, ls=":")
        for c in SESS:
            b.plot(tb, smooth(proj[(c, ev)])[:, 0], color=SCOLS[c], ls=SLS[c], lw=2.0)
        b.set_ylim(lo - pad, hi + pad)
        b.set_xlim(T0, T1)
        b.set_title(T_(ev, lg), color=SCOL[ev], fontsize=12, fontweight="bold", pad=3)
        if i >= 2:
            b.set_xlabel(T_("time", lg))
        else:
            b.set_xticklabels([])
        if i % 2 == 0:
            b.set_ylabel("PC1 (a.u.)")
        else:
            b.set_yticklabels([])
    fig.text(0.375, 0.865, L(lg, f"B  PC1 over time, each session (shading = {wspan()}, used in C)",
                            f"B  시간에 따른 PC1, 세션별 (음영 = {wspan()}, C의 창)"), fontsize=12.5, ha="left", va="bottom")
    # C: PC1 response in the best window, per session, with per-mouse projections
    c_ = fig.add_axes([0.78, 0.115, 0.205, 0.71])
    off = {"vb": -0.3, "vi": -0.1, "sb": 0.1, "si": 0.3}
    for j, ev in enumerate(STIM):
        for c in SESS:
            v = P["resp"][(c, ev)]
            c_.bar(j + off[c], v, 0.18, color=SCOLS[c], alpha=0.75 if c in ("vb", "sb") else 0.9,
                   hatch="///" if c == "vb" else None, edgecolor="white", lw=0.5)
            for m in MO:
                c_.scatter(j + off[c], P["mresp"][m][(c, ev)], marker=MK[m], s=26, color=MC[m], edgecolor="k", lw=0.4, zorder=4)
    c_.axhline(0, color="0.5", lw=0.8)
    c_.set_xticks(range(4)); c_.set_xticklabels([T_(ev, lg).replace(" ", "\n") for ev in STIM])
    c_.set_ylabel(L(lg, f"PC1 response, {wspan()} (a.u.)", f"PC1 반응, {wspan()} (a.u.)"))
    c_.set_title(L(lg, f"C  PC1 in {wspan()}, all stimuli\n    (bar = all, marker = one mouse)",
                   f"C  PC1, 모든 자극 {wspan()}\n    (막대 = 전체, 표식 = 마우스)"), loc="left", fontsize=12.5)
    hs = [Patch(facecolor=SCOLS["vb"], alpha=0.75, hatch="///", edgecolor="white", label=L(lg, "veh-day base", "veh 날 기준")),
          Patch(facecolor=CC["vehicle"], label=L(lg, "after veh", "veh 후")),
          Patch(facecolor=SCOLS["sb"], alpha=0.75, label=L(lg, "SBI-day base", "SBI 날 기준")),
          Patch(facecolor=CC["SBI-553"], label=L(lg, "after SBI", "SBI 후"))]
    fig.legend(handles=sess_handles(lg) + [Line2D([], [], color="none", label=L(lg, "    C markers = one mouse:",
                                                                                 "    C 표식 = 마우스 하나:"))] + mouse_handles(),
               loc="upper left", ncol=9, frameon=False, bbox_to_anchor=(0.005, 0.995), fontsize=11, columnspacing=0.9,
               handlelength=2.2, handletextpad=0.4)
    c_.legend(handles=hs, frameon=False, fontsize=11, loc="upper left", ncol=2, borderaxespad=0.1, columnspacing=0.6,
              handlelength=1.2)
    allm = [v for d_ in P["mresp"].values() for v in d_.values()]
    ymax = max(max(P["resp"].values()), max(allm))
    ymin = min(0, min(allm))
    c_.set_ylim(ymin - 0.1 * (ymax - ymin), ymax + 0.45 * (ymax - ymin))
    nt = P["ntr"]
    trmin = min(min(d_.values()) for d_ in nt.values()); trmax = max(max(d_.values()) for d_ in nt.values())
    pm = P["per_m"]
    foot(fig, L(lg,
                f"PCA of the trial-mean z (each trial minus its own 0.5 s before) of the {P['n']} tracked neurons pooled over 4 mice "
                f"(5941 {pm['5941']}, 5939 {pm['5939']}, 5937 {pm['5937']}, 1191 {pm['1191']}); 100-ms bins, -1 to 2.5 s; one PCA "
                f"on all 16 averages (4 sessions x 4 stimuli; {trmin}-{trmax} trials each). Lines smoothed over 0.3 s for display. "
                f"Marker = the same projection from one mouse's neurons alone. Descriptive, no test (4 mice; drug order confounded "
                f"with day). Split-half check: PC1 responses of odd vs even trials, r = {P['r_split']:.2f} over the 16 averages; "
                f"5939 carries {100 * P['load_share']['5939']:.0f}% of the PC1 weights ({100 * P['frac_neurons']['5939']:.0f}% of "
                f"the neurons). PC1 response {wspan()} (pin-prick window) for all stimuli. Figure: population activity around "
                f"each stimulus in PCA space, modelled on Pennington et al. 2026 Fig 2j / 4h.",
                f"추적 뉴런 {P['n']}개 (5941 {pm['5941']}, 5939 {pm['5939']}, 5937 {pm['5937']}, 1191 {pm['1191']}; 마우스 4마리 "
                f"합산)의 시행 평균 z (각 시행 - 직전 0.5 s)에 PCA; 100 ms 구간, -1 ~ 2.5 s; 16개 평균 (세션 4 x 자극 4; 각 "
                f"{trmin}-{trmax} 시행)에 PCA 하나. 선은 표시용 0.3 s 평활. 표식 = 마우스 하나의 뉴런만으로 같은 투영. 기술적, 검정 "
                f"없음 (마우스 4마리; 약물 순서와 날이 겹침). 반분 확인: 홀수 vs 짝수 시행 PC1 반응 r = {P['r_split']:.2f} (16개 "
                f"평균); PC1 가중치의 {100 * P['load_share']['5939']:.0f}%가 5939 (뉴런의 {100 * P['frac_neurons']['5939']:.0f}%). "
                f"PC1 반응은 모든 자극에 {wspan()} (Pin prick 창). 그림: 자극 주변 집단 활동의 PCA 공간 궤적, Pennington et al. "
                f"2026 그림 2j / 4h 방식."),
         fs=10)
    fig.savefig(fn, dpi=150); plt.close(fig)


# ------------------------------------------------------------------------------------------------ NS2 figure
def fig_ns2(P, lg, fn):
    font(lg)
    fig = plt.figure(figsize=(15, 6.3))
    N, tr, tt, st = P["N"], P["traces"], P["tt"], P["st"]
    w0 = WW.of("Pin prick")
    nv, ns_ = st["vehicle"]["n"], st["SBI-553"]["n"]
    # 10-25: no in-figure title (it repeated the slide title); the citation is in the notes (foot text below)
    pos = [(0.07, 0.52), (0.27, 0.52), (0.07, 0.14), (0.27, 0.14)]       # (footer + title -> notes: taller panels)
    tr = {k: smooth(v, 7) for k, v in tr.items()}
    allm = np.concatenate([np.nanmean(v, 1) for v in tr.values()])
    sem_max = max(np.nanmax(np.nanstd(v, 1) / np.sqrt(v.shape[1])) for v in tr.values())
    lo, hi = np.nanmin(allm) - sem_max, np.nanmax(allm) + sem_max
    for i, ev in enumerate(STIM):
        a = fig.add_axes([pos[i][0], pos[i][1], 0.185, 0.30])
        we = WW.of(ev)                                     # each stimulus's own response window (shading only)
        a.axvspan(we[2], we[3], color="#f3e9c6", lw=0)
        a.axvline(0, color="0.4", lw=0.8, ls=":")
        a.axhline(0, color="0.75", lw=0.6)
        for c in SESS:
            v = tr[(c, ev)]
            mu, se = np.nanmean(v, 1), np.nanstd(v, 1) / np.sqrt(v.shape[1])
            a.fill_between(tt, mu - se, mu + se, color=SCOLS[c], alpha=0.12, lw=0)
            a.plot(tt, mu, color=SCOLS[c], ls=SLS[c], lw=2.0)
        a.set_ylim(lo, hi); a.set_xlim(T0, T1)
        a.set_title(T_(ev, lg), color=SCOL[ev], fontsize=12, fontweight="bold", pad=3)
        if i >= 2:
            a.set_xlabel(T_("time", lg))
        else:
            a.set_xticklabels([])
        if i % 2 == 0:
            a.set_ylabel(L(lg, "z (dF/F / robust SD),\nminus the 0.5 s before", "z (dF/F / robust SD),\n직전 0.5 s 기준"),
                         fontsize=11.5)
        else:
            a.set_yticklabels([])
    fig.text(0.07, 0.87, L(lg, "A  mean ± s.e.m. (band) over neurons; yellow = response window",
                           "A  뉴런 평균 ± s.e.m. (띠); 노랑 = 반응 창"), fontsize=12.5, ha="left", va="bottom")
    # B: each neuron's response per stimulus, four sessions
    b = fig.add_axes([0.535, 0.14, 0.45, 0.625])
    Z = N[N.selected]
    xs = {"vb": 0.0, "vi": 0.85, "sb": 2.05, "si": 2.9}
    rng = np.random.default_rng(1)
    vals = []
    for j, ev in enumerate(STIM):
        x0 = j * 4.2
        for drug, (cb, ci) in (("vehicle", ("vb", "vi")), ("SBI-553", ("sb", "si"))):
            q = Z[Z.drug == drug]
            vb_, vi_ = q[f"base_held|{ev}"].to_numpy(), q[f"inj|{ev}"].to_numpy()
            for u, w_ in zip(vb_, vi_):
                b.plot([x0 + xs[cb], x0 + xs[ci]], [u, w_], color="0.6", lw=0.5, alpha=0.5, zorder=1)
            for c, v in ((cb, vb_), (ci, vi_)):
                jit = rng.uniform(-0.12, 0.12, len(v))
                b.scatter(x0 + xs[c] + jit, v, s=15, color=SCOLS[c] if c != "vb" else "white", edgecolor=SCOLS[c], lw=0.9,
                          alpha=0.85, zorder=2)
                b.plot([x0 + xs[c] - 0.28, x0 + xs[c] + 0.28], [np.nanmedian(v)] * 2, color="k", lw=2.6, zorder=3)
                vals.append(v)
    allv = np.concatenate(vals)
    lo, hi = np.nanpercentile(allv, [0.5, 99.5])
    span = hi - lo
    b.set_ylim(lo - 0.05 * span, hi + 0.30 * span)
    yb = hi + 0.06 * span
    for j, ev in enumerate(STIM):
        x0 = j * 4.2
        PS.bracket(b, x0 + xs["vb"], x0 + xs["vi"], yb, st["vehicle"][ev]["p_base_inj"], color=CC["vehicle"], fs=13)
        PS.bracket(b, x0 + xs["sb"], x0 + xs["si"], yb, st["SBI-553"][ev]["p_base_inj"], color=CC["SBI-553"], fs=13)
    b.axhline(0, color="0.6", lw=0.8)
    b.set_xticks([j * 4.2 + 1.45 for j in range(4)])
    b.set_xticklabels([f"{T_(ev, lg)}\n({win_of(ev)})" for ev in STIM])      # each stimulus's own response window
    for j in range(1, 4):
        b.axvline(j * 4.2 - 0.65, color="0.85", lw=0.8)
    b.set_xlim(-0.6, 3 * 4.2 + 3.5)
    b.set_ylabel(L(lg, "response (z): response window\nminus the 0.5 s before", "반응 (z): 반응 창\n- 직전 0.5 s"), fontsize=11.5)
    fig.text(0.535, 0.87, L(lg, "B  each neuron (dot; grey line = the same neuron), black bar = median",
                            "B  뉴런 하나 = 점 (회색 선 = 같은 뉴런), 검은 막대 = 중앙값"), fontsize=12.5, ha="left", va="bottom")
    # 2-row key above B (column-major): dots of the vehicle day | dots of the SBI-553 day | what the brackets compare
    hb = [Line2D([], [], marker="o", ls="", mfc="white", mec=SCOLS["vb"], mew=1.3, ms=9, label=L(lg, "veh-day base", "veh 날 기준")),
          Line2D([], [], marker="o", ls="", color=CC["vehicle"], ms=9, label=L(lg, f"after veh (n = {nv})", f"veh 후 (n = {nv})")),
          Line2D([], [], marker="o", ls="", color=SCOLS["sb"], ms=9, label=L(lg, "SBI-day base", "SBI 날 기준")),
          Line2D([], [], marker="o", ls="", color=CC["SBI-553"], ms=9, label=L(lg, f"after SBI (n = {ns_})", f"SBI 후 (n = {ns_})")),
          Line2D([], [], color=CC["vehicle"], lw=1.6, label=L(lg, "bracket: base vs after, veh day", "괄호: 기준 vs 후, veh 날")),
          Line2D([], [], color=CC["SBI-553"], lw=1.6, label=L(lg, "bracket: base vs after, SBI day", "괄호: 기준 vs 후, SBI 날"))]
    b.legend(handles=[hb[0], hb[2], hb[1], hb[3], hb[4], hb[5]], frameon=False, fontsize=11, loc="lower left", ncol=3,
             bbox_to_anchor=(0.0, 1.0), columnspacing=1.0, handletextpad=0.3, handlelength=1.5, borderaxespad=0.2)
    fig.legend(handles=sess_handles(lg, 2.4), loc="upper left", ncol=4, frameon=False, bbox_to_anchor=(0.005, 0.995), fontsize=11.5,
               title=None, handlelength=2.6)
    fv, fs_ = st["vehicle"], st["SBI-553"]
    pm = lambda d_: ", ".join(f"{m} {d_['per_mouse'][m]}/{d_['per_mouse_all'][m]}" for m in MO)
    foot(fig, L(lg,
                f"Every neuron of each day; kept if activated by Pin prick or Heat on half of the baseline trials (Wilcoxon signed-rank "
                f"over trials, p < 0.05; two folds: odd / even trials) and shown on the OTHER half (no selection bias); after the "
                f"injection all trials; traces smoothed over 0.23 s for display. Vehicle day {pm(fv)}; SBI-553 day {pm(fs_)} "
                f"(selected / all). Brackets: baseline vs after "
                f"the injection, same neurons, Wilcoxon signed-rank. Friedman over the 4 stimuli: vehicle day "
                f"{ptxt(fv['friedman_base'])} / {ptxt(fv['friedman_inj'])}, SBI-553 day {ptxt(fs_['friedman_base'])} / "
                f"{ptxt(fs_['friedman_inj'])} (baseline / after). Exploratory: neurons pooled, 4 mice, drug order confounded with "
                f"day. Figure: noxious-responsive neurons (Pin prick / Heat at baseline), responses to all 4 stimuli, as "
                f"Pennington et al. 2026 Fig 2h-i, 2n, 4f-g. " + PS.KEY,
                f"각 날의 모든 뉴런; 기준 시행의 절반에서 Pin prick 또는 Heat에 활성화되면 선택 (시행 간 Wilcoxon signed-rank, p < 0.05; "
                f"홀수 / 짝수 두 겹) 하고 나머지 절반으로 표시 (선택 편향 없음); 주사 후는 모든 시행; 궤적은 표시용 0.23 s 평활. "
                f"vehicle 날 {pm(fv)}; SBI-553 날 "
                f"{pm(fs_)} (선택 / 전체). 괄호: 기준 vs 주사 후, 같은 뉴런, Wilcoxon signed-rank. 4 자극 간 Friedman: vehicle 날 "
                f"{ptxt(fv['friedman_base'])} / {ptxt(fv['friedman_inj'])}, SBI-553 날 {ptxt(fs_['friedman_base'])} / "
                f"{ptxt(fs_['friedman_inj'])} (기준 / 후). 탐색적: 뉴런 합산, 마우스 4마리, 약물 순서와 날이 겹침. 그림: 통증 반응 "
                f"뉴런 (기준에서 Pin prick / Heat에 활성화)의 4 자극 반응, Pennington et al. 2026 그림 2h-i, 2n, 4f-g 방식. " + PS.KEY),
         fs=10)
    fig.savefig(fn, dpi=150); plt.close(fig)


# ------------------------------------------------------------------------------------------------ NS3 figure
def fig_ns3(P, lg, fn):
    font(lg)
    fig = plt.figure(figsize=(15, 6.3))
    # 10-25: no in-figure title (it repeated the slide title); the citation is in the notes (foot text below)
    xs = {"vb": 0.0, "vi": 1.0, "sb": 2.5, "si": 3.5}
    spec = (("touch_noxious", L(lg, "A  touch (Light / Mild) vs noxious (Pin prick / Heat)", "A  촉각 (Light / Mild touch) vs 통증 (Pin prick / Heat)"),
             (L(lg, "touch only", "촉각만"), L(lg, "noxious only", "통증만")), ("#aed6f1", "#f5b7b1")),
            ("stimulus_reflex", L(lg, "B  any stimulus vs reflex (withdrawal / flinch)", "B  자극 (아무거나) vs 반사 (withdrawal / flinch)"),
             (L(lg, "stimulus only", "자극만"), L(lg, "reflex only", "반사만")), ("#f9e79f", "#a9dfbf")))
    BOTH = "#6c3483"
    for i, (key, ttl, labs, cols) in enumerate(spec):
        r = P[key]
        a = fig.add_axes([0.06 + i * 0.49, 0.175, 0.40, 0.655])       # (footer + title -> notes: taller axes)
        for c in SESS:
            x, d_ = xs[c], r[c]
            a.bar(x, d_["both"], 0.62, color=BOTH, edgecolor="white", lw=0.5)
            a.bar(x, d_["x_only"], 0.62, bottom=d_["both"], color=cols[0], edgecolor="white", lw=0.5)
            a.bar(x, d_["y_only"], 0.62, bottom=d_["both"] + d_["x_only"], color=cols[1], edgecolor="white", lw=0.5)
            a.plot([x - 0.31, x + 0.31], [d_["expected_both"]] * 2, color="k", lw=1.6, ls=(0, (2, 1.5)), zorder=4)
            for jm, m in enumerate(MO):
                a.scatter(x + 0.40 + 0.06 * jm, d_["mice"][m]["both"], marker=MK[m], s=32, color=MC[m], edgecolor="k", lw=0.4,
                          zorder=5, clip_on=False)
            a.text(x, d_["any"] + 0.8, f"{d_['any']:.0f}%", ha="center", va="bottom", fontsize=11.5, color="0.25")
        top = max(max(r[c]["any"] for c in SESS), max(r[c]["mice"][m]["both"] for c in SESS for m in MO))
        a.set_ylim(0, top * 1.75)
        PS.bracket(a, xs["vb"], xs["vi"], top * 1.18, r["p_vb_vi"], color=CC["vehicle"], fs=13)
        PS.bracket(a, xs["sb"], xs["si"], top * 1.18, r["p_sb_si"], color=CC["SBI-553"], fs=13)
        PS.bracket(a, xs["vi"], xs["si"], top * 1.38, r["p_vi_si"], fs=13)
        a.set_xticks(list(xs.values()))
        a.set_xticklabels([f"{T_(c + '2', lg)}\n(n = {r[c]['n']})" for c in SESS], fontsize=11.5)
        for t_, c in zip(a.get_xticklabels(), SESS):
            t_.set_color(SCOLS[c] if c in ("vi", "si") else "0.3")
        a.set_xlim(-0.6, 4.25)
        a.set_ylabel(L(lg, "% of neurons (activated)", "뉴런 % (활성화)"))
        a.set_title(ttl, loc="left", fontsize=12.5)
        h = [Patch(color=cols[1], label=labs[1]), Patch(color=cols[0], label=labs[0]),
             Patch(color=BOTH, label=L(lg, "both (overlap)", "둘 다 (겹침)")),
             Line2D([], [], color="k", lw=1.6, ls=(0, (2, 1.5)), label=L(lg, "both if independent", "독립이면 예상되는 둘 다"))]
        a.legend(handles=h, frameon=False, fontsize=11.5, loc="upper left", ncol=2, bbox_to_anchor=(0.0, 1.02),
                 columnspacing=0.8, handlelength=1.4)
    fig.legend(handles=mouse_handles(8), loc="upper right", ncol=4, frameon=False, bbox_to_anchor=(0.995, 0.995), fontsize=11.5,
               title=L(lg, "marker = 'both' % of one mouse", "표식 = 마우스 하나의 '둘 다' %"), title_fontsize=11.5)
    pa = max(P[k][c]["p_assoc"] for k in P for c in SESS)
    foot(fig, L(lg,
                "Activated = within-block time-shift test, BH q <= 0.20 per session (cells_classes.csv), best windows (stimuli "
                f"{stim_words()}, reflexes {WW.beh_span()} after vs before). Bar height = % responsive to either class (number on top). "
                "Brackets test the overlap ('both'): baseline vs after the injection = the same neurons, McNemar exact; after vehicle "
                "vs after SBI-553 = different days' neurons pooled, Fisher exact (exploratory). Overlap above the dashed line in every "
                f"session (Fisher exact for association, all {ptxt(pa).replace('p = ', 'p <= ')}). 4 mice; drug order confounded "
                "with day. Figure: neurons activated by one stimulus class, the other or both (all neurons of each day), as "
                "Pennington et al. 2026 Fig 2c, 2g, 4e. " + PS.KEY,
                "활성화 = 블록 내 시간 이동 검정, 세션별 BH q <= 0.20 (cells_classes.csv), 최적 창 (자극 "
                f"{stim_words()}, 반사 후 {WW.beh_span()} vs 직전). 막대 높이 = 둘 중 하나에 반응하는 % (위 숫자). 괄호 = 겹침 ('둘 다') "
                "검정: 기준 vs 주사 후 = 같은 뉴런, McNemar exact; vehicle 후 vs SBI-553 후 = 다른 날 뉴런 합산, Fisher exact (탐색적). "
                f"모든 세션에서 겹침이 점선 (독립 예상) 위 (연관 Fisher exact, 모두 {ptxt(pa).replace('p = ', 'p <= ')}). "
                "마우스 4마리; 약물 순서와 날이 겹침. 그림: 한 자극 종류, 다른 종류, 또는 둘 다에 활성화된 뉴런 (각 날의 모든 "
                "뉴런), Pennington et al. 2026 그림 2c, 2g, 4e 방식. " + PS.KEY), fs=10)
    fig.savefig(fn, dpi=150); plt.close(fig)


# ------------------------------------------------------------------------------------------------ NS4 figure
def fig_ns4(X, st, lg, fn):
    font(lg)
    fig = plt.figure(figsize=(15, 6.3))
    # 10-25: no in-figure title (it repeated the slide title); the citation is in the notes (foot text below)
    X = X.copy()
    X["order"] = X[[f"npi_{c}" for c in SESS]].mean(1)
    X = pd.concat([X[X.mouse == m].sort_values("order", ascending=False) for m in MO])
    cols = [f"{c}|{ev}" for c in SESS for ev in STIM]
    M = X[cols].to_numpy()
    vmax = float(np.nanpercentile(np.abs(M), 97)) or 0.3
    # (footer -> notes, 10-24: taller panels; keys for every colour on the figure)
    a = fig.add_axes([0.1, 0.2, 0.258, 0.65])
    im = a.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
    for j in (4, 8, 12):
        a.axvline(j - 0.5, color="k", lw=1.6 if j == 8 else 0.9)
    y0 = 0
    for m in MO:
        n = int((X.mouse == m).sum())
        if y0:
            a.axhline(y0 - 0.5, color="k", lw=1.0)
        a.text(-0.8, y0 + n / 2, f"{m}\n(n={n})", ha="right", va="center", fontsize=11.5, color=MC[m], fontweight="bold")
        y0 += n
    a.set_yticks([])
    a.set_ylabel(L(lg, "neurons (1 row = 1 neuron),\ngrouped by mouse", "뉴런 (1행 = 뉴런 1개),\n마우스별로 묶음"), fontsize=12)
    a.yaxis.set_label_coords(-0.205, 0.5)
    a.set_xticks(range(16)); a.set_xticklabels(["L", "M", "P", "H"] * 4, fontsize=11.5)
    for j, c in enumerate(SESS):
        a.text(j * 4 + 1.5, -2, T_(c + "2", lg), ha="center", va="bottom", fontsize=11.5, color=SCOLS[c], fontweight="bold")
    a.set_xlabel(L(lg, f"stimulus: L light, M mild, P pin prick, H heat\n(window: {stim_words()})",
                   f"자극: L Light touch, M Mild touch, P Pin prick, H Heat\n(반응 창: {stim_words()})"), fontsize=11.5)
    cax = fig.add_axes([0.364, 0.27, 0.008, 0.51])
    cb = fig.colorbar(im, cax=cax, extend="both")         # |response| above the 97th percentile drawn at the ends
    vt = np.floor(vmax * 100) / 100
    cb.set_ticks([-vt, -vt / 2, 0, vt / 2, vt])
    cb.set_ticklabels([f"{-vt:.2f}", f"{-vt / 2:.2f}", "0", f"{vt / 2:.2f}", f"{vt:.2f}"])
    cb.ax.tick_params(labelsize=11)
    cb.set_label(L(lg, "response, z (dF/F / robust SD):\nresponse window minus the 0.5 s before",
                   "반응, z (dF/F / robust SD):\n반응 창 - 직전 0.5 s"), fontsize=11)
    fig.text(0.1, 0.935, L(lg, "A  response of each neuron (row) to each stimulus", "A  뉴런 (행)별 자극 반응"), fontsize=12.5)
    b = fig.add_axes([0.53, 0.225, 0.235, 0.59])
    xs = {"vb": 0, "vi": 1, "sb": 2.4, "si": 3.4}
    for z in X.itertuples():
        b.plot([0, 1], [z.npi_vb, z.npi_vi], color=MC[z.mouse], lw=0.6, alpha=0.35)
        b.plot([2.4, 3.4], [z.npi_sb, z.npi_si], color=MC[z.mouse], lw=0.6, alpha=0.35)
    rng = np.random.default_rng(1)
    for c in SESS:
        v = X[f"npi_{c}"]
        b.scatter(np.full(len(v), xs[c]) + rng.uniform(-0.07, 0.07, len(v)), v, s=11, color=SCOLS[c], alpha=0.6, lw=0)
        b.plot([xs[c] - 0.22, xs[c] + 0.22], [v.median()] * 2, color="k", lw=3)
    vals = X[[f"npi_{c}" for c in SESS]].to_numpy()
    lo, hi = np.nanpercentile(vals, [1, 99]); span = hi - lo
    b.set_ylim(lo - 0.05 * span, hi + 0.42 * span)
    yb = hi + 0.06 * span
    PS.bracket(b, 0, 1, yb, st["p_vb_vi"], color=CC["vehicle"], fs=13)
    PS.bracket(b, 2.4, 3.4, yb, st["p_sb_si"], color=CC["SBI-553"], fs=13)
    PS.bracket(b, 1, 3.4, yb + 0.15 * span, st["p_vi_si"], fs=13)
    PS.bracket(b, 0, 2.4, yb + 0.28 * span, st["p_vb_sb"], color="0.45", fs=13)
    b.axhline(0, color="0.6", lw=0.8)
    b.set_xticks(list(xs.values())); b.set_xticklabels([T_(c + "2", lg) for c in SESS], fontsize=11.5)
    for t_, c in zip(b.get_xticklabels(), SESS):
        t_.set_color(SCOLS[c])                               # dot colour = session = the colour of its x label
        t_.set_fontweight("bold")
    b.set_ylabel(L(lg, "noxious preference (z)\nmean(P, H) - mean(L, M)", "통증 선호 (z)\nmean(P, H) - mean(L, M)"))
    b.set_title(L(lg, "B  each neuron: dot (colour = session)\n    and line (colour = mouse); black = median",
                  "B  뉴런마다: 점 (색 = 세션)\n    과 선 (색 = 마우스); 검은 선 = 중앙값"), loc="left", fontsize=12)
    c_ = fig.add_axes([0.87, 0.225, 0.12, 0.59])
    for m in MO:
        v = st["mice_chg"][m]
        c_.plot([0, 1], [v["veh"], v["sbi"]], color=MC[m], marker=MK[m], lw=1.5, ms=8, mec="k", mew=0.4)
    c_.axhline(0, color="0.6", lw=0.8)
    c_.set_xticks([0, 1]); c_.set_xticklabels([L(lg, "veh\nday", "veh\n날"), L(lg, "SBI\nday", "SBI\n날")]); c_.set_xlim(-0.4, 1.4)
    c_.set_ylabel(L(lg, "change in preference from the\nsame-day baseline (z), mouse median",
                    "같은 날 기준 대비 선호 변화 (z),\n마우스 중앙값"))
    c_.set_title(L(lg, "C  per mouse", "C  마우스별"), loc="left", fontsize=12.5)
    yl = c_.get_ylim(); c_.set_ylim(yl[0], yl[1] + 0.45 * (yl[1] - yl[0]))
    c_.text(0.03, 0.98, L(lg, f"neurons: median d =\n{st['med_norm']:+.3f} z, Wilcoxon\n{ptxt(st['p_norm'])}",
                          f"뉴런: d 중앙값 =\n{st['med_norm']:+.3f} z, Wilcoxon\n{ptxt(st['p_norm'])}"),
            transform=c_.transAxes, va="top", fontsize=11)
    # mouse key for B (line colour) and C (marker), and the bracket key, both on the figure
    fig.legend(handles=mouse_handles(8), loc="upper right", ncol=4, frameon=False, bbox_to_anchor=(0.995, 0.995), fontsize=11.5,
               title=L(lg, "mouse (B: line colour, C: marker)", "마우스 (B: 선 색, C: 표식)"), title_fontsize=11.5,
               columnspacing=1.0)
    hk = [Line2D([], [], color=CC["vehicle"], lw=1.8, label=L(lg, "veh day: base vs after", "veh 날: 기준 vs 후")),
          Line2D([], [], color=CC["SBI-553"], lw=1.8, label=L(lg, "SBI day: base vs after", "SBI 날: 기준 vs 후")),
          Line2D([], [], color="k", lw=1.8, label=L(lg, "after veh vs after SBI", "veh 후 vs SBI 후")),
          Line2D([], [], color="0.45", lw=1.8, label=L(lg, "the two baselines", "두 기준"))]
    fig.legend(handles=hk, loc="lower left", ncol=2, frameon=False, bbox_to_anchor=(0.47, 0.0), fontsize=11.5,
               title=L(lg, "brackets (Wilcoxon signed-rank, paired by neuron):", "괄호 (Wilcoxon signed-rank, 뉴런별 짝):"),
               title_fontsize=11.5, alignment="left", columnspacing=1.2, handlelength=1.6)
    foot(fig, L(lg,
                f"Neurons followed on both days ({permouse(st)}); response = response-window mean z minus the 0.5 s before, trial "
                f"mean (cells_classes.csv, window {stim_words()}); rows sorted within each mouse by the mean preference over the four "
                f"sessions. B: Wilcoxon signed-rank paired by neuron (blue = vehicle day, red = SBI-553 day, black = after vehicle vs "
                f"after SBI-553, grey = the two baselines); y axis 1st-99th percentile. C: d = (SBI-553 - its baseline) - (vehicle - "
                f"its baseline) per neuron. Exploratory: 4 mice, drug order confounded with day. Figure: tuning across the 4 "
                f"stimuli in the same {st['n']} tracked neurons, before / after each injection, as Pennington et al. 2026 Fig 2h-i, "
                f"2n. " + PS.KEY,
                f"두 날 모두 추적한 뉴런 ({permouse(st)}); 반응 = 반응 창 평균 z - 직전 0.5 s, 시행 평균 (cells_classes.csv, 창 "
                f"{stim_words()}); 행 = 마우스 안에서 네 세션 평균 선호 순. B: 뉴런별 짝지은 Wilcoxon signed-rank (파랑 = vehicle 날, "
                f"빨강 = SBI-553 날, 검은색 = vehicle 후 vs SBI-553 후, 회색 = 두 기준); y축 1-99 백분위. C: d = (SBI-553 - 그날 기준) "
                f"- (vehicle - 그날 기준), 뉴런별. 탐색적: 마우스 4마리, 약물 순서와 날이 겹침. 그림: 같은 추적 뉴런 {st['n']}개의 "
                f"4 자극 튜닝, 각 주사 전 / 후, Pennington et al. 2026 그림 2h-i, 2n 방식. " + PS.KEY), fs=10)
    fig.savefig(fn, dpi=150); plt.close(fig)


# ------------------------------------------------------------------------------------------------ summary
def f2(x):
    return f"{round(x, 2) + 0.0:.2f}"            # no "-0.00"


def f3(x):
    return f"{x:+.3f}"


def summary(P1, P2, P3, P4, DEC):
    """summary.json: per figure (EN file name) the numbers it shows + proposed slide texts (every number computed)"""
    J = {}
    # ---- NS1
    keys = P1["keys"]
    mean_st = {ev: float(np.mean([P1["resp"][(c, ev)] for c in SESS])) for ev in STIM}
    order = sorted(STIM, key=lambda e: -mean_st[e])
    nox_gt = {m: float(np.mean([np.mean([P1["mresp"][m][(c, "Pin prick")], P1["mresp"][m][(c, "Heat")]]) -
                                np.mean([P1["mresp"][m][(c, "Light touch")], P1["mresp"][m][(c, "Mild touch")]]) for c in SESS]))
              for m in MO}
    sess_nox = {c: float(np.mean([P1["resp"][(c, "Pin prick")], P1["resp"][(c, "Heat")]]) -
                         np.mean([P1["resp"][(c, "Light touch")], P1["resp"][(c, "Mild touch")]])) for c in SESS}
    n_mice_nox = int(sum(v > 0 for v in nox_gt.values()))
    n_sess_nox = int(sum(v > 0 for v in sess_nox.values()))
    n_pos = int(sum(P1["resp"][k] > 0 for k in keys))
    chg = {ev: dict(veh=P1["resp"][("vi", ev)] - P1["resp"][("vb", ev)], sbi=P1["resp"][("si", ev)] - P1["resp"][("sb", ev)])
           for ev in STIM}
    k_sbi_more = int(sum(chg[e]["sbi"] > chg[e]["veh"] for e in STIM))
    rose_en = "rose after every stimulus in all 4 sessions" if n_pos == 16 else f"rose in {n_pos}/16 averages"
    rose_kr = "모든 세션에서 모든 자극 후 상승" if n_pos == 16 else f"16개 평균 중 {n_pos}개에서 상승"
    dec_ok4 = int((DEC.stim4_p < 0.05).sum()); dec_ok2 = int((DEC.noxious_vs_touch_p < 0.05).sum())
    shortn = {"Light touch": "Light", "Mild touch": "Mild", "Pin prick": "Pin", "Heat": "Heat"}
    shortk = {"Light touch": "약한 촉각", "Mild touch": "중간 촉각", "Pin prick": "핀", "Heat": "열"}
    J["NS1_pca_trajectories.png"] = dict(
        section="Q3 population", priority="main",
        mirrors="Pennington et al. 2026 Fig 2j and Fig 4h (PCA of the trial-averaged population around each stimulus)",
        numbers=dict(n_neurons=P1["n"], per_mouse=P1["per_m"], var_explained_pc1_pc2_pc3=[float(v) for v in P1["var"][:3]],
                     pc1_response_window=wspan(),
                     pc1_response={f"{c}|{ev}": P1["resp"][(c, ev)] for c, ev in keys},
                     pc1_response_mean_over_sessions=mean_st, pc1_response_per_mouse={m: {f"{c}|{ev}": P1["mresp"][m][(c, ev)]
                                                                                      for c, ev in keys} for m in MO},
                     noxious_minus_touch_per_session=sess_nox, noxious_minus_touch_per_mouse=nox_gt,
                     change_from_same_day_baseline=chg, n_averages_positive=n_pos,
                     split_half_r_odd_even=P1["r_split"], pc1_weight_share_per_mouse=P1["load_share"],
                     neuron_share_per_mouse=P1["frac_neurons"],
                     trials_per_average={f"{c}|{ev}": v_ for (c, ev), v_ in P1["ntr"].items()},
                     decoding=dict(method="LDA (shrinkage), 5-fold stratified CV x 10, label-shuffle null (200), all neurons of a "
                                          "session, best-window single-trial responses",
                                   stim4_acc=DEC.stim4_acc.round(3).tolist(), stim4_chance=0.25,
                                   stim4_sessions_p05=dec_ok4, noxious_vs_touch_acc=DEC.noxious_vs_touch_acc.round(3).tolist(),
                                   noxious_vs_touch_null_mean=DEC.noxious_vs_touch_null_mean.round(3).tolist(),
                                   noxious_vs_touch_sessions_p05=dec_ok2, sessions=[f"{m}-d{d}-{ph}" for m, d, ph in
                                                                                     zip(DEC.mouse, DEC.day, DEC.phase)],
                                   stim4_acc_range=[float(DEC.stim4_acc.min()), float(DEC.stim4_acc.max())],
                                   noxious_vs_touch_acc_range=[float(DEC.noxious_vs_touch_acc.min()),
                                                               float(DEC.noxious_vs_touch_acc.max())])),
        title_en="Population trajectories: every stimulus moves BLA activity along one axis",
        title_kr="집단 궤적: 모든 자극이 BLA 활동을 한 축으로 움직임",
        key_en=(f"PC1 ({100 * P1['var'][0]:.0f}% var.) {rose_en}; mean {' > '.join(shortn[e] for e in order)}; change larger "
                f"on SBI-553 day for {k_sbi_more}/4 stimuli"),
        key_kr=(f"PC1 (분산 {100 * P1['var'][0]:.0f}%)은 {rose_kr}; 평균 {' > '.join(shortk[e] for e in order)}; SBI-553 날 "
                f"변화가 더 큰 자극 {k_sbi_more}/4"),
        bottom_en=(f"PCA of trial-mean z, {P1['n']} tracked neurons (4 mice), 4 sessions x 4 stimuli, -1 to 2.5 s. Descriptive, "
                   f"no test; split-half r = {P1['r_split']:.2f}. PC1 response {wspan()} (pin-prick window) for all stimuli."),
        bottom_kr=(f"추적 뉴런 {P1['n']}개 (마우스 4) 시행 평균 z PCA, 세션 4 x 자극 4, -1 ~ 2.5 s. 기술적, 검정 없음; 반분 r = "
                   f"{P1['r_split']:.2f}. PC1 반응은 모든 자극에 {wspan()} (Pin prick 창)."),
        notes_en=(f"Mirrors the paper's Fig 2j / 4h, where PC1 (~25% of variance) ordered stimuli by intensity. Here PC1 explains "
                  f"{100 * P1['var'][0]:.0f}% and PC2 {100 * P1['var'][1]:.0f}%; the PC1 response in the {'pin-prick window, used for all stimuli' if WW.mixed() else 'best window'} ({wspan()}) "
                  f"was positive in {n_pos}/16 averages, larger for noxious (Pin, Heat) than touch (Light, Mild) in "
                  f"{n_sess_nox}/4 sessions and in {n_mice_nox}/4 mice (each mouse's neurons alone). Session-by-session ordering "
                  f"is noisy: odd vs even trials r = {P1['r_split']:.2f} over the 16 averages; the SBI-553-day baseline Heat "
                  f"average is the outlier ({f2(P1['resp'][('sb', 'Heat')])}). 5939 carries "
                  f"{100 * P1['load_share']['5939']:.0f}% of the PC1 weights with {100 * P1['frac_neurons']['5939']:.0f}% of the "
                  f"neurons. Change from the same-day baseline (PC1, a.u.): "
                  + "; ".join(f"{shortn[e]} veh {chg[e]['veh']:+.2f} / SBI {chg[e]['sbi']:+.2f}" for e in STIM)
                  + f". Single-trial decoding (numbers only, no figure): 4-way stimulus accuracy "
                  f"{DEC.stim4_acc.min():.2f}-{DEC.stim4_acc.max():.2f} (chance 0.25), above the shuffle null (p < 0.05) in "
                  f"{dec_ok4}/16 sessions; noxious vs touch {DEC.noxious_vs_touch_acc.min():.2f}-"
                  f"{DEC.noxious_vs_touch_acc.max():.2f}, p < 0.05 in {dec_ok2}/16. Stimuli come in one-type blocks, so a decoder "
                  f"can also use block state - therefore no decoding-over-time figure. Drug order confounded with day."),
        notes_kr=(f"논문 그림 2j / 4h (PC1 ~25%가 자극 강도 순서)와 같은 분석. 여기서 PC1 = 분산 {100 * P1['var'][0]:.0f}%, PC2 "
                  f"{100 * P1['var'][1]:.0f}%; {'pin prick 창 (모든 자극에 같은 창)' if WW.mixed() else '최적 창'} ({wspan()}) PC1 반응은 16개 평균 중 {n_pos}개에서 양수, 통증 (핀, 열) > 촉각 "
                  f"(약한, 중간)은 세션 {n_sess_nox}/4, 마우스 {n_mice_nox}/4 (마우스 하나의 뉴런만). 세션별 순서는 잡음이 큼: 홀수 vs "
                  f"짝수 시행 r = {P1['r_split']:.2f} (16개 평균); SBI-553 날 기준 열 평균이 예외 ({f2(P1['resp'][('sb', 'Heat')])}). "
                  f"PC1 가중치의 {100 * P1['load_share']['5939']:.0f}%가 5939 (뉴런의 {100 * P1['frac_neurons']['5939']:.0f}%). "
                  f"같은 날 기준 대비 변화 (PC1, a.u.): "
                  + "; ".join(f"{shortk[e]} veh {chg[e]['veh']:+.2f} / SBI {chg[e]['sbi']:+.2f}" for e in STIM)
                  + f". 단일 시행 해독 (숫자만, 그림 없음): 4 자극 정확도 {DEC.stim4_acc.min():.2f}-{DEC.stim4_acc.max():.2f} "
                  f"(우연 0.25), 섞기 귀무보다 높음 (p < 0.05) 세션 {dec_ok4}/16; 통증 vs 촉각 {DEC.noxious_vs_touch_acc.min():.2f}-"
                  f"{DEC.noxious_vs_touch_acc.max():.2f}, p < 0.05 세션 {dec_ok2}/16. 자극이 한 종류씩 블록으로 와서 해독기가 블록 "
                  f"상태도 쓸 수 있음 - 그래서 시간별 해독 그림은 없음. 약물 순서와 날이 겹침."))
    # ---- NS2
    st = P2["st"]; v, s_ = st["vehicle"], st["SBI-553"]
    lt_v, lt_s = v["Light touch"], s_["Light touch"]
    mice_lt = {m: dict(veh=lt_v["mice_chg"][m], sbi=lt_s["mice_chg"][m]) for m in MO}
    n_lt_up = int(sum(1 for m in MO if mice_lt[m]["sbi"] is not None and mice_lt[m]["veh"] is not None
                      and mice_lt[m]["sbi"] > mice_lt[m]["veh"]))
    top_m = max(MO, key=lambda m: s_["per_mouse"][m])
    kept = all(st[d][e]["p_inj_vs_0"] < 0.05 for d in st for e in ("Pin prick", "Heat"))
    kept_en = "kept their noxious responses after both injections" if kept else "noxious responses did not survive every injection"
    kept_kr = "두 주사 후에도 통증 반응 유지" if kept else "통증 반응이 모든 주사 후 유지되지는 않음"
    veh_list = ", ".join(f"{shortn[e]} {ptxt(v[e]['p_base_inj'])}" for e in STIM)
    sbi_list = ", ".join(f"{shortn[e]} {ptxt(s_[e]['p_base_inj'])}" for e in STIM[1:])
    J["NS2_noxious_neurons.png"] = dict(
        section="Q3 neurons", priority="backup",
        mirrors="Pennington et al. 2026 Fig 2h-i, 4f-g (responsive neurons across stimulus amplitudes: time course + each "
                "neuron's mean) and Fig 2n (the shock-responsive subpopulation's response to every stimulus)",
        numbers=dict(selection="activated by Pin prick or Heat on half of the baseline trials (Wilcoxon signed-rank p < 0.05, "
                               "mean > 0), two folds, measured on the held-out half",
                     n_selected=dict(vehicle_day=v["n"], sbi_day=s_["n"]), n_all=dict(vehicle_day=v["n_all"], sbi_day=s_["n_all"]),
                     per_mouse_selected=dict(vehicle_day=v["per_mouse"], sbi_day=s_["per_mouse"]),
                     per_stimulus={d: {ev: {k: st[d][ev][k] for k in ("med_base", "med_inj", "mean_base", "mean_inj", "p_base_vs_0",
                                                                    "p_inj_vs_0", "p_base_inj", "med_chg", "mice_chg")}
                                       for ev in STIM} for d in st},
                     friedman=dict(vehicle_day=dict(base=v["friedman_base"], inj=v["friedman_inj"]),
                                   sbi_day=dict(base=s_["friedman_base"], inj=s_["friedman_inj"])),
                     light_touch_change_per_mouse=mice_lt, mice_light_touch_rise_larger_with_sbi=n_lt_up),
        title_en="Noxious-responsive neurons: responses to all four stimuli before / after each injection",
        title_kr="통증 반응 뉴런: 각 주사 전 / 후 네 자극 모두에 대한 반응",
        key_en=(f"Pin/Heat neurons {kept_en}; Light touch response rose after SBI-553 "
                f"({f2(lt_s['med_base'])}->{f2(lt_s['med_inj'])} z), mainly in {top_m}"),
        key_kr=(f"핀/열 뉴런은 {kept_kr}; 약한 촉각 반응은 SBI-553 후 상승 ({f2(lt_s['med_base'])}->"
                f"{f2(lt_s['med_inj'])} z), 주로 {top_m}"),
        bottom_en=(f"Neurons activated by Pin/Heat on half the baseline trials, shown on the other half (veh day n = {v['n']}, SBI "
                   f"day n = {s_['n']}); baseline vs after, same neurons, Wilcoxon signed-rank."),
        bottom_kr=(f"기준 시행 절반에서 핀/열에 활성화된 뉴런, 나머지 절반으로 표시 (veh 날 n = {v['n']}, SBI 날 n = {s_['n']}); "
                   f"기준 vs 후, 같은 뉴런, Wilcoxon signed-rank."),
        notes_en=(f"Selection is cross-validated, so the baseline responses are unbiased: held-out baseline Pin prick median "
                  f"{f2(v['Pin prick']['med_base'])} / {f2(s_['Pin prick']['med_base'])} z and Heat {f2(v['Heat']['med_base'])} / "
                  f"{f2(s_['Heat']['med_base'])} z (vehicle / SBI-553 day; vs 0 Wilcoxon {ptxt(v['Pin prick']['p_base_vs_0'])}"
                  f", {ptxt(s_['Pin prick']['p_base_vs_0'])}, {ptxt(v['Heat']['p_base_vs_0'])}, {ptxt(s_['Heat']['p_base_vs_0'])})"
                  f". These neurons also respond to Mild touch (median {f2(v['Mild touch']['med_base'])} / "
                  f"{f2(s_['Mild touch']['med_base'])} z), i.e. they are not noxious-only. Baseline vs after: vehicle day "
                  f"({veh_list}); "
                  f"SBI-553 day Light touch {ptxt(lt_s['p_base_inj'])}, others "
                  f"{sbi_list}. "
                  f"The Light touch rise is carried by {top_m} ({s_['per_mouse'][top_m]} of {s_['n']} SBI-day neurons); per-mouse "
                  f"median change after SBI-553 was larger than after vehicle in {n_lt_up}/4 mice, and the SBI-day baseline Light "
                  f"touch response was lower ({f2(lt_s['med_base'])} vs {f2(lt_v['med_base'])} z). The two days are different "
                  f"neuron sets (no cross-day z test, user rule). Friedman across the 4 stimuli (analogue of the paper's RM-ANOVA): "
                  f"vehicle day {ptxt(v['friedman_base'])} / {ptxt(v['friedman_inj'])}, SBI day {ptxt(s_['friedman_base'])} / "
                  f"{ptxt(s_['friedman_inj'])} (baseline / after). Exploratory; drug order confounded with day."),
        notes_kr=(f"선택이 교차 검증이라 기준 반응에 편향 없음: 남겨둔 기준 핀 자극 중앙값 {f2(v['Pin prick']['med_base'])} / "
                  f"{f2(s_['Pin prick']['med_base'])} z, 열 {f2(v['Heat']['med_base'])} / {f2(s_['Heat']['med_base'])} z (vehicle / "
                  f"SBI-553 날). 이 뉴런들은 중간 촉각에도 반응 (중앙값 {f2(v['Mild touch']['med_base'])} / "
                  f"{f2(s_['Mild touch']['med_base'])} z) - 통증만의 뉴런은 아님. 기준 vs 후: vehicle 날 ({veh_list}); SBI-553 날 약한 촉각 "
                  f"{ptxt(lt_s['p_base_inj'])}. 약한 촉각 상승은 주로 {top_m} (SBI 날 뉴런 {s_['n']}개 중 "
                  f"{s_['per_mouse'][top_m]}개); 마우스별 중앙값 변화가 SBI-553 후 > vehicle 후인 마우스 {n_lt_up}/4, SBI 날 기준 "
                  f"약한 촉각 반응이 더 낮았음 ({f2(lt_s['med_base'])} vs {f2(lt_v['med_base'])} z). 두 날은 다른 뉴런 집합 (날 간 "
                  f"z 검정 없음, 사용자 규칙). 4 자극 간 Friedman (논문의 RM-ANOVA에 해당): vehicle 날 {ptxt(v['friedman_base'])} / "
                  f"{ptxt(v['friedman_inj'])}, SBI 날 {ptxt(s_['friedman_base'])} / {ptxt(s_['friedman_inj'])} (기준 / 후). 탐색적; "
                  f"약물 순서와 날이 겹침."))
    # ---- NS3
    tn, sr = P3["touch_noxious"], P3["stimulus_reflex"]
    def mice_norm(r):
        out = {}
        for m in MO:
            out[m] = dict(veh=r["vi"]["mice"][m]["both"] - r["vb"]["mice"][m]["both"],
                          sbi=r["si"]["mice"][m]["both"] - r["sb"]["mice"][m]["both"])
        return out
    mn_tn, mn_sr = mice_norm(tn), mice_norm(sr)
    k_tn = int(sum(mn_tn[m]["sbi"] > mn_tn[m]["veh"] for m in MO)); k_sr = int(sum(mn_sr[m]["sbi"] > mn_sr[m]["veh"] for m in MO))
    J["NS3_overlap.png"] = dict(
        section="Q4 stimulus or movement", priority="main",
        mirrors="Pennington et al. 2026 Fig 2c, 2g, 2m, 4e (% of neurons responsive to sound, shock or both)",
        numbers=dict(touch_noxious={c: {k: tn[c][k] for k in ("n", "x_only", "both", "y_only", "any", "expected_both", "x", "y",
                                                               "p_assoc", "mice")} for c in SESS},
                     stimulus_reflex={c: {k: sr[c][k] for k in ("n", "x_only", "both", "y_only", "any", "expected_both", "x", "y",
                                                                 "p_assoc", "mice")} for c in SESS},
                     tests=dict(touch_noxious={k: tn[k] for k in ("p_vb_vi", "p_sb_si", "p_vi_si")},
                                stimulus_reflex={k: sr[k] for k in ("p_vb_vi", "p_sb_si", "p_vi_si")}),
                     overlap_change_per_mouse=dict(touch_noxious=mn_tn, stimulus_reflex=mn_sr),
                     mice_larger_overlap_rise_with_sbi=dict(touch_noxious=k_tn, stimulus_reflex=k_sr)),
        title_en="Touch / noxious and stimulus / reflex neurons overlap more after SBI-553 (same day)",
        title_kr="SBI-553 후 촉각 / 통증, 자극 / 반사 뉴런의 겹침 증가 (같은 날)",
        key_en=(f"Touch+noxious overlap: vehicle day {tn['vb']['both']:.0f}->{tn['vi']['both']:.0f}%, SBI-553 day "
                f"{tn['sb']['both']:.0f}->{tn['si']['both']:.0f}% (McNemar {ptxt(tn['p_sb_si'])}); after veh vs after SBI "
                f"{ptxt(tn['p_vi_si'])}; bigger rise with SBI in {k_tn}/4 mice"),
        key_kr=(f"촉각+통증 겹침: vehicle 날 {tn['vb']['both']:.0f}->{tn['vi']['both']:.0f}%, SBI-553 날 {tn['sb']['both']:.0f}->"
                f"{tn['si']['both']:.0f}% (McNemar {ptxt(tn['p_sb_si'])}); veh 후 vs SBI 후 {ptxt(tn['p_vi_si'])}; SBI 날 상승이 더 큰 마우스 "
                f"{k_tn}/4"),
        bottom_en=(f"Activated neurons (time-shift test, q <= 0.20), all neurons (veh day {tn['vb']['n']}, SBI day {tn['sb']['n']}). "
                   f"Overlap: McNemar within day, Fisher after veh vs after SBI; 4 mice."),
        bottom_kr=(f"활성화 뉴런 (시간 이동 검정, q <= 0.20), 모든 뉴런 (veh 날 {tn['vb']['n']}, SBI 날 {tn['sb']['n']}). 겹침: 날 "
                   f"안 McNemar, veh 후 vs SBI 후 Fisher; 마우스 4."),
        notes_en=(f"Paper: ~70% of sound-responsive AHN neurons also responded to shock. Here, overlap is above independence in "
                  f"every session (expected touch+noxious {tn['vb']['expected_both']:.1f} / {tn['vi']['expected_both']:.1f} / "
                  f"{tn['sb']['expected_both']:.1f} / {tn['si']['expected_both']:.1f}%). Stimulus+reflex overlap: vehicle day "
                  f"{sr['vb']['both']:.0f}->{sr['vi']['both']:.0f}% (McNemar {ptxt(sr['p_vb_vi'])}), SBI-553 day "
                  f"{sr['sb']['both']:.0f}->{sr['si']['both']:.0f}% ({ptxt(sr['p_sb_si'])}); after vehicle vs after SBI-553 Fisher "
                  f"{ptxt(sr['p_vi_si'])}. Caveat: the SBI-553-day baseline had fewer responsive neurons ({tn['sb']['any']:.0f}% "
                  f"vs {tn['vb']['any']:.0f}% for touch / noxious), so part of the within-day rise is a low starting point; the "
                  f"after-injection comparison is not significant. Per mouse, the overlap rose more on the SBI-553 day than on the "
                  f"vehicle day in {k_tn}/4 mice (touch / noxious) and {k_sr}/4 (stimulus / reflex). Vehicle day: touch-responsive "
                  f"fell {tn['vb']['x']:.0f}->{tn['vi']['x']:.0f}% and noxious-responsive rose {tn['vb']['y']:.0f}->"
                  f"{tn['vi']['y']:.0f}%. Drug order confounded with day."),
        notes_kr=(f"논문: 소리 반응 AHN 뉴런의 ~70%가 전기충격에도 반응. 여기서는 모든 세션에서 겹침이 독립 예상보다 큼 (예상 촉각+통증 "
                  f"{tn['vb']['expected_both']:.1f} / {tn['vi']['expected_both']:.1f} / {tn['sb']['expected_both']:.1f} / "
                  f"{tn['si']['expected_both']:.1f}%). 자극+반사 겹침: vehicle 날 {sr['vb']['both']:.0f}->{sr['vi']['both']:.0f}% "
                  f"(McNemar {ptxt(sr['p_vb_vi'])}), SBI-553 날 {sr['sb']['both']:.0f}->{sr['si']['both']:.0f}% "
                  f"({ptxt(sr['p_sb_si'])}); vehicle 후 vs SBI-553 후 Fisher {ptxt(sr['p_vi_si'])}. 주의: SBI-553 날 기준의 반응 "
                  f"뉴런이 더 적었음 (촉각 / 통증 {tn['sb']['any']:.0f}% vs {tn['vb']['any']:.0f}%) - 날 안 상승의 일부는 낮은 출발점; "
                  f"주사 후끼리 비교는 유의하지 않음. 마우스별로 겹침 상승이 SBI-553 날 > vehicle 날인 마우스 {k_tn}/4 (촉각 / 통증), "
                  f"{k_sr}/4 (자극 / 반사). vehicle 날: 촉각 반응 {tn['vb']['x']:.0f}->{tn['vi']['x']:.0f}% 감소, 통증 반응 "
                  f"{tn['vb']['y']:.0f}->{tn['vi']['y']:.0f}% 증가. 약물 순서와 날이 겹침."))
    # ---- NS4
    J["NS4_tuning_crossover.png"] = dict(
        section="Q3 neurons", priority="backup",
        mirrors="Pennington et al. 2026 Fig 2h-i, 2n (tuning of single neurons across stimuli), in the same tracked neurons "
                "through the crossover",
        numbers=dict(n=P4["n"], per_mouse=P4["per_mouse"], median_npi=P4["med"],
                     tests={k: P4[k] for k in ("p_vb_vi", "p_sb_si", "p_vi_si", "p_vb_sb", "p_vb_0", "p_sb_0")},
                     med_chg_veh=P4["med_chg_veh"], med_chg_sbi=P4["med_chg_sbi"], med_norm=P4["med_norm"],
                     ci_norm=list(P4["ci_norm"]), p_norm=P4["p_norm"], mice_chg=P4["mice_chg"],
                     mice_sbi_lower=P4["mice_sbi_lower"], track_class_A_only=P4["A_only"],
                     tuning_stability_spearman=P4["stability"]),
        title_en="Stimulus tuning of the same neurons does not shift toward noxious with either injection",
        title_kr="같은 뉴런의 자극 튜닝은 어느 주사 후에도 통증 쪽으로 이동하지 않음",
        key_en=(f"Noxious preference (Pin/Heat - Light/Mild) of {P4['n']} tracked neurons: medians {f2(min(P4['med'].values()))} to "
                f"{f2(max(P4['med'].values()))} z; normalized SBI vs veh d = {f3(P4['med_norm'])} z, {ptxt(P4['p_norm'])}"),
        key_kr=(f"추적 뉴런 {P4['n']}개의 통증 선호 (핀/열 - 약한/중간): 중앙값 {f2(min(P4['med'].values()))} ~ "
                f"{f2(max(P4['med'].values()))} z; 기준 정규화 SBI vs veh d = {f3(P4['med_norm'])} z, {ptxt(P4['p_norm'])}"),
        bottom_en=(f"Same {P4['n']} tracked neurons in 4 sessions; response = each stimulus's window ({WW._family(WW.STIM)}); Wilcoxon signed-rank paired "
                   f"by neuron; per-mouse medians shown (4 mice)."),
        bottom_kr=(f"같은 추적 뉴런 {P4['n']}개, 4 세션; 반응 = 자극별 창 ({WW._family(WW.STIM, True)}); 뉴런별 짝지은 Wilcoxon signed-rank; 마우스별 "
                   f"중앙값 표시 (마우스 4)."),
        notes_en=(f"Median preference: vehicle-day baseline {f3(P4['med']['vb'])}, after vehicle {f3(P4['med']['vi'])}, SBI-day "
                  f"baseline {f3(P4['med']['sb'])}, after SBI-553 {f3(P4['med']['si'])} z. Within day: vehicle {ptxt(P4['p_vb_vi'])}, "
                  f"SBI-553 {ptxt(P4['p_sb_si'])}; after vehicle vs after SBI-553 {ptxt(P4['p_vi_si'])}; the two baselines "
                  f"{ptxt(P4['p_vb_sb'])} (a day difference before any drug). Baseline-normalized d median {f3(P4['med_norm'])} z "
                  f"(95% CI {P4['ci_norm'][0]:+.3f} to {P4['ci_norm'][1]:+.3f}), class A only (n = {P4['A_only']['n']}) "
                  f"{f3(P4['A_only']['med_norm'])}, {ptxt(P4['A_only']['p_norm'])}. Per mouse the change was smaller on the "
                  f"SBI-553 day in {P4['mice_sbi_lower']}/4 mice. Response rank stability across neurons (Spearman, baseline vs "
                  f"after): " + "; ".join(f"{shortn[e]} veh {P4['stability'][e]['veh']:.2f} / SBI {P4['stability'][e]['sbi']:.2f}"
                                          for e in STIM) + ". Drug order confounded with day."),
        notes_kr=(f"선호 중앙값: vehicle 날 기준 {f3(P4['med']['vb'])}, vehicle 후 {f3(P4['med']['vi'])}, SBI 날 기준 "
                  f"{f3(P4['med']['sb'])}, SBI-553 후 {f3(P4['med']['si'])} z. 날 안: vehicle {ptxt(P4['p_vb_vi'])}, SBI-553 "
                  f"{ptxt(P4['p_sb_si'])}; vehicle 후 vs SBI-553 후 {ptxt(P4['p_vi_si'])}; 두 기준 {ptxt(P4['p_vb_sb'])} (약물 전 날 "
                  f"차이). 기준 정규화 d 중앙값 {f3(P4['med_norm'])} z (95% CI {P4['ci_norm'][0]:+.3f} ~ {P4['ci_norm'][1]:+.3f}), "
                  f"A 등급만 (n = {P4['A_only']['n']}) {f3(P4['A_only']['med_norm'])}, {ptxt(P4['A_only']['p_norm'])}. 마우스별로 "
                  f"SBI-553 날 변화가 더 작은 마우스 {P4['mice_sbi_lower']}/4. 뉴런 간 반응 순위 안정성 (Spearman, 기준 vs 후): "
                  + "; ".join(f"{shortk[e]} veh {P4['stability'][e]['veh']:.2f} / SBI {P4['stability'][e]['sbi']:.2f}" for e in STIM)
                  + ". 약물 순서와 날이 겹침."))
    for fn, d_ in J.items():
        for k in ("key_en", "key_kr"):
            if len(d_[k]) > 150:
                print(f"  WARNING {fn} {k} {len(d_[k])} chars")
        for k in ("bottom_en", "bottom_kr"):
            if len(d_[k]) > 230:
                print(f"  WARNING {fn} {k} {len(d_[k])} chars")
    return J


# ------------------------------------------------------------------------------------------------ main
def main():
    os.makedirs(OUT, exist_ok=True)
    print("loading the 16 sessions ...", flush=True)
    S = E.load_all()
    D = build_trials(S)
    P1 = ns1_pca(S, D)
    P2 = ns2_noxious(S, D)
    up, P3 = ns3_overlap()
    X4, P4 = ns4_npi()
    print("decoding (numbers only) ...", flush=True)
    DEC = decoding(S, D)
    P1["table"].to_csv(os.path.join(OUT, "ns1_pca.csv"), index=False)
    P2["N"].to_csv(os.path.join(OUT, "ns2_neurons.csv"), index=False)
    up.to_csv(os.path.join(OUT, "ns3_overlap.csv"), index=False)
    X4.to_csv(os.path.join(OUT, "ns4_npi.csv"), index=False)
    DEC.to_csv(os.path.join(OUT, "ns_decoding.csv"), index=False)
    for lg in ("en", "kr"):
        sfx = "_kr" if lg == "kr" else ""
        fig_ns1(P1, lg, os.path.join(OUT, f"NS1_pca_trajectories{sfx}.png"))
        fig_ns2(P2, lg, os.path.join(OUT, f"NS2_noxious_neurons{sfx}.png"))
        fig_ns3(P3, lg, os.path.join(OUT, f"NS3_overlap{sfx}.png"))
        fig_ns4(X4, P4, lg, os.path.join(OUT, f"NS4_tuning_crossover{sfx}.png"))
    J = summary(P1, P2, P3, P4, DEC)
    json.dump(J, open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=jconv)
    for fn, d_ in J.items():
        print(f"{fn} [{d_['priority']}]\n  {d_['key_en']}\n  {d_['key_kr']}")
    print("->", OUT)


if __name__ == "__main__":
    main()
