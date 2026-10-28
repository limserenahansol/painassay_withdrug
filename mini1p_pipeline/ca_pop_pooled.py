"""ca_pop_pooled.py - population response of ALL FOUR MICE TOGETHER, per event family (2026-10-23, PI request).

What
  The per-mouse slide '<m> Day d: population response, baseline vs after SBI-553' (make_per_mouse_v2.fig_population: top =
  every neuron around each event, grey = baseline, colour = after the injection, mean +- SEM, windows shaded; bottom = bars of
  the best-window response, base vs inj, Wilcoxon) redrawn with the neurons of all mice pooled, plus baseline-normalized and
  responsive-only versions and the pooled UP / DOWN counts, for every event family:
    stim = the 4 stimuli; behA = withdrawal, flinch, movement onset; behB = escape / rearing, grooming, attending onsets
  (behaviour B events exist only in some sessions: SBI-553 sessions have no escape / grooming; drawn where the data exist).

  1  POP_<fam>_vehday.png / POP_<fam>_sbiday.png  every neuron of the vehicle day (216) / SBI-553 day (246), 4 mice pooled:
     top = traces baseline vs after the injection (trace cache), bottom = bars of the response (cells_classes.csv resp), mean
     +- SEM, mouse means as markers, Wilcoxon signed-rank vs 0 per bar and paired (same neurons) baseline vs injection.
  2  POP_<fam>_norm.png  baseline-normalized: per neuron (injection - same-day baseline): trace (top) and best-window change
     (bars), vehicle-day neurons (blue) vs SBI-553-day neurons (red): Mann-Whitney U (every neuron, independent groups),
     Wilcoxon vs 0 per bar; under each panel the paired Wilcoxon of the tracked neurons (crossday pairs, 185): change on the
     SBI-553 day vs change on the vehicle day, each neuron with itself.
  3  POP_<fam>_respOnly.png  only the responsive neurons: per event the neurons UP (row 1) / DOWN (row 2) in the baseline OR
     the injection session of that day; traces + bars for base (veh day), veh, base (SBI day), SBI. The selection uses both
     sessions, so base vs inj is biased toward the session that selected the neuron (said in the figure); the unbiased check
     (P5_fate.csv of ca_event_analysis.plot5_fate: selected on the ODD baseline trials, measured on the EVEN baseline trials
     and after the injection) is in the notes (summary.json).
  4  UPDOWN_pooled.png  per event horizontal stacked bars DOWN / no response / UP (% of neurons, counts written), four rows:
     vehicle-day baseline, after vehicle, SBI-553-day baseline, after SBI-553.
     UPDOWN_pooled_norm.png  change from the same-day baseline in % UP and % DOWN, vehicle day vs SBI-553 day, mouse markers;
     McNemar (same neurons, baseline vs injection, 'vs 0' over each bar) and Fisher exact (black bracket: neurons that gained
     vs lost the class, vehicle day vs SBI-553 day).

Why
  The per-mouse slides cannot be compared across the crossover; the PI asked for the same format with every mouse together and
  for the baseline-normalized comparison (user rule 10-23: % UP / DOWN may use every neuron of a day; cross-day comparisons of
  neuron z use the tracked neurons, each with itself; show (injection - its own same-day baseline) vehicle vs SBI-553).
  Drug order is confounded with day (5941, 5939 got SBI-553 on Day 1; 5937, 1191 on Day 2): per-mouse markers are shown.

Inputs (all timing-corrected, read only)
  analysis\\event_analysis\\cells_classes.csv     resp (best-window response z) and cls (time-shift test, BH q <= 0.20)
  analysis\\new_decks\\psth_cache_win3_tf.npz     trial-mean traces per neuron (z minus its own -3..0 s mean)
  analysis\\crossday\\<m>_pairs.csv               tracked neurons (day1_id, day2_id, track_class A / B)
  analysis\\event_analysis\\P5_fate.csv           odd / even split of the baseline trials (the unbiased check)
  analysis\\event_analysis\\summary.json          best windows (winwords.of)

Outputs  analysis\\pi_extra\\population\\  <figure>.png (EN) and <figure>_kr.png (Korean), summary.json (numbers + slide texts per figure)

Run
  python -I -X utf8 C:\\Users\\hsollim\\Documents\\HEAL_mini1p_SBI553\\mini1p_pipeline\\ca_pop_pooled.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest, fisher_exact, mannwhitneyu, wilcoxon

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import safe_io  # noqa: E402,F401  (savefig retries a transient Windows write error)
import pairstars as PS  # noqa: E402  (every star shows the pair it compares)
import winwords as WW  # noqa: E402  (the best windows of the event analysis)
import figfoot as FF  # noqa: E402  (10-24: the methods text under a figure goes to the slide notes)

B = r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553"
EA = os.path.join(B, "analysis", "event_analysis")
CACHE = os.environ.get("PSTH_CACHE", os.path.join(B, "analysis", "new_decks", "psth_cache_win3_tf.npz"))   # 10-24: variants
CD = os.path.join(B, "analysis", "crossday")
OUT = os.path.join(B, "analysis", "pi_extra", "population")
MO = ["5941", "5939", "5937", "1191"]
SEX = {"5941": "F", "1191": "F", "5939": "M", "5937": "M"}
SBI_DAY = {"5941": 1, "5939": 1, "5937": 2, "1191": 2}
CC = {"baseline": "#7f8c8d", "vehicle": "#2471a3", "SBI-553": "#c0392b"}
MC = {"5941": "#e67e22", "5939": "#8e44ad", "5937": "#16a085", "1191": "#34495e"}
MK = {"5941": "o", "5939": "s", "5937": "^", "1191": "D"}
UPC, DNC, NOC = "#e74c3c", "#2e86c1", "#c8cdd2"
DRUG = {"veh": "vehicle", "sbi": "SBI-553"}
STIM = ["Light touch", "Mild touch", "Pin prick", "Heat"]
FAMS = [("stim", "stimuli", "자극", STIM),
        ("behA", "behaviours A (withdrawal, flinch, movement)", "행동 A (withdrawal, flinch, movement)",
         ["withdrawal", "flinch", "movement onset"]),
        ("behB", "behaviours B (escape / rearing, grooming, attending)", "행동 B (escape / rearing, grooming, attending)",
         ["escape/rearing onset", "grooming onset", "attending onset"])]
ALL = [e for f in FAMS for e in f[3]]
SH = {"Light touch": "light touch", "Mild touch": "mild touch", "Pin prick": "pin prick", "Heat": "heat", "withdrawal": "withdrawal",
      "flinch": "flinch", "movement onset": "movement onset", "escape/rearing onset": "escape / rearing onset",
      "grooming onset": "grooming onset", "attending onset": "attending onset"}
SH2 = {"Light touch": "light\ntouch", "Mild touch": "mild\ntouch", "Pin prick": "pin\nprick", "Heat": "heat", "withdrawal": "withdrawal",
       "flinch": "flinch", "movement onset": "movement\nonset", "escape/rearing onset": "escape /\nrearing", "grooming onset": "grooming\nonset",
       "attending onset": "attending\nonset"}
FS = (15, 6.3)       # shown 12.4 in wide on the slide: effective pt = pt x 0.825, so >= 11 pt here is >= 9 pt there
FONT = 11.5
KEYFS = 11.5         # the one key line under a figure
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "xtick.labelsize": 11.5, "ytick.labelsize": 11.5,
                     "legend.fontsize": 11.5, "axes.spines.top": False, "axes.spines.right": False})


def font(lg):
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"] if lg == "kr" else ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def vday(dk, m):
    """the day mouse m got the drug of dk ('veh' / 'sbi')"""
    return SBI_DAY[m] if dk == "sbi" else 3 - SBI_DAY[m]


# ------------------------------------------------------------------------------------------------ stats
def wil(d):
    d = np.asarray(d, float); d = d[np.isfinite(d)]
    if len(d) < 6 or np.all(d == 0):
        return np.nan
    return float(wilcoxon(d).pvalue)


def mwu(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if min(len(a), len(b)) < 6:
        return np.nan
    return float(mannwhitneyu(a, b).pvalue)


def mcnemar(b_, c_):
    n = int(b_ + c_)
    return float(binomtest(int(b_), n, 0.5).pvalue) if n else 1.0       # no neuron changed: no evidence of a change


def fisher(t):
    t = np.asarray(t, int)
    if t.sum(1).min() == 0 or t.sum(0).min() == 0:
        return np.nan
    return float(fisher_exact(t)[1])


def sem(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan


def ptxt(p, lg="en"):
    if p is None or not np.isfinite(p):
        return "n.a." if lg != "kr" else "검정 안 됨"
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}" if p < 0.1 else f"p = {p:.2f}"


def sm(A, k=6):
    """0.2-s running mean along time (axis 0), display only (as CA.sm)"""
    return pd.DataFrame(A).rolling(k, center=True, min_periods=1).mean().values


def fnum(x, nd=2):
    return "n.a." if x is None or not np.isfinite(x) else f"{x:+.{nd}f}"


# ------------------------------------------------------------------------------------------------ text
def foot(fig, txt, lg=None, key=None, key_color="0.2", star=True):
    """10-24 (user: "all details put in note not main slides"): the methods text goes to the slide notes (figfoot writes it
    next to the PNG as <name>.png.notes.txt); the figure keeps ONE readable line at the bottom: `key` (a short legend line,
    left, optional) and the star key (right; left when there is no `key`) when the text has one. Returns the number of lines
    drawn (0, 1, or 2 when the two did not fit side by side)."""
    FF.note(fig, txt, lg, star=False)
    has_star = star and bool(FF.STAR.search(" ".join(str(txt).split())))
    arts = []
    if key:
        arts.append(fig.text(0.008, 0.008, key, fontsize=KEYFS, color=key_color, ha="left", va="bottom"))
    if has_star:
        arts.append(fig.text(0.992 if key else 0.008, 0.008, FF.KEY_KR if lg == "kr" else FF.KEY_EN, fontsize=KEYFS, color="0.25",
                             ha="right" if key else "left", va="bottom"))
    if len(arts) == 2:
        r = fig.canvas.get_renderer()
        if arts[0].get_window_extent(r).x1 + 15 > arts[1].get_window_extent(r).x0:
            arts[0].set_y(0.008 + 1.35 * KEYFS / 72 / fig.get_figheight())
            return 2
    return len(arts)


def wtxt(ev, lg):
    pre, gap, a, b = WW.of(ev)
    a0 = min(a, 0.0) - gap
    return (f"{a:g}-{b:g} s vs {a0 - pre:g}..{a0:g} s" if lg != "kr" else f"반응 {a:g}-{b:g} s vs 기준 {a0 - pre:g}..{a0:g} s")


def yticks(ax, n=4):
    """at most n + 1 y ticks, written without trailing zeros (0.05, not 0.050)"""
    ax.yaxis.set_major_locator(MaxNLocator(n))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}" if abs(v) > 1e-12 else "0"))


def shade(ax, ev):
    pre, gap, a, b = WW.of(ev)
    a0 = min(a, 0.0) - gap
    ax.axvspan(a, b, color="0.86", zorder=0, lw=0)
    ax.axvspan(a0 - pre, a0, color="#dbe9f6", zorder=0, lw=0)
    ax.axvline(0, color="k", lw=0.7)
    ax.axhline(0, color="0.6", lw=0.5)


K = {   # EN / KR
    "base": ("baseline", "기준"), "after_vehicle": ("after vehicle", "vehicle 후"), "after_SBI-553": ("after SBI-553", "SBI-553 후"),
    "vday": ("vehicle day", "vehicle 날"), "sday": ("SBI-553 day", "SBI-553 날"),
    "time": ("time from event (s)", "사건부터 시간 (s)"),
    "zall": ("z, minus 3 s before\nmean ± SEM", "z, 직전 3 s 평균 뺌\n평균 ± SEM"),
    "resp": ("response (z)\nmean ± SEM", "반응 (z)\n평균 ± SEM"),
    "dz": ("z, injection − base\nmean ± SEM", "z, 주사 − 기준\n평균 ± SEM"),
    "dresp": ("change in response (z)\nmean ± SEM", "반응 변화 (z)\n평균 ± SEM"),
    "rwin": ("response window", "반응 창"), "bwin": ("baseline window", "기준 창"),
    "mmean": ("marker = mean of one mouse", "표식 = 마우스 하나의 평균"),
    "nosess": ("no session\nwith >= 5 events", "사건 5회 이상\n세션 없음"),
}


def T_(k, lg):
    return K[k][1 if lg == "kr" else 0]


def cond_lab(dk, ph, lg):
    return T_("base", lg) if ph == "pre" else T_("after_" + DRUG[dk], lg)


# ------------------------------------------------------------------------------------------------ data
class Data:
    def __init__(self):
        T = pd.read_csv(os.path.join(EA, "cells_classes.csv"))
        T["mouse"] = T.mouse.astype(str)
        T["dk"] = ["sbi" if SBI_DAY[m] == d else "veh" for m, d in zip(T.mouse, T.day)]
        T["k"] = T.cls.map({"activated": "UP", "inhibited": "DOWN"}).fillna("no response")
        self.T = T
        self.D = {k: v for k, v in np.load(CACHE, allow_pickle=True)["D"].item().items() if v is not None}
        P = []
        for m in MO:
            p = pd.read_csv(os.path.join(CD, f"{m}_pairs.csv")); p["mouse"] = m
            P.append(p)
        self.P = pd.concat(P, ignore_index=True)
        self.F = pd.read_csv(os.path.join(EA, "P5_fate.csv")); self.F["mouse"] = self.F.mouse.astype(str)
        self.n_day = {dk: int(T[(T.dk == dk) & (T.phase == "pre")].groupby(["mouse"]).cell.nunique().sum()) for dk in ("veh", "sbi")}

    def rows(self, dk, ph, ev):
        T = self.T
        return T[(T.dk == dk) & (T.phase == ph) & (T.event == ev)]

    def traces(self, dk, ph, ev, sel=None):
        """pooled trial-mean traces (time x neurons) of the dk day of every mouse; sel = {mouse: set of ids} or None"""
        Y, mice, tt = [], [], None
        for m in MO:
            v = self.D.get((m, vday(dk, m), ph, ev))
            if v is None:
                continue
            ids = np.asarray(v["ids"]).astype(int)
            k = np.ones(len(ids), bool) if sel is None else np.isin(ids, list(sel.get(m, ())))
            if not k.any():
                continue
            Y.append(np.asarray(v["mean"], float)[:, k]); mice += [m] * int(k.sum()); tt = v["tt"]
        if not Y:
            return None
        return tt, np.concatenate(Y, 1), np.array(mice)

    def diff_traces(self, dk, ev):
        """per neuron (injection - same-day baseline) trace, pooled over the mice"""
        Y, mice, tt = [], [], None
        for m in MO:
            d = vday(dk, m)
            a, b = self.D.get((m, d, "pre", ev)), self.D.get((m, d, "post", ev))
            if a is None or b is None:
                continue
            ia, ib = np.asarray(a["ids"]).astype(int), np.asarray(b["ids"]).astype(int)
            c = np.intersect1d(ia, ib)
            ja = np.searchsorted(ia, c) if np.all(np.diff(ia) > 0) else np.array([np.flatnonzero(ia == x)[0] for x in c])
            jb = np.searchsorted(ib, c) if np.all(np.diff(ib) > 0) else np.array([np.flatnonzero(ib == x)[0] for x in c])
            Y.append(np.asarray(b["mean"], float)[:, jb] - np.asarray(a["mean"], float)[:, ja]); mice += [m] * len(c); tt = a["tt"]
        if not Y:
            return None
        return tt, np.concatenate(Y, 1), np.array(mice)

    def paired(self, dk, ev):
        """same neurons, baseline and injection session of the dk day: mouse, day, cell, resp_b, resp_i, k_b, k_i"""
        a, b = self.rows(dk, "pre", ev), self.rows(dk, "post", ev)
        return a.merge(b, on=["mouse", "day", "cell"], suffixes=("_b", "_i"))

    def tracked_change(self, ev):
        """tracked neurons: change (injection - baseline) on the SBI-553 day and on the vehicle day, each neuron with itself"""
        R = self.T[self.T.event == ev].set_index(["mouse", "day", "phase", "cell"]).resp
        out = []
        for r in self.P.itertuples():
            ids = {1: int(r.day1_id), 2: int(r.day2_id)}
            sd, vd = SBI_DAY[r.mouse], 3 - SBI_DAY[r.mouse]
            g = lambda d, ph: R.get((r.mouse, d, ph, ids[d]), np.nan)
            out.append(dict(mouse=r.mouse, track_class=r.track_class, chg_sbi=g(sd, "post") - g(sd, "pre"),
                            chg_veh=g(vd, "post") - g(vd, "pre")))
        X = pd.DataFrame(out)
        return X[np.isfinite(X.chg_sbi) & np.isfinite(X.chg_veh)]


def mean_trace(Y):
    Ys = sm(Y)
    n = np.sum(np.isfinite(Ys), 1)
    return np.nanmean(Ys, 1), np.nanstd(Ys, 1, ddof=1) / np.sqrt(np.maximum(n, 1))


def mice_txt(mice, lg):
    ms = [m for m in MO if m in set(mice)]
    return ", ".join(ms)


# ------------------------------------------------------------------------------------------------ bar panel
def bar_panel(b, groups, pairs, vs0=True, f_hi=None, width=0.6, mk_dx=0.11, fs=FONT):
    """groups: dict(x, v, col, fill, mice={m: mean}); pairs: (x1, x2, p, colour, level 1 / 2).
    The data take the lower part of the axes; the 'vs 0' marks sit over each bar, the brackets above them (axes fraction)."""
    tops, bots = [0.0], [0.0]
    for g in groups:
        v = np.asarray(g["v"], float); v = v[np.isfinite(v)]
        g["n"] = len(v)
        if not len(v):
            continue
        mu, se = float(v.mean()), sem(v)
        se0 = se if np.isfinite(se) else 0.0
        b.bar(g["x"], mu, yerr=se0, width=width, color=g["col"] if g.get("fill", True) else "white", edgecolor=g["col"], lw=1.6,
              capsize=3, error_kw=dict(lw=1.1, ecolor="k"), zorder=2)
        mv = []
        for j, m in enumerate(MO):
            if m in g.get("mice", {}) and np.isfinite(g["mice"][m]):
                b.scatter(g["x"] + (j - 1.5) * mk_dx, g["mice"][m], marker=MK[m], color=MC[m], s=34, edgecolor="white", lw=0.6,
                          zorder=5, clip_on=False)            # a mouse mean of exactly 0 stays visible over the 0 line
                mv.append(g["mice"][m])
        g["top"] = max([mu + se0, 0.0] + mv); g["bot"] = min([mu - se0, 0.0] + mv)
        tops.append(g["top"]); bots.append(g["bot"])
    lo, hi = min(bots), max(tops)
    span = (hi - lo) or 1.0
    if f_hi is None:
        f_hi = 0.5 if vs0 else (0.62 if any(p[4] == 2 for p in pairs) else 0.72)
    R = span / (f_hi - 0.04)
    y0 = lo - 0.04 * R
    b.set_ylim(y0, y0 + R)
    b.axhline(0, color="0.4", lw=0.6)
    if vs0:
        for g in groups:
            if g.get("n", 0):
                PS.vs_ref(b, g["x"], g["top"] + 0.025 * R, wil(g["v"]), "vs 0", g.get("vcol", "k"), fs=fs)
    for x1, x2, p, c, lev in pairs:
        if vs0:
            y = f_hi + 0.31 if lev == 1 else f_hi + 0.42
        else:
            y = f_hi + 0.06 if lev == 1 else f_hi + 0.2
        PS.bracket(b, x1, x2, y, p, c, fs=fs + 1, axes_y=True, inset=0.04)


# ------------------------------------------------------------------------------------------------ 1  vehicle day / SBI-553 day
def fig_day(C, fam, dk, lg, fn):
    font(lg)
    key, fen, fkr, evs = fam
    drug = DRUG[dk]
    col = CC[drug]
    n = len(evs)
    fig = plt.figure(figsize=FS)
    left, right, gap = 0.075, 0.992, 0.055 if n == 4 else 0.07
    w = (right - left - (n - 1) * gap) / n
    res = {}
    for k, ev in enumerate(evs):
        x0 = left + k * (w + gap)
        a = fig.add_axes([x0, 0.585, w, 0.28])               # 10-24: footer moved to the notes, its room given to the plots
        b = fig.add_axes([x0, 0.15, w, 0.33])
        r = dict(event=ev)
        for ph, cc in (("pre", CC["baseline"]), ("post", col)):
            tr = C.traces(dk, ph, ev)
            if tr is None:
                continue
            tt, Y, mice = tr
            mu, se = mean_trace(Y)
            s = (tt >= -3) & (tt <= 6)
            a.fill_between(tt[s], (mu - se)[s], (mu + se)[s], color=cc, alpha=0.25, lw=0)
            a.plot(tt[s], mu[s], color=cc, lw=1.7)
            r[f"n_trace_{ph}"] = int(Y.shape[1])
        shade(a, ev)
        a.set_xlim(-3, 6)
        lo_, hi_ = a.get_ylim(); a.set_ylim(lo_, hi_ + 0.08 * (hi_ - lo_))
        yticks(a)
        a.set_title(f"{SH[ev]}\n{wtxt(ev, lg)}", fontsize=13)          # black: red / blue mean the drug
        a.set_xlabel(T_("time", lg), labelpad=1)
        groups, labels = [], []
        for i, ph in enumerate(("pre", "post")):
            X = C.rows(dk, ph, ev)
            cc = CC["baseline"] if ph == "pre" else col
            mm = X.groupby("mouse").resp.mean().to_dict()
            groups.append(dict(x=i, v=X.resp.values, col=cc, mice=mm))
            labels.append(f"{cond_lab(dk, ph, lg)}\n(n = {len(X)})")
            r[f"n_{ph}"] = int(len(X)); r[f"mean_{ph}"] = float(X.resp.mean()) if len(X) else np.nan
            r[f"sem_{ph}"] = sem(X.resp); r[f"p_vs0_{ph}"] = wil(X.resp); r[f"mice_{ph}"] = {m: float(v) for m, v in mm.items()}
            r[f"mice_list_{ph}"] = [m for m in MO if m in mm]
            if not len(X):
                b.text(i, 0.5, T_("nosess", lg), transform=b.get_xaxis_transform(), ha="center", va="center", fontsize=FONT,
                       color="0.35", bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none"), zorder=5)
        M = C.paired(dk, ev)
        pp = wil(M.resp_i - M.resp_b) if len(M) else np.nan
        r["n_paired"], r["p_paired"] = int(len(M)), pp
        r["mean_change"] = float((M.resp_i - M.resp_b).mean()) if len(M) else np.nan
        bar_panel(b, groups, [(0, 1, pp, "k", 1)] if len(M) else [], vs0=True)
        yticks(b)
        b.set_xticks([0, 1], labels)
        b.set_xlim(-0.6, 1.6)
        if k == 0:
            a.set_ylabel(T_("zall", lg)); b.set_ylabel(T_("resp", lg))
        res[ev] = r
    h = [Line2D([], [], color=CC["baseline"], lw=2.2, label=T_("base", lg)),
         Line2D([], [], color=col, lw=2.2, label=T_("after_" + drug, lg)),
         Patch(color="0.86", label=T_("rwin", lg)), Patch(color="#dbe9f6", label=T_("bwin", lg))]
    h += [Line2D([], [], marker=MK[m], ls="", color=MC[m], ms=7, label=m) for m in MO]
    fig.legend(handles=h, loc="upper center", ncol=8, frameon=False, bbox_to_anchor=(0.5, 0.995), columnspacing=1.3,
               handlelength=1.6)
    nn = C.n_day[dk]
    days = ", ".join(f"{m} D{vday(dk, m)}" for m in MO)
    if lg != "kr":
        t = (f"Every neuron of the {T_('vday' if dk == 'veh' else 'sday', lg)} ({nn} neurons; {days}), 4 mice pooled; n = neurons "
             f"of the sessions with >= 5 events. Top: mean ± SEM of the neurons (z = dF/F / robust SD of the session; trial means "
             f"minus the 3 s before, 0.2-s running mean, display). Bottom: response in each event's window (panel title: response "
             f"window vs baseline window; event analysis), markers = mouse means. Wilcoxon signed-rank over "
             f"neurons: 'vs 0' = that bar vs 0; bracket = baseline vs after {drug}, the same neurons (paired). Neurons pooled over "
             f"mice = exploratory. " + PS.KEY)
    else:
        t = (f"{T_('vday' if dk == 'veh' else 'sday', lg)}의 모든 뉴런 ({nn}개; {days}), 4마리 합산; n = 사건 5회 이상 세션의 뉴런. "
             f"위: 뉴런 평균 ± SEM (z = dF/F / 세션의 robust SD; trial 평균, 직전 3 s 평균 뺌, 0.2 s 이동 평균, 표시용). 아래: 사건별 창의 "
             f"반응 (패널 제목: 반응 창 vs 기준 창; 이벤트 분석), 표식 = 마우스 "
             f"평균. 뉴런 Wilcoxon signed-rank: 'vs 0' = 막대 vs 0; 괄호 = 기준 vs {drug} 후, 같은 뉴런 (짝). 마우스 합산 뉴런 = "
             f"탐색적. * p < 0.05, ** p < 0.01, *** p < 0.001, ns = 유의하지 않음")
    foot(fig, t, lg)
    fig.savefig(fn, dpi=150); plt.close(fig)
    return res


# ------------------------------------------------------------------------------------------------ 2  baseline-normalized
def fig_norm(C, fam, lg, fn):
    font(lg)
    key, fen, fkr, evs = fam
    n = len(evs)
    fig = plt.figure(figsize=FS)
    left, right, gap = 0.075, 0.992, 0.055 if n == 4 else 0.07
    w = (right - left - (n - 1) * gap) / n
    res = {}
    for k, ev in enumerate(evs):
        x0 = left + k * (w + gap)
        a = fig.add_axes([x0, 0.6, w, 0.265])                # 10-24: footer moved to the notes, its room given to the plots
        b = fig.add_axes([x0, 0.19, w, 0.305])
        r = dict(event=ev)
        for dk in ("veh", "sbi"):
            tr = C.diff_traces(dk, ev)
            if tr is None:
                continue
            tt, Y, mice = tr
            mu, se = mean_trace(Y)
            s = (tt >= -3) & (tt <= 6)
            a.fill_between(tt[s], (mu - se)[s], (mu + se)[s], color=CC[DRUG[dk]], alpha=0.22, lw=0)
            a.plot(tt[s], mu[s], color=CC[DRUG[dk]], lw=1.7)
            r[f"n_trace_{dk}"] = int(Y.shape[1])
        shade(a, ev)
        a.set_xlim(-3, 6)
        lo_, hi_ = a.get_ylim(); a.set_ylim(lo_, hi_ + 0.08 * (hi_ - lo_))
        yticks(a)
        a.set_title(f"{SH[ev]}\n{wtxt(ev, lg)}", fontsize=13)          # black: red / blue mean the drug
        a.set_xlabel(T_("time", lg), labelpad=1)
        groups, labels, ch = [], [], {}
        for i, dk in enumerate(("veh", "sbi")):
            M = C.paired(dk, ev)
            d = (M.resp_i - M.resp_b).values
            ch[dk] = d
            mm = (M.assign(d=d).groupby("mouse").d.mean().to_dict()) if len(M) else {}
            groups.append(dict(x=i, v=d, col=CC[DRUG[dk]], mice=mm, vcol=CC[DRUG[dk]]))
            labels.append(f"{T_('vday' if dk == 'veh' else 'sday', lg)}\n(n = {len(M)})")
            r[f"n_{dk}"] = int(len(M)); r[f"mean_{dk}"] = float(np.mean(d)) if len(d) else np.nan; r[f"sem_{dk}"] = sem(d)
            r[f"p_vs0_{dk}"] = wil(d); r[f"mice_{dk}"] = {m: float(v) for m, v in mm.items()}
            if not len(M):
                b.text(i, 0.5, T_("nosess", lg), transform=b.get_xaxis_transform(), ha="center", va="center", fontsize=FONT, color="0.35",
                       bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none"), zorder=5)
        p_mw = mwu(ch["veh"], ch["sbi"])
        r["p_mw"] = p_mw
        X = C.tracked_change(ev)
        r["n_tracked"] = int(len(X)); r["p_tracked"] = wil(X.chg_sbi - X.chg_veh) if len(X) else np.nan
        r["med_tracked"] = float(np.median(X.chg_sbi - X.chg_veh)) if len(X) else np.nan
        r["mean_tracked_sbi"] = float(X.chg_sbi.mean()) if len(X) else np.nan
        r["mean_tracked_veh"] = float(X.chg_veh.mean()) if len(X) else np.nan
        XA = X[X.track_class == "A"]
        r["n_tracked_A"] = int(len(XA)); r["p_tracked_A"] = wil(XA.chg_sbi - XA.chg_veh) if len(XA) else np.nan
        bar_panel(b, groups, [(0, 1, p_mw, "k", 1)] if np.isfinite(p_mw) else [], vs0=True)
        yticks(b)
        b.set_xticks([0, 1], labels)
        b.set_xlim(-0.6, 1.6)
        if len(X):
            tl = (f"tracked n = {len(X)}: {ptxt(r['p_tracked'])}" if lg != "kr" else f"추적 n = {len(X)}: {ptxt(r['p_tracked'], lg)}")
        else:
            tl = "tracked: not testable" if lg != "kr" else "추적: 검정 불가"
        b.set_xlabel(tl, fontsize=FONT, labelpad=3, color="0.15")
        if k == 0:
            a.set_ylabel(T_("dz", lg)); b.set_ylabel(T_("dresp", lg))
        res[ev] = r
    h = [Line2D([], [], color=CC["vehicle"], lw=2.2, label=("vehicle-day neurons" if lg != "kr" else "vehicle 날 뉴런")),
         Line2D([], [], color=CC["SBI-553"], lw=2.2, label=("SBI-553-day neurons" if lg != "kr" else "SBI-553 날 뉴런")),
         Patch(color="0.86", label=T_("rwin", lg)), Patch(color="#dbe9f6", label=T_("bwin", lg))]
    h += [Line2D([], [], marker=MK[m], ls="", color=MC[m], ms=7, label=m) for m in MO]
    fig.legend(handles=h, loc="upper center", ncol=8, frameon=False, bbox_to_anchor=(0.5, 0.995), columnspacing=1.3, handlelength=1.6)
    if lg != "kr":
        t = ("Baseline-normalized: per neuron, after the injection minus the baseline session of the same day (the same neuron). "
             f"Vehicle day {C.n_day['veh']} neurons, SBI-553 day {C.n_day['sbi']}, 4 mice pooled (n = neurons whose day had both "
             "sessions with >= 5 events); markers = mouse means. Black bracket = vehicle-day vs SBI-553-day neurons, Mann-Whitney U "
             "(independent neurons, exploratory); 'vs 0' = Wilcoxon signed-rank (= baseline vs injection). Under each panel: tracked "
             "neurons (followed on both days, 185 in all), change on the SBI-553 day vs on the vehicle day, paired Wilcoxon "
             "signed-rank. Drug order = day for 2 + 2 mice. " + PS.KEY)
    else:
        t = ("기준 정규화: 뉴런마다 주사 후 - 같은 날 기준 세션 (같은 뉴런). "
             f"vehicle 날 뉴런 {C.n_day['veh']}, SBI-553 날 {C.n_day['sbi']}, 4마리 합산 (n = 두 세션 모두 사건 5회 이상인 날의 뉴런); "
             "표식 = 마우스 평균. 검은 괄호 = vehicle 날 vs SBI-553 날 뉴런, Mann-Whitney U (독립 뉴런, 탐색적); 'vs 0' = Wilcoxon "
             "signed-rank (= 기준 vs 주사). 각 패널 아래: 두 날 모두 추적한 뉴런 (전체 185), SBI-553 날 변화 vs vehicle 날 변화, 짝 "
             "Wilcoxon signed-rank. 약물 순서 = 날 (2 + 2마리). * p < 0.05, ** p < 0.01, *** p < 0.001, ns = 유의하지 않음")
    foot(fig, t, lg, key=("black bracket = vehicle vs SBI-553 day (Mann-Whitney U);   'vs 0' = Wilcoxon signed-rank" if lg != "kr" else
                          "검은 괄호 = vehicle 날 vs SBI-553 날 (Mann-Whitney U);   'vs 0' = Wilcoxon signed-rank"))
    fig.savefig(fn, dpi=150); plt.close(fig)
    return res


# ------------------------------------------------------------------------------------------------ 3  responsive neurons only
G4 = [("veh", "pre"), ("veh", "post"), ("sbi", "pre"), ("sbi", "post")]
X4 = {("veh", "pre"): 0.0, ("veh", "post"): 1.0, ("sbi", "pre"): 2.35, ("sbi", "post"): 3.35}


def select(C, dk, ev, cls):
    """neurons of the dk day classified cls (UP / DOWN) in the baseline OR the injection session (sessions with the event)"""
    X = C.T[(C.T.dk == dk) & (C.T.event == ev)]
    S = X[X.k == cls]
    return {m: set(S[S.mouse == m].cell.astype(int)) for m in MO}, {ph: int((X.phase == ph).any()) for ph in ("pre", "post")}


MIN_N = 5      # fewer selected neurons than this: no mean drawn ("n < 5")


def fig_resp(C, fam, lg, fn):
    font(lg)
    key, fen, fkr, evs = fam
    n = len(evs)
    fig = plt.figure(figsize=FS)
    left, right = 0.074, 0.995
    blk = (right - left) / n
    tw, gp, bw = (0.11, 0.05, 0.058) if n == 4 else (0.165, 0.05, 0.08)      # room between blocks: no '4-2' tick clash
    rows_y = {"UP": (0.525, 0.31), "DOWN": (0.16, 0.31)}        # 10-24: footer moved to the notes, its room given to the plots
    res = {}
    TA = {"UP": [], "DOWN": []}
    YR = {"UP": [0.0, 0.0], "DOWN": [0.0, 0.0]}                  # data range of the traces per row (shared y scale)
    for k, ev in enumerate(evs):
        x0 = left + k * blk
        for cls, (y0, hh) in rows_y.items():
            a = fig.add_axes([x0, y0, tw, hh])
            b = fig.add_axes([x0 + tw + gp, y0, bw, hh])
            TA[cls].append(a)
            r, ns, groups, vals = {}, {}, [], {}
            for dk in ("veh", "sbi"):
                sel, have = select(C, dk, ev, cls)
                ns[dk] = sum(len(v) for v in sel.values())
                r[f"n_{dk}"] = ns[dk]
                r[f"sessions_{dk}"] = have
                ok = ns[dk] >= MIN_N
                for ph in ("pre", "post"):
                    col = CC[DRUG[dk]]
                    tr = C.traces(dk, ph, ev, sel) if ok else None
                    if tr is not None:
                        tt, Y, mice = tr
                        mu, se = mean_trace(Y)
                        s = (tt >= -2) & (tt <= 4)
                        a.fill_between(tt[s], (mu - se)[s], (mu + se)[s], color=col, alpha=0.13, lw=0)
                        YR[cls] = [min(YR[cls][0], np.nanmin((mu - se)[s])), max(YR[cls][1], np.nanmax((mu + se)[s]))]
                        a.plot(tt[s], mu[s], color=col, lw=1.3 if ph == "pre" else 1.9, ls="--" if ph == "pre" else "-")
                    X = C.rows(dk, ph, ev)
                    X = X[np.array([c in sel[m] for m, c in zip(X.mouse, X.cell)], bool)] if ok else X.iloc[:0]
                    vals[(dk, ph)] = X.resp.values
                    groups.append(dict(x=X4[(dk, ph)], v=X.resp.values, col=col, fill=ph == "post"))
                    r[f"mean_{dk}_{ph}"] = float(X.resp.mean()) if len(X) else np.nan
                    r[f"sem_{dk}_{ph}"] = sem(X.resp)
                    r[f"nrow_{dk}_{ph}"] = int(len(X))
                    if ok and not len(X):
                        b.text(X4[(dk, ph)], 0.32, ("no session" if lg != "kr" else "세션 없음"), transform=b.get_xaxis_transform(),
                               ha="center", va="center", fontsize=FONT, color="0.4", rotation=90)
                if not ok:
                    b.text((X4[(dk, "pre")] + X4[(dk, "post")]) / 2, 0.3, f"n < {MIN_N}",
                           transform=b.get_xaxis_transform(), ha="center", va="center", fontsize=FONT, color="0.4", rotation=90)
            pr = []
            for dk in ("veh", "sbi"):
                sel, _ = select(C, dk, ev, cls)
                M = C.paired(dk, ev)
                M = M[np.array([c in sel[m] for m, c in zip(M.mouse, M.cell)], bool)] if ns[dk] >= MIN_N else M.iloc[:0]
                p = wil(M.resp_i - M.resp_b) if len(M) else np.nan
                r[f"p_{dk}"] = p; r[f"npair_{dk}"] = int(len(M))
                r[f"mean_change_{dk}"] = float((M.resp_i - M.resp_b).mean()) if len(M) else np.nan
                pr.append((X4[(dk, "pre")], X4[(dk, "post")], p, CC[DRUG[dk]], 1))
            pv = mwu(vals[("veh", "post")], vals[("sbi", "post")])
            r["p_veh_vs_sbi"] = pv
            pr.append((X4[("veh", "post")], X4[("sbi", "post")], pv, "k", 2))
            bar_panel(b, groups, [q for q in pr if np.isfinite(q[2])], vs0=False, width=0.72)
            if max(ns.values()) < MIN_N:                 # nothing to draw for this event and class
                a.text(1.0, 0.45, (f"fewer than {MIN_N} {cls} neurons\non either day" if lg != "kr" else
                                   f"두 날 모두 {cls} 뉴런\n{MIN_N}개 미만"), transform=a.get_xaxis_transform(), ha="center",
                       va="center", fontsize=FONT, color="0.35",
                       bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="0.7"), zorder=8)
                b.set_yticks([])
            shade(a, ev)
            a.set_xlim(-2, 4)
            a.set_xticks([-2, 0, 2, 4])
            up = cls == "UP"
            a.text(0.03 if up else 0.97, 0.97, (f"veh day n = {ns['veh']}\nSBI day n = {ns['sbi']}" if lg != "kr" else
                                                f"veh 날 n = {ns['veh']}\nSBI 날 n = {ns['sbi']}"),
                   transform=a.transAxes, ha="left" if up else "right", va="top", fontsize=FONT, color="0.1", linespacing=1.1,
                   bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85), zorder=7)
            b.set_xlim(-0.65, 4.0)
            if max(ns.values()) >= MIN_N:
                yticks(b, 3)
            if up:     # event titles black: red / blue mean the drug (pin prick used to be red)
                a.set_title(f"{SH[ev]}  ({wtxt(ev, lg)})" if n == 3 else f"{SH[ev]}\n{wtxt(ev, lg)}", fontsize=13,
                            x=(tw + gp + bw) / tw / 2, ha="center")
                a.set_xticklabels([]); b.set_xticks([0.5, 2.85], ["", ""])
            else:
                a.set_xlabel(T_("time", lg), labelpad=1)
                b.set_xticks([0.5, 2.85], ["veh\nday", "SBI\nday"] if lg != "kr" else ["veh\n날", "SBI\n날"])
            b.set_ylabel(("response (z)" if lg != "kr" else "반응 (z)"), labelpad=1)
            if k == 0:
                a.set_ylabel((f"{cls} neurons\nz, minus 3 s before\nmean ± SEM" if lg != "kr" else
                              f"{cls} 뉴런\nz, 직전 3 s 평균 뺌\n평균 ± SEM"), labelpad=2)
            res[(ev, cls)] = r
    for cls, axs in TA.items():                    # one y scale per row (tick labels on the first traces only)
        lo, hi = YR[cls]; rg = (hi - lo) or 1.0
        for i, ax in enumerate(axs):
            ax.set_ylim(lo - 0.03 * rg, hi + (0.42 if cls == "UP" else 0.36) * rg)
            if i:
                ax.set_yticklabels([])
    h = [Line2D([], [], color=CC["vehicle"], lw=1.4, ls="--", label=("vehicle-day baseline" if lg != "kr" else "vehicle 날 기준")),
         Line2D([], [], color=CC["vehicle"], lw=2.2, label=T_("after_vehicle", lg)),
         Line2D([], [], color=CC["SBI-553"], lw=1.4, ls="--", label=("SBI-553-day baseline" if lg != "kr" else "SBI-553 날 기준")),
         Line2D([], [], color=CC["SBI-553"], lw=2.2, label=T_("after_SBI-553", lg)),
         Patch(facecolor="white", edgecolor="0.3", label=("bar = mean ± SEM, open = baseline" if lg != "kr" else
                                                          "막대 = 평균 ± SEM, 빈 막대 = 기준")),
         Patch(color="0.86", label=T_("rwin", lg)), Patch(color="#dbe9f6", label=T_("bwin", lg))]
    # 10-24: legend a little below the figure top (slide 86: the slide's key line touched it); with one-line titles (3 events)
    # there is room to sit it lower still
    fig.legend(handles=h, loc="upper center", ncol=7, frameon=False, bbox_to_anchor=(0.5, 0.985 if n == 4 else 0.955),
               columnspacing=1.0, handlelength=1.8, handletextpad=0.5)
    if lg != "kr":
        t = ("SELECTION USES BOTH SESSIONS: a neuron is shown if it was UP (row 1) or DOWN (row 2) in the baseline OR after the "
             "injection of that day (time-shift test, BH q <= 0.20), so the baseline-vs-injection difference is biased toward the "
             "session that selected the neuron; the unbiased check (selected on odd baseline trials, measured on even trials) is in "
             f"the notes. Mean ± SEM of the selected neurons, 4 mice pooled (n < {MIN_N} neurons: not drawn). Coloured bracket = that "
             "day's baseline vs after the injection, the same neurons (Wilcoxon signed-rank); black = after vehicle vs after SBI-553 "
             "(Mann-Whitney U). " + PS.KEY)
    else:
        t = ("선택에 두 세션을 모두 사용: 그날 기준 또는 주사 후 세션에서 UP (1행) 또는 DOWN (2행)이었던 뉴런 (시간 이동 검정, BH q <= 0.20). "
             "따라서 기준 vs 주사 차이는 뉴런을 고른 세션 쪽으로 치우침; 편향 없는 확인 (홀수 기준 trial로 선택, 짝수 trial로 측정)은 노트에. "
             f"선택된 뉴런의 평균 ± SEM, 4마리 합산 (뉴런 {MIN_N}개 미만: 표시 안 함). 색 괄호 = 그날 기준 vs 주사 후, 같은 뉴런 "
             "(Wilcoxon signed-rank); 검정 괄호 = vehicle 후 vs SBI-553 후 (Mann-Whitney U). "
             "* p < 0.05, ** p < 0.01, *** p < 0.001, ns = 유의하지 않음")
    # the full caveat goes to the notes; ONE short caveat line stays on the figure
    foot(fig, t, lg, key=("Selection uses both sessions (baseline OR injection): biased, see notes" if lg != "kr" else
                          "선택에 두 세션 (기준 또는 주사) 모두 사용: 편향 있음, 노트 참조"), key_color="#7b241c")
    fig.savefig(fn, dpi=150); plt.close(fig)
    return res


def oddeven(C, evs):
    """the unbiased check from P5_fate.csv: neurons activated / inhibited on the ODD baseline trials (Wilcoxon p < 0.05), their
    response on the EVEN baseline trials vs after the injection (paired Wilcoxon, >= 6 neurons), change veh vs SBI (Mann-Whitney)"""
    F = C.F
    out = {}
    for ev in evs:
        for g in ("activated", "inhibited"):
            r = {}
            ch = {}
            for drug in ("vehicle", "SBI-553"):
                Z = F[(F.event == ev) & (F.group == g) & (F.drug == drug)].dropna(subset=["base_heldout", "inj"])
                r[drug] = dict(n=int(len(Z)), base_even=float(Z.base_heldout.mean()) if len(Z) else np.nan,
                               inj=float(Z.inj.mean()) if len(Z) else np.nan, p=wil(Z.inj - Z.base_heldout) if len(Z) else np.nan)
                ch[drug] = (Z.inj - Z.base_heldout).values
            r["p_change_veh_vs_sbi"] = mwu(ch["vehicle"], ch["SBI-553"])
            out[(ev, g)] = r
    return out


# ------------------------------------------------------------------------------------------------ 4  UP / DOWN counts
def counts(C, dk, ph, ev):
    X = C.rows(dk, ph, ev)
    return dict(n=int(len(X)), up=int((X.k == "UP").sum()), down=int((X.k == "DOWN").sum()),
                mice=[m for m in MO if (X.mouse == m).any()])


def fig_updown(C, lg, fn):
    font(lg)
    fig = plt.figure(figsize=FS)
    xs, w = [0.14, 0.47, 0.80], 0.18
    top, bot_stim = 0.975, 0.09                      # 10-24: footer moved to the notes, its room given to the bars
    BLK = 5.35                                       # header + 4 rows + gap, in row units
    units = {f[0]: 0.6 + (len(f[3]) - 1) * BLK + 4 + 0.6 for f in FAMS}
    CNT = 12                                         # counts on the bars (>= 9.9 pt on the slide)
    res = {}
    lab = {("veh", "pre"): ("vehicle-day base", "vehicle 날 기준"), ("veh", "post"): ("after vehicle", "vehicle 후"),
           ("sbi", "pre"): ("SBI-day base", "SBI-553 날 기준"), ("sbi", "post"): ("after SBI-553", "SBI-553 후")}
    few = False
    ybot = {}
    for f_i, (key, fen, fkr, evs) in enumerate(FAMS):
        hgt_ax = (top - bot_stim) * units[key] / units["stim"]
        a = fig.add_axes([xs[f_i], top - hgt_ax, w, hgt_ax])
        ybot[key] = top - hgt_ax
        y = 0.0
        yt, yl, yc = [], [], []
        for e_i, ev in enumerate(evs):
            pre_, gap_, wa, wb = WW.of(ev)           # the event's response window (the UP / DOWN test uses it)
            a.text(-0.3, y, f"{SH[ev]}  ({wa:g}-{wb:g} s)", transform=a.get_yaxis_transform(), fontsize=12, fontweight="bold",
                   color="k", va="center", ha="left")
            for j, (dk, ph) in enumerate(G4):
                yy = y - 1 - j
                c = counts(C, dk, ph, ev)
                res[(ev, dk, ph)] = c
                star_ = "*" if 0 < len(c["mice"]) < 4 else ""
                few |= bool(star_)
                yt.append(yy); yl.append(f"{lab[(dk, ph)][1 if lg == 'kr' else 0]} ({c['n']}){star_}")
                yc.append(CC[DRUG[dk]] if ph == "post" else "0.25")
                if not c["n"]:
                    a.text(50, yy, ("no session with >= 5 events" if lg != "kr" else "사건 5회 이상 세션 없음"), ha="center",
                           va="center", fontsize=FONT, color="0.4")
                    continue
                pdn, pup = 100 * c["down"] / c["n"], 100 * c["up"] / c["n"]
                hb = 0.8
                a.barh(yy, pdn, color=DNC, height=hb)
                a.barh(yy, 100 - pdn - pup, left=pdn, color=NOC, height=hb)
                a.barh(yy, pup, left=100 - pup, color=UPC, height=hb)
                if ph == "post":
                    a.barh(yy, 100, color="none", edgecolor=CC[DRUG[dk]], lw=1.6, height=hb)
                # count inside its segment when it fits, else just beside it on the grey part, in the segment's dark colour
                # (not outside the bar, where it read as part of the row label)
                for val, wd, x_in, x_out, ha_out, dark in ((c["down"], pdn, pdn / 2, pdn + 1.2, "left", "#1b4f72"),
                                                           (c["up"], pup, 100 - pup / 2, 100 - pup - 1.2, "right", "#922b21")):
                    if wd >= 4.0 * len(str(val)) + 2.5:          # room inside the segment for the digits at CNT pt
                        a.text(x_in, yy, str(val), ha="center", va="center", fontsize=CNT, color="white", fontweight="bold")
                    else:
                        a.text(x_out, yy, str(val), ha=ha_out, va="center", fontsize=CNT, color=dark, fontweight="bold")
            y -= BLK
        a.set_yticks(yt, yl, fontsize=FONT)
        for t_, cc in zip(a.get_yticklabels(), yc):
            t_.set_color(cc)
        a.set_ylim(yt[-1] - 0.6, 0.6)
        a.set_xlim(-2, 102)
        a.set_xticks([0, 25, 50, 75, 100])
        a.set_xlabel("% of neurons" if lg != "kr" else "뉴런 %", labelpad=1)
        a.spines["left"].set_visible(False); a.tick_params(axis="y", length=0)
    h = [Patch(color=DNC, label="DOWN (inhibited)" if lg != "kr" else "DOWN (억제)"),
         Patch(color=NOC, label="no response" if lg != "kr" else "반응 없음"),
         Patch(color=UPC, label="UP (activated)" if lg != "kr" else "UP (활성)"),
         Patch(facecolor="white", edgecolor=CC["vehicle"], lw=1.6, label=("outline: after vehicle" if lg != "kr" else "테두리: vehicle 후")),
         Patch(facecolor="white", edgecolor=CC["SBI-553"], lw=1.6, label=("outline: after SBI-553" if lg != "kr" else "테두리: SBI-553 후"))]
    yl0 = ybot["behA"] - 0.085                       # free space under the behaviour A column (below its x label)
    xc = (xs[1] + xs[2] + w) / 2 - 0.03
    fig.legend(handles=h, loc="center", ncol=3, frameon=False, bbox_to_anchor=(xc, yl0 * 0.62), fontsize=FONT)
    num = ("numbers = neurons DOWN (left) and UP (right);  (n) = neurons in the session" if lg != "kr" else
           "숫자 = DOWN (왼쪽), UP (오른쪽) 뉴런 수;  (n) = 세션의 뉴런 수")
    if few:
        num += ("\n* = fewer than 4 mice had >= 5 events" if lg != "kr" else "\n* = 사건 5회 이상인 마우스가 4마리 미만")
    fig.text(xc, yl0 * 0.62 - 0.07, num, ha="center", va="top", fontsize=FONT, color="0.2", linespacing=1.35)
    if lg != "kr":
        t = (f"Every neuron of each day, 4 mice pooled (vehicle day {C.n_day['veh']}, SBI-553 day {C.n_day['sbi']}; n in brackets = "
             "neurons of the sessions with >= 5 events). Numbers on the bars = neurons DOWN (left) and UP (right); UP / DOWN = "
             "within-block time-shift test per neuron, 1000 shuffles, BH q <= 0.20 per session. Window (next to each event name): "
             + WW.en() + "." + (" * fewer than 4 mice had >= 5 events (which mice: notes)." if few else "")
             + " Descriptive (tests: next figure).")
    else:
        t = (f"각 날의 모든 뉴런, 4마리 합산 (vehicle 날 {C.n_day['veh']}, SBI-553 날 {C.n_day['sbi']}; 괄호 = 사건 5회 이상 세션의 뉴런). "
             "막대 위 숫자 = DOWN (왼쪽), UP (오른쪽) 뉴런 수; UP / DOWN = 뉴런별 블록 내 시간 이동 검정, 1000회, 세션별 BH q <= 0.20. "
             "창 (사건 이름 옆): " + WW.kr() + "." + (" * 사건 5회 이상인 마우스가 4마리 미만 (어떤 마우스: 노트)." if few else "")
             + " 기술 통계 (검정: 다음 그림).")
    foot(fig, t, lg)
    fig.savefig(fn, dpi=150); plt.close(fig)
    return res


def updown_norm_numbers(C):
    res = {}
    for ev in ALL:
        for kind, lab in (("up", "UP"), ("down", "DOWN")):
            r = {}
            dis = {}
            for dk in ("veh", "sbi"):
                M = C.paired(dk, ev)
                bb, ii = (M.k_b == lab).values, (M.k_i == lab).values
                gain, lose = int((~bb & ii).sum()), int((bb & ~ii).sum())
                dis[dk] = (gain, lose)
                mm = {}
                for m in MO:
                    q = M[M.mouse == m]
                    if len(q):
                        mm[m] = float(100 * ((q.k_i == lab).mean() - (q.k_b == lab).mean()))
                r[dk] = dict(n=int(len(M)), base_pct=float(100 * bb.mean()) if len(M) else np.nan,
                             inj_pct=float(100 * ii.mean()) if len(M) else np.nan,
                             chg=float(100 * (ii.mean() - bb.mean())) if len(M) else np.nan,
                             gained=gain, lost=lose, p_mcnemar=mcnemar(gain, lose), mice=mm)
            r["p_fisher_gain_lose"] = fisher([[dis["veh"][0], dis["veh"][1]], [dis["sbi"][0], dis["sbi"][1]]]) \
                if r["veh"]["n"] and r["sbi"]["n"] else np.nan
            res[(ev, kind)] = r
    return res


def fig_updown_norm(C, N, lg, fn):
    font(lg)
    fig = plt.figure(figsize=FS)
    # 10-24: footer moved to the notes, its room given to the panels
    axes = {"up": fig.add_axes([0.08, 0.55, 0.91, 0.32]), "down": fig.add_axes([0.08, 0.175, 0.91, 0.32])}
    x = np.arange(len(ALL)) * 1.0
    dx = 0.2
    for kind, a in axes.items():
        tops, bots = [0.0], [0.0]
        for i, ev in enumerate(ALL):
            r = N[(ev, kind)]
            for s_, dk in ((-1, "veh"), (1, "sbi")):
                q = r[dk]
                xx = x[i] + s_ * dx
                if not q["n"]:
                    a.text(xx, 0.45, ("no\ndata" if lg != "kr" else "자료\n없음"), transform=a.get_xaxis_transform(), ha="center",
                           va="center", fontsize=FONT, color="0.45", linespacing=1.2)
                    continue
                a.bar(xx, q["chg"], width=0.36, color=CC[DRUG[dk]], zorder=2)
                mv = []
                for j, m in enumerate(MO):
                    if m in q["mice"]:     # clip_on False + white edge + above the 0 line: a mouse at exactly 0 stays visible
                        a.scatter(xx + (j - 1.5) * 0.07, q["mice"][m], marker=MK[m], color=MC[m], s=30, edgecolor="white", lw=0.6,
                                  zorder=5, clip_on=False)
                        mv.append(q["mice"][m])
                q["_top"] = max([q["chg"], 0.0] + mv); q["_bot"] = min([q["chg"], 0.0] + mv)
                tops.append(q["_top"]); bots.append(q["_bot"])
        lo, hi = min(bots), max(tops)
        span = (hi - lo) or 1.0
        f_hi = 0.6
        R = span / (f_hi - 0.04)
        y0 = lo - 0.04 * R
        a.set_ylim(y0, y0 + R)
        a.axhline(0, color="0.4", lw=0.8, zorder=3)
        for i, ev in enumerate(ALL):
            r = N[(ev, kind)]
            for s_, dk in ((-1, "veh"), (1, "sbi")):
                q = r[dk]
                if q["n"]:      # one row of 'vs 0' marks per panel, above every bar and marker (the data end at f_hi)
                    PS.vs_ref(a, x[i] + s_ * dx, f_hi + 0.035, q["p_mcnemar"], "vs 0", CC[DRUG[dk]], fs=FONT, axes_y=True)
            if np.isfinite(r["p_fisher_gain_lose"]):
                PS.bracket(a, x[i] - dx, x[i] + dx, 0.855, r["p_fisher_gain_lose"], "k", fs=FONT + 1, axes_y=True, inset=0.0)
        for xb in x[:-1] + 0.5:
            a.axvline(xb, color="0.9", lw=0.8, zorder=0)
        a.set_xlim(-0.55, x[-1] + 0.55)
        yticks(a, 4)
        a.set_ylabel((f"change in % {kind.upper()}\n(after − same-day base)" if lg != "kr" else f"{kind.upper()} % 변화\n(후 − 같은 날 기준)"))
        a.set_title(("A  UP (activated) neurons" if kind == "up" else "B  DOWN (inhibited) neurons") if lg != "kr" else
                    ("A  UP (활성) 뉴런" if kind == "up" else "B  DOWN (억제) 뉴런"), loc="left", fontsize=13, pad=2)
        if kind == "up":
            a.set_xticks(x, [""] * len(x))
        else:            # event names black (red / blue mean the drug), each with its response window
            a.set_xticks(x, [f"{SH2[e]}\n{WW.of(e)[2]:g}-{WW.of(e)[3]:g} s" for e in ALL], fontsize=FONT)
    h = [Patch(color=CC["vehicle"], label=f"vehicle day ({C.n_day['veh']} neurons)" if lg != "kr" else f"vehicle 날 (뉴런 {C.n_day['veh']})"),
         Patch(color=CC["SBI-553"], label=f"SBI-553 day ({C.n_day['sbi']} neurons)" if lg != "kr" else f"SBI-553 날 (뉴런 {C.n_day['sbi']})"),
         Line2D([], [], ls="", label=("markers = one mouse each:" if lg != "kr" else "표식 = 마우스 하나:"))]
    h += [Line2D([], [], marker=MK[m], ls="", color=MC[m], ms=7, label=m) for m in MO]
    fig.legend(handles=h, loc="upper center", ncol=7, frameon=False, bbox_to_anchor=(0.5, 0.99), columnspacing=1.3,
               handlelength=1.4, handletextpad=0.5)
    if lg != "kr":
        t = ("Bars = change in the % of neurons UP (A) or DOWN (B) after the injection vs the baseline session of the same day, every "
             "neuron of the days with both sessions >= 5 events, 4 mice pooled; markers = mice. 'vs 0' = McNemar (the same neurons, "
             "baseline vs injection, exact). Black bracket = Fisher exact on the neurons that gained vs lost the class, vehicle day vs "
             "SBI-553 day (independent neurons, exploratory). no data = no session pair with >= 5 events. " + PS.KEY)
    else:
        t = ("막대 = 같은 날 기준 세션 대비 주사 후 UP (A) 또는 DOWN (B) 뉴런 % 변화, 두 세션 모두 사건 5회 이상인 날의 모든 뉴런, 4마리 "
             "합산; 표식 = 마우스. 'vs 0' = McNemar (같은 뉴런, 기준 vs 주사, exact). 검은 괄호 = 분류를 얻은 vs 잃은 뉴런의 Fisher exact, "
             "vehicle 날 vs SBI-553 날 (독립 뉴런, 탐색적). 자료 없음 = 사건 5회 이상 세션 쌍 없음. "
             "* p < 0.05, ** p < 0.01, *** p < 0.001, ns = 유의하지 않음")
    foot(fig, t, lg, key=("'vs 0' = McNemar (baseline vs injection);   black bracket = Fisher exact, vehicle vs SBI-553 day" if lg != "kr"
                          else "'vs 0' = McNemar (기준 vs 주사);   검은 괄호 = Fisher exact, vehicle 날 vs SBI-553 날"))
    fig.savefig(fn, dpi=150); plt.close(fig)


# ------------------------------------------------------------------------------------------------ slide texts (summary.json)
SHS = {"Light touch": "light", "Mild touch": "mild", "Pin prick": "pin", "Heat": "heat", "withdrawal": "withdrawal",
       "flinch": "flinch", "movement onset": "movement", "escape/rearing onset": "escape", "grooming onset": "grooming",
       "attending onset": "attending"}
FAMKR = {"stim": "자극", "behA": "행동 A", "behB": "행동 B"}
FAMEN = {"stim": "stimuli", "behA": "behaviours A", "behB": "behaviours B"}
FAMANY = {"stim": ("stimulus", "자극"), "behA": ("behaviour", "행동"), "behB": ("behaviour", "행동")}
CONF_EN = ("Drug order is confounded with day (5941, 5939 got SBI-553 on Day 1; 5937, 1191 on Day 2); with 4 mice no test across "
           "mice can reach p < 0.05, so the mouse markers are descriptive; neurons pooled over mice = exploratory.")
CONF_KR = ("약물 순서와 날이 겹침 (5941, 5939는 Day 1에 SBI-553; 5937, 1191은 Day 2에); 4마리로는 마우스 간 검정이 p < 0.05에 도달할 수 "
           "없으므로 마우스 표식은 기술 통계; 마우스 합산 뉴런 = 탐색적.")
TRIAL_EN = ("Traces come from the trace cache (psth_cache_win3_tf.npz), bars from cells_classes.csv; in 11 of 160 session x event "
            "entries the trace has 1-2 fewer trials (trials too close to the session edge for the -5..8 s trace window).")
TRIAL_KR = ("trace는 trace cache (psth_cache_win3_tf.npz), 막대는 cells_classes.csv; 160개 세션 x 사건 중 11개에서 trace의 trial이 "
            "1-2개 적음 (-5..8 s trace 창에 비해 세션 끝에 너무 가까운 trial).")


def _p(p):
    return ptxt(p)


def _pk(p):
    return ptxt(p, "kr")


def _s3(x):
    return "n.a." if x is None or not np.isfinite(x) else f"{x:.3f}"


def _minp(ps):
    ps = [p for p in ps if p is not None and np.isfinite(p)]
    return min(ps) if ps else np.nan


def _ge(p):
    """'p >= x' for the smallest p of a set (all >= 0.05)"""
    return "n.a." if not np.isfinite(p) else f"p ≥ {p:.3f}" if p < 0.1 else f"p ≥ {p:.2f}"


def _short(s, n):
    if len(s) > n:
        print(f"  !! text {len(s)} > {n}: {s}")
    return s


def sessions_txt(C, dk, ev, lg):
    out = []
    for ph in ("pre", "post"):
        X = C.rows(dk, ph, ev)
        ms = [f"{m} D{vday(dk, m)}" for m in MO if (X.mouse == m).any()]
        lab = cond_lab(dk, ph, lg)
        out.append(f"{lab}: " + (", ".join(ms) if ms else ("none" if lg != "kr" else "없음")))
    return "; ".join(out)


def text_day(C, fam, dk, R):
    key, fen, fkr, evs = fam
    drug = DRUG[dk]
    den, dkr = ("vehicle day", "vehicle 날") if dk == "veh" else ("SBI-553 day", "SBI-553 날")
    kb = [e for e in evs if R[e]["n_pre"]]; ki = [e for e in evs if R[e]["n_post"]]
    nb = [e for e in kb if np.isfinite(R[e]["p_vs0_pre"]) and R[e]["p_vs0_pre"] < 0.05 and R[e]["mean_pre"] > 0]
    ni = [e for e in ki if np.isfinite(R[e]["p_vs0_post"]) and R[e]["p_vs0_post"] < 0.05 and R[e]["mean_post"] > 0]
    ch = [e for e in evs if np.isfinite(R[e]["p_paired"]) and R[e]["p_paired"] < 0.05]
    pp = _minp([R[e]["p_paired"] for e in evs])
    k1 = f"Response > 0 (p < 0.05): baseline {len(nb)}/{len(kb)}, after {drug} {len(ni)}/{len(ki)} events; "
    k1k = f"반응 > 0 (p < 0.05): 기준 {len(nb)}/{len(kb)}, {drug} 후 {len(ni)}/{len(ki)}개 사건; "
    if ch:
        k2 = "changed: " + ", ".join(f"{SHS[e]} {R[e]['mean_pre']:+.3f} → {R[e]['mean_post']:+.3f} z ({_p(R[e]['p_paired'])})" for e in ch)
        k2k = "변화: " + ", ".join(f"{SHS[e]} {R[e]['mean_pre']:+.3f} → {R[e]['mean_post']:+.3f} z ({_pk(R[e]['p_paired'])})" for e in ch)
    elif np.isfinite(pp):
        k2 = f"no event differs, baseline vs after {drug} (paired {_ge(pp)})"
        k2k = f"기준 vs {drug} 후 차이 있는 사건 없음 (짝 {_ge(pp)})"
    else:
        k2 = "no event has both sessions"; k2k = "두 세션이 모두 있는 사건 없음"
    nn = C.n_day[dk]
    det_en, det_kr = [], []
    for e in evs:
        r = R[e]
        det_en.append(f"{SH[e]}: baseline {fnum(r['mean_pre'], 3)} ± {_s3(r['sem_pre'])} z "
                      f"(n = {r['n_pre']}, vs 0 {_p(r['p_vs0_pre'])}), after {drug} {fnum(r['mean_post'], 3)} ± "
                      f"{_s3(r['sem_post'])} z (n = {r['n_post']}, vs 0 {_p(r['p_vs0_post'])}), "
                      f"paired n = {r['n_paired']}: {_p(r['p_paired'])}; mouse means base "
                      + ", ".join(f"{m} {v:+.3f}" for m, v in r["mice_pre"].items()) + "; inj "
                      + ", ".join(f"{m} {v:+.3f}" for m, v in r["mice_post"].items()) + f" [{sessions_txt(C, dk, e, 'en')}]")
        det_kr.append(f"{SH[e]}: 기준 {fnum(r['mean_pre'], 3)} ± {_s3(r['sem_pre'])} z "
                      f"(n = {r['n_pre']}, vs 0 {_pk(r['p_vs0_pre'])}), {drug} 후 {fnum(r['mean_post'], 3)} ± "
                      f"{_s3(r['sem_post'])} z (n = {r['n_post']}, vs 0 {_pk(r['p_vs0_post'])}), "
                      f"짝 n = {r['n_paired']}: {_pk(r['p_paired'])}; 마우스 평균 기준 "
                      + ", ".join(f"{m} {v:+.3f}" for m, v in r["mice_pre"].items()) + "; 주사 "
                      + ", ".join(f"{m} {v:+.3f}" for m, v in r["mice_post"].items()) + f" [{sessions_txt(C, dk, e, 'kr')}]")
    return dict(
        title_en=f"All mice, {den}: population response, baseline vs after {drug} ({FAMEN[key]})",
        title_kr=f"모든 마우스, {dkr}: 집단 반응, 기준 vs {drug} 후 ({FAMKR[key]})",
        key_en=_short(k1 + k2, 150), key_kr=_short(k1k + k2k, 150),
        bottom_en=_short(f"Every neuron of the {den} ({nn}), 4 mice pooled; bars = best-window response; Wilcoxon signed-rank: each bar "
                         f"vs 0 and baseline vs after {drug} (same neurons); markers = mouse means.", 230),
        bottom_kr=_short(f"{dkr}의 모든 뉴런 ({nn}), 4마리 합산; 막대 = 최적 창 반응; Wilcoxon signed-rank: 막대 vs 0, 기준 vs {drug} 후 "
                         f"(같은 뉴런); 표식 = 마우스 평균.", 230),
        notes_en="Same format as the per-mouse slide (make_per_mouse_v2.fig_population) with every mouse pooled. Window: " + WW.en()
        + ". " + " | ".join(det_en).rstrip(".") + ". " + CONF_EN + " " + TRIAL_EN,
        notes_kr="마우스별 슬라이드 (make_per_mouse_v2.fig_population)와 같은 형식, 모든 마우스 합산. 창: " + WW.kr() + ". "
        + " | ".join(det_kr).rstrip(".") + ". " + CONF_KR + " " + TRIAL_KR,
        section="Q3 population")


def text_norm(C, fam, R, base_note):
    key, fen, fkr, evs = fam
    any_en, any_kr = FAMANY[key]
    both = [e for e in evs if R[e]["n_veh"] and R[e]["n_sbi"]]
    smw = [e for e in both if np.isfinite(R[e]["p_mw"]) and R[e]["p_mw"] < 0.05]
    str_ = [e for e in both if np.isfinite(R[e]["p_tracked"]) and R[e]["p_tracked"] < 0.05]
    pmw, ptr = _minp([R[e]["p_mw"] for e in both]), _minp([R[e]["p_tracked"] for e in both])
    if not both:
        only_v = [SHS[e] for e in evs if R[e]["n_veh"] and not R[e]["n_sbi"]]
        only_s = [SHS[e] for e in evs if R[e]["n_sbi"] and not R[e]["n_veh"]]
        k = (f"No event has both drug days ({', '.join(only_v)} only on the vehicle day, {', '.join(only_s)} only on the SBI-553 day): "
             "no comparison, descriptive")
        kk = (f"두 약물 날이 모두 있는 사건 없음 ({', '.join(only_v)}: vehicle 날만, {', '.join(only_s)}: SBI-553 날만): 비교 불가, "
              "기술 통계")
    elif not smw and not str_:
        k = (f"Change from the same-day baseline did not differ, SBI-553 vs vehicle day, for any {any_en} (Mann-Whitney {_ge(pmw)}; "
             f"tracked {_ge(ptr)})")
        kk = f"같은 날 기준 대비 변화: 어느 {any_kr}에서도 SBI-553 날과 vehicle 날 차이 없음 (Mann-Whitney {_ge(pmw)}; 추적 뉴런 {_ge(ptr)})"
    else:
        it = [f"{SHS[e]} veh {R[e]['mean_veh']:+.3f} vs SBI {R[e]['mean_sbi']:+.3f} z (MW {_p(R[e]['p_mw'])}, tracked "
              f"{_p(R[e]['p_tracked'])})" for e in sorted(set(smw) | set(str_), key=lambda e: min(R[e]['p_mw'], R[e]['p_tracked']))]
        itk = [f"{SHS[e]} veh {R[e]['mean_veh']:+.3f} vs SBI {R[e]['mean_sbi']:+.3f} z (MW {_pk(R[e]['p_mw'])}, 추적 "
               f"{_pk(R[e]['p_tracked'])})" for e in sorted(set(smw) | set(str_), key=lambda e: min(R[e]['p_mw'], R[e]['p_tracked']))]
        k = "Change from the same-day baseline differed for " + "; ".join(it[:2])
        kk = "같은 날 기준 대비 변화가 다름: " + "; ".join(itk[:2])
    det_en, det_kr = [], []
    for e in evs:
        r = R[e]
        mv = lambda d: ", ".join(f"{m} {v:+.3f}" for m, v in r[f"mice_{d}"].items())
        det_en.append(f"{SH[e]}: change vehicle day {fnum(r['mean_veh'], 3)} z (n = {r['n_veh']}, vs 0 {_p(r['p_vs0_veh'])}; mice "
                      f"{mv('veh') or '-'}), SBI-553 day {fnum(r['mean_sbi'], 3)} z (n = {r['n_sbi']}, vs 0 {_p(r['p_vs0_sbi'])}; mice "
                      f"{mv('sbi') or '-'}), Mann-Whitney {_p(r['p_mw'])}; tracked n = {r['n_tracked']}: mean change vehicle "
                      f"{fnum(r['mean_tracked_veh'], 3)}, SBI-553 {fnum(r['mean_tracked_sbi'], 3)}, median difference "
                      f"{fnum(r['med_tracked'], 3)} z, paired Wilcoxon {_p(r['p_tracked'])} (class A only, n = {r['n_tracked_A']}: "
                      f"{_p(r['p_tracked_A'])})")
        det_kr.append(f"{SH[e]}: 변화 vehicle 날 {fnum(r['mean_veh'], 3)} z (n = {r['n_veh']}, vs 0 {_pk(r['p_vs0_veh'])}; 마우스 "
                      f"{mv('veh') or '-'}), SBI-553 날 {fnum(r['mean_sbi'], 3)} z (n = {r['n_sbi']}, vs 0 {_pk(r['p_vs0_sbi'])}; 마우스 "
                      f"{mv('sbi') or '-'}), Mann-Whitney {_pk(r['p_mw'])}; 추적 n = {r['n_tracked']}: 평균 변화 vehicle "
                      f"{fnum(r['mean_tracked_veh'], 3)}, SBI-553 {fnum(r['mean_tracked_sbi'], 3)}, 차이 중앙값 {fnum(r['med_tracked'], 3)} z, "
                      f"짝 Wilcoxon {_pk(r['p_tracked'])} (A 등급만, n = {r['n_tracked_A']}: {_pk(r['p_tracked_A'])})")
    return dict(
        title_en=f"All mice, baseline-normalized: change after the injection, vehicle day vs SBI-553 day ({FAMEN[key]})",
        title_kr=f"모든 마우스, 기준 정규화: 주사 후 변화, vehicle 날 vs SBI-553 날 ({FAMKR[key]})",
        key_en=_short(k, 150), key_kr=_short(kk, 150),
        bottom_en=_short(f"Per neuron, injection minus same-day baseline (best window); vehicle day {C.n_day['veh']} vs SBI-553 day "
                         f"{C.n_day['sbi']} neurons, 4 mice pooled: Mann-Whitney U; 185 tracked neurons: paired Wilcoxon signed-rank.", 230),
        bottom_kr=_short(f"뉴런마다 주사 - 같은 날 기준 (최적 창); vehicle 날 {C.n_day['veh']} vs SBI-553 날 {C.n_day['sbi']} 뉴런, 4마리 합산: "
                         "Mann-Whitney U; 추적 뉴런 185개: 짝 Wilcoxon signed-rank.", 230),
        notes_en="User rule 10-23: (injection - its own same-day baseline), vehicle day vs SBI-553 day; cross-day neuron comparisons "
        "only with the tracked neurons (crossday pairs: A 101 + B 84 = 185; n per event = tracked neurons with the event in all four "
        "sessions; identical to ca_tracked_crossover.py p_norm). Window: " + WW.en() + ". " + " | ".join(det_en).rstrip(".") + ". "
        + base_note[0] + " " + CONF_EN,
        notes_kr="사용자 규칙 10-23: (주사 - 그 날 기준), vehicle 날 vs SBI-553 날; 날 사이 뉴런 비교는 추적 뉴런만 (crossday pairs: "
        "A 101 + B 84 = 185; 사건별 n = 네 세션 모두에 그 사건이 있는 추적 뉴런; ca_tracked_crossover.py p_norm과 동일). 창: " + WW.kr()
        + ". " + " | ".join(det_kr).rstrip(".") + ". " + base_note[1] + " " + CONF_KR,
        section="Q3 population")


def text_resp(C, fam, R, OE):
    key, fen, fkr, evs = fam
    gl = {"activated": ("UP", "UP"), "inhibited": ("DOWN", "DOWN")}
    mwi, wi = [], []
    for e in evs:
        for g in ("activated", "inhibited"):
            o = OE.get((e, g))
            if not o:
                continue
            if np.isfinite(o["p_change_veh_vs_sbi"]) and o["p_change_veh_vs_sbi"] < 0.05:
                dv = o["vehicle"]["inj"] - o["vehicle"]["base_even"]; ds = o["SBI-553"]["inj"] - o["SBI-553"]["base_even"]
                mwi.append((o["p_change_veh_vs_sbi"],
                            f"{SHS[e]} {gl[g][0]} change veh {dv:+.2f} vs SBI {ds:+.2f} z ({_p(o['p_change_veh_vs_sbi'])})",
                            f"{SHS[e]} {gl[g][1]} 변화 veh {dv:+.2f} vs SBI {ds:+.2f} z ({_pk(o['p_change_veh_vs_sbi'])})"))
            for drug in ("vehicle", "SBI-553"):
                q = o[drug]
                if np.isfinite(q["p"]) and q["p"] < 0.05:
                    stronger = (q["inj"] > q["base_even"]) == (g == "activated")
                    ds_ = "SBI" if drug == "SBI-553" else "veh"
                    wi.append((q["p"], f"{SHS[e]} {gl[g][0]} {'stronger' if stronger else 'weaker'} after {ds_} (n = {q['n']}, "
                                        f"{_p(q['p'])})",
                               f"{SHS[e]} {gl[g][1]} {ds_} 후 {'강해짐' if stronger else '약해짐'} (n = {q['n']}, {_pk(q['p'])})"))
    pick = sorted(mwi)[:2] if mwi else sorted(wi)[:2]
    if pick:
        k = "Unbiased odd/even check: " + "; ".join(x[1] for x in pick)
        kk = "편향 없는 홀수/짝수 확인: " + "; ".join(x[2] for x in pick)
    elif any(np.isfinite(o[d]["p"]) for (e, g), o in OE.items() if e in evs for d in ("vehicle", "SBI-553")):
        k = "Selection-biased view; the unbiased odd/even check shows no change after either injection (Wilcoxon p ≥ 0.05)"
        kk = "선택 편향 있는 보기; 편향 없는 홀수/짝수 확인에서 어느 주사 후에도 변화 없음 (Wilcoxon p ≥ 0.05)"
    else:
        k = "Selection-biased view; the unbiased odd/even check is not testable here (fewer than 6 neurons in every group)"
        kk = "선택 편향 있는 보기; 편향 없는 홀수/짝수 확인은 검정 불가 (모든 그룹 뉴런 6개 미만)"
    det_en, det_kr = [], []
    for e in evs:
        for cls in ("UP", "DOWN"):
            r = R[(e, cls)]
            det_en.append(f"{SH[e]} {cls}: vehicle day n = {r['n_veh']} (base {fnum(r['mean_veh_pre'], 3)}, after {fnum(r['mean_veh_post'], 3)}, "
                          f"{_p(r['p_veh'])}), SBI-553 day n = {r['n_sbi']} (base {fnum(r['mean_sbi_pre'], 3)}, after "
                          f"{fnum(r['mean_sbi_post'], 3)}, {_p(r['p_sbi'])}), after veh vs after SBI {_p(r['p_veh_vs_sbi'])}")
            det_kr.append(f"{SH[e]} {cls}: vehicle 날 n = {r['n_veh']} (기준 {fnum(r['mean_veh_pre'], 3)}, 후 {fnum(r['mean_veh_post'], 3)}, "
                          f"{_pk(r['p_veh'])}), SBI-553 날 n = {r['n_sbi']} (기준 {fnum(r['mean_sbi_pre'], 3)}, 후 "
                          f"{fnum(r['mean_sbi_post'], 3)}, {_pk(r['p_sbi'])}), veh 후 vs SBI 후 {_pk(r['p_veh_vs_sbi'])}")
    oe_en, oe_kr = [], []
    for e in evs:
        for g in ("activated", "inhibited"):
            o = OE.get((e, g))
            if not o or not (o["vehicle"]["n"] or o["SBI-553"]["n"]):
                continue
            f_ = lambda q: (f"n = {q['n']}, even base {fnum(q['base_even'], 3)} → inj {fnum(q['inj'], 3)} z, {_p(q['p'])}" if q["n"]
                            else "n = 0")
            fk = lambda q: (f"n = {q['n']}, 짝수 기준 {fnum(q['base_even'], 3)} → 주사 {fnum(q['inj'], 3)} z, {_pk(q['p'])}" if q["n"]
                            else "n = 0")
            oe_en.append(f"{SH[e]} {g}: vehicle {f_(o['vehicle'])}; SBI-553 {f_(o['SBI-553'])}; change veh vs SBI Mann-Whitney "
                         f"{_p(o['p_change_veh_vs_sbi'])}")
            oe_kr.append(f"{SH[e]} {'활성' if g == 'activated' else '억제'}: vehicle {fk(o['vehicle'])}; SBI-553 {fk(o['SBI-553'])}; "
                         f"변화 veh vs SBI Mann-Whitney {_pk(o['p_change_veh_vs_sbi'])}")
    return dict(
        title_en=f"All mice, responsive neurons only: UP / DOWN in either session of the day ({FAMEN[key]})",
        title_kr=f"모든 마우스, 반응 뉴런만: 그날 어느 한 세션에서 UP / DOWN ({FAMKR[key]})",
        key_en=_short(k, 150), key_kr=_short(kk, 150),
        bottom_en=_short("Neurons UP or DOWN (time-shift test) in the baseline OR injection session of their day, 4 mice pooled; Wilcoxon "
                         "signed-rank (same neurons), Mann-Whitney U (veh vs SBI). Biased toward the selecting session.", 230),
        bottom_kr=_short("그날 기준 또는 주사 세션에서 UP 또는 DOWN인 뉴런 (시간 이동 검정), 4마리 합산; Wilcoxon signed-rank (같은 뉴런), "
                         "Mann-Whitney U (veh vs SBI). 뉴런을 고른 세션 쪽으로 치우침.", 230),
        notes_en=("Why biased: a neuron classified UP only after the injection enters the set because its injection response was "
                  "large, so the injection bar is inflated (and vice versa for baseline-only neurons); the SBI-553-day baselines had few "
                  "UP neurons for light touch and heat, so those sets are mostly selected by the injection session. Unbiased check "
                  "(P5_fate.csv of ca_event_analysis.plot5_fate): neurons activated / inhibited on the ODD baseline trials (Wilcoxon "
                  "p < 0.05, baseline >= 10 trials, injection >= 5), response on the EVEN baseline trials vs after the injection, paired "
                  "Wilcoxon (>= 6 neurons), change vehicle vs SBI-553 Mann-Whitney (both >= 6): " + " | ".join(oe_en)
                  + f". Figure numbers (n < {MIN_N} neurons not drawn): " + " | ".join(det_en) + ". Window: " + WW.en() + ". " + CONF_EN),
        notes_kr=("편향 이유: 주사 후에만 UP인 뉴런은 주사 반응이 커서 선택되므로 주사 막대가 부풀려짐 (기준에서만 UP인 뉴런은 반대); "
                  "SBI-553 날 기준에서 light touch, heat의 UP 뉴런이 적어서 그 집합은 대부분 주사 세션으로 선택됨. 편향 없는 확인 "
                  "(ca_event_analysis.plot5_fate의 P5_fate.csv): 홀수 기준 trial에서 활성 / 억제된 뉴런 (Wilcoxon p < 0.05, 기준 >= 10 trial, "
                  "주사 >= 5), 짝수 기준 trial vs 주사 후 반응, 짝 Wilcoxon (뉴런 >= 6), 변화 vehicle vs SBI-553 Mann-Whitney (둘 다 >= 6): "
                  + " | ".join(oe_kr) + f". 그림 수치 (뉴런 {MIN_N}개 미만은 표시 안 함): " + " | ".join(det_kr) + ". 창: " + WW.kr()
                  + ". " + CONF_KR),
        section="Q3 neurons")


def text_updown(C, R):
    s = {(dk, ph): (sum(R[(e, dk, ph)]["up"] for e in STIM), sum(R[(e, dk, ph)]["down"] for e in STIM)) for dk, ph in G4}
    nv, ns_ = C.n_day["veh"], C.n_day["sbi"]
    k = (f"UP over the 4 stimuli: vehicle day {s[('veh', 'pre')][0]} → {s[('veh', 'post')][0]} (of 4 × {nv}), SBI-553 day "
         f"{s[('sbi', 'pre')][0]} → {s[('sbi', 'post')][0]} (of 4 × {ns_})")
    kk = (f"4개 자극 UP 합계: vehicle 날 {s[('veh', 'pre')][0]} → {s[('veh', 'post')][0]} (4 × {nv} 중), SBI-553 날 "
          f"{s[('sbi', 'pre')][0]} → {s[('sbi', 'post')][0]} (4 × {ns_} 중)")
    det_en, det_kr = [], []
    for e in ALL:
        parts, partk = [], []
        for dk, ph in G4:
            c = R[(e, dk, ph)]
            lab = {("veh", "pre"): "veh-day base", ("veh", "post"): "after veh", ("sbi", "pre"): "SBI-day base",
                   ("sbi", "post"): "after SBI"}[(dk, ph)]
            labk = {("veh", "pre"): "veh 날 기준", ("veh", "post"): "veh 후", ("sbi", "pre"): "SBI 날 기준",
                    ("sbi", "post"): "SBI 후"}[(dk, ph)]
            if c["n"]:
                ms = "" if len(c["mice"]) == 4 else f" [{', '.join(c['mice'])}]"
                parts.append(f"{lab} {c['up']} UP / {c['down']} DOWN of {c['n']}{ms}")
                partk.append(f"{labk} UP {c['up']} / DOWN {c['down']} / {c['n']}{ms}")
            else:
                parts.append(f"{lab} no session"); partk.append(f"{labk} 세션 없음")
        det_en.append(f"{SH[e]}: " + ", ".join(parts)); det_kr.append(f"{SH[e]}: " + ", ".join(partk))
    return dict(
        title_en="All mice: how many neurons go up or down, per event (four sessions)",
        title_kr="모든 마우스: 사건별 상승·하강 뉴런 수 (네 세션)",
        key_en=_short(k, 150), key_kr=_short(kk, 150),
        bottom_en=_short(f"Every neuron of each day, 4 mice pooled (vehicle day {nv}, SBI-553 day {ns_}); UP / DOWN = within-block "
                         "time-shift test, BH q ≤ 0.20 per session; counts on the bars; descriptive (tests: next slide).", 230),
        bottom_kr=_short(f"각 날의 모든 뉴런, 4마리 합산 (vehicle 날 {nv}, SBI-553 날 {ns_}); UP / DOWN = 블록 내 시간 이동 검정, 세션별 "
                         "BH q ≤ 0.20; 막대 위 숫자 = 뉴런 수; 기술 통계 (검정: 다음 슬라이드).", 230),
        notes_en="Per-mouse slide 'how many neurons go up or down, per event' with every mouse together. Counts (mice in brackets "
        "when fewer than 4 had >= 5 events): " + " | ".join(det_en) + ". Window: " + WW.en() + ". " + CONF_EN,
        notes_kr="마우스별 슬라이드 '사건별 상승·하강 뉴런 수'를 모든 마우스 합산으로. 수 (사건 5회 이상 마우스가 4마리 미만이면 괄호에 "
        "마우스): " + " | ".join(det_kr) + ". 창: " + WW.kr() + ". " + CONF_KR,
        section="Q3 neurons")


def text_updown_norm(C, N, base_note):
    hi_s, hi_v, mice_cons = [], [], {}
    for e in ALL:
        r = N[(e, "up")]
        if np.isfinite(r["p_fisher_gain_lose"]) and r["p_fisher_gain_lose"] < 0.05:
            (hi_s if r["sbi"]["chg"] > r["veh"]["chg"] else hi_v).append(SHS[e])
    k = (f"Change in % UP was higher on the SBI-553 day for {', '.join(hi_s) or 'none'}, higher on the vehicle day for "
         f"{', '.join(hi_v) or 'none'} "
         "(Fisher p < 0.05)")
    kk = (f"UP % 변화가 SBI-553 날에 더 큼: {', '.join(hi_s) or '없음'}; vehicle 날에 더 큼: {', '.join(hi_v) or '없음'} (Fisher p < 0.05)")
    det_en, det_kr = [], []
    for e in ALL:
        for kind in ("up", "down"):
            r = N[(e, kind)]
            if not (r["veh"]["n"] or r["sbi"]["n"]):
                continue
            cm = [m for m in MO if m in r["veh"]["mice"] and m in r["sbi"]["mice"]]
            nh = sum(r["sbi"]["mice"][m] > r["veh"]["mice"][m] for m in cm)
            f_ = lambda q: (f"{q['base_pct']:.1f}% → {q['inj_pct']:.1f}% ({q['chg']:+.1f}; gained {q['gained']}, lost {q['lost']}, "
                            f"McNemar {_p(q['p_mcnemar'])}, n = {q['n']})" if q["n"] else "no data")
            fk = lambda q: (f"{q['base_pct']:.1f}% → {q['inj_pct']:.1f}% ({q['chg']:+.1f}; 얻음 {q['gained']}, 잃음 {q['lost']}, "
                            f"McNemar {_pk(q['p_mcnemar'])}, n = {q['n']})" if q["n"] else "자료 없음")
            det_en.append(f"{SH[e]} {kind.upper()}: vehicle day {f_(r['veh'])}; SBI-553 day {f_(r['sbi'])}; Fisher {_p(r['p_fisher_gain_lose'])}"
                          + (f"; mice with a larger change on the SBI-553 day {nh}/{len(cm)}" if cm else ""))
            det_kr.append(f"{SH[e]} {kind.upper()}: vehicle 날 {fk(r['veh'])}; SBI-553 날 {fk(r['sbi'])}; Fisher {_pk(r['p_fisher_gain_lose'])}"
                          + (f"; SBI-553 날 변화가 더 큰 마우스 {nh}/{len(cm)}" if cm else ""))
    return dict(
        title_en="All mice: change in % UP and % DOWN from the same-day baseline, vehicle day vs SBI-553 day",
        title_kr="모든 마우스: 같은 날 기준 대비 UP %, DOWN % 변화, vehicle 날 vs SBI-553 날",
        key_en=_short(k, 150), key_kr=_short(kk, 150),
        bottom_en=_short("Every neuron of the days with both sessions, 4 mice pooled; McNemar (baseline vs injection, same neurons); "
                         "Fisher exact on gained vs lost neurons, vehicle vs SBI-553 day; markers = mice.", 230),
        bottom_kr=_short("두 세션이 모두 있는 날의 모든 뉴런, 4마리 합산; McNemar (기준 vs 주사, 같은 뉴런); 분류를 얻은 vs 잃은 뉴런의 "
                         "Fisher exact, vehicle 날 vs SBI-553 날; 표식 = 마우스.", 230),
        notes_en="User rule 10-23: % UP / DOWN may use every neuron of each day. Change = % after the injection minus % in the same "
        "day's baseline (the same neurons). McNemar = exact binomial on the neurons that changed class (no neuron changed: p = 1). "
        "Fisher = 2 x 2 of gained vs lost neurons, vehicle day vs SBI-553 day (does the direction of switching differ). " + " | ".join(det_en).rstrip(".")
        + ". " + base_note[0] + " " + CONF_EN,
        notes_kr="사용자 규칙 10-23: UP / DOWN %는 각 날의 모든 뉴런 사용 가능. 변화 = 주사 후 % - 같은 날 기준 % (같은 뉴런). McNemar = "
        "분류가 바뀐 뉴런의 exact binomial (바뀐 뉴런 없음: p = 1). Fisher = 얻은 vs 잃은 뉴런의 2 x 2, vehicle 날 vs SBI-553 날 (바뀌는 "
        "방향이 다른가). " + " | ".join(det_kr).rstrip(".") + ". " + base_note[1] + " " + CONF_KR,
        section="Q3 neurons")


PRIORITY = {"POP_stim_norm.png": "main", "POP_behA_norm.png": "main", "POP_stim_vehday.png": "backup", "POP_stim_sbiday.png": "backup",
            "POP_stim_respOnly.png": "backup", "POP_behA_vehday.png": "backup", "POP_behA_sbiday.png": "backup",
            "POP_behA_respOnly.png": "backup", "UPDOWN_pooled.png": "backup", "UPDOWN_pooled_norm.png": "backup",
            "POP_behB_vehday.png": "full", "POP_behB_sbiday.png": "full", "POP_behB_norm.png": "full", "POP_behB_respOnly.png": "full"}


def build_summary(C, NUM):
    ud = NUM["UPDOWN_pooled.png"]
    nv, ns_ = C.n_day["veh"], C.n_day["sbi"]
    # stimuli whose SBI-553-day baseline had less than half the vehicle-day baseline % UP
    lo = [(e, ud[(e, "veh", "pre")]["up"], ud[(e, "sbi", "pre")]["up"]) for e in STIM
          if ud[(e, "sbi", "pre")]["up"] / ns_ < 0.5 * ud[(e, "veh", "pre")]["up"] / nv]
    base_note = ("Baseline caveat: the SBI-553-day baselines had far fewer UP neurons than the vehicle-day baselines for "
                 + ", ".join(f"{SHS[e]} ({s} of {ns_} = {100 * s / ns_:.1f}% vs {v} of {nv} = {100 * v / nv:.1f}%)" for e, v, s in lo)
                 + "; a change from a low starting point is easier upward (floor) and is part of the day / order confound.",
                 "기준 주의: SBI-553 날 기준의 UP 뉴런이 vehicle 날 기준보다 훨씬 적음: "
                 + ", ".join(f"{SHS[e]} ({ns_} 중 {s} = {100 * s / ns_:.1f}% vs {nv} 중 {v} = {100 * v / nv:.1f}%)" for e, v, s in lo)
                 + "; 낮은 출발점에서는 증가가 쉬움 (바닥 효과), 날 / 순서 교란의 일부.")
    S = {}
    for fam in FAMS:
        key = fam[0]
        for dk in ("veh", "sbi"):
            fn = f"POP_{key}_{dk}day.png"
            S[fn] = dict(numbers=NUM[fn], **text_day(C, fam, dk, NUM[fn]))
        fn = f"POP_{key}_norm.png"
        S[fn] = dict(numbers=NUM[fn], **text_norm(C, fam, NUM[fn], base_note))
        fn = f"POP_{key}_respOnly.png"
        oe = {k: v for k, v in NUM["oddeven"].items() if k[0] in fam[3]}
        S[fn] = dict(numbers=dict(figure=NUM[fn], oddeven_check=oe), **text_resp(C, fam, NUM[fn], NUM["oddeven"]))
    S["UPDOWN_pooled.png"] = dict(numbers=ud, **text_updown(C, ud))
    S["UPDOWN_pooled_norm.png"] = dict(numbers=NUM["UPDOWN_pooled_norm.png"],
                                       **text_updown_norm(C, NUM["UPDOWN_pooled_norm.png"], base_note))
    for fn, v in S.items():
        v["priority"] = PRIORITY[fn]
        v["files"] = dict(en=fn, kr=fn.replace(".png", "_kr.png"))
    return dict(script="mini1p_pipeline\\ca_pop_pooled.py", folder=OUT, n_neurons=dict(vehicle_day=C.n_day["veh"], sbi_day=C.n_day["sbi"],
                tracked=int(len(C.P))), windows={e: list(WW.of(e)) for e in ALL}, figures=S)


# ------------------------------------------------------------------------------------------------ main
def jconv(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 5)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, set):
        return sorted(o)
    return str(o)


def clean(o):
    if isinstance(o, dict):
        return {(k if isinstance(k, str) else " | ".join(map(str, k)) if isinstance(k, tuple) else str(k)): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else round(float(o), 5)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", nargs="+", default=["en", "kr"])
    A = ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    C = Data()
    NUM = {}
    for lg in A.lang:
        sfx = "_kr" if lg == "kr" else ""
        for fam in FAMS:
            key = fam[0]
            for dk in ("veh", "sbi"):
                r = fig_day(C, fam, dk, lg, os.path.join(OUT, f"POP_{key}_{dk}day{sfx}.png"))
                NUM[f"POP_{key}_{dk}day.png"] = r
            NUM[f"POP_{key}_norm.png"] = fig_norm(C, fam, lg, os.path.join(OUT, f"POP_{key}_norm{sfx}.png"))
            NUM[f"POP_{key}_respOnly.png"] = fig_resp(C, fam, lg, os.path.join(OUT, f"POP_{key}_respOnly{sfx}.png"))
        NUM["UPDOWN_pooled.png"] = fig_updown(C, lg, os.path.join(OUT, f"UPDOWN_pooled{sfx}.png"))
        N = updown_norm_numbers(C)
        fig_updown_norm(C, N, lg, os.path.join(OUT, f"UPDOWN_pooled_norm{sfx}.png"))
        NUM["UPDOWN_pooled_norm.png"] = N
        print(lg, "figures done", flush=True)
    NUM["oddeven"] = oddeven(C, ALL)
    S = build_summary(C, NUM)
    json.dump(clean(S), open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=jconv)
    for fn, v in S["figures"].items():
        print(f"{fn} [{v['priority']}]\n  {v['key_en']}\n  {v['key_kr']}")
    print("->", OUT)


if __name__ == "__main__":
    main()
