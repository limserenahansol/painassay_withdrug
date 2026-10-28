"""ca_responsive_traces.py - how many neurons respond (UP / DOWN) AND what those neurons look like (2026-10-23, PI extra).

What
  1  RESP_<family>.png          top: % UP (solid) / % DOWN (hatched) in the four sessions of the crossover - base (vehicle
                                day), after vehicle, base (SBI-553 day), after SBI-553 - every neuron of each day, mouse
                                markers; brackets: blue / red = that day's baseline vs after the injection (the same neurons,
                                McNemar), black = after vehicle vs after SBI-553 (Fisher exact, neurons pooled = exploratory);
                                brackets above the bars test % UP, below test % DOWN.
                                bottom (new): mean +- SEM trace of the UP neurons (solid) and of the DOWN neurons (dashed) in
                                each session; left sub-panel = vehicle day, right = SBI-553 day (under its two bars).
     families: stim (Light touch, Mild touch, Pin prick, Heat), behA (withdrawal, flinch, movement onset), behB (escape /
     rearing, grooming, attending onsets - only the sessions with >= 5 events exist).
  2  RESP_<family>_tracked.png  the same with only the 185 tracked neurons (analysis\\crossday\\<m>_pairs.csv, the same cells
                                on both days); vehicle vs SBI-553 is then paired too -> McNemar for every bracket.
  3  GROUPS_<tag>.png           every event: baseline / vehicle / SBI-553, mean trace of the UP, DOWN and other neurons
                                (n in the legend; baseline = both days' baseline sessions).
     GROUPS_<tag>_tracked.png   the tracked neurons grouped by their class in the baseline of each day and shown in that day's
                                baseline and injection session (vehicle-day baseline, after vehicle, SBI-553-day baseline,
                                after SBI-553): how the SAME neurons' traces change.
  Every figure in English and Korean (<name>_kr.png); summary.json (numbers + slide text per figure), resp_tests.csv,
  groups_tracked.csv.

Why
  The PI slide 'How many neurons respond to each stimulus' gives only percentages; the trace row shows what an UP / DOWN neuron
  of each session looks like. User rule 10-23: % UP / DOWN may use every neuron of each day; cross-day comparisons of the
  same neurons use only the tracked neurons (each neuron paired with itself).

Inputs (all timing-corrected, 10-23)
  analysis\\event_analysis\\cells_classes.csv  (class per neuron x session x event: within-block time-shift test, 1000
      shuffles, BH q <= 0.20 per session; resp = response-window mean z - baseline-window mean z; window of the window search)
  analysis\\new_decks\\psth_cache_win3_tf.npz   (trial-mean z per neuron, minus its own mean over -3..0 s; -5..8 s, 30 Hz)
  analysis\\crossday\\<mouse>_pairs.csv         (tracked neurons: day1_id, day2_id, track_class A / B)
  windows: winwords.of(event) -> (pre, gap, a, b)
Drug order (crossover, confounded with day): Day 1 (20261019) SBI-553 = 5941, 5939, vehicle = 5937, 1191; Day 2 (20261021)
  the reverse.

Outputs  analysis\\pi_extra\\responsive\\  (nothing else is written)

Run
  python -I -X utf8 C:\\Users\\hsollim\\Documents\\HEAL_mini1p_SBI553\\mini1p_pipeline\\ca_responsive_traces.py
"""
import json
import os
import sys
import textwrap
import warnings

import numpy as np
import pandas as pd
from scipy.stats import binomtest, fisher_exact, wilcoxon

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea, VPacker

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import safe_io  # noqa: E402,F401
import figfoot  # noqa: E402  (10-24: the methods text goes to the slide notes, <name>.png.notes.txt)
import pairstars as PS  # noqa: E402  (every star shows the pair it compares)
import winwords as WW  # noqa: E402  (the response window of the event analysis)

warnings.filterwarnings("ignore", category=RuntimeWarning)
B = r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553"
EA = os.path.join(B, "analysis", "event_analysis")
CACHE = os.environ.get("PSTH_CACHE", os.path.join(B, "analysis", "new_decks", "psth_cache_win3_tf.npz"))   # 10-24: variants
CD = os.path.join(B, "analysis", "crossday")
OUT = os.path.join(B, "analysis", "pi_extra", "responsive")
MO = ["5941", "5939", "5937", "1191"]
SEX = {"5941": "F", "1191": "F", "5939": "M", "5937": "M"}
ARMS = {1: {"5941": "SBI-553", "5939": "SBI-553", "5937": "vehicle", "1191": "vehicle"},
        2: {"5941": "vehicle", "5939": "vehicle", "5937": "SBI-553", "1191": "SBI-553"}}
CC = {"baseline": "#7f8c8d", "vehicle": "#2471a3", "SBI-553": "#c0392b"}
MC = {"5941": "#e67e22", "5939": "#8e44ad", "5937": "#16a085", "1191": "#34495e"}
MK = {"5941": "o", "5939": "s", "5937": "^", "1191": "D"}
KC = {"UP": "#e74c3c", "DOWN": "#2e86c1", "rest": "#aab2b9"}     # rest: house #c8cdd2, one step darker for a visible line
STIM = ["Light touch", "Mild touch", "Pin prick", "Heat"]
FAM = {"stim": STIM, "behA": ["withdrawal", "flinch", "movement onset"],
       "behB": ["escape/rearing onset", "grooming onset", "attending onset"]}
ALL_EV = STIM + FAM["behA"] + FAM["behB"]
G4 = ["base_veh", "vehicle", "base_sbi", "SBI-553"]
X4 = {"base_veh": 0.0, "vehicle": 1.0, "base_sbi": 2.35, "SBI-553": 3.35}
C4 = {"base_veh": CC["baseline"], "vehicle": CC["vehicle"], "base_sbi": CC["baseline"], "SBI-553": CC["SBI-553"]}
DAYP = {"veh": ("base_veh", "vehicle"), "sbi": ("base_sbi", "SBI-553")}
W_IN, H_IN, DPI = 15.0, 6.3, 150
# 10-24: the deck scales a 15 x 6.3 in figure by 0.825 into its 12.4 x 5.2 in box -> every font >= 11 pt here (>= 9 pt on the slide)
FS = 11.5
GFS = 12.5                            # the GROUPS figures (fewer, larger panels)
KEY_FS = 11.5                         # the one-line star key that figfoot draws
TR_X = (-2.0, 4.0)                    # trace x range under the bars (narrow panels)
GR_X = (-3.0, 6.0)                    # trace x range of the GROUPS figures (house style)
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12.5, "xtick.labelsize": 12, "ytick.labelsize": 12,
                     "legend.fontsize": 12, "axes.spines.top": False, "axes.spines.right": False})

TXT = {   # key: (English, Korean)
    "base_veh": ("base\n(veh day)", "기준\n(veh 날)"), "vehicle": ("veh", "veh"), "base_sbi": ("base\n(SBI day)", "기준\n(SBI 날)"),
    "SBI-553": ("SBI", "SBI"),
    "veh_day": ("vehicle day", "vehicle 날"), "sbi_day": ("SBI-553 day", "SBI-553 날"),
    "ylab_bar": ("% of neurons\nUP (solid) / DOWN (hatched)", "뉴런 %\nUP (채움) / DOWN (빗금)"),
    # 10-24: what the traces are (psth cache: z = dF/F / robust SD of the session; every trial minus its own mean over -3..0 s)
    "ylab_tr": ("z (dF/F / robust SD),\nminus mean of 3 s before", "z (dF/F / robust SD),\n직전 3 s 평균을 뺌"),
    "noud": ("no UP / DOWN\nneurons", "UP / DOWN\n뉴런 없음"),
    "ylab_gr": ("z (dF/F / robust SD),\nminus the mean of the 3 s before", "z (dF/F / robust SD),\n사건 직전 3 s 평균을 뺌"),
    "veh_day_s": ("veh day", "veh 날"), "sbi_day_s": ("SBI day", "SBI 날"),
    "B_key": ("n = neurons, UP / DOWN = how many of them; each: baseline → after the injection",
              "n = 뉴런 수, UP / DOWN = 그중 수; 각각 기준 → 주사 후"),
    "ev_on": ("event onset (t = 0)", "사건 시작 (t = 0)"),
    "sem_band": ("band = ± SEM over neurons", "띠 = 뉴런 간 ± SEM"),
    "solid_b": ("solid = this day's baseline session", "실선 = 그날 기준 세션"),
    "dash_i": ("dashed = after the injection (same neurons)", "파선 = 주사 후 (같은 뉴런)"),
    "sex": ("F / M = sex", "F / M = 성별"),
    "time": ("time from event (s)", "사건부터 시간 (s)"),
    "noev": ("< 5\nevents", "사건\n< 5"),
    "notr": ("no session with\n>= 5 events", "사건 5번 이상\n세션 없음"),
    "up_s": ("% UP (solid)", "% UP (채움)"), "dn_h": ("% DOWN (hatched)", "% DOWN (빗금)"),
    "base_line": ("baseline session", "기준 세션"), "veh_line": ("after vehicle", "vehicle 후"),
    "sbi_line": ("after SBI-553", "SBI-553 후"), "up_line": ("UP neurons (solid)", "UP 뉴런 (실선)"),
    "dn_line": ("DOWN neurons (dashed)", "DOWN 뉴런 (파선)"), "rw": ("response window", "반응 창"),
    "bw": ("baseline window", "기준 창"), "mk": ("marker = mouse:", "표식 = 마우스:"),
    "traces": ("traces (B):", "trace (B):"), "bars": ("bars (A):", "막대 (A):"),
    "A_all": ("A  % of neurons UP / DOWN - every neuron of each day", "A  UP / DOWN 뉴런 % - 그날의 모든 뉴런"),
    "A_tr": ("A  % of neurons UP / DOWN - the 185 tracked neurons only", "A  UP / DOWN 뉴런 % - 추적 뉴런 185개만"),
    "B_all": ("B  mean trace of those UP and DOWN neurons in each session", "B  각 세션 UP / DOWN 뉴런의 평균 trace"),
    "baseline": ("baseline", "기준"), "vehicle_c": ("vehicle", "vehicle"), "SBI-553_c": ("SBI-553", "SBI-553"),
    "rest": ("rest", "나머지"),
    "vb": ("vehicle-day baseline", "vehicle 날 기준"), "vi": ("after vehicle", "vehicle 후"),
    "sb": ("SBI-553-day baseline", "SBI-553 날 기준"), "si": ("after SBI-553", "SBI-553 후"),
    "grp_by_vb": ("neurons grouped by their class at this day's baseline", "그날 기준 세션의 분류로 묶은 뉴런"),
    "grp_by_sb": ("neurons grouped by their class at this day's baseline", "그날 기준 세션의 분류로 묶은 뉴런"),
    "noref": ("this session exists,\nbut no baseline session with\n>= 5 events to group by",
              "세션은 있으나 묶을 기준 세션\n(사건 5번 이상)이 없음"),
    "nosess": ("no session with >= 5 events\nin this condition", "이 조건에 사건 5번 이상\n세션 없음"),
}


def T_(k, lg):
    return TXT[k][1 if lg == "kr" else 0]


def font(lg):
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"] if lg == "kr" else ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def ptxt(p):
    return "n.a." if p is None or not np.isfinite(p) else "p < 0.001" if p < 0.001 else f"p = {p:.3f}" if p < 0.01 else f"p = {p:.2f}"


def mtxt(n):
    return "1 mouse" if n == 1 else f"{n} mice"


def fnum(x, f="{:.0f}"):
    return "n.a." if x is None or not np.isfinite(x) else f.format(x)


def jclean(o):
    if isinstance(o, dict):
        return {str(k): jclean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jclean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 5)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


# ------------------------------------------------------------------------------------------------ data
def load():
    T = pd.read_csv(os.path.join(EA, "cells_classes.csv"))
    T["mouse"] = T.mouse.astype(str)
    T["drug"] = [ARMS[int(d)][m] for m, d in zip(T.mouse, T.day)]
    T["g4"] = np.where(T.phase == "pre", np.where(T.drug == "vehicle", "base_veh", "base_sbi"), T.drug)
    T["k"] = T.cls.map({"activated": "UP", "inhibited": "DOWN"}).fillna("rest")
    rows = []
    for m in MO:
        P = pd.read_csv(os.path.join(CD, f"{m}_pairs.csv"))
        for p in P.itertuples():
            nid = f"{m}_{int(p.day1_id)}_{int(p.day2_id)}"
            rows += [(m, 1, int(p.day1_id), nid, p.track_class), (m, 2, int(p.day2_id), nid, p.track_class)]
    K = pd.DataFrame(rows, columns=["mouse", "day", "cell", "nid", "track_class"])
    T = T.merge(K, on=["mouse", "day", "cell"], how="left")
    T["tracked"] = T.nid.notna()
    D = np.load(CACHE, allow_pickle=True)["D"].item()
    IDX = {k: {int(c): j for j, c in enumerate(v["ids"])} for k, v in D.items() if v is not None}
    problems = []
    for (m, d, ph, ev), Z in T.groupby(["mouse", "day", "phase", "event"]):
        v = D.get((m, int(d), ph, ev))
        if v is None:
            problems.append(f"{m} Day {d} {ph} {ev}: classes exist but the trace cache has no entry")
            continue
        miss = set(Z.cell.astype(int)) - set(IDX[(m, int(d), ph, ev)])
        if miss:
            problems.append(f"{m} Day {d} {ph} {ev}: {len(miss)} classified neurons missing from the trace cache")
        if int(Z.n_trials.iloc[0]) != int(v["n"]):
            problems.append(f"{m} Day {d} {ph} {ev}: {int(Z.n_trials.iloc[0])} trials in the classes vs {int(v['n'])} in the cache")
    for (m, d, ph, ev), v in D.items():
        if v is not None and not ((T.mouse == m) & (T.day == d) & (T.phase == ph) & (T.event == ev)).any():
            problems.append(f"{m} Day {d} {ph} {ev}: trace cache entry without classes")
    n_tr = K.nid.nunique()
    tt = next(v["tt"] for v in D.values() if v is not None)
    return T, D, IDX, tt, problems, n_tr


def tr_mat(R, D, IDX):
    """time x neurons trial-mean traces of the rows R (mouse, day, phase, event, cell)"""
    cols = []
    for r in R.itertuples():
        key = (r.mouse, int(r.day), r.phase, r.event)
        v = D.get(key)
        if v is None:
            continue
        j = IDX[key].get(int(r.cell))
        if j is not None:
            cols.append(v["mean"][:, j])
    return np.array(cols, float).T if cols else None


def sm(y, k=6):
    """0.2-s running mean (display only, as CA.sm)"""
    return pd.Series(y).rolling(k, center=True, min_periods=1).mean().values


def msem(Y):
    n = Y.shape[1]
    mu = sm(np.nanmean(Y, 1))
    se = sm(np.nanstd(Y, 1, ddof=1) / np.sqrt(n)) if n > 1 else np.zeros(len(mu))
    return mu, se


def wins(ev):
    pre, gap, a, b = WW.of(ev)
    a0 = min(a, 0.0) - gap
    return (a0 - pre, a0), (a, b)


def wtxt(ev, lg="en"):
    (p0, p1), (a, b) = wins(ev)
    return f"{a:g}-{b:g} s vs {p0:g}..{p1:g} s" if lg == "en" else f"{a:g}-{b:g} s vs {p0:g}..{p1:g} s"


def mcn(b, c):
    n = int(b + c)
    return float(binomtest(int(b), n, 0.5).pvalue) if n else np.nan


def wsr(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) < 6 or np.all(d == 0):
        return np.nan
    return float(wilcoxon(d, zero_method="wilcox").pvalue)


# ------------------------------------------------------------------------------------------------ stats
def stats_event(X, tracked):
    """counts and tests of one event (X = its rows; all neurons or the tracked ones)"""
    r = {"n_mice": int(X.mouse.nunique())}
    for g in G4:
        Y = X[X.g4 == g]
        n = len(Y)
        up, dn = int((Y.k == "UP").sum()), int((Y.k == "DOWN").sum())
        r[g] = dict(n=n, up=up, down=dn, pct_up=100 * up / n if n else np.nan, pct_down=100 * dn / n if n else np.nan,
                    resp_up=float(Y[Y.k == "UP"].resp.mean()) if up else np.nan,
                    resp_down=float(Y[Y.k == "DOWN"].resp.mean()) if dn else np.nan,
                    mice={m: dict(n=int((Y.mouse == m).sum()), pct_up=100 * float((Y[Y.mouse == m].k == "UP").mean()),
                                  pct_down=100 * float((Y[Y.mouse == m].k == "DOWN").mean())) for m in MO if (Y.mouse == m).any()})
    for drug, (gb, gi) in (("vehicle", DAYP["veh"]), ("SBI-553", DAYP["sbi"])):
        M = X[X.g4 == gb].merge(X[X.g4 == gi], on=["mouse", "day", "cell"], suffixes=("_b", "_i"))
        for k in ("UP", "DOWN"):
            b, c = int(((M.k_b == k) & (M.k_i != k)).sum()), int(((M.k_b != k) & (M.k_i == k)).sum())
            r[f"mcnemar_{k}_{drug}"] = dict(p=mcn(b, c), lost=b, gained=c, n_pairs=len(M))
    for k in ("UP", "DOWN"):
        if tracked:
            M = X[X.g4 == "vehicle"].merge(X[X.g4 == "SBI-553"], on="nid", suffixes=("_v", "_s"))
            b, c = int(((M.k_v == k) & (M.k_s != k)).sum()), int(((M.k_v != k) & (M.k_s == k)).sum())
            r[f"vs_{k}"] = dict(test="McNemar (same neurons)", p=mcn(b, c), only_vehicle=b, only_sbi=c, n_pairs=len(M))
        else:
            nv, ns = r["vehicle"]["n"], r["SBI-553"]["n"]
            kv = r["vehicle"]["up" if k == "UP" else "down"]
            ks = r["SBI-553"]["up" if k == "UP" else "down"]
            p = float(fisher_exact([[ks, ns - ks], [kv, nv - kv]])[1]) if nv and ns else np.nan
            r[f"vs_{k}"] = dict(test="Fisher exact (neurons pooled)", p=p, k_vehicle=kv, n_vehicle=nv, k_sbi=ks, n_sbi=ns)
    if tracked:                                   # baseline-normalized: per neuron (UP after - UP before) SBI day vs vehicle day
        W = {}
        for g in G4:
            Z = X[X.g4 == g].set_index("nid").k
            W[g] = Z[~Z.index.duplicated()]
        common = W["base_veh"].index.intersection(W["vehicle"].index).intersection(W["base_sbi"].index).intersection(W["SBI-553"].index)
        for k in ("UP", "DOWN"):
            f = {g: (W[g].loc[common] == k).astype(int).values for g in G4}
            dv, ds = f["vehicle"] - f["base_veh"], f["SBI-553"] - f["base_sbi"]
            r[f"norm_{k}"] = dict(n=len(common), chg_vehicle_pts=100 * float(dv.mean()) if len(common) else np.nan,
                                  chg_sbi_pts=100 * float(ds.mean()) if len(common) else np.nan,
                                  p_wilcoxon=wsr(ds - dv), test="Wilcoxon signed-rank on (UP after - UP before) per neuron, "
                                                               "SBI-553 day vs vehicle day")
    return r


# ------------------------------------------------------------------------------------------------ drawing helpers
def ctext(a, pieces, loc, fs=FS):
    """coloured pieces of text in one line, anchored inside the axes (loc as for a legend)"""
    boxes = [TextArea(s, textprops=dict(color=c, fontsize=fs, fontweight="bold" if c != "k" else "normal")) for s, c in pieces]
    box = HPacker(children=boxes, align="baseline", pad=0, sep=0)
    ab = AnchoredOffsetbox(loc=loc, child=box, pad=0.12, frameon=True, borderpad=0.15)
    ab.patch.set(facecolor="w", edgecolor="none", alpha=0.85)
    ab.set_zorder(6)
    a.add_artist(ab)


def wrap_px(fig, txt, fs, width_frac=0.98):
    """wrap txt to a share of the figure width, measured with the renderer (works for Korean too)"""
    r = fig.canvas.get_renderer()
    maxw = width_frac * fig.get_figwidth() * fig.dpi
    out = []
    for para in txt.split("\n"):
        words, line = para.split(" "), ""
        for w in words:
            cand = (line + " " + w).strip()
            t = fig.text(0, 0, cand, fontsize=fs)
            wpx = t.get_window_extent(renderer=r).width
            t.remove()
            if wpx > maxw and line:
                out.append(line)
                line = w
            else:
                line = cand
        out.append(line)
    return "\n".join(out)


def foot(fig, txt, fs=KEY_FS, y=0.006, lg=None):
    """10-24: the methods text goes to the slide notes (figfoot writes <name>.png.notes.txt next to the PNG); the figure keeps
    only the one-line star key when the text has one. Returns the number of lines left in the figure (0 or 1)."""
    return 0 if figfoot.note(fig, txt, lg, fs=fs) is None else 1


RW_C, BW_C, ON_C = "0.86", "#dbe9f6", "0.35"     # response window, baseline window, event-onset line


def shade(a, ev):
    (p0, p1), (ra, rb) = wins(ev)
    a.axvspan(ra, rb, color=RW_C, zorder=0, lw=0)
    a.axvspan(p0, p1, color=BW_C, zorder=0, lw=0)
    a.axvline(0, color=ON_C, lw=1.0, ls="--", zorder=1)
    a.axhline(0, color="0.6", lw=0.6, zorder=1)


def rng(a, b, lg):
    return (f"{a:g} to {b:g} s" if lg == "en" else f"{a:g}~{b:g} s") if a < 0 else f"{a:g}-{b:g} s"


def sem_handle(lg):
    """round 2 (10-25): every trace carries a shaded mean +- SEM band -> say so in the legend"""
    # a line inside a light band (like the traces), so it cannot be mistaken for the grey response-window patch
    return (Patch(facecolor="0.45", alpha=0.25, edgecolor="none", label=T_("sem_band", lg)),
            Line2D([], [], color="0.3", lw=1.8))


def hl(handles):
    """handles + labels; a tuple handle (band + line) takes the label of its first element"""
    return list(handles), [(h[0] if isinstance(h, tuple) else h).get_label() for h in handles]


def band_handles(evs, lg):
    """legend handles of the shading: response window, baseline window (per event when the events differ), event onset"""
    out = []
    for i, k, col in ((1, "rw", RW_C), (0, "bw", BW_C)):
        G = {}
        for ev in evs:
            G.setdefault(wins(ev)[i], []).append(ev)
        ws = sorted(G, key=lambda w: -len(G[w]))
        s = rng(*ws[0], lg) + "".join(f"; {', '.join(G[w])} {rng(*w, lg)}" for w in ws[1:])
        out.append(Patch(color=col, label=f"{T_(k, lg)} ({s})"))
    out.append(Line2D([], [], color=ON_C, lw=1.2, ls="--", label=T_("ev_on", lg)))
    return out


def absfmt(a):
    a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{abs(v):g}"))


# ------------------------------------------------------------------------------------------------ figure 1 / 2
def fig_resp(T, D, IDX, tt, fam, tracked, lg, n_tr):
    evs = FAM[fam]
    X0 = T[T.tracked] if tracked else T
    font(lg)
    fig = plt.figure(figsize=(W_IN, H_IN))
    ne = len(evs)
    left, right, gap, sgap = 1.12, 0.12, 0.58, 0.16
    colw = (W_IN - left - right - gap * (ne - 1)) / ne
    # 10-24: no footer block any more -> taller trace panels; the UP / DOWN counts sit in a header above each trace panel
    bar_top, bar_h = H_IN - 0.84, 1.55          # inches from the bottom of the figure
    tr_top, tr_h = bar_top - bar_h - 1.36, 1.32
    trw = (colw - sgap) / 2
    S, axb, axt = {}, [], []
    # shared y of the bars: highest % UP / DOWN (bars and mouse markers) over the events
    gtop, gbot = 5.0, 5.0
    for ev in evs:
        X = X0[X0.event == ev]
        for g in G4:
            Y = X[X.g4 == g]
            for m in [None] + MO:
                Z = Y if m is None else Y[Y.mouse == m]
                if len(Z):
                    gtop = max(gtop, 100 * (Z.k == "UP").mean()); gbot = max(gbot, 100 * (Z.k == "DOWN").mean())
    R = gtop + gbot
    curves = []
    for i, ev in enumerate(evs):
        X = X0[X0.event == ev]
        x0 = left + i * (colw + gap)
        a = fig.add_axes([x0 / W_IN, (bar_top - bar_h) / H_IN, colw / W_IN, bar_h / H_IN])
        axb.append(a)
        st = stats_event(X, tracked)
        S[ev] = st
        for g in G4:
            q = st[g]
            if not q["n"]:
                a.text(X4[g], 0, T_("noev", lg), ha="center", va="center", fontsize=FS, color="0.4", zorder=4,
                       bbox=dict(facecolor="w", edgecolor="none", pad=1.5))
                continue
            a.bar(X4[g], q["pct_up"], color=C4[g], width=0.7)
            a.bar(X4[g], -q["pct_down"], color=C4[g], width=0.7, alpha=0.45, hatch="//", edgecolor="w")
            for j, m in enumerate(MO):
                if m in q["mice"]:
                    a.scatter(X4[g] + (j - 1.5) * 0.13, q["mice"][m]["pct_up"], marker=MK[m], color=MC[m], s=24, zorder=3,
                              edgecolor="w", lw=0.4)
                    a.scatter(X4[g] + (j - 1.5) * 0.13, -q["mice"][m]["pct_down"], marker=MK[m], color=MC[m], s=24, zorder=3,
                              edgecolor="w", lw=0.4)
        a.axhline(0, color="k", lw=0.8)
        a.set_ylim(-gbot - 0.56 * R, gtop + 0.40 * R)     # round 2: room under the lowest bracket label (clear of the spine)
        a.set_xlim(-0.6, 3.95)
        a.set_xticks([X4[g] for g in G4], [T_(g, lg) for g in G4], fontsize=FS)
        a.tick_params(axis="x", pad=2)
        a.tick_params(axis="y", labelsize=FS)
        for tl, g in zip(a.get_xticklabels(), G4):
            tl.set_color(C4[g] if g in ("vehicle", "SBI-553") else "0.3")
        absfmt(a)
        hb = 0.022 * (2 * 0.40 * R + R)
        for k, y1, y2, below in (("UP", gtop + 0.06 * R, gtop + 0.25 * R, False), ("DOWN", -gbot - 0.06 * R, -gbot - 0.25 * R, True)):
            PS.bracket(a, X4["base_veh"], X4["vehicle"], y1, st[f"mcnemar_{k}_vehicle"]["p"], CC["vehicle"], fs=FS, below=below, h=hb)
            PS.bracket(a, X4["base_sbi"], X4["SBI-553"], y1, st[f"mcnemar_{k}_SBI-553"]["p"], CC["SBI-553"], fs=FS, below=below, h=hb)
            PS.bracket(a, X4["vehicle"], X4["SBI-553"], y2, st[f"vs_{k}"]["p"], "k", fs=FS, below=below, h=hb)
        a.set_title(ev, fontsize=13.5, pad=9)
        if i == 0:
            a.set_ylabel(T_("ylab_bar", lg), fontsize=FS)
        else:
            a.set_yticklabels([])
        # traces: vehicle day (left), SBI-553 day (right)
        for s_, (dk, (gb, gi)) in enumerate(DAYP.items()):
            b = fig.add_axes([(x0 + s_ * (trw + sgap)) / W_IN, (tr_top - tr_h) / H_IN, trw / W_IN, tr_h / H_IN])
            axt.append(b)
            shade(b, ev)
            nn = {}
            for g in (gb, gi):
                col = CC["baseline"] if g.startswith("base") else CC[g]
                for k, ls in (("UP", "-"), ("DOWN", "--")):
                    Y = tr_mat(X[(X.g4 == g) & (X.k == k)], D, IDX)
                    nn[(g, k)] = 0 if Y is None else Y.shape[1]
                    if Y is None:
                        continue
                    mu, se = msem(Y)
                    b.plot(tt, mu, color=col, ls=ls, lw=1.9, zorder=3)
                    b.fill_between(tt, mu - se, mu + se, color=col, alpha=0.18, lw=0, zorder=2)
                    win = (tt >= TR_X[0]) & (tt <= TR_X[1])
                    curves.append((i, np.nanmax((mu + se)[win]), np.nanmin((mu - se)[win])))
                    S[ev].setdefault("trace_n", {})[f"{g}_{k}"] = int(Y.shape[1])
            b.set_xlim(*TR_X)
            b.set_xticks([-1, 0, 1, 2, 3])
            b.tick_params(axis="x", labelsize=FS, pad=2)
            nb, ni = st[gb]["n"], st[gi]["n"]
            ntx = f"n = {nb}" if nb == ni else f"n = {nb} → {ni}"
            if nb + ni == 0 or all(nn.get((g, k), 0) == 0 for g in (gb, gi) for k in ("UP", "DOWN")):
                b.text(0.62, 0.5, T_("notr", lg) if nb + ni == 0 else T_("noud", lg), transform=b.transAxes, ha="center",
                       va="center", fontsize=FS, color="0.4", zorder=5, bbox=dict(facecolor="w", edgecolor="none", pad=1.5))
            # 10-24: header ABOVE the panel (nothing on the traces): day + n, then the UP and DOWN counts, baseline -> injection
            icol = CC[gi]
            dcol = CC["vehicle" if dk == "veh" else "SBI-553"]
            rows = [HPacker(children=[TextArea(f"{T_('veh_day_s' if dk == 'veh' else 'sbi_day_s', lg)}, {ntx}",
                                               textprops=dict(color=dcol, fontsize=FS))], pad=0, sep=0)]
            for k in ("UP", "DOWN"):
                rows.append(HPacker(children=[TextArea(f"{k} ", textprops=dict(color="k", fontsize=FS)),
                                              TextArea(str(nn.get((gb, k), 0)), textprops=dict(color=CC["baseline"], fontsize=FS,
                                                                                               fontweight="bold")),
                                              TextArea(" → ", textprops=dict(color="k", fontsize=FS)),
                                              TextArea(str(nn.get((gi, k), 0)), textprops=dict(color=icol, fontsize=FS,
                                                                                               fontweight="bold"))],
                                    align="baseline", pad=0, sep=0))
            hb_ = AnchoredOffsetbox(loc="lower center", child=VPacker(children=rows, align="center", pad=0, sep=2.5), frameon=False,
                                    bbox_to_anchor=(0.5, 1.0), bbox_transform=b.transAxes, pad=0, borderpad=0.25)
            b.add_artist(hb_)
            if i == 0 and s_ == 0:
                b.set_ylabel(T_("ylab_tr", lg), fontsize=FS, labelpad=3)
        fig.text((x0 + colw / 2) / W_IN, (tr_top - tr_h - 0.27) / H_IN, T_("time", lg), fontsize=FS, ha="center", va="top")
    # y of the traces: shared by the two day panels of one event (each event its own scale); every curve fully inside
    for i in range(ne):
        ci = [c for c in curves if c[0] == i]
        hi = max(c[1] for c in ci) if ci else 1.0
        lo = min(c[2] for c in ci) if ci else -1.0
        hi, lo = max(hi, 0.05), min(lo, -0.05)
        pad = 0.30 * (hi - lo)
        for s_ in (0, 1):
            b = axt[2 * i + s_]
            b.set_ylim(lo - pad, hi + pad)
            b.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3, symmetric=True))
            b.tick_params(axis="y", labelsize=FS)
            if s_ == 1:
                b.set_yticklabels([])
    # panel labels, legends, notes
    fig.text(0.006, (bar_top + 0.50) / H_IN, T_("A_tr" if tracked else "A_all", lg), fontsize=12.5, fontweight="bold",
             ha="left", va="bottom")
    tB = fig.text(0.006, (tr_top + 0.74) / H_IN, T_("B_all", lg), fontsize=12.5, fontweight="bold", ha="left", va="bottom")
    fig.canvas.draw()
    xb = tB.get_window_extent().x1 / (fig.dpi * W_IN)
    fig.text(xb + 0.012, (tr_top + 0.74) / H_IN, "(" + T_("B_key", lg) + ")", fontsize=FS, color="0.2", ha="left", va="bottom")
    h1 = [Patch(facecolor="0.45", label=T_("up_s", lg)), Patch(facecolor="0.82", hatch="//", edgecolor="w", label=T_("dn_h", lg)),
          Patch(visible=False, label=T_("sex", lg) + ":")]
    h1 += [Line2D([], [], ls="", marker=MK[m], color=MC[m], ms=7.5, label=f"{m} {SEX[m]}") for m in MO]
    fig.legend(handles=h1, loc="lower right", bbox_to_anchor=(1 - right / W_IN, (bar_top + 0.47) / H_IN), ncol=7, frameon=False,
               fontsize=FS, handlelength=1.4, handletextpad=0.5, columnspacing=0.9, borderaxespad=0)
    tr_bot = tr_top - tr_h
    h2 = [Line2D([], [], color=CC["baseline"], lw=2.2, label=T_("base_line", lg)),
          Line2D([], [], color=CC["vehicle"], lw=2.2, label=T_("veh_line", lg)),
          Line2D([], [], color=CC["SBI-553"], lw=2.2, label=T_("sbi_line", lg)),
          Line2D([], [], color="k", lw=1.9, label=T_("up_line", lg)), Line2D([], [], color="k", lw=1.9, ls="--", label=T_("dn_line", lg))]
    h3 = band_handles(evs, lg) + [sem_handle(lg)]
    if lg == "kr":
        key = ("기준 = 그날의 기준 세션. 괄호: 파랑 / 빨강 = 그날 기준 vs 주사 후, 같은 뉴런 (McNemar); 검은 괄호 = vehicle 후 vs "
               "SBI-553 후 (" + ("McNemar, 같은 추적 뉴런" if tracked else "Fisher exact, 뉴런 합산 = 탐색적") + "); 막대 위 괄호 = % UP, "
               "아래 = % DOWN. UP / DOWN = 뉴런별 블록 내 시간 이동 검정, 세션 안 BH q <= 0.20. Trace = 뉴런 평균 ± SEM, 0.2 s 평활, 사건마다 y 축척; "
               "n 숫자 = 기준 → 주사 세션. * p < 0.05, ** p < 0.01, *** p < 0.001, ns = 유의하지 않음.")
    else:
        key = ("base = the baseline session of that day. Brackets: blue / red = that day's baseline vs after the injection, the same "
               "neurons (McNemar); black = after vehicle vs after SBI-553 (" + ("McNemar, the same tracked neurons" if tracked else
               "Fisher exact, neurons pooled = exploratory") + "); brackets above the bars test % UP, below test % DOWN. UP / DOWN = "
               "within-block time-shift test per neuron, BH q <= 0.20 per session. Traces = mean ± SEM over neurons, 0.2-s smoothing, y scale per event; "
               "n numbers = baseline → injection session. Trace y = z (dF/F / robust SD of the session), each trial minus its "
               "own mean over the 3 s before the event, then averaged over trials. " + PS.KEY + ".")
    if lg == "kr":
        key = key.replace("Trace = 뉴런 평균", "Trace y = z (dF/F / 세션 robust SD), trial마다 사건 직전 3 s 평균을 빼고 trial 평균; Trace = 뉴런 평균")
    nl = foot(fig, key, lg=lg)
    # two legend rows under the traces: colours / line styles, then the shading (window per event when the events differ)
    lg2 = fig.legend(handles=h2, loc="lower left", bbox_to_anchor=(0.005, 0.49 / H_IN), ncol=5, frameon=False,
                     fontsize=FS, handlelength=1.8, columnspacing=1.2, borderaxespad=0)
    lg3 = fig.legend(*hl(h3), loc="lower left", bbox_to_anchor=(0.005, 0.26 / H_IN), ncol=4, frameon=False,
                     fontsize=FS, handlelength=1.8, columnspacing=1.2, borderaxespad=0)
    fig.canvas.draw()                                   # check: the legends must sit above the star key and inside the figure
    lb = lg3.get_window_extent().y0 / fig.dpi
    ftop = 0.05 + nl * KEY_FS * 1.25 / 72
    for L_ in (lg2, lg3):
        if L_.get_window_extent().x1 / fig.dpi > W_IN:
            print(f"WARNING {fam} tracked={tracked} {lg}: legend runs past the right edge")
    if lb < ftop:
        print(f"WARNING {fam} tracked={tracked} {lg}: legend bottom {lb:.2f} in < star key top {ftop:.2f} in")
    name = f"RESP_{fam}" + ("_tracked" if tracked else "") + ("_kr" if lg == "kr" else "")
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=DPI)
    plt.close(fig)
    return S


# ------------------------------------------------------------------------------------------------ figure 3
def gr_legend(a, handles, lg):
    a.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=GFS - 0.5, handlelength=1.5,
             handletextpad=0.5, columnspacing=1.2, borderaxespad=0.15, labelspacing=0.25)


def gr_axes(a, lg):
    """10-24: x ticks every 2 s from -2 (no '-3' next to the neighbour's '6'), minor ticks every 1 s; readable sizes"""
    a.set_xlim(*GR_X)
    a.set_xticks([-2, 0, 2, 4, 6])
    a.set_xticks(np.arange(GR_X[0], GR_X[1] + 0.1, 1.0), minor=True)
    a.tick_params(axis="both", labelsize=GFS - 0.5)
    a.set_xlabel(T_("time", lg), fontsize=GFS)


def gr_bottom_legend(fig, handles, ncol):
    """the key of the shading (and line styles) under the panels, centred"""
    hs_, ls_ = hl(handles)
    fig.legend(hs_, ls_, loc="lower center", bbox_to_anchor=(0.5, 0.06 / H_IN), ncol=ncol, frameon=False,
               fontsize=GFS - 0.5, handlelength=2.0, columnspacing=1.6, borderaxespad=0)


def fig_groups(T, D, IDX, tt, ev, lg):
    """every neuron: baseline / vehicle / SBI-553, the UP, DOWN and other neurons' mean traces"""
    font(lg)
    X = T[T.event == ev]
    nm = int(X.mouse.nunique())
    fig, ax = plt.subplots(1, 3, figsize=(W_IN, H_IN), sharey=True)
    res, curves = {"n_mice": nm}, []
    conds = [("baseline", X.phase == "pre"), ("vehicle", X.g4 == "vehicle"), ("SBI-553", X.g4 == "SBI-553")]
    for c_, (c, sel) in enumerate(conds):
        a = ax[c_]
        Xc = X[sel]
        shade(a, ev)
        hs = []
        res[c] = dict(n=int(len(Xc)))
        for k in ("UP", "DOWN", "rest"):
            Y = tr_mat(Xc[Xc.k == k], D, IDX)
            n = 0 if Y is None else Y.shape[1]
            res[c][k] = dict(n=n, pct=100 * n / len(Xc) if len(Xc) else np.nan,
                             resp=float(Xc[Xc.k == k].resp.mean()) if n else np.nan,
                             mice=sorted(Xc[Xc.k == k].mouse.unique().tolist()))
            hs.append(Line2D([], [], color=KC[k], lw=2.4, label=f"{T_('rest', lg) if k == 'rest' else k} (n = {n})"))
            if Y is None:
                continue
            mu, se = msem(Y)
            a.plot(tt, mu, color=KC[k], lw=2.2, zorder=3)
            a.fill_between(tt, mu - se, mu + se, color=KC[k], alpha=0.2, lw=0, zorder=2)
            w = (tt >= GR_X[0]) & (tt <= GR_X[1])
            curves.append((n, np.nanmax((mu + se)[w]), np.nanmin((mu - se)[w])))
        if not len(Xc):
            a.text(0.5, 0.5, T_("nosess", lg), transform=a.transAxes, ha="center", va="center", fontsize=FS + 1, color="0.4",
                   zorder=5, bbox=dict(facecolor="w", edgecolor="none", pad=2))
        else:
            gr_legend(a, hs, lg)
        gr_axes(a, lg)
        ttl = {"baseline": T_("baseline", lg), "vehicle": T_("vi", lg), "SBI-553": T_("si", lg)}[c]
        nses = "" if not len(Xc) else (f" - {len(Xc)} " + ("neuron-sessions" if lg == "en" else "뉴런-세션"))
        a.set_title(ttl + nses, color=CC[c], fontsize=13.5, fontweight="bold", pad=46)
    ax[0].set_ylabel(T_("ylab_gr", lg), fontsize=GFS)
    ylim_from(ax, curves)
    if lg == "kr":
        sup = f"{ev}: UP, DOWN, 나머지 뉴런의 평균 trace (그날의 모든 뉴런; 기준 = 두 날의 기준 세션)"
        key = (f"UP / DOWN = 반응 창({wtxt(ev)})의 뉴런별 블록 내 시간 이동 검정 (1000회), 세션 안 BH q <= 0.20; 나머지 = 무반응. 선 = 뉴런 평균 "
               "± SEM (각 뉴런의 trial 평균 z, 사건 직전 3 s 기준, 0.2 s 평활); 회색 띠 = 반응 창, 파랑 띠 = 기준 창. 마우스 " + str(nm) + ". 기술 통계 (검정 없음).")
    else:
        sup = f"{ev}: mean trace of the UP, DOWN and other neurons (every neuron of each day; baseline = both days' baseline sessions)"
        key = (f"UP / DOWN = within-block time-shift test per neuron in the response window ({wtxt(ev)}; 1000 shuffles), BH q <= 0.20 per "
               "session; rest = non-responsive. Lines = mean ± SEM over neurons (each neuron's trial-mean z minus the 3 s before the "
               "event, 0.2-s smoothing; z = dF/F / robust SD of the session); grey band = response window, blue band = baseline window. " + mtxt(nm) + ". Descriptive (no test).")
    fig.suptitle(sup, fontsize=13.5, y=0.985)
    foot(fig, key, lg=lg)                       # -> slide notes
    gr_bottom_legend(fig, band_handles([ev], lg) + [sem_handle(lg)], 4)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.785, bottom=1.02 / H_IN, wspace=0.12)
    fig.savefig(os.path.join(OUT, f"GROUPS_{tag(ev)}" + ("_kr" if lg == "kr" else "") + ".png"), dpi=DPI)
    plt.close(fig)
    return res


def fig_groups_tracked(T, D, IDX, tt, ev, lg):
    """the tracked neurons grouped by their class in the baseline of each day, in that day's baseline and injection session"""
    font(lg)
    X = T[(T.event == ev) & T.tracked]
    nm = int(X.mouse.nunique())
    fig = plt.figure(figsize=(W_IN, H_IN))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 0.16, 1, 1], wspace=0.07)
    ax = [fig.add_subplot(gs[0, i]) for i in (0, 1, 3, 4)]
    res, curves = {"n_mice": nm}, []
    cols = [("vb", "base_veh", "base_veh"), ("vi", "vehicle", "base_veh"), ("sb", "base_sbi", "base_sbi"), ("si", "SBI-553", "base_sbi")]
    for c_, (cn, g, gref) in enumerate(cols):
        a = ax[c_]
        shade(a, ev)
        ref = X[X.g4 == gref][["mouse", "day", "cell", "k", "resp"]].rename(columns={"k": "k_ref", "resp": "resp_ref"})
        Xc = X[X.g4 == g].merge(ref, on=["mouse", "day", "cell"])
        hs = []
        res[cn] = dict(n=int(len(Xc)))
        for k in ("UP", "DOWN", "rest"):
            Z = Xc[Xc.k_ref == k]
            Y = tr_mat(Z, D, IDX)
            n = 0 if Y is None else Y.shape[1]
            res[cn][k] = dict(n=n, resp=float(Z.resp.mean()) if n else np.nan, resp_ref=float(Z.resp_ref.mean()) if n else np.nan,
                              p_wilcoxon_vs_baseline=wsr(Z.resp.values - Z.resp_ref.values) if g != gref else np.nan,
                              still_same=int((Z.k == k).sum()) if g != gref else n)
            lab = f"{T_('rest', lg) if k == 'rest' else k} (n = {n})"
            hs.append(Line2D([], [], color=KC[k], lw=2.4, label=lab, ls="-" if g == gref else (0, (5, 1.5))))
            if Y is None:
                continue
            mu, se = msem(Y)
            a.plot(tt, mu, color=KC[k], lw=2.2, zorder=3, ls="-" if g == gref else (0, (5, 1.5)))
            a.fill_between(tt, mu - se, mu + se, color=KC[k], alpha=0.2, lw=0, zorder=2)
            w = (tt >= GR_X[0]) & (tt <= GR_X[1])
            curves.append((n, np.nanmax((mu + se)[w]), np.nanmin((mu - se)[w])))
        if not len(Xc):
            no_ref = (X.g4 == g).any() and not (X.g4 == gref).any()
            a.text(0.5, 0.5, T_("noref" if no_ref else "nosess", lg), transform=a.transAxes, ha="center", va="center",
                   fontsize=GFS - 0.5, color="0.4", zorder=5, bbox=dict(facecolor="w", edgecolor="none", pad=2))
        else:
            a.legend(handles=hs, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=GFS - 1, handlelength=1.5,
                     handletextpad=0.4, columnspacing=0.6, borderaxespad=0.15, labelspacing=0.25)
        gr_axes(a, lg)
        ccol = {"vb": CC["baseline"], "vi": CC["vehicle"], "sb": CC["baseline"], "si": CC["SBI-553"]}[cn]
        a.set_title(T_(cn, lg), color=ccol, fontsize=13, fontweight="bold", pad=44)
    ax[0].set_ylabel(T_("ylab_gr", lg), fontsize=GFS)
    ylim_from(ax, curves)
    for a in (ax[1], ax[3]):                    # 10-24: the SBI-553-day pair (after the gap) keeps its y tick labels
        a.tick_params(labelleft=False)
    ntr = X.nid.nunique()
    if lg == "kr":
        sup = f"{ev}: 같은 추적 뉴런 ({ntr}개), 그날 기준의 UP / DOWN / 나머지로 묶어 기준과 주사 세션에 표시"
        key = (f"추적 뉴런 = 두 날 모두 있는 같은 세포 (analysis/crossday). 그룹 = 그날 기준 세션의 분류 (반응 창 {wtxt(ev)}, 시간 이동 검정, "
               "BH q <= 0.20); 주사 열(파선)은 같은 뉴런을 같은 그룹으로 표시. 선 = 뉴런 평균 ± SEM (trial 평균 z, 직전 3 s 기준, 0.2 s 평활). "
               "주의: 기준에서 고른 그룹은 다음 세션에서 평균으로 회귀하므로, vehicle 날과 SBI-553 날의 변화를 서로 비교. 마우스 " + str(nm) + ". 기술 통계 (검정 없음).")
    else:
        sup = f"{ev}: the same tracked neurons ({ntr}), grouped by their class at the baseline of each day"
        key = (f"Tracked = the same cell on both days (analysis/crossday). Groups = the class in that day's baseline session (response window "
               f"{wtxt(ev)}, time-shift test, BH q <= 0.20); the injection columns (dashed) show the same neurons in the same groups. Lines "
               "= mean ± SEM over neurons (trial-mean z minus the 3 s before, 0.2-s smoothing). Note: groups chosen at baseline regress "
               "to the mean in the next session, so compare the vehicle-day change with the SBI-553-day change. " + mtxt(nm) + ". Descriptive (no test).")
    fig.suptitle(sup, fontsize=13.5, y=0.985)
    foot(fig, key, lg=lg)                       # -> slide notes
    ls_h = [Line2D([], [], color="0.25", lw=2.2, ls="-", label=T_("solid_b", lg)),
            Line2D([], [], color="0.25", lw=2.2, ls=(0, (5, 1.5)), label=T_("dash_i", lg))]
    gr_bottom_legend(fig, ls_h + band_handles([ev], lg) + [sem_handle(lg)], 3)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.74, bottom=1.30 / H_IN)
    for (a1, a2), hk, dk, col in (((ax[0], ax[1]), "grp_by_vb", "veh_day", CC["vehicle"]),
                                  ((ax[2], ax[3]), "grp_by_sb", "sbi_day", CC["SBI-553"])):
        p1, p2 = a1.get_position(), a2.get_position()
        xm = (p1.x0 + p2.x1) / 2
        fig.text(xm, p1.y1 + 0.135, f"{T_(dk, lg)}: {T_(hk, lg)}", ha="center", va="bottom", fontsize=12.5, color=col,
                 fontweight="bold")
        fig.add_artist(Line2D([p1.x0 + 0.01, p2.x1 - 0.01], [p1.y1 + 0.128] * 2, color=col, lw=1.2, transform=fig.transFigure))
    fig.savefig(os.path.join(OUT, f"GROUPS_{tag(ev)}_tracked" + ("_kr" if lg == "kr" else "") + ".png"), dpi=DPI)
    plt.close(fig)
    return res


def ylim_from(ax, curves):
    if not curves:
        return
    hi, lo = max(c[1] for c in curves), min(c[2] for c in curves)    # every curve fully inside (nothing cut off)
    pad = 0.08 * (hi - lo)
    for a in np.atleast_1d(ax):
        a.set_ylim(lo - pad, hi + pad)


def tag(ev):
    return ev.replace(" ", "_").replace("/", "_")


# ------------------------------------------------------------------------------------------------ summary text
DRUG_ORDER_EN = ("Drug order (crossover, confounded with day): Day 1 (20261019) SBI-553 = 5941 F, 5939 M, vehicle = 5937 M, 1191 F; "
                 "Day 2 (20261021) SBI-553 = 5937, 1191, vehicle = 5941, 5939 - 2 mice got SBI-553 first, 2 second.")
DRUG_ORDER_KR = ("투여 순서 (crossover, 날과 교락): Day 1 (20261019) SBI-553 = 5941 F, 5939 M, vehicle = 5937 M, 1191 F; Day 2 (20261021) "
                 "SBI-553 = 5937, 1191, vehicle = 5941, 5939 - 2마리는 SBI-553 먼저, 2마리는 나중.")


def short(ev):
    return {"attending onset": "attending onset"}.get(ev, ev)          # 10-24: keep "escape/rearing onset" (same name as the title)


def summ_resp(S, fam, tracked, n_tr):
    evs = FAM[fam]
    ev_ok = [e for e in evs if S[e]["vehicle"]["n"] and S[e]["SBI-553"]["n"]
             and S[e]["base_veh"]["n"] and S[e]["base_sbi"]["n"]]
    dif = {e: abs(S[e]["SBI-553"]["pct_up"] - S[e]["vehicle"]["pct_up"]) for e in ev_ok}
    pick = max(ev_ok, key=lambda e: dif[e]) if ev_ok and max(dif.values()) > 0 else None
    alt = None
    if pick is None:                         # no informative vehicle vs SBI-553 contrast: the largest same-day change instead
        cand = [(e, dr, gb, gi) for e in evs for dr, (gb, gi) in (("vehicle", DAYP["veh"]), ("SBI-553", DAYP["sbi"]))
                if S[e][gb]["n"] and S[e][gi]["n"]]
        alt = max(cand, key=lambda c: abs(S[c[0]][c[3]]["pct_up"] - S[c[0]][c[2]]["pct_up"])) if cand else None
        pick = alt[0] if alt else evs[0]
    s = S[pick]
    nv = sorted({S[e]["vehicle"]["n"] for e in evs}); ns = sorted({S[e]["SBI-553"]["n"] for e in evs})
    nvt = "/".join(str(x) for x in nv); nst = "/".join(str(x) for x in ns)
    pv = s["vs_UP"]["p"]
    tst = "McNemar" if tracked else "Fisher"
    nms = sorted({S[e]["n_mice"] for e in evs})
    mi_en = mtxt(nms[0]) if len(nms) == 1 else f"{nms[0]}-{nms[-1]} mice per event"
    mi_kr = f"마우스 {nms[0]}" if len(nms) == 1 else f"사건당 마우스 {nms[0]}-{nms[-1]}"
    if alt:
        e, dr, gb, gi = alt
        q = S[e]
        pre = f"Same {n_tr} neurons, " if tracked else ""
        prk = f"같은 뉴런 {n_tr}개, " if tracked else ""
        key_en = (f"{pre}{short(e)}: {q[gb]['pct_up']:.0f}% -> {q[gi]['pct_up']:.0f}% UP after {dr} (same neurons, McNemar "
                  f"{ptxt(q[f'mcnemar_UP_{dr}']['p'])}); too few events for a vehicle vs SBI-553 comparison")
        key_kr = (f"{prk}{short(e)}: {dr} 후 UP {q[gb]['pct_up']:.0f}% -> {q[gi]['pct_up']:.0f}% (같은 뉴런, McNemar "
                  f"{ptxt(q[f'mcnemar_UP_{dr}']['p'])}); 사건이 적어 vehicle vs SBI-553 비교 불가")
    elif tracked:
        key_en = (f"Same {n_tr} neurons, {short(pick)}: {s['SBI-553']['pct_up']:.0f}% UP after SBI-553 vs {s['vehicle']['pct_up']:.0f}% "
                  f"after vehicle ({tst} {ptxt(pv)})")
        key_kr = (f"같은 뉴런 {n_tr}개, {short(pick)}: SBI-553 후 UP {s['SBI-553']['pct_up']:.0f}% vs vehicle 후 {s['vehicle']['pct_up']:.0f}% "
                  f"({tst} {ptxt(pv)})")
    else:
        key_en = (f"{short(pick)}: {s['SBI-553']['pct_up']:.0f}% UP after SBI-553 vs {s['vehicle']['pct_up']:.0f}% after vehicle "
                  f"(same-day baselines {s['base_sbi']['pct_up']:.0f}% / {s['base_veh']['pct_up']:.0f}%; {tst} {ptxt(pv)})")
        key_kr = (f"{short(pick)}: SBI-553 후 UP {s['SBI-553']['pct_up']:.0f}% vs vehicle 후 {s['vehicle']['pct_up']:.0f}% "
                  f"(같은 날 기준 {s['base_sbi']['pct_up']:.0f}% / {s['base_veh']['pct_up']:.0f}%; {tst} {ptxt(pv)})")
    if tracked:
        bottom_en = (f"We counted UP / DOWN among the {n_tr} tracked neurons in all four sessions and averaged their traces; n = {mi_en}; "
                     "time-shift test per neuron, McNemar (same neurons) for every bracket.")
        bottom_kr = (f"추적 뉴런 {n_tr}개의 네 세션 UP / DOWN을 세고 trace를 평균; {mi_kr}; 뉴런별 시간 이동 검정, 모든 괄호 McNemar (같은 뉴런).")
    elif len(nv) > 1 or len(ns) > 1:
        nall = [S[e][g]["n"] for e in evs for g in G4 if S[e][g]["n"]]
        bottom_en = (f"We counted UP / DOWN neurons in every session and averaged their traces; n = {min(nall)}-{max(nall)} neurons per "
                     f"session (sessions with >= 5 events), {mi_en}; time-shift test per neuron, McNemar, Fisher exact.")
        bottom_kr = (f"모든 세션에서 UP / DOWN 뉴런을 세고 trace를 평균; 세션당 뉴런 {min(nall)}-{max(nall)} (사건 5번 이상 세션), {mi_kr}; "
                     "뉴런별 시간 이동 검정, McNemar, Fisher exact.")
    else:
        bottom_en = (f"We counted UP / DOWN neurons in every session and averaged their traces; n = {nvt} (vehicle day) / {nst} "
                     f"(SBI-553 day) neurons, {mi_en}; time-shift test per neuron, McNemar, Fisher exact.")
        bottom_kr = (f"모든 세션에서 UP / DOWN 뉴런을 세고 trace를 평균; 뉴런 {nvt} (vehicle 날) / {nst} (SBI-553 날), {mi_kr}; "
                     "뉴런별 시간 이동 검정, McNemar, Fisher exact.")
    det_en, det_kr = [], []
    for e in evs:
        q = S[e]
        def pc(g, k):
            return fnum(q[g]["pct_up" if k == "UP" else "pct_down"])
        det_en.append(f"{e}: UP base {pc('base_veh', 'UP')}% -> veh {pc('vehicle', 'UP')}% (McNemar {ptxt(q['mcnemar_UP_vehicle']['p'])}), "
                      f"base {pc('base_sbi', 'UP')}% -> SBI {pc('SBI-553', 'UP')}% ({ptxt(q['mcnemar_UP_SBI-553']['p'])}), veh vs SBI "
                      f"{ptxt(q['vs_UP']['p'])}; DOWN {pc('base_veh', 'DOWN')} -> {pc('vehicle', 'DOWN')}% "
                      f"({ptxt(q['mcnemar_DOWN_vehicle']['p'])}), {pc('base_sbi', 'DOWN')} -> {pc('SBI-553', 'DOWN')}% "
                      f"({ptxt(q['mcnemar_DOWN_SBI-553']['p'])}), veh vs SBI {ptxt(q['vs_DOWN']['p'])}; n = {q['base_veh']['n']}/"
                      f"{q['vehicle']['n']}/{q['base_sbi']['n']}/{q['SBI-553']['n']}.")
        det_kr.append(f"{e}: UP 기준 {pc('base_veh', 'UP')}% -> veh {pc('vehicle', 'UP')}% (McNemar {ptxt(q['mcnemar_UP_vehicle']['p'])}), "
                      f"기준 {pc('base_sbi', 'UP')}% -> SBI {pc('SBI-553', 'UP')}% ({ptxt(q['mcnemar_UP_SBI-553']['p'])}), veh vs SBI "
                      f"{ptxt(q['vs_UP']['p'])}; DOWN {pc('base_veh', 'DOWN')} -> {pc('vehicle', 'DOWN')}% "
                      f"({ptxt(q['mcnemar_DOWN_vehicle']['p'])}), {pc('base_sbi', 'DOWN')} -> {pc('SBI-553', 'DOWN')}% "
                      f"({ptxt(q['mcnemar_DOWN_SBI-553']['p'])}), veh vs SBI {ptxt(q['vs_DOWN']['p'])}; n = {q['base_veh']['n']}/"
                      f"{q['vehicle']['n']}/{q['base_sbi']['n']}/{q['SBI-553']['n']}.")
        if tracked and q.get("norm_UP", {}).get("n"):
            nu = q["norm_UP"]
            det_en.append(f"  baseline-normalized (same {nu['n']} neurons): % UP change vehicle day {nu['chg_vehicle_pts']:+.0f} pts, "
                          f"SBI-553 day {nu['chg_sbi_pts']:+.0f} pts (Wilcoxon signed-rank by neuron {ptxt(nu['p_wilcoxon'])}).")
            det_kr.append(f"  기준 정규화 (같은 뉴런 {nu['n']}): % UP 변화 vehicle 날 {nu['chg_vehicle_pts']:+.0f}%p, SBI-553 날 "
                          f"{nu['chg_sbi_pts']:+.0f}%p (뉴런 쌍 Wilcoxon {ptxt(nu['p_wilcoxon'])}).")
        up_r = " / ".join(fnum(q[g]["resp_up"], "{:+.2f}") for g in G4)
        dn_r = " / ".join(fnum(q[g]["resp_down"], "{:+.2f}") for g in G4)
        det_en.append(f"  mean response (z, best window) of the UP neurons base-veh / veh / base-SBI / SBI: {up_r}; DOWN: {dn_r}.")
        det_kr.append(f"  UP 뉴런 평균 반응 (z, 최적 창) 기준-veh / veh / 기준-SBI / SBI: {up_r}; DOWN: {dn_r}.")
    win_en = f"Window: {WW.en()} (window search)."
    win_kr = f"창: {WW.kr()} (창 탐색)."
    pm_en = ("Per-mouse markers are descriptive (4 mice: a sign / rank test across mice cannot reach p < 0.05). "
             + ("Tracked = 185 neurons followed on both days (A 101 same, B 84 probably the same): 5941 55, 5939 43, 5937 22, 1191 65."
                if tracked else "Neurons pooled over the mice: the brackets are exploratory."))
    pm_kr = ("마우스별 표식은 기술 통계 (4마리: 마우스 간 부호 / 순위 검정은 p < 0.05 불가). "
             + ("추적 = 두 날 모두 따라간 185 뉴런 (A 101 같은 뉴런, B 84 아마 같은 뉴런): 5941 55, 5939 43, 5937 22, 1191 65."
                if tracked else "마우스를 합친 뉴런: 괄호는 탐색적."))
    if fam == "behB":
        pm_en += (" escape / grooming: no SBI-553 session has >= 5 onsets; grooming only 5937; attending onsets: only sessions with "
                  ">= 5 (no vehicle-day baseline).")
        pm_kr += " escape / grooming: SBI-553 세션에 5번 이상 없음; grooming은 5937만; attending: 5번 이상인 세션만 (vehicle 날 기준 없음)."
    notes_en = " ".join([DRUG_ORDER_EN, win_en, "UP / DOWN = within-block time-shift test per neuron (1000 shuffles), BH q <= 0.20 "
                         "within the session.", " ".join(det_en), pm_en])
    notes_kr = " ".join([DRUG_ORDER_KR, win_kr, "UP / DOWN = 뉴런별 블록 내 시간 이동 검정 (1000회), 세션 안 BH q <= 0.20.", " ".join(det_kr), pm_kr])
    fam_en = {"stim": "stimulus", "behA": "reflex / movement", "behB": "escape, grooming, attending"}[fam]
    fam_kr = {"stim": "자극", "behA": "반사 / 움직임", "behB": "escape, grooming, attending"}[fam]
    if tracked:
        title_en = f"The same tracked neurons: UP / DOWN per {fam_en} and their traces"
        title_kr = f"같은 추적 뉴런: {fam_kr}별 UP / DOWN과 그 trace"
    else:
        title_en = f"How many neurons respond to each {fam_en}, and what they look like"
        title_kr = f"{fam_kr}마다 반응하는 뉴런 수와 그 모양"
    pr = "main" if fam == "stim" else "backup" if fam == "behA" else "full"
    return dict(title_en=title_en, title_kr=title_kr, key_en=key_en, key_kr=key_kr, bottom_en=bottom_en, bottom_kr=bottom_kr,
                notes_en=notes_en, notes_kr=notes_kr, section="Q3 neurons", priority=pr, numbers=S, key_event=pick)


def summ_groups(R, ev, n_ev):
    c3 = ["baseline", "vehicle", "SBI-553"]
    have = [c for c in c3 if R[c]["n"]]
    def pc(c, k):
        return fnum(R[c][k]["pct"]) if R[c]["n"] else "-"
    key_en = f"{short(ev)}: UP / DOWN " + ", ".join(f"{c} {pc(c, 'UP')}% / {pc(c, 'DOWN')}%" for c in have)
    key_kr = f"{short(ev)}: UP / DOWN " + ", ".join(f"{ {'baseline': '기준'}.get(c, c)} {pc(c, 'UP')}% / {pc(c, 'DOWN')}%" for c in have)
    nn = " / ".join(str(R[c]["n"]) for c in c3)
    bottom_en = (f"We split the neurons by their test result (UP, DOWN, rest) and averaged each group's trace; n = {nn} neuron-sessions "
                 f"(baseline / vehicle / SBI-553), {mtxt(R['n_mice'])}; time-shift test, BH q <= 0.20.")
    bottom_kr = (f"검정 결과로 뉴런을 나눠 (UP, DOWN, 나머지) 그룹별 trace를 평균; 뉴런-세션 {nn} (기준 / vehicle / SBI-553), 마우스 {R['n_mice']}; "
                 "시간 이동 검정, BH q <= 0.20.")
    det_en = "; ".join(f"{c}: UP {R[c]['UP']['n']} (mean response {fnum(R[c]['UP']['resp'], '{:+.2f}')} z), DOWN {R[c]['DOWN']['n']} "
                       f"({fnum(R[c]['DOWN']['resp'], '{:+.2f}')} z), rest {R[c]['rest']['n']}" for c in c3)
    det_kr = "; ".join(f"{ {'baseline': '기준'}.get(c, c)}: UP {R[c]['UP']['n']} (평균 반응 {fnum(R[c]['UP']['resp'], '{:+.2f}')} z), DOWN "
                       f"{R[c]['DOWN']['n']} ({fnum(R[c]['DOWN']['resp'], '{:+.2f}')} z), 나머지 {R[c]['rest']['n']}" for c in c3)
    miss_en = " ".join(f"No {c} session with >= 5 events." for c in c3 if not R[c]["n"])
    miss_kr = " ".join(f"{c} 세션에 사건 5번 이상 없음." for c in c3 if not R[c]["n"])
    notes_en = (f"{DRUG_ORDER_EN} Window {wtxt(ev)}. Baseline = both days' baseline sessions pooled. {det_en}. {miss_en} Traces: "
                "trial-mean z minus the 3 s before (display); numbers = the response in the best window (event analysis).").replace("  ", " ")
    notes_kr = (f"{DRUG_ORDER_KR} 창 {wtxt(ev)}. 기준 = 두 날 기준 세션 합침. {det_kr}. {miss_kr} Trace: trial 평균 z, 직전 3 s 기준 (표시용); "
                "수치 = 최적 창의 반응 (event analysis).").replace("  ", " ")
    pr = "backup" if ev in ("Pin prick", "Heat") else "full"
    return dict(title_en=f"{ev}: UP, DOWN and the other neurons", title_kr=f"{ev}: UP, DOWN, 나머지 뉴런", key_en=key_en, key_kr=key_kr,
                bottom_en=bottom_en, bottom_kr=bottom_kr, notes_en=notes_en, notes_kr=notes_kr,
                section="Q3 neurons" if ev in STIM else "Q4 stimulus or movement", priority=pr, numbers=R)


def summ_groups_tr(R, ev, n_tr_ev):
    def f2(cn, k, w="resp"):
        return fnum(R[cn][k][w], "{:+.2f}")
    if R["vi"]["UP"]["n"] and R["si"]["UP"]["n"]:
        key_en = (f"Same neurons, {short(ev)}: baseline-UP response {f2('vi', 'UP', 'resp_ref')} -> {f2('vi', 'UP')} z after vehicle, "
                  f"{f2('si', 'UP', 'resp_ref')} -> {f2('si', 'UP')} z after SBI-553")
        key_kr = (f"같은 뉴런, {short(ev)}: 기준 UP 뉴런 반응 vehicle 후 {f2('vi', 'UP', 'resp_ref')} -> {f2('vi', 'UP')} z, SBI-553 후 "
                  f"{f2('si', 'UP', 'resp_ref')} -> {f2('si', 'UP')} z")
    elif R["vi"]["UP"]["n"] or R["si"]["UP"]["n"]:               # only one drug day has baseline-UP neurons with an injection session
        cn, dr = ("vi", "vehicle") if R["vi"]["UP"]["n"] else ("si", "SBI-553")
        miss = "SBI-553" if cn == "vi" else "vehicle"
        key_en = (f"Same neurons, {short(ev)}: baseline-UP response {f2(cn, 'UP', 'resp_ref')} -> {f2(cn, 'UP')} z after {dr} "
                  f"(n = {R[cn]['UP']['n']}); no {miss} comparison (< 5 events or no UP neurons)")
        key_kr = (f"같은 뉴런, {short(ev)}: 기준 UP 뉴런 반응 {dr} 후 {f2(cn, 'UP', 'resp_ref')} -> {f2(cn, 'UP')} z (n = {R[cn]['UP']['n']}); "
                  f"{miss} 비교 없음 (사건 < 5 또는 UP 뉴런 없음)")
    else:
        nb = {c: R[c]["UP"]["n"] + R[c]["DOWN"]["n"] for c in ("vb", "sb")}
        key_en = (f"Same neurons, {short(ev)}: too few responsive neurons to compare (baseline UP + DOWN: vehicle day {nb['vb']}, "
                  f"SBI-553 day {nb['sb']}; {mtxt(R['n_mice'])})")
        key_kr = (f"같은 뉴런, {short(ev)}: 비교할 반응 뉴런이 너무 적음 (기준 UP + DOWN: vehicle 날 {nb['vb']}, SBI-553 날 {nb['sb']}; "
                  f"마우스 {R['n_mice']})")
    nn = " / ".join(str(R[c]["n"]) for c in ("vb", "vi", "sb", "si"))
    bottom_en = (f"Tracked neurons grouped by their class at each day's baseline, shown in that day's baseline and injection session; n = "
                 f"{nn} (4 columns), {mtxt(R['n_mice'])}; time-shift test, BH q <= 0.20; Wilcoxon signed-rank in the notes.")
    bottom_kr = (f"추적 뉴런을 그날 기준의 분류로 묶어 기준·주사 세션에 표시; n = {nn} (4열), 마우스 {R['n_mice']}; 시간 이동 검정, BH q <= 0.20; "
                 "Wilcoxon signed-rank는 노트.")
    det_en, det_kr = [], []
    for cn, ref in (("vi", "vehicle"), ("si", "SBI-553")):
        for k in ("UP", "DOWN"):
            q = R[cn][k]
            if q["n"]:
                det_en.append(f"{ref} day, baseline-{k} neurons (n = {q['n']}): response {fnum(q['resp_ref'], '{:+.2f}')} -> "
                              f"{fnum(q['resp'], '{:+.2f}')} z (Wilcoxon signed-rank {ptxt(q['p_wilcoxon_vs_baseline'])}); still {k} after "
                              f"the injection {q['still_same']}/{q['n']}")
                det_kr.append(f"{ref} 날, 기준 {k} 뉴런 (n = {q['n']}): 반응 {fnum(q['resp_ref'], '{:+.2f}')} -> {fnum(q['resp'], '{:+.2f}')} z "
                              f"(Wilcoxon signed-rank {ptxt(q['p_wilcoxon_vs_baseline'])}); 주사 후에도 {k} {q['still_same']}/{q['n']}")
    notes_en = (f"{DRUG_ORDER_EN} Window {wtxt(ev)}. {'; '.join(det_en)}. Groups selected at baseline regress toward the mean in the next "
                "session (selection on noise), so the decline itself is expected; compare the vehicle-day with the SBI-553-day decline. "
                f"Neurons pooled over {mtxt(R['n_mice'])} = exploratory.")
    notes_kr = (f"{DRUG_ORDER_KR} 창 {wtxt(ev)}. {'; '.join(det_kr)}. 기준에서 고른 그룹은 다음 세션에서 평균으로 회귀 (잡음으로 선택)하므로 "
                f"감소 자체는 예상됨; vehicle 날과 SBI-553 날의 감소를 비교. {R['n_mice']}마리 뉴런 합산 = 탐색적.")
    pr = "backup" if ev in ("Pin prick", "Heat") else "full"
    return dict(title_en=f"{ev}: the same neurons before and after each injection", title_kr=f"{ev}: 같은 뉴런, 각 주사 전후",
                key_en=key_en, key_kr=key_kr, bottom_en=bottom_en, bottom_kr=bottom_kr, notes_en=notes_en, notes_kr=notes_kr,
                section="Q3 neurons" if ev in STIM else "Q4 stimulus or movement", priority=pr, numbers=R)


# ------------------------------------------------------------------------------------------------ main
def main():
    os.makedirs(OUT, exist_ok=True)
    T, D, IDX, tt, problems, n_tr = load()
    print(f"neurons: {T[['mouse', 'day', 'cell']].drop_duplicates().shape[0]}; tracked neuron rows {T.tracked.sum()}, tracked neurons {n_tr}")
    for p in problems:
        print("DATA:", p)
    J, rows = {}, []
    for fam in FAM:
        for tracked in (False, True):
            for lg in ("en", "kr"):
                S = fig_resp(T, D, IDX, tt, fam, tracked, lg, n_tr)
            name = f"RESP_{fam}" + ("_tracked" if tracked else "") + ".png"
            J[name] = summ_resp(S, fam, tracked, n_tr)
            for ev, st in S.items():
                for k, v in st.items():
                    if isinstance(v, dict) and "p" in v:
                        rows.append(dict(figure=name, event=ev, test_key=k, **{kk: vv for kk, vv in v.items() if not isinstance(vv, dict)}))
            print("wrote", name)
    gt = []
    for ev in ALL_EV:
        for lg in ("en", "kr"):
            R = fig_groups(T, D, IDX, tt, ev, lg)
        J[f"GROUPS_{tag(ev)}.png"] = summ_groups(R, ev, None)
        for lg in ("en", "kr"):
            Rt = fig_groups_tracked(T, D, IDX, tt, ev, lg)
        J[f"GROUPS_{tag(ev)}_tracked.png"] = summ_groups_tr(Rt, ev, None)
        for cn, v in Rt.items():
            if not isinstance(v, dict):
                continue
            for k in ("UP", "DOWN", "rest"):
                gt.append(dict(event=ev, column=cn, group_at_baseline=k, **v[k]))
        print("wrote GROUPS", ev)
    for k, v in J.items():
        for f_ in ("key_en", "key_kr"):
            if len(v[f_]) > 150:
                print(f"LONG {f_} ({len(v[f_])}) {k}: {v[f_]}")
        for f_ in ("bottom_en", "bottom_kr"):
            if len(v[f_]) > 230:
                print(f"LONG {f_} ({len(v[f_])}) {k}: {v[f_]}")
    J["_meta"] = dict(script=os.path.abspath(__file__), inputs=[os.path.join(EA, "cells_classes.csv"), CACHE, CD],
                      n_tracked=n_tr, data_problems=problems, drug_order_en=DRUG_ORDER_EN, drug_order_kr=DRUG_ORDER_KR,
                      windows={ev: WW.of(ev) for ev in ALL_EV})
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(jclean(J), fh, ensure_ascii=False, indent=1)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "resp_tests.csv"), index=False)
    pd.DataFrame(gt).drop(columns=["mice"], errors="ignore").to_csv(os.path.join(OUT, "groups_tracked.csv"), index=False)
    print("done:", OUT)


if __name__ == "__main__":
    main()
