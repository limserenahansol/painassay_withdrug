"""make_new_deck.py - the decks in the layout of your own two decks (2026-10-22): Day 1 + Day 2 (crossover), Day 1 alone,
Day 2 alone; FULL or a short PI version; English or Korean.

Your references: 11WT_SBI553_FINAL_hansol_checked_v3_summary (every mouse / every behaviour, per stimulus, the same
mouse in both conditions, affective vs reflexive, movement before any stimulus) and
Ntsr1 neurons+TRUFACT_MarkGreg_updated_somascore_HS (responsive cells per event on the FOV, UP / DOWN / rest averages,
population AUC + PSTH, a summary table). Your rules: few intuitive slides; title on top, ONE key sentence + an arrow on
the panel to look at, one or two sentences at the bottom (what we did / found); legends, n, tests and definitions in the
NOTES; big fonts. Every neuron is shown: one that exists on one day only is left blank on the other day.
Units (as in the Ntsr1 deck): z = dF/F / robust SD (MAD) of the session; PSTH = mean +- SEM over neurons of the trial-
averaged z, each trial baselined on [-2, 0] s; AUC (z*s) = area of that trial-baselined z over [0, 2] s after the event.
Conditions: baseline = the session before the injection; vehicle / SBI-553 = the injection session (crossover: 5941 + 5939
SBI-553 on Day 1, 5937 + 1191 on Day 2). Responsive = time-shift test of ca_day1_analysis (q <= 0.20).
Day-2 selections are PROVISIONAL until you confirm them (red mark on every slide).

  python make_new_deck.py --base <BASE> --out <deck.pptx> [--days 1 2] [--short] [--lang kr] [--psth-cache <npz>]
"""
import argparse
import json
import re
import os
import sys

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import binomtest, fisher_exact, mannwhitneyu, wilcoxon
from scipy.stats import beta as _beta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import safe_io  # noqa: E402,F401
import ca_day1_analysis as CA  # noqa: E402
import pairstars as PS  # noqa: E402  (user rule 10-22: every star shows the pair it compares)

MO = ["5941", "5939", "5937", "1191"]
ARMS = {1: {"5941": "SBI-553", "5939": "SBI-553", "5937": "vehicle", "1191": "vehicle"},
        2: {"5941": "vehicle", "5939": "vehicle", "5937": "SBI-553", "1191": "SBI-553"}}
COND = ["baseline", "vehicle", "SBI-553"]
CC = {"baseline": "#7f8c8d", "vehicle": "#2471a3", "SBI-553": "#c0392b"}
MC = {"5941": "#e67e22", "5939": "#8e44ad", "5937": "#16a085", "1191": "#34495e"}
MK = {"5941": "o", "5939": "s", "5937": "^", "1191": "D"}
STIM = ["Light touch", "Mild touch", "Pin prick", "Heat"]
BEHEV = ["withdrawal", "flinch", "attending onset", "movement onset", "grooming onset", "escape/rearing onset"]
RESP_EV = STIM + ["withdrawal", "movement onset"]
ALL_EV = STIM + BEHEV                     # every stimulus and every behaviour (user 10-22: show all of them)
DISP_PRE = (-3.0, 0.0)                    # display baseline of the traces (as the event analysis); every NUMBER uses WIN
WIN = {}                                  # event -> (pre, gap, a, b): the best window of the window search (ca_event_analysis)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import winwords as WW  # noqa: E402  (10-23: the window the event analysis used, in words)
import figfoot as FF  # noqa: E402  (10-24: figure methods text -> slide notes)
WINDOC = WW.prefix_en() + ": " + WW.en()
WINDOC_KR = WW.prefix_kr() + ": " + WW.kr()
BEHS = ["withdrawal", "flinch", "attending", "lick/bite", "escape/rearing", "general movement"]
SW, SH = 13.333, 7.5
RED = "#c0392b"
RC = {"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11,
      "legend.fontsize": 10.5, "axes.spines.top": False, "axes.spines.right": False}
plt.rcParams.update(RC)
DAYS = [1, 2]
LANG = "en"                               # deck language (main); the figures stay English, their notes text follows LANG
ZLAB = "z (dF/F / robust SD),\nminus mean of the 3 s before"   # display baseline of every trace (DISP_PRE, per trial)
ZLAB3 = "z (dF/F / robust SD),\nminus mean of\nthe 3 s before"   # the same for short axes


def _rc():
    """this module's figure style again: modules imported later in main() (make_longitudinal_deck -> make_day1_ca_slides)
    change the global fonts, which shrank the text of every figure drawn after them (10-24)"""
    plt.rcParams.update(RC)
    plt.rcParams["font.family"] = plt.rcParamsDefault["font.family"]


def ptxt(p):
    if p is None or not np.isfinite(p):
        return "n.a."
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}" if p < 0.01 else f"p = {p:.2f}"


def star(p):
    if p is None or not np.isfinite(p):
        return "n.a."
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


_TWO = [("_m1", "5941 and 5939", "5941, 5939"), ("_m2", "5937 and 1191", "5937, 1191")]
_SB = [("_stim", "stimuli", "자극"), ("_beh", "behaviours", "행동")]
_P2 = [("_p1", "part 1", "1부"), ("_p2", "part 2", "2부")]
_S3 = [("_stim", "stimuli", "자극"), ("_beh1", "withdrawal, flinch, movement", "withdrawal, flinch, movement"),
       ("_beh2", "escape, grooming, attending", "escape, grooming, attending")]
SPLIT = {                     # figure stem -> its parts (suffix, label EN, label KR); used when every part exists
    "f07_population_behaviour": [("_a", "withdrawal, flinch, attending", "withdrawal, flinch, attending"),
                                 ("_b", "movement, grooming, escape", "movement, grooming, escape")],
    "f09c_responsive_pct_behaviour": [("_a", "withdrawal, flinch, attending", "withdrawal, flinch, attending"),
                                      ("_b", "movement, grooming, escape", "movement, grooming, escape")],
    "f10_maps": _TWO, "e_maps_*": _TWO, "f05b_fov_change": _TWO,
    "f11_responsive_change": _SB,
    "f13_behaviour_stimulus": [("_a", "withdrawal, flinch, attending", "withdrawal, flinch, attending"),
                               ("_b", "escape, movement, lick / bite", "escape, movement, lick / bite")],
    "P3a_responsive_pct": _S3, "P3e_sessions": _S3, "P5_fate": _S3,
    # 10-22 evening: the 4-behaviour part (_a, fonts at 82 %) shown as two 2-behaviour parts (_w, _x); _a stays on disk
    "P6_behaviour_per_stimulus": [("_w", "withdrawal, flinch", "withdrawal, flinch"),
                                  ("_x", "attending, lick / bite", "attending, lick / bite"),
                                  ("_b", "escape, movement, grooming", "escape, movement, grooming")],
    "O1_near_simultaneous": [("_p1", "A, B: responses and the encoding model", "A, B: 반응과 encoding model"),
                             ("_p2", "C, D: what each neuron follows", "C, D: 뉴런마다 따르는 것")],
    "S1_behaviour_by_sex": [("_a", "withdrawal, flinch, attending", "withdrawal, flinch, attending"),
                            ("_b", "lick / bite, escape, movement", "lick / bite, escape, movement")],
    "S2_drug_effect_by_sex": [("_a", "withdrawal, flinch, attending", "withdrawal, flinch, attending"),
                              ("_b", "escape, movement", "escape, movement")],
    "S4_responsive_by_sex": [("_stim", "stimuli", "자극"), ("_beh", "withdrawal, flinch, movement", "withdrawal, flinch, movement")],
    "S5_traces_by_sex": _SB,
    "P2b_average_4panels_SBIday_vehicleday": [("_sbi", "SBI-553 day", "SBI-553 날"), ("_veh", "vehicle day", "vehicle 날")],
    "P1_whole_*": [("_d1", "Day 1 sessions", "Day 1 세션"), ("_d2", "Day 2 sessions", "Day 2 세션")],
    "W1_window_search": [("_p1", "stimuli", "자극"), ("_p2", "behaviours + the held-out check", "행동 + 빠진 마우스 확인")],
    "W1b_window_search_slow": [("_p1", "stimuli", "자극"), ("_p2", "behaviours", "행동")],
    "W2_named_windows": _P2, "W3_timing": _P2, "Q1_motion_check": _P2, "P6b_behaviour_seconds": _P2, "P7_brain_behaviour": _P2,
    "A1_rules_compared": _P2, "A2_functional_types": _P2, "A2b_types_responsive": _P2, "V4_key_findings_per_mouse": _P2,
    "P4b_maps_*": [("_q1", "Day 1, stimuli", "Day 1, 자극"), ("_q2", "Day 1, behaviours", "Day 1, 행동"),
                   ("_q3", "Day 2, stimuli", "Day 2, 자극"), ("_q4", "Day 2, behaviours", "Day 2, 행동")],
}
PAIRNOTE = ("brackets = the 2 conditions compared: base-veh and base-SBI = the same neurons, baseline of that day vs after the "
            "injection; veh-SBI = after vehicle vs after SBI-553.  " + PS.KEY)


def pairnote(fig, txt=PAIRNOTE, fs=10.5, kr=None):
    """the figure's methods key (which pair each bracket compares, ...): 10-24 it goes to the slide notes (figfoot); the
    figure keeps only the one-line star key at the bottom left (about 0.25 in high). kr = the Korean text for the KR deck"""
    return FF.note(fig, kr if (LANG == "kr" and kr) else txt, LANG, fs=fs)


KEY_KR = "* p < 0.05, ** p < 0.01, *** p < 0.001, ns = 유의하지 않음"


def ticks4g(a, fs=10.5, n=None, short=False):
    """the 4 positions as two short labels per day + the day under each pair (10-24: 'base (veh day)' and 'veh' ran into each
    other at a readable size): base | veh, base | SBI, then 'vehicle day' / 'SBI-553 day' centred under each pair
    (short=True: 'veh day' / 'SBI day', for narrow panels)"""
    a.set_xticks([X4[g] for g in G4], ["base", "veh", "base", "SBI"] if not n else
                 [f"{s}\nn = {n[g]}" for s, g in zip(["base", "veh", "base", "SBI"], G4)], fontsize=fs)
    a.set_xlim(-0.6, 3.95)
    tr = a.get_xaxis_transform()
    off = -(8 + 1.3 * fs * (2 if n else 1))                       # points under the axis: below the tick labels
    for x, lab, c in (((X4["base_veh"] + X4["vehicle"]) / 2, "veh day" if short else "vehicle day", CC["vehicle"]),
                      ((X4["base_sbi"] + X4["SBI-553"]) / 2, "SBI day" if short else "SBI-553 day", CC["SBI-553"])):
        a.annotate(lab, xy=(x, 0), xycoords=tr, xytext=(0, off), textcoords="offset points", ha="center", va="top", fontsize=fs,
                   color=c, annotation_clip=False)


N_ANIMALS = re.compile(r"\bn\s*=\s*\d+\s*(mice|mouse|females|males)|\b\d+\s*(mice|mouse)\b|\b2\s*\+\s*2\b|\d+\s*마리|마우스\s*\d|한 마리|one mouse",
                       re.I)
# 10-25: matched case-insensitively (was case-sensitive per pattern, so 'Binomial' / 'Time-shift' fell through to
# 'descriptive (no test)'); 'random draws' = permutation; shuffle on its own; KR synonyms; 'no drug test' counts as descriptive
TEST_NAMES = [(r"Wilcoxon", "Wilcoxon signed-rank"), (r"Mann-Whitney", "Mann-Whitney U"), (r"McNemar", "McNemar"),
              (r"Fisher", "Fisher exact"), (r"Mantel|CMH|Cochran", "CMH"),
              (r"time-shift|time shift|shift test|shifted inside|Same test as|시간 이동", "time-shift test"), (r"max-T", "max-T"),
              (r"binomial|이항", "binomial test"), (r"Spearman", "Spearman"), (r"GLM|encoding model", "GLM"), (r"bootstrap", "bootstrap"),
              (r"random draws|permutation|순열|무작위 추출", "permutation test"), (r"shuffle|셔플", "shuffle test"),
              (r"split-half|split half|odd / even|odd/even", "split-half replication"),
              (r"sham", "sham control"), (r"k-means|silhouette|clustering", "k-means clustering"), (r"decod", "decoding (cross-validated)"),
              (r"[Dd]escriptive|기술 통계|no drug test|약물 검정 없음", "descriptive")]
TEST_ANY = re.compile("|".join(t for t, _ in TEST_NAMES), re.I)


def complete_n_test(bottom, key, notes, n_txt, lang):
    """the bottom sentences + n animals (from the deck) and the test (from the notes) when the slide does not say them"""
    bottom = bottom or ""
    txt = f"{bottom} {key or ''}"
    add = []
    if n_txt and not N_ANIMALS.search(txt):
        add.append(n_txt)
    if not TEST_ANY.search(txt):
        ts = list(dict.fromkeys(lab for pat, lab in TEST_NAMES if re.search(pat, notes or "", re.I)))
        if ts:
            add.append(("검정: " if lang == "kr" else "Test: ") + ", ".join(ts))
        else:                                          # nothing tested on this slide: say so
            add.append("기술 통계 (검정 없음)" if lang == "kr" else "descriptive (no test)")
    if not add:
        return bottom
    return (bottom.rstrip() + " " if bottom else "") + "; ".join(add) + "."


def zz(v, kr=False):
    """a mean response in z for a key line; a condition without events (nan) says so (10-24: was '+nan z')"""
    return f"{v:+.3f} z" if v is not None and np.isfinite(v) else ("n/a (사건 없음)" if kr else "n/a (no events)")


def whisker_top(v):
    """top of the upper whisker of a boxplot of v (1.5 IQR rule, as matplotlib draws it)"""
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if not len(v):
        return 0.0
    q1, q3 = np.percentile(v, [25, 75])
    w = v[v <= q3 + 1.5 * (q3 - q1)]
    return float(w.max()) if len(w) else float(q3)


def cond_of(m, day, phase):
    return "baseline" if phase == "pre" else ARMS[day][m]


def safe_mwu(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    return mannwhitneyu(a, b).pvalue if min(len(a), len(b)) > 3 else np.nan


def bracket(ax, x1, x2, y, p, fs=13, show_ns=True):
    """significance bracket between two bars: * p < 0.05, ** p < 0.01, *** p < 0.001 ('ns' otherwise)"""
    st = star(p)
    if st == "n.a." or (st == "ns" and not show_ns):
        return
    lo, hi = ax.get_ylim()
    h = 0.02 * (hi - lo)
    ax.plot([x1, x1, x2, x2], [y - h, y, y, y - h], color="k", lw=1, clip_on=False)
    ax.text((x1 + x2) / 2, y + 0.2 * h, st, ha="center", va="bottom", fontsize=fs if st != "ns" else fs - 2.5, clip_on=False)


def mark(ax, x, y, p, col, fs=14, va="bottom"):
    """a star on one bar = that injection vs the same neurons' (mice's) baseline of the same day; nothing when p >= 0.05"""
    st = star(p)
    if st in ("ns", "n.a."):
        return
    ax.text(x, y, st, ha="center", va=va, fontsize=fs, color=col, fontweight="bold", clip_on=False)


def within_day(X, drug, col=None):
    """the same neurons: the baseline session and the injection session of the same day (the days of that drug only)"""
    pre = X[(X.phase == "pre") & np.array([ARMS[int(d)][str(m)] == drug for m, d in zip(X.mouse, X.day)], bool)]
    post = X[(X.phase == "post") & (X.condition == drug)]
    M = pre.merge(post, on=["mouse", "day", "cell"], suffixes=("_b", "_i"))
    return M[[f"{col}_b", f"{col}_i"]].dropna() if col else M


def wsr_p(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 6 or np.allclose(a[ok], b[ok]):
        return np.nan
    return float(wilcoxon(a[ok], b[ok]).pvalue)


def mcnemar(b_, c_):
    n = int(b_ + c_)
    return float(binomtest(int(b_), n, 0.5).pvalue) if n else np.nan


def mh_rate_p(k1, n1, k2, n2):
    """Mantel-Haenszel test of a rate ratio stratified by mouse (k = events, n = deliveries; one stratum per mouse)"""
    from scipy.stats import chi2
    k1, n1, k2, n2 = (np.asarray(v, float) for v in (k1, n1, k2, n2))
    ok = (n1 > 0) & (n2 > 0) & (k1 + k2 > 0)
    if not ok.any():
        return np.nan
    K, p0 = (k1 + k2)[ok], (n1 / (n1 + n2))[ok]
    num, var = (k1[ok] - K * p0).sum(), (K * p0 * (1 - p0)).sum()
    if var <= 0:
        return np.nan
    return float(chi2.sf(max(abs(num) - 0.5, 0) ** 2 / var, 1))


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


# 10-22 night (triple check): each coloured bracket tests an injection against the baseline of ITS OWN day, so the plots show
# that baseline: base (veh day) | vehicle | base (SBI day) | SBI-553; the black bracket = vehicle vs SBI-553
G4 = ["base_veh", "vehicle", "base_sbi", "SBI-553"]
X4 = {"base_veh": 0.0, "vehicle": 1.0, "base_sbi": 2.35, "SBI-553": 3.35}
L4 = {"base_veh": "base\n(veh day)", "vehicle": "veh", "base_sbi": "base\n(SBI day)", "SBI-553": "SBI"}
C4 = {"base_veh": "#7f8c8d", "vehicle": "#2471a3", "base_sbi": "#7f8c8d", "SBI-553": "#c0392b"}


def sub4(X, g):
    """rows of X in one of the 4 groups (X has mouse, day, phase, condition)"""
    if g in ("vehicle", "SBI-553"):
        return X[X.condition == g]
    drug = "vehicle" if g == "base_veh" else "SBI-553"
    return X[(X.phase == "pre") & np.array([ARMS[int(d)][str(m)] == drug for m, d in zip(X.mouse, X.day)], bool)]


def ticks4(a, fs=None, n=None):
    a.set_xticks([X4[g] for g in G4], [L4[g] + (f"\nn = {n[g]}" if n else "") for g in G4], fontsize=fs)
    a.set_xlim(-0.6, 3.95)


def brackets4(a, y1, y2, p_veh, p_sbi, p_vs, fs=11, show_vs=True):
    PS.bracket(a, X4["base_veh"], X4["vehicle"], y1, p_veh, CC["vehicle"], fs=fs)
    PS.bracket(a, X4["base_sbi"], X4["SBI-553"], y1, p_sbi, CC["SBI-553"], fs=fs)
    if show_vs:
        PS.bracket(a, X4["vehicle"], X4["SBI-553"], y2, p_vs, "k", fs=fs)


PAIRNOTE4 = ("bars: base = the baseline session of that day (vehicle day / SBI-553 day). Blue / red bracket = the same neurons, that "
             "day's baseline vs after the injection; black = after vehicle vs after SBI-553.  " + PS.KEY)
PAIRNOTE4_KR = ("base = 그날의 기준 세션 (vehicle 날 / SBI-553 날). 파랑 / 빨강 괄호 = 같은 뉴런, 그날 기준 vs 주사 후; 검정 괄호 = "
                "vehicle 후 vs SBI-553 후.  " + KEY_KR)


def callout(ax, xy, text="look here", y=0.965, inside=False):
    """the key-message arrow: bold red label in the figure's top margin above the panel (or inside it), pointing at xy"""
    if inside:
        ax.annotate(text, xy=xy, xytext=(0.5, 0.95), textcoords="axes fraction", ha="center", va="top", fontsize=13,
                    fontweight="bold", color=RED, arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.2, shrinkA=2, shrinkB=4))
        return
    bb = ax.get_position()
    ax.annotate(text, xy=xy, xycoords="data", xytext=((bb.x0 + bb.x1) / 2, y), textcoords="figure fraction", ha="center",
                va="top", fontsize=13, fontweight="bold", color=RED,
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.2, shrinkA=2, shrinkB=4), annotation_clip=False)


def conds_present():
    """the conditions that exist in the selected days (one day: baseline + both drugs, 2 mice each)"""
    return COND


# ------------------------------------------------------------------------------------------------ data
def load_windows(B):
    js = os.path.join(B, "analysis", "event_analysis", "summary.json")
    if os.path.isfile(js):
        for ev, w in json.load(open(js, encoding="utf-8")).get("windows", {}).items():
            WIN[ev] = tuple(float(x) for x in w["1"])


def wwin(ev):
    """(baseline start, end), (response start, end) in s from the event, of the best window"""
    pre, gap, a, b = WIN.get(ev, (2.0, 0.0, 0.0, 2.0))
    a0 = min(a, 0.0) - gap
    return (a0 - pre, a0), (a, b)


def wtxt(ev):
    (p0, p1), (a, b) = wwin(ev)
    return f"{a:g}-{b:g} s vs {p0:g}..{p1:g} s"


def load_tests(B):
    """per neuron x session x event: the response in the best window and its class, from the event analysis
    (cells_classes.csv: within-block time-shift test, 1000 shuffles, BH q <= 0.20 per session). auc_post = that response
    (z: response-window mean minus baseline-window mean, trial-averaged); cls_shift = UP / DOWN / no response."""
    T = pd.read_csv(os.path.join(B, "analysis", "event_analysis", "cells_classes.csv"))
    T["mouse"] = T.mouse.astype(str)
    T = T[T.day.isin(DAYS)].copy()
    T["auc_post"] = T.resp
    T["cls_shift"] = T.cls.map({"activated": "UP", "inhibited": "DOWN"}).fillna("no response")
    T["condition"] = [cond_of(m, d, p) for m, d, p in zip(T.mouse, T.day, T.phase)]
    return T.reset_index(drop=True)


def load_rates(B):
    out = []
    for d in DAYS:
        W = pd.read_csv(os.path.join(B, "analysis", f"ca_day{d}", f"ca_d{d}_cells_whole_session.csv"))
        W["mouse"] = W.mouse.astype(str); W["day"] = d
        num = sum(W[f"still rate | {s}"].fillna(0) * W[f"still min | {s}"].fillna(0) for s in STIM)
        den = sum(W[f"still min | {s}"].fillna(0) for s in STIM)
        W["rate_blocks"] = num / den.replace(0, np.nan)
        W["condition"] = [cond_of(m, d, p) for m, p in zip(W.mouse, W.phase)]
        out.append(W[["mouse", "day", "phase", "cell", "condition", "rate_blocks"]])
    return pd.concat(out, ignore_index=True)


def load_pairs(B):
    P = {}
    for m in MO:
        f = os.path.join(B, "analysis", "crossday", f"{m}_pairs.csv")
        P[m] = pd.read_csv(f) if os.path.isfile(f) else pd.DataFrame(columns=["day1_id", "day2_id"])
    return P


def psth_all(B, events, cache):
    """per session and event: time x cells trial-mean of z minus its 3 s before the event (display baseline, as the event
    analysis) and the cell ids; cached. An event with too few trials (< CA.MIN_TRIALS) is kept as None so it is not tried
    again. Attending onsets come from your scoring (code 1, a bout after >= 2 s without it), as in the event analysis."""
    D = {}
    if cache and os.path.isfile(cache):
        D = np.load(cache, allow_pickle=True)["D"].item()
    need = [(m, d, ph) for m in MO for d in DAYS for ph in ("pre", "post") if any((m, d, ph, ev) not in D for ev in events)]
    if need:
        sfx = os.environ.get("SCORING_SUFFIX", "_timefix")     # 10-23: the timing-corrected scoring copies (scoring_timefix.py)
        score = {1: os.path.join(B, "scoring_mini1p", "20261019_final" + sfx), 2: os.path.join(B, "scoring_mini1p", "20261021_final" + sfx)}
        root = {1: os.path.join(B, "mini1p_out", "20261019"), 2: os.path.join(B, "mini1p_out", "20261021")}
        for m, d, ph in need:
            s = CA.load_session(root[d], m, ph, score[d], day=d)
            if s is None or s["B"] is None:
                for ev in events:
                    D.setdefault((m, d, ph, ev), None)
                continue
            Sm, _ = CA.scoring_for(score[d], m, ph)
            s["B"]["ev"]["attending onset"] = CA.onsets(np.atleast_1d(Sm["score"]).astype(int) == 1) + 1
            ids = s["T"]["cell_ids"].astype(int)
            for ev in events:
                if (m, d, ph, ev) in D:
                    continue
                X, tt, used = CA.peri(s, s["B"]["ev"].get(ev, []))
                if X is None or len(used) < CA.MIN_TRIALS:
                    D[(m, d, ph, ev)] = None
                    continue
                pre = (tt >= DISP_PRE[0]) & (tt < DISP_PRE[1])
                Xb = X - np.nanmean(X[:, pre, :], 1, keepdims=True)
                D[(m, d, ph, ev)] = dict(mean=np.nanmean(Xb, 0).astype(np.float32), tt=tt, ids=ids, n=len(used))
        if cache:
            np.savez_compressed(cache, D=np.array(D, dtype=object))
    return {k: v for k, v in D.items() if k[1] in DAYS and k[3] in events and v is not None}


def neuron_rows(B, P, D, ev):
    """every neuron of the selected days, per mouse: (status, {(day, phase): cell id or None}); status = 'both days' (tracked),
    'Day 1 only', 'Day 2 only' (combined deck) or 'Day d' (one day)"""
    rows = {}
    for m in MO:
        ids = {d: (D[(m, d, "pre", ev)]["ids"] if (m, d, "pre", ev) in D else np.array([], int)) for d in DAYS}
        R = []
        if len(DAYS) == 2:
            pr = P[m]
            t1, t2 = set(pr.day1_id.astype(int)), set(pr.day2_id.astype(int))
            for r in pr.itertuples():
                R.append(("both days", {1: int(r.day1_id), 2: int(r.day2_id)}))
            R += [("Day 1 only", {1: int(c), 2: None}) for c in ids[1] if int(c) not in t1]
            R += [("Day 2 only", {1: None, 2: int(c)}) for c in ids[2] if int(c) not in t2]
        else:
            d = DAYS[0]
            R = [(f"Day {d}", {d: int(c)}) for c in ids[d]]
        rows[m] = R
    return rows


# ------------------------------------------------------------------------------------------------ figures: neurons
def fig_rate(R, out):
    _rc()
    fig, ax = plt.subplots(1, 2, figsize=(12.6, 4.9), gridspec_kw=dict(width_ratios=[1.5, 1]))
    rng = np.random.default_rng(0)
    res = {}
    for c in COND:
        res[c] = R[R.condition == c].rate_blocks.dropna().values
    n4 = {}
    for g in G4:
        Xg = sub4(R, g)
        v = Xg.rate_blocks.dropna().values
        n4[g] = len(v)
        if len(v):
            ax[0].boxplot(v, positions=[X4[g]], widths=0.55, showfliers=False, medianprops=dict(color="k", lw=2.5),
                          boxprops=dict(color=C4[g], lw=2), whiskerprops=dict(color=C4[g]), capprops=dict(color=C4[g]))
        for m in MO:
            w = Xg[Xg.mouse == m].rate_blocks.dropna().values
            ax[0].scatter(X4[g] + rng.uniform(-0.22, 0.22, len(w)), w, s=9, color=MC[m], alpha=0.45, lw=0, clip_on=False)
    ticks4(ax[0], n=n4)
    ax[0].set_ylabel("Ca transients / min\n(stimulus blocks, still frames)")
    ax[0].set_title("A  all neurons pooled (dot = one neuron)", loc="left")
    pv = safe_mwu(res["SBI-553"], res["vehicle"])
    top = max([whisker_top(sub4(R, g).rate_blocks.dropna().values) for g in G4] + [1.0])
    ax[0].set_ylim(0, top * 1.42)
    pw = {}
    for drug, x in (("vehicle", 1), ("SBI-553", 2)):
        M_ = within_day(R, drug, "rate_blocks")
        pw[drug] = wsr_p(M_.rate_blocks_b, M_.rate_blocks_i)
    # every star with its pair: base-veh / base-SBI = the same neurons, that day's baseline vs the injection (Wilcoxon);
    # veh-SBI = after vehicle vs after SBI-553 (Mann-Whitney U over neurons)
    brackets4(ax[0], top * 1.05, top * 1.24, pw["vehicle"], pw["SBI-553"], pv, fs=12)
    med = R.groupby(["mouse", "condition"]).rate_blocks.median().unstack().reindex(columns=COND)
    for m in MO:                                               # one line per mouse and day: that day's baseline -> injection
        for gb, gi in (("base_veh", "vehicle"), ("base_sbi", "SBI-553")):
            yb, yi = sub4(R, gb), sub4(R, gi)
            yb, yi = yb[yb.mouse == m].rate_blocks.median(), yi[yi.mouse == m].rate_blocks.median()
            if np.isfinite(yb) and np.isfinite(yi):
                ax[1].plot([X4[gb], X4[gi]], [yb, yi], marker=MK[m], color=MC[m], lw=2, ms=9, label=m if gb == "base_veh" else None)
    ticks4(ax[1]); ax[1].set_ylabel("median transients / min\n(of the mouse's neurons)")
    ax[1].set_title("B  one line per mouse", loc="left")
    d_m = {m: med.loc[m, "SBI-553"] - med.loc[m, "baseline"] for m in MO if m in med.index}
    fig.tight_layout(rect=(0, 0.05, 1, 0.85))
    # 10-24: the box elements and the mouse colours keyed ON the figure (the footer called the boxes 'bars'; the colour key of
    # the dots in A was only in B)
    fig.legend(handles=[Patch(facecolor="w", edgecolor="0.35", lw=2, label="A: box = middle 50 % (IQR)"),
                        Line2D([], [], color="k", lw=2.5, label="line = median"),
                        Line2D([], [], color="0.35", lw=1.2, marker="_", ms=12, mew=1.2, label="whiskers = data within 1.5 x IQR")],
               loc="upper left", ncol=3, frameon=False, fontsize=10.5, bbox_to_anchor=(0.005, 0.955),
               handlelength=1.6, columnspacing=1.2)
    fig.legend(handles=[Line2D([], [], color=MC[m], marker=MK[m], lw=2, ms=8, label=m) for m in MO], loc="upper right", ncol=4,
               frameon=False, fontsize=10.5, bbox_to_anchor=(0.995, 1.0), title="colour = mouse (dots in A, lines in B)",
               title_fontsize=10.5, handlelength=1.6, columnspacing=1.0)
    pairnote(fig, PAIRNOTE4, kr=PAIRNOTE4_KR)
    fig.savefig(out, dpi=150); plt.close(fig)
    return dict(p=pv, p_within=pw, med={c: float(np.nanmedian(res[c])) if len(res[c]) else np.nan for c in COND},
                n={c: len(res[c]) for c in COND}, med_m=med)


def fig_rate_matrix(R, P, out):
    """every neuron in the three conditions (rows = neurons grouped by mouse; tracked neurons first, then the neurons of one
    day only, blank where a neuron does not exist); colour = transient rate"""
    blocks = []
    for m in MO:
        Rm = R[R.mouse == m].set_index(["day", "phase", "cell"]).rate_blocks
        def g(d, ph, c):
            try:
                return float(Rm[(d, ph, c)])
            except KeyError:
                return np.nan
        rows, st = [], []
        if len(DAYS) == 2:
            pr = P[m]
            t1, t2 = set(pr.day1_id.astype(int)), set(pr.day2_id.astype(int))
            items = [("both days", int(r.day1_id), int(r.day2_id)) for r in pr.itertuples()]
            items += [("Day 1 only", int(c), None) for c in sorted(set(R[(R.mouse == m) & (R.day == 1)].cell.astype(int)) - t1)]
            items += [("Day 2 only", None, int(c)) for c in sorted(set(R[(R.mouse == m) & (R.day == 2)].cell.astype(int)) - t2)]
            for s_, c1, c2 in items:
                v = {1: (g(1, "pre", c1), g(1, "post", c1)) if c1 else (np.nan, np.nan),
                     2: (g(2, "pre", c2), g(2, "post", c2)) if c2 else (np.nan, np.nan)}
                base = np.nanmean([v[1][0], v[2][0]]) if np.isfinite([v[1][0], v[2][0]]).any() else np.nan
                inj = {ARMS[1][m]: v[1][1], ARMS[2][m]: v[2][1]}
                bd = {ARMS[1][m]: v[1][0], ARMS[2][m]: v[2][0]}          # the baseline of the same day (right panel)
                rows.append([base, inj["vehicle"], inj["SBI-553"], bd["vehicle"], bd["SBI-553"]]); st.append(s_)
        else:
            d = DAYS[0]
            for c in sorted(set(R[(R.mouse == m) & (R.day == d)].cell.astype(int))):
                inj = {"vehicle": np.nan, "SBI-553": np.nan}; inj[ARMS[d][m]] = g(d, "post", c)
                bd = {"vehicle": np.nan, "SBI-553": np.nan}; bd[ARMS[d][m]] = g(d, "pre", c)
                rows.append([g(d, "pre", c), inj["vehicle"], inj["SBI-553"], bd["vehicle"], bd["SBI-553"]]); st.append(f"Day {d}")
        if rows:
            A = np.array(rows, float); st = np.array(st)
            o = np.lexsort((-np.nan_to_num(A[:, 0], nan=-1), [{"both days": 0, "Day 1 only": 1, "Day 2 only": 2}.get(s, 0) for s in st]))
            blocks.append((m, A[o], st[o]))
    if not blocks:
        return None
    Y = np.vstack([b[1][:, :3] for b in blocks]); S = np.concatenate([b[2] for b in blocks])
    _rc()
    fig = plt.figure(figsize=(12.6, 5.6))
    gs = fig.add_gridspec(1, 5, width_ratios=[0.25, 2.2, 0.07, 0.85, 1.7], wspace=0.05)
    a0, a1, acb, a2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2]), fig.add_subplot(gs[4])
    scol = {"both days": "#2ecc40", "Day 1 only": "#f012be", "Day 2 only": "#00d0ff"}
    a0.imshow(np.array([[matplotlib.colors.to_rgb(scol.get(s, "#2ecc40"))] for s in S]), aspect="auto", interpolation="nearest")
    a0.set_xticks([]); a0.set_yticks([])
    a0.set_title("recorded\non", fontsize=10.5)
    a0.set_ylabel("neurons (1 row = 1 neuron), grouped by mouse", fontsize=11, labelpad=44)
    cm = plt.get_cmap("magma").copy(); cm.set_bad("#eeeeee")
    vmax = np.nanpercentile(Y, 97) if np.isfinite(Y).any() else 1
    im = a1.imshow(np.ma.masked_invalid(Y), aspect="auto", cmap=cm, vmin=0, vmax=vmax, interpolation="nearest")
    a1.set_xticks(range(3), COND, fontsize=11.5); a1.set_yticks([])
    y0 = 0
    for m, A, _ in blocks:
        for a_ in (a0, a1):
            a_.axhline(y0 - 0.5, color="w" if a_ is a1 else "k", lw=1.5)
        a0.text(-0.7, y0 + len(A) / 2, m, ha="right", va="center", fontsize=12, fontweight="bold", color=MC[m])
        y0 += len(A)
    for t_, c in zip(a1.get_xticklabels(), COND):
        t_.set_color(CC[c]); t_.set_fontweight("bold")
    cb = fig.colorbar(im, cax=acb); cb.set_label("Ca transients / min\n(stimulus blocks, still frames)", fontsize=11)
    cb.ax.tick_params(labelsize=10.5)
    a1.set_title("every neuron; grey = " + ("the neuron does not exist that day" if len(DAYS) == 2 else
                 "that drug was not given to this mouse on this day"), fontsize=11.5)
    # right: per mouse, the change of each neuron that has both values
    labs0 = []
    for k, (m, A, _) in enumerate(blocks):
        for j, (c, x) in enumerate((("vehicle", 1), ("SBI-553", 2))):
            bc = 3 if c == "vehicle" else 4                            # same-day baseline of that drug's day
            ok = np.isfinite(A[:, bc]) & np.isfinite(A[:, x])
            d = np.log2((A[ok, x] + 0.1) / (A[ok, bc] + 0.1))
            if len(d):
                pos = k * 3 + j
                a2.boxplot(d, positions=[pos], widths=0.7, showfliers=False, medianprops=dict(color="k", lw=2),
                           boxprops=dict(color=CC[c], lw=2), whiskerprops=dict(color=CC[c]), capprops=dict(color=CC[c]))
                if len(d) >= 6 and np.any(d != 0):
                    labs0.append((pos, whisker_top(d), wilcoxon(d).pvalue, CC[c]))
    lo_, hi_ = a2.get_ylim()
    top_ = max([hi_] + [w_ + 0.22 * (hi_ - lo_) for _, w_, _, _ in labs0])
    a2.set_ylim(lo_, top_)
    for pos, w_, pp, cc in labs0:
        PS.vs_ref(a2, pos, w_ + 0.03 * (top_ - lo_), pp, "vs 0", cc, fs=10)
    a2.axhline(0, color="0.4", lw=0.8)
    a2.set_xticks([k * 3 + 0.5 for k in range(len(blocks))], [b[0] for b in blocks])
    a2.set_ylabel("log2 (injection / baseline)\nper neuron")
    a2.set_title("change per neuron: blue = vehicle, red = SBI-553\nstar over 'vs 0' = that change differs from 0", fontsize=11)
    lab_ = {"both days": "neuron found on both days", "Day 1 only": "Day 1 only", "Day 2 only": "Day 2 only"}
    fig.legend(handles=[Patch(color=v, label=lab_[k]) for k, v in scol.items() if k in set(S)], loc="lower left", ncol=3,
               frameon=False, fontsize=10.5, bbox_to_anchor=(0.005, 0.045), handlelength=1.4, columnspacing=1.0)
    fig.legend(handles=[Patch(facecolor="w", edgecolor="0.35", lw=2, label="box = middle 50 %"),
                        Line2D([], [], color="k", lw=2, label="median"),
                        Line2D([], [], color="0.35", lw=1.2, marker="_", ms=12, label="whiskers = 1.5 x IQR")],
               loc="lower right", ncol=3, frameon=False, fontsize=10.5, bbox_to_anchor=(0.995, 0.045), handlelength=1.4,
               columnspacing=1.0)
    fig.subplots_adjust(left=0.115, right=0.985, top=0.88, bottom=0.2)
    FF.note(fig, "Left: one row per neuron, grouped by mouse; colour = transient rate in the stimulus blocks (still frames); "
                 "baseline = mean of the neuron's baseline sessions. Right: log2 ((injection + 0.1) / (same-day baseline + 0.1)) "
                 "per neuron; star over 'vs 0' = Wilcoxon signed-rank vs 0 over the mouse's neurons. " + PS.KEY if LANG != "kr" else
            "왼쪽: 뉴런당 한 줄, 마우스별; 색 = 자극 블록(움직임 없는 프레임) transient 빈도; 기준 = 뉴런의 기준 세션 평균. 오른쪽: 뉴런별 "
            "log2 ((주사 + 0.1) / (같은 날 기준 + 0.1)); 'vs 0' 위 별 = 마우스 뉴런에 대한 Wilcoxon (0 대비). " + KEY_KR, LANG)
    fig.savefig(out, dpi=150); plt.close(fig)
    return dict(n=len(S), n_both=int((S == "both days").sum()), n_one=int((S != "both days").sum()))


def fig_population(E, D, events, out):
    _rc()
    n = len(events)
    fig, ax = plt.subplots(2, n, figsize=(12.8, 5.6), gridspec_kw=dict(height_ratios=[1, 1.05]))
    res = []
    for k, ev in enumerate(events):
        a = ax[0, k]
        X = E[E.event == ev]
        vals = {c: X[X.condition == c].auc_post.dropna().values for c in COND}
        v4 = {g: sub4(X, g) for g in G4}
        for g in G4:
            v = v4[g].auc_post.dropna().values
            if len(v):
                a.bar(X4[g], np.mean(v), yerr=np.std(v) / np.sqrt(len(v)), color=C4[g], alpha=0.85, capsize=4, width=0.7)
            for j, m in enumerate(MO):
                w = v4[g][v4[g].mouse == m].auc_post.dropna()
                if len(w):
                    a.scatter(X4[g] + (j - 1.5) * 0.12, w.mean(), marker=MK[m], color=MC[m], s=30, zorder=4, edgecolor="w", lw=0.6,
                              clip_on=False)
        a.axhline(0, color="0.3", lw=0.8)
        p = safe_mwu(vals["SBI-553"], vals["vehicle"])
        ticks4g(a, fs=10.5)
        a.set_title(f"{ev}\n{wtxt(ev)}", fontsize=11.5)
        tops = [v4[g].auc_post.mean() + (v4[g].auc_post.std() / np.sqrt(v4[g].auc_post.notna().sum())
                                         if v4[g].auc_post.notna().sum() > 1 else 0) for g in G4 if v4[g].auc_post.notna().any()]
        tops += [v4[g][v4[g].mouse == m].auc_post.mean() for g in G4 for m in MO if v4[g][v4[g].mouse == m].auc_post.notna().any()]
        lo, hi = a.get_ylim()
        top = max(tops + [0])
        span = hi - lo
        a.set_ylim(lo, top + 0.52 * span)
        pwd = {}
        for drug, x in (("vehicle", 1), ("SBI-553", 2)):
            M_ = within_day(X, drug, "auc_post")
            pwd[drug] = wsr_p(M_.auc_post_b, M_.auc_post_i)
        brackets4(a, top + 0.06 * span, top + 0.28 * span, pwd["vehicle"], pwd["SBI-553"], p, fs=11.5)
        if k == 0:
            a.set_ylabel("response (z):\nwindow minus\nbaseline window", fontsize=11)
        se_ = lambda v: np.std(v) / np.sqrt(len(v)) if len(v) > 1 else np.nan
        res.append(dict(event=ev, p=p, p_within_vehicle=pwd["vehicle"], p_within_sbi=pwd["SBI-553"],
                        **{f"mean_{c}": float(np.mean(vals[c])) if len(vals[c]) else np.nan for c in COND},
                        **{f"n_{c}": len(vals[c]) for c in COND},
                        zdiff=(np.mean(vals["SBI-553"]) - np.mean(vals["vehicle"])) / np.sqrt(se_(vals["SBI-553"]) ** 2 + se_(vals["vehicle"]) ** 2)
                        if min(len(vals["SBI-553"]), len(vals["vehicle"])) > 1 else np.nan))
        b = ax[1, k]
        for c in COND:
            M = [D[key]["mean"] for key in D if key[3] == ev and cond_of(key[0], key[1], key[2]) == c]
            if not M:
                continue
            tt = next(D[key]["tt"] for key in D if key[3] == ev and cond_of(key[0], key[1], key[2]) == c)
            Y = np.concatenate(M, 1)
            mu = CA.sm(np.nanmean(Y, 1)[:, None], 6)[:, 0]
            se = CA.sm((np.nanstd(Y, 1) / np.sqrt(np.sum(np.isfinite(Y), 1)))[:, None], 6)[:, 0]
            b.plot(tt, mu, color=CC[c], lw=2, label=f"{c} (n={Y.shape[1]})")
            b.fill_between(tt, mu - se, mu + se, color=CC[c], alpha=0.2, lw=0)
        (p0_, p1_), (a_, b_) = wwin(ev)
        b.axvspan(a_, b_, color="0.86", zorder=0); b.axvspan(p0_, p1_, color="#dbe9f6", zorder=0)
        b.axvline(0, color="0.4", lw=0.8, ls="--"); b.axhline(0, color="0.6", lw=0.6)
        b.set_xlim(-3, 6); b.set_xlabel("time from event (s)")      # 10-23: the per-panel legend hid the traces (one shared legend)
        if k == 0:
            b.set_ylabel(ZLAB3, fontsize=11)
        b.set_xticks([-2, 0, 2, 4, 6])
    T = pd.DataFrame(res)
    T["diff"] = T["mean_SBI-553"] - T["mean_vehicle"]
    sig = T[T.p < 0.05]
    pick = (sig if len(sig) else T).assign(ad=lambda q: q["zdiff"].abs().fillna(0)).sort_values("ad").iloc[-1]
    k = events.index(pick.event)
    fig.tight_layout(rect=(0, 0.055, 1, 0.9), h_pad=0.6)
    fig.legend(handles=[Line2D([], [], ls="", marker=MK[m], color=MC[m], ms=7, label=m) for m in MO], loc="upper right", ncol=4,
               frameon=False, fontsize=10.5, title="marker = mouse mean (bars)", title_fontsize=10.5, bbox_to_anchor=(0.995, 1.0))
    fig.legend(handles=[Line2D([], [], color=CC[c], lw=2.2, label=c) for c in COND], loc="upper left", ncol=3, frameon=False,
               fontsize=10.5, title="colour = condition; bars and traces = mean +- SEM of the neurons", title_fontsize=10.5,
               bbox_to_anchor=(0.005, 1.0))
    # 10-24: the shaded bands and the dashed line of the traces keyed on the figure (they were explained only in the footer)
    fig.legend(handles=[Patch(color="0.86", label="traces: response window"), Patch(color="#dbe9f6", label="baseline window"),
                        Line2D([], [], color="0.4", lw=1, ls="--", label="event onset (0 s)")], loc="lower right", ncol=3,
               frameon=False, fontsize=10.5, bbox_to_anchor=(0.995, 0.0), handlelength=1.6, columnspacing=1.2)
    pairnote(fig, "bars = the response in the best window (grey = response window, blue = baseline window in the traces). "
                  + PAIRNOTE4, kr="막대 = 최적 창의 반응 (trace에서 회색 = 반응 창, 파랑 = 기준 창). " + PAIRNOTE4_KR)
    fig.savefig(out, dpi=150); plt.close(fig)
    return T, pick


def fig_groups(E, D, events, out):
    """Ntsr1 'group averages: up, down, rest' - per condition, the mean trace of the UP, the DOWN and the other neurons"""
    _rc()
    H = 2.75 * len(events) + 0.6                                       # + the key of the bands on top (10-24)
    fig, ax = plt.subplots(len(events), 3, figsize=(12.4, H), sharex=True)
    ax = np.atleast_2d(ax)
    gcol = {"UP": "#e74c3c", "DOWN": "#2e86c1", "rest": "#95a5a6"}
    res = []
    for r, ev in enumerate(events):
        for c_, c in enumerate(COND):
            a = ax[r, c_]
            tr = {"UP": [], "DOWN": [], "rest": []}
            tt = None
            for key, K in D.items():
                if key[3] != ev or cond_of(key[0], key[1], key[2]) != c:
                    continue
                m, d, ph, _ = key
                X = E[(E.mouse == m) & (E.day == d) & (E.phase == ph) & (E.event == ev)].set_index("cell").cls_shift
                tt = K["tt"]
                for j, cid in enumerate(K["ids"]):
                    g = X.get(int(cid), "no response")
                    tr["UP" if g == "UP" else "DOWN" if g == "DOWN" else "rest"].append(K["mean"][:, j])
            nn_ = {}
            for g, L_ in tr.items():
                if not L_:
                    continue
                Y = np.array(L_).T
                mu = CA.sm(np.nanmean(Y, 1)[:, None], 6)[:, 0]
                se = CA.sm((np.nanstd(Y, 1) / np.sqrt(Y.shape[1]))[:, None], 6)[:, 0]
                a.plot(tt, mu, color=gcol[g], lw=2)
                a.fill_between(tt, mu - se, mu + se, color=gcol[g], alpha=0.2, lw=0)
                res.append(dict(event=ev, condition=c, group=g, n=Y.shape[1]))
                nn_[g] = Y.shape[1]
            (p0_, p1_), (a_, b_) = wwin(ev)                                # this event's own window (WIN_MODE=mix differs per event)
            a.axvspan(a_, b_, color="0.88", zorder=0); a.axvspan(p0_, p1_, color="#dbe9f6", zorder=0)
            a.axvline(0, color="0.4", lw=0.8, ls="--"); a.axhline(0, color="0.6", lw=0.6)
            a.set_xlim(-3, 6)
            # 10-24: the n of each group as the panel's title line (the in-plot legends sat on the traces)
            ttl = (f"n: UP {nn_.get('UP', 0)}, DOWN {nn_.get('DOWN', 0)}, rest {nn_.get('rest', 0)}" if nn_ else
                   f"no session with >= {CA.MIN_TRIALS} events")
            a.set_title(ttl, fontsize=11.5, color="0.15")
            if r == 0:                                                     # the condition name, bold and coloured, above it
                a.annotate(c, xy=(0.5, 1.0), xycoords="axes fraction", xytext=(0, 23), textcoords="offset points", ha="center",
                           va="bottom", fontsize=14, fontweight="bold", color=CC[c])
            if c_ == 0:
                a.set_ylabel(f"{ev}\nwindow {a_:g}-{b_:g} s vs {p0_:g}..{p1_:g} s", fontsize=11.5)
            if r == len(events) - 1:
                a.set_xlabel("time from event (s)")
    for a in ax.ravel():
        if not a.lines or all(len(l_.get_xdata()) <= 2 for l_ in a.lines):    # empty panel: no meaningless 0-1 y axis
            a.set_yticks([])
    fig.supylabel(ZLAB, fontsize=11.5, x=0.005)
    fig.tight_layout(rect=(0.02, 0, 1, 1 - 0.42 / H))
    fig.legend(handles=[Line2D([], [], color="#e74c3c", lw=2, label="UP"), Line2D([], [], color="#2e86c1", lw=2, label="DOWN"),
                        Line2D([], [], color="#95a5a6", lw=2, label="rest (mean +- SEM)"),
                        Patch(color="0.88", label="response window"), Patch(color="#dbe9f6", label="baseline window"),
                        Line2D([], [], color="0.4", lw=1, ls="--", label="event onset (0 s)")],
               loc="upper center", ncol=6, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 1.0), handlelength=1.6,
               columnspacing=1.4)
    fig.savefig(out, dpi=150); plt.close(fig)
    return pd.DataFrame(res)


def fig_heatmap(D, rows, ev, out):
    """every neuron (rows, grouped by mouse; neurons of one day only are blank on the other day) in the three conditions"""
    tt = next((K["tt"] for k, K in D.items() if k[3] == ev), None)
    if tt is None:
        return None
    blocks = []
    (p0_, p1_), (a_, b_) = wwin(ev)
    rw = (tt >= a_) & (tt < b_)                                       # the response window of this event
    for m in MO:
        def col(d, ph, cid):
            K = D.get((m, d, ph, ev))
            if K is None or cid is None:
                return np.full(len(tt), np.nan)
            i = np.flatnonzero(K["ids"] == cid)
            return K["mean"][:, i[0]] if len(i) else np.full(len(tt), np.nan)
        T3, st = [], []
        for s_, ids in rows[m]:
            base = [col(d, "pre", ids.get(d)) for d in DAYS]
            b = np.nanmean(np.vstack(base), 0) if np.isfinite(np.vstack(base)).any() else np.full(len(tt), np.nan)
            inj = {"vehicle": np.full(len(tt), np.nan), "SBI-553": np.full(len(tt), np.nan)}
            for d in DAYS:
                if ids.get(d) is not None:
                    inj[ARMS[d][m]] = col(d, "post", ids.get(d))
            T3.append((b, inj["vehicle"], inj["SBI-553"])); st.append(s_)
        if not T3:
            continue
        base = np.array([t[0] for t in T3]); veh = np.array([t[1] for t in T3]); sbi = np.array([t[2] for t in T3])
        st = np.array(st)
        order = np.lexsort((-np.nan_to_num(np.nanmean(base[:, rw], 1), nan=-9),
                            [{"both days": 0, "Day 1 only": 1, "Day 2 only": 2}.get(s, 0) for s in st]))
        blocks.append((m, base[order], veh[order], sbi[order], st[order]))
    if not blocks:
        return None
    # 10-24: fixed layout instead of bbox 'tight' around a legend hanging under the figure (that made the PNG tall, so the deck
    # shrank every font to ~7 pt); y label, strip title, strip key, colour-bar unit; an empty condition says why it is empty
    _rc()
    fig = plt.figure(figsize=(12.8, 5.4))
    # 10-25: no suptitle (it repeated the slide title) -> the maps get that height
    gs = fig.add_gridspec(1, 4, width_ratios=[0.12, 1, 1, 1], wspace=0.06, left=0.135, right=0.86, top=0.925, bottom=0.2)
    a0 = fig.add_subplot(gs[0]); ax = [fig.add_subplot(gs[i]) for i in (1, 2, 3)]
    S = np.concatenate([b[4] for b in blocks])
    scol = {"both days": "#2ecc40", "Day 1 only": "#f012be", "Day 2 only": "#00d0ff", "Day 1": "#2ecc40", "Day 2": "#2ecc40"}
    a0.imshow(np.array([[matplotlib.colors.to_rgb(scol.get(s, "#2ecc40"))] for s in S]), aspect="auto", interpolation="nearest")
    a0.set_xticks([]); a0.set_yticks([])
    if len(DAYS) == 2:
        a0.set_title("recorded\non", fontsize=11)
    a0.set_ylabel("neurons (1 row = 1 neuron),\ngrouped by mouse", fontsize=11.5, labelpad=46)
    cm = plt.get_cmap("RdBu_r").copy(); cm.set_bad("#e6e6e6")
    vmax = 0.5                                                        # 10-25: responses are ~0.002-0.3 z; +-1.5 looked white
    means = []
    for k, (c, idx) in enumerate(zip(COND, (1, 2, 3))):
        Y = np.vstack([b[idx] for b in blocks])
        Ys = CA.sm(np.nan_to_num(Y.T), 6).T
        Ys[~np.isfinite(Y)] = np.nan
        im = ax[k].imshow(np.ma.masked_invalid(Ys), aspect="auto", cmap=cm, vmin=-vmax, vmax=vmax,
                          extent=[tt[0], tt[-1], len(Y), 0], interpolation="nearest")
        ax[k].axvline(0, color="k", lw=0.8, ls="--"); ax[k].set_xlim(-3, 6); ax[k].set_xlabel("time from event (s)", fontsize=12)
        ax[k].set_xticks([-2, 0, 2, 4])                                # 10-25: same ticks in every panel (no '6' next to '-2')
        ax[k].tick_params(labelsize=11)
        ax[k].axvline(b_, color="k", lw=0.7, ls=":")
        ax[k].set_title(c, color=CC[c], fontsize=14, fontweight="bold"); ax[k].set_yticks([])
        y0 = 0
        for m, b, *_ in blocks:
            ax[k].axhline(y0, color="k", lw=0.8)
            y0 += len(b)
        if not np.isfinite(Y).any():                                  # e.g. grooming after SBI-553: no session had enough events
            ax[k].text(1.5, len(Y) / 2, f"no session with\n>= {CA.MIN_TRIALS} events\n" + ("at baseline" if c == "baseline" else f"after {c}"),
                       ha="center", va="center", fontsize=12, color="0.25",
                       bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.6", lw=0.8))
        means.append(float(np.nanmean(Y[:, rw])) if np.isfinite(Y[:, rw]).any() else np.nan)
    y0 = 0
    for m, b, *_ in blocks:
        a0.axhline(y0 - 0.5, color="k", lw=1.2)
        a0.text(-0.8, y0 + len(b) / 2, m, ha="right", va="center", fontsize=12, fontweight="bold", color=MC[m])
        y0 += len(b)
    cax = fig.add_axes([0.875, 0.3, 0.012, 0.55])
    cb = fig.colorbar(im, cax=cax, ticks=[-0.5, -0.25, 0, 0.25, 0.5], extend="both")   # 10-25: clipped at +-0.5 z
    cb.ax.set_yticklabels(["<= -0.5", "-0.25", "0", "+0.25", ">= +0.5"])
    cb.set_label("z (dF/F / robust SD), trial mean,\nminus the mean of the 3 s before", fontsize=11)
    cb.ax.tick_params(labelsize=11)
    if len(DAYS) == 2:
        lab_ = {"both days": "neuron found on both days", "Day 1 only": "Day 1 only", "Day 2 only": "Day 2 only"}
        fig.legend(handles=[Patch(color=v, label=lab_[k]) for k, v in scol.items() if k in set(S) and k in lab_],
                   loc="lower left", ncol=3, frameon=False, fontsize=11, bbox_to_anchor=(0.005, 0.0), handlelength=1.4,
                   columnspacing=1.2)
    fig.legend(handles=[Patch(color="#e6e6e6", label="grey = not in that condition"),
                        Line2D([], [], color="k", lw=0.8, ls="--", label="event onset"),
                        Line2D([], [], color="k", lw=0.8, ls=":", label="end of the response window")],
               loc="lower right", ncol=3, frameon=False, fontsize=11, bbox_to_anchor=(0.995, 0.0), handlelength=1.6,
               columnspacing=1.2)
    fig.savefig(out, dpi=150); plt.close(fig)
    return dict(n=len(S), n_both=int((S == "both days").sum()), mean=dict(zip(COND, means)))


def fig_responsive(E, events, out):
    _rc()
    fig, ax = plt.subplots(1, len(events), figsize=(12.8, 4.8), sharey=True)
    res = []
    gtop, gbot = 5.0, 5.0
    for ev in events:
        X = E[E.event == ev]
        for c in COND:
            for m in [None] + MO:
                Y = X[(X.condition == c) & ((X.mouse == m) if m else True)]
                if len(Y):
                    gtop = max(gtop, 100 * (Y.cls_shift == "UP").mean())
                    gbot = max(gbot, 100 * (Y.cls_shift == "DOWN").mean())
    for k, ev in enumerate(events):
        a = ax[k]
        X = E[E.event == ev]
        for c in COND:                                          # the numbers for the deck text (baseline = both days)
            Y = X[X.condition == c]
            res.append(dict(event=ev, condition=c, n=int(len(Y)), up=int((Y.cls_shift == "UP").sum()),
                            down=int((Y.cls_shift == "DOWN").sum())))
        for g in G4:                                            # the plot: each injection next to its own day's baseline
            Y = sub4(X, g)
            up, dn, n = (Y.cls_shift == "UP").sum(), (Y.cls_shift == "DOWN").sum(), len(Y)
            a.bar(X4[g], 100 * up / max(n, 1), color=C4[g], width=0.7)
            a.bar(X4[g], -100 * dn / max(n, 1), color=C4[g], width=0.7, alpha=0.45, hatch="//", edgecolor="w")
            for j, m in enumerate(MO):
                Z = Y[Y.mouse == m]
                if len(Z):                                      # 10-24: white edge + above the zero line (markers at 0 stay visible)
                    a.scatter(X4[g] + (j - 1.5) * 0.12, 100 * (Z.cls_shift == "UP").mean(), marker=MK[m], color=MC[m], s=30,
                              zorder=5, edgecolor="w", lw=0.6, clip_on=False)
                    a.scatter(X4[g] + (j - 1.5) * 0.12, -100 * (Z.cls_shift == "DOWN").mean(), marker=MK[m], color=MC[m], s=30,
                              zorder=5, edgecolor="w", lw=0.6, clip_on=False)
        a.axhline(0, color="k", lw=0.8, zorder=2)
        ticks4g(a, fs=10.5); a.set_title(ev, fontsize=13)
        Rv = {c: (X[X.condition == c].cls_shift == "UP").sum() for c in COND}
        Nv = {c: (X.condition == c).sum() for c in COND}
        p = fisher_exact([[Rv["SBI-553"], Nv["SBI-553"] - Rv["SBI-553"]], [Rv["vehicle"], Nv["vehicle"] - Rv["vehicle"]]])[1] \
            if Nv["SBI-553"] and Nv["vehicle"] else np.nan
        res[-1]["p_up"] = p
        a.set_ylim(-gbot - 6, gtop + 31)
        pm_ = {}
        for drug, x in (("vehicle", 1), ("SBI-553", 2)):
            M_ = within_day(X, drug)
            ub, ui = (M_.cls_shift_b == "UP").values, (M_.cls_shift_i == "UP").values
            pm_[drug] = pm = mcnemar((ub & ~ui).sum(), (~ub & ui).sum())
            res[-1][f"p_mcnemar_{drug}"] = pm
        brackets4(a, gtop + 4, gtop + 18, pm_["vehicle"], pm_["SBI-553"], p, fs=11)
    ax[0].set_ylabel("% of neurons\nUP (solid, up) / DOWN (hatched, down)")
    ax[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{abs(v):g}"))   # DOWN % shown as positive
    T = pd.DataFrame(res)
    T["pct_up"] = 100 * T.up / T.n.clip(lower=1); T["pct_down"] = 100 * T.down / T.n.clip(lower=1)
    W = T.pivot(index="event", columns="condition", values="pct_up").reindex(columns=COND)
    W["d"] = W["SBI-553"] - W["vehicle"]
    ev_pick = W.d.abs().idxmax()
    k = events.index(ev_pick)
    fig.tight_layout(rect=(0, 0.05, 1, 0.84))
    # 10-24: the bar colours and solid / hatched keyed on the figure (only blue and red were explained, in the footer)
    fig.legend(handles=[Patch(color=CC["baseline"], label="baseline"), Patch(color=CC["vehicle"], label="vehicle"),
                        Patch(color=CC["SBI-553"], label="SBI-553"), Patch(facecolor="0.45", label="UP (solid)"),
                        Patch(facecolor="0.45", alpha=0.45, hatch="//", edgecolor="w", label="DOWN (hatched)")],
               loc="upper left", ncol=5, frameon=False, fontsize=10.5, bbox_to_anchor=(0.005, 1.0), title="bars = % of all neurons",
               title_fontsize=10.5, handlelength=1.5, columnspacing=1.1)
    fig.legend(handles=[Line2D([], [], ls="", marker=MK[m], color=MC[m], ms=7, label=m) for m in MO], loc="upper right", ncol=4,
               frameon=False, fontsize=10.5, title="marker = mouse", title_fontsize=10.5, bbox_to_anchor=(0.995, 1.0))
    pairnote(fig, "% UP: base = the baseline session of that day. Blue / red bracket = the same neurons, that day's baseline vs after "
                  "the injection (McNemar); black = after vehicle vs after SBI-553 (Fisher).  " + PS.KEY,
             kr="% UP: base = 그날의 기준 세션. 파랑 / 빨강 괄호 = 같은 뉴런, 그날 기준 vs 주사 후 (McNemar); 검정 괄호 = vehicle 후 vs "
                "SBI-553 후 (Fisher).  " + KEY_KR)
    fig.savefig(out, dpi=150); plt.close(fig)
    return T, ev_pick, W


def fig_table(TR, events, out):
    """Ntsr1-deck 'Summary': UP / DOWN counts per event and condition, all neurons pooled"""
    _rc()
    cells, colc = [], []
    for ev in events:
        r_, c_ = [], []
        for c in COND:
            q = TR[(TR.event == ev) & (TR.condition == c)]
            if len(q) and int(q.n.iloc[0]) == 0:                    # 10-24: say why the cell is empty
                r_.append(f"not tested\n(< {CA.MIN_TRIALS} events)"); c_.append("#f4f4f4")
            elif len(q):
                u, d_, n = int(q.up.iloc[0]), int(q.down.iloc[0]), int(q.n.iloc[0])
                r_.append(f"{u} UP ({100 * u / max(n, 1):.0f}%)\n{d_} DOWN ({100 * d_ / max(n, 1):.0f}%)\nof {n}")
                c_.append(matplotlib.colors.to_rgba("#e74c3c", min(0.15 + 2.5 * u / max(n, 1), 0.9)))
            else:
                r_.append("-"); c_.append("w")
        cells.append(r_); colc.append(c_)
    if len(events) > 6:                                       # many events: one column per event, one row per condition
        cells = [list(r) for r in zip(*cells)]; colc = [list(r) for r in zip(*colc)]
        fig, ax = plt.subplots(figsize=(12.6, 4.2))
        ax.set_axis_off()
        tb = ax.table(cellText=cells, rowLabels=COND, colLabels=[e.replace(" onset", "").replace("escape/rearing", "escape")
                                                                     for e in events], cellColours=colc, loc="center", cellLoc="center")
        tb.auto_set_font_size(False); tb.set_fontsize(9.5); tb.scale(1, 3.6)
        for (r, c), cell in tb.get_celld().items():
            if r == 0:
                cell.set_text_props(fontweight="bold", fontsize=10.5)
            if c == -1:
                cell.set_text_props(fontweight="bold", color=CC[COND[r - 1]] if r >= 1 else "k", fontsize=11)
    else:
        fig, ax = plt.subplots(figsize=(12.4, 0.95 * len(events) + 0.9))
        ax.set_axis_off()
        tb = ax.table(cellText=cells, rowLabels=events, colLabels=COND, cellColours=colc, loc="center", cellLoc="center")
        tb.auto_set_font_size(False); tb.set_fontsize(12.5); tb.scale(1, 3.2)
        for (r, c), cell in tb.get_celld().items():
            if r == 0:
                cell.set_text_props(fontweight="bold", color=CC[COND[c]] if c >= 0 else "k", fontsize=14)
            if c == -1:
                cell.set_text_props(fontweight="bold", fontsize=13)
    fig.tight_layout()
    # 10-24: the shading key ON the figure (it was only in the caption): cell colour = the same alpha rule as above, 0 to 30 % UP
    from matplotlib.colors import ListedColormap, Normalize
    pp_ = np.linspace(0, 0.30, 256)
    al_ = np.minimum(0.15 + 2.5 * pp_, 0.9)[:, None]
    red_ = np.array(matplotlib.colors.to_rgb("#e74c3c"))[None, :]
    cmk = ListedColormap(al_ * red_ + (1 - al_) * np.ones((1, 3)))
    fig.canvas.draw()
    bb = tb.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())   # the table, figure fraction
    cax = fig.add_axes([bb.x1 - 0.3 * bb.width, bb.y0 - 0.17, 0.3 * bb.width, 0.04])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 30), cmap=cmk), cax=cax, orientation="horizontal", ticks=[0, 10, 20, 30])
    cb.ax.set_xticklabels(["0", "10", "20", ">= 30"]); cb.ax.tick_params(labelsize=10.5)
    cax.set_title("cell shade = % of neurons UP", fontsize=10.5, loc="left")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)


def footprints(B, m, d):
    t = f"{m}_d{d}"
    root = os.path.join(B, "mini1p_out", "20261019" if d == 1 else "20261021", t)
    S = sio.loadmat(os.path.join(root, "3_extract", f"{t}_extract.mat"), variable_names=["spatial_weights"])["spatial_weights"]
    mx = sio.loadmat(os.path.join(root, "3_extract", f"{t}_projections.mat"), variable_names=["max_dff"])["max_dff"]
    return np.asarray(S, np.float32), np.asarray(mx, np.float32)


def fig_maps(B, E, ev, out, FPc, mice=None):
    """Ntsr1 'responsive cells per event': per mouse its sessions; red = UP, blue = DOWN, grey = the other neurons;
    mice = two mice -> one row per mouse, bigger panels (10-22, legibility)"""
    _rc()
    cols = [(d, ph) for d in DAYS for ph in ("pre", "post")]
    nc = len(cols)
    MM = mice or MO
    NOC_ = "#d9d9d9"                                  # 10-24: 'no response' outlines lighter and thicker (were hard to see)
    if mice:                                          # the deck shows it 5.2 in high: 14 pt -> ~9.6 pt on the slide
        fig, ax = plt.subplots(len(MM), nc, figsize=(3.6 * nc, 3.5 * len(MM) + 0.6), squeeze=False)
        fs_t, fs_l = 14, 14
    else:
        fig, ax = plt.subplots(2, 2 * nc, figsize=(16 if nc == 4 else 9.5, 5.6))
        fs_t, fs_l = 12, 12
    for a in ax.ravel():
        a.set_anchor("N")                             # the image right under its title (10-24: titles ran into the row above)
    for r_, m in enumerate(MM):
        r, c0 = (r_, 0) if mice else (r_ // 2, nc * (r_ % 2))
        for c_, (d, ph) in enumerate(cols):
            a = ax[r, c0 + c_]; a.set_axis_off()
            if (m, d) not in FPc:
                FPc[(m, d)] = footprints(B, m, d)
            S, mx = FPc[(m, d)]
            lo, hi = np.percentile(mx, [1, 99.8])
            X = E[(E.mouse == m) & (E.day == d) & (E.phase == ph) & (E.event == ev)]
            cnd = cond_of(m, d, ph)
            if not len(X):                            # this event was not tested in this session: greyed, says why
                a.imshow(np.clip((mx - lo) / (hi - lo), 0, 1) ** 0.7, cmap="gray", alpha=0.25)
                a.text(0.5, 0.5, f"fewer than {CA.MIN_TRIALS}\nevents in this\nsession: not tested", transform=a.transAxes,
                       ha="center", va="center", fontsize=fs_t - 0.5, color="0.2", linespacing=1.2)
                a.set_title(f"{m}  D{d} {cnd}\nnot tested", fontsize=fs_t, color="0.5")
                continue
            a.imshow(np.clip((mx - lo) / (hi - lo), 0, 1) ** 0.7, cmap="gray")
            nu = nd = 0
            for q in X.itertuples():
                j = int(q.cell) - 1
                if j >= S.shape[2] or S[:, :, j].max() <= 0:
                    continue
                f = S[:, :, j]
                col = {"UP": "#ff2d2d", "DOWN": "#2f8cff"}.get(q.cls_shift, NOC_)
                a.contour(f >= 0.3 * f.max(), levels=[0.5], colors=[col], linewidths=1.8 if q.cls_shift in ("UP", "DOWN") else 0.9)
                nu += q.cls_shift == "UP"; nd += q.cls_shift == "DOWN"
            cnt = f"{nu} UP, {nd} DOWN of {len(X)} neurons" if mice else f"{nu} UP, {nd} DOWN\nof {len(X)} neurons"
            a.set_title(f"{m}  D{d} {cnd}\n{cnt}", fontsize=fs_t, color="0.1",
                        fontweight="bold" if ph == "post" else "normal")   # 10-23: red / blue = UP / DOWN only
    # 10-24: outline-style key (the cells are outlines), readable size, and what the background is; no scale bar: the um per
    # pixel of this rig is not measured (PIPELINE_mini1p.md, 'Open')
    oc = lambda c, w, lab: Line2D([], [], ls="none", marker="o", ms=13, mfc="none", mec=c, mew=w, label=lab)
    fig.legend(handles=[oc("#ff2d2d", 2.2, "UP"), oc("#2f8cff", 2.2, "DOWN"), oc("#b0b0b0", 1.4, "no response"),
                        Patch(facecolor="0.25", edgecolor="0.25", label="background = max dF/F projection of that day")],
               loc="upper center", ncol=4, frameon=False, fontsize=fs_l, columnspacing=1.6, handletextpad=0.4)
    H_ = fig.get_figheight()
    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.01, top=1 - (0.55 + (0.62 if mice else 0.75)) / H_, wspace=0.06,
                        hspace=0.3 if mice else 0.42)
    fig.savefig(out, dpi=130); plt.close(fig)


def fig_fov_change(B, out, FPc, mice):
    """old A5 for the crossover: every neuron's footprint coloured by its whole-session rate change after the injection
    (log2 (injection + 0.1) / (baseline of the same day + 0.1)); per mouse the vehicle day and the SBI-553 day"""
    from matplotlib.colors import Normalize
    _rc()
    W = {}
    for d in DAYS:
        X = pd.read_csv(os.path.join(B, "analysis", f"ca_day{d}", f"ca_d{d}_cells_whole_session.csv")); X["mouse"] = X.mouse.astype(str)
        W[d] = X.pivot_table(index=["mouse", "cell"], columns="phase", values="rate")
    cm, nm = plt.get_cmap("RdBu_r"), Normalize(-2, 2)
    drugs = ["vehicle", "SBI-553"]
    per = len(drugs) if len(DAYS) == 2 else 1
    fig, ax = plt.subplots(1, per * len(mice), figsize=(3.5 * per * len(mice), 5.0), squeeze=False,
                           gridspec_kw=dict(wspace=0.12))
    res = {}
    for r, m in enumerate(mice):
        for c_, drug in enumerate(drugs if len(DAYS) == 2 else [ARMS[DAYS[0]][m]]):
            a = ax[0, r * per + c_]; a.set_axis_off()
            d = [dd for dd in DAYS if ARMS[dd][m] == drug]
            if not d:
                continue
            d = d[0]
            if (m, d) not in FPc:
                FPc[(m, d)] = footprints(B, m, d)
            S, mx = FPc[(m, d)]
            lo, hi = np.percentile(mx, [1, 99.8])
            a.imshow(np.clip((mx - lo) / (hi - lo), 0, 1) ** 0.7, cmap="gray")
            ov = np.zeros(mx.shape + (4,))
            V = W[d].loc[m] if m in W[d].index.get_level_values(0) else pd.DataFrame()
            ch = []
            for cid, row in V.iterrows():
                if not {"pre", "post"} <= set(row.index) or not np.isfinite(row.pre) or not np.isfinite(row.post):
                    continue
                j = int(cid) - 1
                if j >= S.shape[2] or S[:, :, j].max() <= 0:
                    continue
                v = float(np.log2((row.post + 0.1) / (row.pre + 0.1)))
                ch.append(v)
                ov[S[:, :, j] >= 0.3 * S[:, :, j].max()] = (*cm(nm(v))[:3], 0.95)
            a.imshow(ov)
            ch = np.array(ch)
            res[(m, drug)] = dict(n=len(ch), up=int((ch > 0).sum()), down=int((ch < 0).sum()),
                                  median=float(np.median(ch)) if len(ch) else np.nan)
            # three short lines (10-22: the longer titles ran into the next panel's title)
            a.set_title(f"{m} {drug}, Day {d}\n{res[(m, drug)]['up']} higher, {res[(m, drug)]['down']} lower\nmedian log2 "
                        f"{res[(m, drug)]['median']:+.2f}", fontsize=12, color=CC[drug], linespacing=1.25)
    sm_ = plt.cm.ScalarMappable(norm=nm, cmap=cm)
    cb = fig.colorbar(sm_, ax=ax, shrink=0.9, pad=0.01)
    cb.set_label("log2 (injection / same-day baseline)\nwhole-session transient rate", fontsize=12)
    cb.ax.tick_params(labelsize=11.5)
    fig.canvas.draw()                                                  # under the images, centred (not widening the PNG)
    r_ = fig.canvas.get_renderer()
    bbs = [a.get_images()[0].get_window_extent(r_).transformed(fig.transFigure.inverted()) for a in ax.ravel() if a.get_images()]
    if bbs:
        fig.text((min(b.x0 for b in bbs) + max(b.x1 for b in bbs)) / 2, min(b.y0 for b in bbs) - 0.03,
                 "background = max dF/F projection of that day; coloured patch = one neuron's footprint",
                 fontsize=12, color="0.25", ha="center", va="top")
    fig.savefig(out, dpi=220, bbox_inches="tight"); plt.close(fig)        # 10-25: was 130 dpi (soft when stretched to 12.4 in)
    return res


def fig_resp_change(E, events, out):
    """EVERY neuron, within its day (baseline -> injection of the same day): A = of the neurons responsive at baseline, %
    with the same response after the injection; B = of the others, % responsive after the injection"""
    a_ = E[E.phase == "pre"][["mouse", "day", "cell", "event", "cls_shift"]].rename(columns={"cls_shift": "base"})
    b_ = E[E.phase == "post"][["mouse", "day", "cell", "event", "cls_shift"]].rename(columns={"cls_shift": "inj"})
    T = a_.merge(b_, on=["mouse", "day", "cell", "event"])
    T["drug"] = [ARMS[d][m] for m, d in zip(T.mouse, T.day)]
    _rc()
    fig, ax = plt.subplots(1, 2, figsize=(12.6, 4.8))
    rot = 90 if len(events) > 6 else 0                                   # k/n over narrow bars: vertical (no overlap)
    out_rows = []
    for k, (title, sel) in enumerate((("A  responsive at baseline: the same after the injection", "kept"),
                                       ("B  not responsive at baseline: responsive after the injection", "new"))):
        a = ax[k]
        for i, ev in enumerate(events):
            for j, drug in enumerate(("vehicle", "SBI-553")):
                Z = T[(T.event == ev) & (T.drug == drug)]
                if sel == "kept":
                    Z = Z[Z.base.isin(["UP", "DOWN"])]; v = (Z.base == Z.inj)
                else:
                    Z = Z[~Z.base.isin(["UP", "DOWN"])]; v = Z.inj.isin(["UP", "DOWN"])
                pct = 100 * v.mean() if len(Z) else np.nan
                a.bar(i + (j - 0.5) * 0.38, pct, width=0.36, color=CC[drug], label=drug if i == 0 else None)
                a.text(i + (j - 0.5) * 0.38, (pct if np.isfinite(pct) else 0) + 1, f"{int(v.sum())}/{len(Z)}", ha="center",
                       va="bottom", fontsize=9.5, rotation=rot)
                out_rows.append(dict(kind=sel, event=ev, drug=drug, n=len(Z), k=int(v.sum()), pct=pct))
            o2 = [r for r in out_rows if r["kind"] == sel and r["event"] == ev]
            if len(o2) == 2 and min(r["n"] for r in o2) > 0:
                kv, ks = o2[0], o2[1]
                pf = fisher_exact([[ks["k"], ks["n"] - ks["k"]], [kv["k"], kv["n"] - kv["k"]]])[1]
                out_rows[-1]["p_fisher"] = pf
                ytop = max(np.nan_to_num(kv["pct"]), np.nan_to_num(ks["pct"]))
                if star(pf) not in ("ns", "n.a."):
                    bracket(a, i - 0.19, i + 0.19, ytop + (19 if rot else 9), pf, fs=12)   # above the k/n labels
        a.set_xticks(range(len(events)), [e.replace(" onset", "").replace("escape/rearing", "escape") for e in events],
                     rotation=30 if rot else 15, ha="right" if rot else "center")
        a.set_ylabel("% of neurons"); a.set_title(title, loc="left", fontsize=12.5)
        a.set_ylim(top=a.get_ylim()[1] * 1.1)
    # 10-24: one key above the panels (the per-panel legends sat on the bars)
    fig.legend(handles=[Patch(color=CC["vehicle"], label="vehicle"), Patch(color=CC["SBI-553"], label="SBI-553")], loc="upper right",
               ncol=2, frameon=False, fontsize=10.5, bbox_to_anchor=(0.995, 1.0), title="bar = % of neurons, k/n on top",
               title_fontsize=10.5)
    O = pd.DataFrame(out_rows)
    K = O[O.kind == "kept"].pivot(index="event", columns="drug", values="pct")
    N = O[O.kind == "kept"].pivot(index="event", columns="drug", values="n")
    K["d"] = K["SBI-553"] - K["vehicle"]
    ok = (N.min(1) >= 10) & K.d.notna()
    pick = K.d[ok].abs().idxmax() if ok.any() else (K.d.abs().idxmax() if K.d.notna().any() else events[0])
    i = events.index(pick)
    fig.tight_layout(rect=(0, 0.05, 1, 0.86))
    callout(ax[0], (i + 0.19, float(K.loc[pick, "SBI-553"]) if np.isfinite(K.loc[pick, "SBI-553"]) else 0))
    FF.note(fig, "A: neurons UP or DOWN at the baseline of their day, % with the same class after the injection; B: neurons without "
                 "a response at baseline, % responsive after the injection; k/n on each bar. Bracket = SBI-553 vs vehicle, Fisher "
                 "exact on k/n (drawn only when p < 0.05). " + PS.KEY if LANG != "kr" else
            "A: 그날 기준에서 UP / DOWN 뉴런 중 주사 후 같은 반응 %; B: 기준에 반응 없던 뉴런 중 주사 후 반응 %; 막대 위 k/n. 괄호 = SBI-553 vs "
            "vehicle, k/n Fisher exact (p < 0.05일 때만). " + KEY_KR, LANG)
    fig.savefig(out, dpi=150); plt.close(fig)
    return O, pick, K


# ------------------------------------------------------------------------------------------------ figures: behaviour
def behaviour_table(B):
    S = pd.read_csv(os.path.join(B, "analysis", "behavior_crossover", "crossover_per_session.csv"))
    S["mouse"] = S.mouse.astype(str)
    S = S[S.day.isin(DAYS)].copy()
    for b in BEHS:
        S[f"{b} /del"] = S[b] / S.deliveries.replace(0, np.nan)
    S["affective"] = S[["attending", "lick/bite", "guarding"]].sum(1) / S.deliveries.replace(0, np.nan)
    S["reflexive"] = S[["withdrawal", "flinch"]].sum(1) / S.deliveries.replace(0, np.nan)
    return S


def per_mouse_conditions(S, col):
    M = {}
    for m in MO:
        X = S[S.mouse == m]
        M[m] = [X[X.condition == c][col].mean() if (X.condition == c).any() else np.nan for c in COND]
    return M


def fig_behaviour(S, out):
    _rc()
    fig, ax = plt.subplots(1, len(BEHS), figsize=(12.8, 4.6))
    res = []
    for k, b in enumerate(BEHS):
        a = ax[k]
        M = per_mouse_conditions(S, f"{b} /del")              # the numbers for the deck text (baseline = both days)
        Y = np.array(list(M.values()), float)
        mu = np.nanmean(Y, 0); se = np.nanstd(Y, 0) / np.sqrt(np.sum(np.isfinite(Y), 0).clip(min=1))
        # the plot: per mouse, each day's baseline -> its injection; mean +- SEM of the mice at the 4 positions
        Q = np.full((len(MO), 4), np.nan)
        for i_, m in enumerate(MO):
            Xm = S[S.mouse == m]
            for j_, (dd, ph) in enumerate((("vehicle", "baseline"), ("vehicle", "injection"), ("SBI-553", "baseline"), ("SBI-553", "injection"))):
                Z = Xm[(Xm.drug_day == dd) & (Xm.phase == ph)]
                if len(Z) and Z.deliveries.sum() > 0:
                    Q[i_, j_] = Z[b].sum() / Z.deliveries.sum()
            for j0, j1 in ((0, 1), (2, 3)):
                if np.isfinite(Q[i_, j0]) and np.isfinite(Q[i_, j1]):
                    a.plot([X4[G4[j0]], X4[G4[j1]]], [Q[i_, j0], Q[i_, j1]], color=MC[m], lw=1, alpha=0.7, marker=MK[m], ms=5,
                           clip_on=False)                     # 10-24: markers at 0 stay visible
        mu4 = np.nanmean(Q, 0); se4 = np.nanstd(Q, 0) / np.sqrt(np.sum(np.isfinite(Q), 0).clip(min=1))
        for j0, j1 in ((0, 1), (2, 3)):
            a.errorbar([X4[G4[j0]], X4[G4[j1]]], mu4[[j0, j1]], yerr=se4[[j0, j1]], color="k", lw=2.6, capsize=4, marker="o", ms=7,
                       zorder=3, clip_on=False)
        both = np.isfinite(Y[:, 1]) & np.isfinite(Y[:, 2])
        p = wilcoxon(Y[both, 2], Y[both, 1]).pvalue if both.sum() >= 4 and np.any(Y[both, 2] != Y[both, 1]) else np.nan
        ticks4g(a, fs=9.6, short=True); a.set_xlim(-0.45, 3.8); a.set_title(b, fontsize=12.5)
        top = np.nanmax(Q) if np.isfinite(Q).any() else 1
        a.set_ylim(0, top * 1.62 + 1e-9)
        # pooled over deliveries, stratified by mouse (Mantel-Haenszel rate test)
        pw_ = {}
        for drug, x in (("vehicle", 1), ("SBI-553", 2)):
            k1, n1, k2, n2 = [], [], [], []
            for m in MO:
                X = S[(S.mouse == m) & (S.drug_day == drug)]
                i1, i0 = X[X.phase == "injection"], X[X.phase == "baseline"]
                if len(i1) and len(i0):
                    k1.append(i1[b].sum()); n1.append(i1.deliveries.sum()); k2.append(i0[b].sum()); n2.append(i0.deliveries.sum())
            pw_[drug] = mh_rate_p(k1, n1, k2, n2) if k1 else np.nan
        pmh = np.nan
        if len(DAYS) == 2:
            k1, n1, k2, n2 = [], [], [], []
            for m in MO:
                X = S[S.mouse == m]
                s1, s2 = X[X.condition == "SBI-553"], X[X.condition == "vehicle"]
                if len(s1) and len(s2):
                    k1.append(s1[b].sum()); n1.append(s1.deliveries.sum()); k2.append(s2[b].sum()); n2.append(s2.deliveries.sum())
            pmh = mh_rate_p(k1, n1, k2, n2) if k1 else np.nan
        brackets4(a, top * 1.06, top * 1.32, pw_["vehicle"], pw_["SBI-553"], pmh, fs=11, show_vs=len(DAYS) == 2)
        if k == 0:
            a.set_ylabel("events per delivery")
        res.append(dict(behaviour=b, base=mu[0], vehicle=mu[1], sbi=mu[2], p=p, p_mh=pmh, p_mh_vehicle=pw_["vehicle"],
                        p_mh_sbi=pw_["SBI-553"],
                        n_lower=int(np.sum(Y[both, 2] < Y[both, 1])), n_pairs=int(both.sum())))
    fig.legend(handles=[Line2D([], [], color=MC[m], marker=MK[m], label=m) for m in MO]
               + [Line2D([], [], color="k", lw=2.6, marker="o", label="mean +- SEM of the 4 mice")],
               loc="upper center", ncol=5, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 1.0), title="thin line = one mouse",
               title_fontsize=11)
    T = pd.DataFrame(res)
    T["rel"] = (T.sbi - T.vehicle) / T.vehicle.replace(0, np.nan)
    C_ = T[(T.vehicle >= 0.05) & T.rel.notna()]
    pick = (C_ if len(C_) else T).sort_values(["n_lower", "rel"], ascending=[False, True]).iloc[0]
    k = BEHS.index(pick.behaviour)
    fig.tight_layout(rect=(0, 0.05, 1, 0.84), w_pad=1.2)          # 10-24: the footer went to the notes; space back to the plots
    pairnote(fig, "base = the baseline session of that day. Blue / red bracket = that day's baseline vs after the injection; black = "
                  "after vehicle vs after SBI-553 (deliveries pooled, each mouse its own stratum).\n" + PS.KEY,
             kr="base = 그날의 기준 세션. 파랑 / 빨강 괄호 = 그날 기준 vs 주사 후; 검정 괄호 = vehicle 후 vs SBI-553 후 (자극을 합치고 "
                "마우스마다 층). " + KEY_KR)
    fig.savefig(out, dpi=150); plt.close(fig)
    return T, pick


def fig_behaviour_stim(B, out, only=None):
    """% of deliveries followed by each behaviour, per stimulus: bars baseline / vehicle / SBI-553 (mean of the mice, dot =
    mouse); brackets (only p < 0.05) = the pair compared: base-veh, base-SBI (same mice, same day) and veh-SBI (CMH by mouse)"""
    rows = []
    for d in DAYS:
        T = pd.read_csv(os.path.join(B, "analysis", f"behavior_day{d}", f"behav_d{d}_per_session_stimulus.csv"))
        T["mouse"] = T.mouse.astype(str); T["day"] = d
        T["condition"] = [("baseline" if ph == "baseline" else ARMS[d][m]) for m, ph in zip(T.mouse, T.phase)]
        rows.append(T)
    T = pd.concat(rows, ignore_index=True)
    cols = [("withdrawal_pct", "withdrawal"), ("flinch_pct", "flinch"), ("attending_after_pct", "attending"),
            ("escape/rearing_after_pct", "escape / rearing"), ("general movement_after_pct", "general movement"),
            ("lick/bite_after_pct", "lick / bite")]
    XO = {"baseline": -0.27, "vehicle": 0.0, "SBI-553": 0.27}           # bar offsets inside one stimulus
    _rc()
    if only:
        cols = [c for c in cols if c[1] in only]
        fig, ax = plt.subplots(1, len(cols), figsize=(12.6, 5.0), squeeze=False)
        fsx, fsb = 11, 10.5                                             # shown at 98 %
    else:
        fig, ax = plt.subplots(2, 3, figsize=(12.6, 6.0))
        fsx, fsb = 12, 11.5                                             # shown at 87 % (10-24: >= 9 pt on the slide)
    best = None
    for k, (cname, lab) in enumerate(cols):
        a = ax.ravel()[k]
        mus, per = {}, {}
        for c in COND:
            G = T[T.condition == c].groupby(["mouse", "stim"])[cname].mean().unstack().reindex(columns=STIM)
            mu, se = G.mean(0), G.std(0) / np.sqrt(G.notna().sum(0).clip(lower=1))
            mus[c], per[c] = mu, G
            a.bar(np.arange(4) + XO[c], mu.fillna(0).values, width=0.25, color=CC[c], yerr=se.fillna(0).values, capsize=2,
                  error_kw=dict(lw=0.8), label=c)
            for i, st in enumerate(STIM):
                v = G[st].dropna().values
                a.scatter(np.full(len(v), i + XO[c]), v, s=11, color="k", zorder=4, lw=0, clip_on=False)
        tops = []
        for st in STIM:
            vv = [float(np.nanmax(per[c][st].values)) for c in COND if per[c][st].notna().any()]
            vv += [float(mus[c][st]) for c in COND if np.isfinite(mus[c][st])]
            tops.append(max(vv + [0.0]))
        ymax = max(tops + [1.0])
        R_ = ymax * 1.45 + 6
        a.set_ylim(0, R_)
        for i, st in enumerate(STIM):
            dv = mus["SBI-553"][st] - mus["vehicle"][st]
            if mus["vehicle"][st] >= 5 and np.isfinite(dv) and (best is None or abs(dv) > abs(best[3])):
                best = (k, i, lab, dv, st, mus["vehicle"][st], mus["SBI-553"][st])
            # counts back from the per-session % (n_deliveries x % / 100); Cochran-Mantel-Haenszel by mouse
            Z = T[T.stim == st].copy()
            Z["k"] = np.round(Z[cname] * Z.n_deliveries / 100)
            pcs = {}
            for drug in ("vehicle", "SBI-553"):
                tabs = []
                for m in MO:
                    for d in DAYS:
                        if ARMS[d][m] != drug:
                            continue
                        z1 = Z[(Z.mouse == m) & (Z.day == d) & (Z.phase != "baseline")]
                        z0 = Z[(Z.mouse == m) & (Z.day == d) & (Z.phase == "baseline")]
                        if len(z1) and len(z0):
                            tabs.append([[z1.k.sum(), z1.n_deliveries.sum() - z1.k.sum()], [z0.k.sum(), z0.n_deliveries.sum() - z0.k.sum()]])
                pcs[drug] = cmh_p(tabs) if tabs else np.nan
            pvs = np.nan
            if len(DAYS) == 2:
                tabs = []
                for m in MO:
                    z1 = Z[(Z.mouse == m) & (Z.condition == "SBI-553")]
                    z0 = Z[(Z.mouse == m) & (Z.condition == "vehicle")]
                    if len(z1) and len(z0):
                        tabs.append([[z1.k.sum(), z1.n_deliveries.sum() - z1.k.sum()], [z0.k.sum(), z0.n_deliveries.sum() - z0.k.sum()]])
                pvs = cmh_p(tabs) if tabs else np.nan
            y1 = tops[i] + 0.06 * R_
            # 4 stimuli x 3 pairs per panel: only the pairs with p < 0.05 get a bracket (no bracket = ns)
            PS.bracket(a, i + XO["baseline"], i + XO["vehicle"], y1, pcs["vehicle"], CC["vehicle"], fs=fsb, show_ns=False, inset=0.01)
            PS.bracket(a, i + XO["vehicle"], i + XO["SBI-553"], y1, pvs, "k", fs=fsb, show_ns=False, inset=0.01)
            PS.bracket(a, i + XO["baseline"], i + XO["SBI-553"], y1 + 0.16 * R_, pcs["SBI-553"], CC["SBI-553"], fs=fsb,
                       show_ns=False, inset=0.01)
        a.set_xticks(range(4), ["light", "mild", "pin", "heat"], fontsize=fsx); a.set_title(lab, fontsize=fsx + 1.5)
        a.tick_params(axis="y", labelsize=fsx)
        if k % 3 == 0:
            a.set_ylabel("% of deliveries", fontsize=fsx + 0.5)
    fig.tight_layout(rect=(0, 0.045, 1, 0.9))
    # 10-24: the key above the panels (the legend sat on the bars of the first panel)
    fig.legend(handles=[Patch(color=CC[c], label=c) for c in COND]
               + [Line2D([], [], ls="none", marker="o", color="k", ms=4, label="dot = one mouse; bar = mean +- SEM of the mice"),
                  Line2D([], [], color="k", lw=1.1, label="bracket = pair with p < 0.05 (no bracket = ns)")],
               loc="upper center", ncol=5, frameon=False, fontsize=fsx - 0.5, bbox_to_anchor=(0.5, 1.0), handlelength=1.4,
               columnspacing=1.2)
    pairnote(fig, "bar = mean of the mice, dot = mouse. Brackets = the 2 conditions compared, drawn only when p < 0.05 (no bracket = "
                  "ns): base-veh and base-SBI = the same mice, baseline of that day vs after the injection; veh-SBI = after vehicle "
                  "vs after SBI-553 (Cochran-Mantel-Haenszel, each mouse its own stratum).  " + PS.KEY, fs=10.5,
             kr="막대 = 마우스 평균, 점 = 마우스. 괄호 = 비교한 두 조건, p < 0.05일 때만 (괄호 없음 = ns): 파랑 / 빨강 = 같은 마우스, 그날 기준 vs "
                "주사 후; 검정 = vehicle 후 vs SBI-553 후 (Cochran-Mantel-Haenszel, 마우스마다 층).  " + KEY_KR)
    if best:
        callout(ax.ravel()[best[0]], (best[1] + XO["SBI-553"], best[6]), inside=True)
    fig.savefig(out, dpi=150); plt.close(fig)
    return T, best


def rate_ratio(k1, n1, k2, n2):
    """exact conditional (binomial) rate ratio (k1 / n1) / (k2 / n2), 95 % CI and two-sided p"""
    if k1 + k2 == 0 or n1 <= 0 or n2 <= 0:
        return np.nan, np.nan, np.nan, np.nan
    K = k1 + k2
    p0 = n1 / (n1 + n2)
    lo = _beta.ppf(0.025, k1, K - k1 + 1) if k1 > 0 else 0.0
    hi = _beta.ppf(0.975, k1 + 1, K - k1) if k1 < K else 1.0
    conv = lambda q: (q / (1 - q)) * (n2 / n1) if q < 1 else np.inf
    return conv(k1 / K) if k1 < K else np.inf, conv(lo), conv(hi), binomtest(int(k1), int(K), p0).pvalue


def fig_every_mouse(S, out):
    """WT-deck 'every mouse, every behaviour': per mouse the rate ratio of each behaviour per delivery with its exact 95 %
    CI - combined deck: SBI-553 injection session / vehicle injection session; one day: injection / baseline"""
    _rc()
    fig, ax = plt.subplots(1, len(BEHS), figsize=(12.8, 4.4), sharey=True)
    res = []
    for k, b in enumerate(BEHS):
        a = ax[k]
        for i, m in enumerate(MO):
            X = S[S.mouse == m]
            if len(DAYS) == 2:
                s1, s2 = X[X.condition == "SBI-553"], X[X.condition == "vehicle"]
                col = "k"
            else:
                s1, s2 = X[X.phase == "injection"], X[X.phase == "baseline"]
                col = CC[ARMS[DAYS[0]][m]]
            if not len(s1) or not len(s2):
                continue
            rr, lo, hi, p = rate_ratio(int(s1[b].sum()), int(s1.deliveries.sum()), int(s2[b].sum()), int(s2.deliveries.sum()))
            if np.isfinite(rr) and rr > 0:
                a.plot([max(lo, 1 / 64), min(hi, 64)], [i, i], color=col, lw=2)
                a.plot(rr, i, "o", color=col, ms=8)
                a.text(min(min(hi, 64) * 1.15, 11.5), i, star(p), va="center", fontsize=10.5)
            res.append(dict(behaviour=b, mouse=m, ratio=rr, lo=lo, hi=hi, p=p))
        if len(a.lines) == 0:                                          # 10-24: say why the panel is empty
            a.text(1, 1.5, "no ratio:\n0 events in\na session", ha="center", va="center", fontsize=10.5, color="0.4")
        a.axvline(1, color="0.5", ls="--", lw=1)
        a.set_xscale("log"); a.set_xlim(1 / 16, 16)
        a.set_xticks([1 / 4, 1, 4], ["1/4", "1", "4"]); a.minorticks_off()
        a.set_title(b, fontsize=12.5)
    ax[0].set_yticks(range(len(MO)), MO); ax[0].invert_yaxis()
    xl = "SBI-553 / vehicle (injection sessions)" if len(DAYS) == 2 else "injection / baseline"
    pair_ = "SBI-553 vs vehicle" if len(DAYS) == 2 else "injection vs baseline"
    fig.supxlabel(f"rate ratio per delivery: {xl}\nstar / ns next to a mouse = {pair_} in that mouse (exact test)", fontsize=11.5,
                  y=0.065)
    T = pd.DataFrame(res)
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    FF.note(fig, "Rate ratio per delivery with its exact 95 % CI (bar); star / ns = exact conditional (binomial) test in that mouse, "
                 "uncorrected. " + PS.KEY if LANG != "kr" else
            "자극당 비율과 정확 95 % CI (막대); 별 / ns = 그 마우스의 정확 이항 검정, 보정 없음. " + KEY_KR, LANG)
    fig.savefig(out, dpi=150); plt.close(fig)
    return T


def fig_affective(S, out):
    """WT-deck 'affective and reflexive': one arrow per mouse (combined: vehicle -> SBI-553 injection session; one day:
    baseline -> injection); a selective effect on affective pain would point straight DOWN"""
    _rc()
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    for m in MO:
        X = S[S.mouse == m]
        if len(DAYS) == 2:
            a0, a1 = X[X.condition == "vehicle"], X[X.condition == "SBI-553"]
            col = MC[m]
        else:
            a0, a1 = X[X.phase == "baseline"], X[X.phase == "injection"]
            col = CC[ARMS[DAYS[0]][m]]
        if not len(a0) or not len(a1):
            continue
        p0 = (a0.reflexive.mean(), a0.affective.mean()); p1 = (a1.reflexive.mean(), a1.affective.mean())
        ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="-|>", color=col, lw=2.2))
        ax.plot(*p0, "o", mfc="w", mec=col, ms=9); ax.plot(*p1, "o", color=col, ms=9)
        ax.text(p1[0], p1[1], f"  {m}", color=col, fontsize=11, va="center")
    ax.set_xlabel("REFLEXIVE (withdrawal + flinch) per delivery")
    ax.set_ylabel("AFFECTIVE (attending + lick/bite + guarding)\nper delivery")
    lab0, lab1 = (("vehicle", "SBI-553") if len(DAYS) == 2 else ("baseline", "injection"))
    ax.legend(handles=[Line2D([], [], marker="o", ls="", mfc="w", mec="k", label=lab0),
                       Line2D([], [], marker="o", ls="", color="k", label=lab1)], frameon=False)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_movement(S, out):
    """WT-deck 'already slower before any stimulus': % of the first 3 min (no stimulus) moving, baseline -> injection"""
    _rc()
    fig, ax = plt.subplots(1, 2, figsize=(11.4, 4.6))
    res = []
    for j, drug in enumerate(("vehicle", "SBI-553")):
        a = ax[j]
        Y = []
        for m in MO:
            X = S[(S.mouse == m) & (S.drug_day == drug)]
            b0 = X[X.phase == "baseline"].movement_pct_nostim; b1 = X[X.phase == "injection"].movement_pct_nostim
            if len(b0) and len(b1):
                y = [float(b0.iloc[0]), float(b1.iloc[0])]
                a.plot([0, 1], y, marker=MK[m], color=MC[m], lw=1.6, ms=8, label=m)
                Y.append(y)
        if Y:
            Y = np.array(Y)
            a.errorbar([0, 1], Y.mean(0), yerr=Y.std(0) / np.sqrt(len(Y)), color="k", lw=2.6, capsize=4, marker="s", ms=8)
            res.append(dict(drug=drug, base=Y[:, 0].mean(), inj=Y[:, 1].mean(), n=len(Y), n_lower=int((Y[:, 1] < Y[:, 0]).sum())))
        a.set_xticks([0, 1], ["baseline", f"after {drug}"]); a.set_xlim(-0.3, 1.3)
        a.set_title(f"{drug} day", color=CC[drug], fontsize=14, fontweight="bold")
        a.set_ylabel("% of the first 3 min moving\n(no stimulus yet)")
        a.set_ylim(bottom=0)
    ax[0].legend(frameon=False, fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return pd.DataFrame(res)


# ------------------------------------------------------------------------------------------------ slides
class Deck:
    def __init__(self, lang, prov):
        from pptx import Presentation
        from pptx.util import Inches
        self.prs = Presentation(); self.prs.slide_width, self.prs.slide_height = Inches(SW), Inches(SH)
        self.lang = lang; self.prov = prov
        self.font = "Malgun Gothic" if lang == "kr" else "Arial"
        self.n_txt = ""                                # e.g. 'n = 4 mice (crossover)'; added at the bottom when a slide does not say n

    def box(self, s, x, y, w, h, text, pt, bold=False, rgb=(0x33, 0x33, 0x33)):
        from pptx.util import Inches, Pt
        from pptx.dml.color import RGBColor
        tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = tb.text_frame; tf.word_wrap = True
        for i, line in enumerate(text.split("\n")):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = line; p.font.size = Pt(pt); p.font.bold = bold; p.font.name = self.font
            p.font.color.rgb = RGBColor(*rgb)
        return tb

    def head(self, section, title, key):
        s = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        self.box(s, 0.45, 0.08, 6, 0.3, section, 11, True, (0x8A, 0x8F, 0x98))
        if self.prov:
            self.box(s, 8.6, 0.06, 4.4, 0.3, self.prov, 10.5, True, (0xC0, 0x39, 0x2B))
        tpt = 26 if len(title) <= 55 else 22 if len(title) <= 68 else 19           # one line, never over the key line
        self.box(s, 0.45, 0.32, 12.4, 0.6, title, tpt, True, (0x1F, 0x2D, 0x3D))
        if key:
            kpt = 16 if len(key) <= 115 else 14 if len(key) <= 150 else 12.5
            self.box(s, 0.45, 0.92, 12.4, 0.45, key, kpt, True, (0xC0, 0x39, 0x2B))
        return s

    def slide(self, section, title, key, fig, bottom, notes):
        """one slide; when the figure has split parts on disk (SPLIT: <stem><suffix>.png), one slide per part (title + part)"""
        if fig:
            d_, (stem, ext) = os.path.dirname(fig), os.path.splitext(os.path.basename(fig))
            parts = SPLIT.get(stem) or next((v for k, v in SPLIT.items() if k.endswith("*") and stem.startswith(k[:-1])), None)
            if parts:
                have = [(p_, os.path.join(d_, stem + p_[0] + ext)) for p_ in parts]
                have = [(p_, fx) for p_, fx in have if os.path.isfile(fx)]       # a part can be absent (too few events)
                if have:
                    for i, ((suf, en, kr), fx) in enumerate(have, 1):
                        s = self._slide(section, f"{title} ({i}/{len(have)}: {kr if self.lang == 'kr' else en})", key, fx, bottom, notes)
                    return s
        return self._slide(section, title, key, fig, bottom, notes)

    def _slide(self, section, title, key, fig, bottom, notes):
        from pptx.util import Inches
        from PIL import Image
        s = self.head(section, title, key)
        top, boxh = 1.42, 5.2
        if fig and os.path.isfile(fig):
            im = Image.open(fig)
            w = 12.4; h = w * im.size[1] / im.size[0]
            if h > boxh:
                h = boxh; w = h * im.size[0] / im.size[1]
            s.shapes.add_picture(fig, Inches(0.45 + (12.4 - w) / 2), Inches(top), Inches(w), Inches(h))
        ff = FF.notes_for(fig) if fig else ""                    # 10-24: the figure's methods text, moved out of the figure
        if fig:
            bottom = complete_n_test(bottom, key, (notes or "") + " " + ff, self.n_txt, self.lang)
        if bottom:
            self.box(s, 0.45, 6.68, 12.4, 0.8, bottom, 13.5 if len(bottom) <= 230 else 12.5)
        if ff:
            fk = ("그림 설명 (그림에서 옮김): " if self.lang == "kr" else "Figure key (moved from the figure): ") + ff
            nb = notes or ""                                     # a [BACKUP] tag stays first (assemble_master hides these)
            notes = ("[BACKUP] " + fk + "\n\n" + nb[len("[BACKUP]"):].lstrip()) if nb.startswith("[BACKUP]") else (fk + "\n\n" + nb)
        s.notes_slide.notes_text_frame.text = notes
        return s

    def table(self, section, title, key, header, rows, bottom, notes, colw=None):
        from pptx.util import Inches, Pt
        from pptx.dml.color import RGBColor
        s = self.head(section, title, key)
        nr, nc = len(rows) + 1, len(header)
        sh = s.shapes.add_table(nr, nc, Inches(0.45), Inches(1.5), Inches(12.4), Inches(0.42 * nr))
        tb = sh.table
        for j, h in enumerate(header):
            if colw:
                tb.columns[j].width = Inches(colw[j])
            c = tb.cell(0, j); c.text = h
            for p in c.text_frame.paragraphs:
                p.font.size = Pt(13); p.font.bold = True; p.font.name = self.font
        for i, r in enumerate(rows, 1):
            for j, v in enumerate(r):
                c = tb.cell(i, j); c.text = str(v)
                for p in c.text_frame.paragraphs:
                    p.font.size = Pt(12.5); p.font.name = self.font
                    if j and "UP" in str(v) and not str(v).startswith("0 UP"):
                        p.font.color.rgb = RGBColor(0xC0, 0x39, 0x2B)
        if bottom:
            self.box(s, 0.45, 6.3, 12.4, 1.0, bottom, 15)
        s.notes_slide.notes_text_frame.text = notes
        return s

    def cards(self, section, title, key, cards, bottom, notes):
        from pptx.util import Inches
        from pptx.dml.color import RGBColor
        from pptx.enum.shapes import MSO_SHAPE
        s = self.head(section, title, key)
        for i, (big, head, txt) in enumerate(cards):
            x = 0.45 + (i % 2) * 6.25; y = 1.55 + (i // 2) * 2.3
            r = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(6.05), Inches(2.1))
            r.fill.solid(); r.fill.fore_color.rgb = RGBColor(0xF3, 0xF5, 0xF8); r.line.fill.background()
            self.box(s, x + 0.15, y + 0.15, 1.9, 1.0, big, 34, True, (0xC0, 0x39, 0x2B))
            self.box(s, x + 2.1, y + 0.15, 3.8, 0.8, head, 15, True, (0x1F, 0x2D, 0x3D))
            self.box(s, x + 2.1, y + 0.95, 3.8, 1.1, txt, 12.5)
        if bottom:
            self.box(s, 0.45, 6.3, 12.4, 1.0, bottom, 15)
        s.notes_slide.notes_text_frame.text = notes
        return s


def L(lang, en, kr):
    return kr if lang == "kr" else en


def main(argv=None):
    global DAYS, LANG
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lang", default="en", choices=["en", "kr"])
    ap.add_argument("--days", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--short", action="store_true", help="the PI version (summary, design, the key slides, limits)")
    ap.add_argument("--psth-cache", default="", help="npz shared between the decks (made on the first build)")
    ap.add_argument("--event-dir", default="", help="ca_event_analysis.py output (default <base>/analysis/event_analysis)")
    a = ap.parse_args(argv)
    DAYS = sorted(a.days)
    LANG = a.lang                                               # 10-24: the language of the figure notes text
    B, lg = a.base, a.lang
    od = os.path.splitext(a.out)[0] + "_figures"
    os.makedirs(od, exist_ok=True)
    f = lambda nm: os.path.join(od, nm)
    both = len(DAYS) == 2
    dtag = "Day 1 + Day 2" if both else f"Day {DAYS[0]}"
    prov = "" if (DAYS == [1] or os.path.isfile(os.path.join(B, "mini1p_out", "20261021", "ALL_CONFIRMED.flag"))) else \
        L(lg, "PROVISIONAL - Day-2 neuron selection not yet confirmed", "잠정 - Day 2 뉴런 선택 미확정")
    load_windows(B)
    E = load_tests(B); R = load_rates(B); P = load_pairs(B)
    D = psth_all(B, ALL_EV, a.psth_cache or f("psth_cache.npz"))
    S = behaviour_table(B)
    dk = Deck(lg, prov)
    dk.n_txt = (L(lg, "n = 4 mice (crossover)", "n = 4마리 (crossover)") if both else
                L(lg, f"n = 2 mice per drug (Day {DAYS[0]})", f"n = 약물당 2마리 (Day {DAYS[0]})"))
    FPc = {}
    # ------------------------------------------------------------------ figures + numbers
    rr = fig_rate(R, f("f04_rate_pooled.png"))
    rm = fig_rate_matrix(R, P, f("f05_rate_every_neuron.png"))
    TP, pk = fig_population(E, D, STIM, f("f06_population_stimuli.png"))
    TB, pkb = fig_population(E, D, BEHEV, f("f07_population_behaviour.png"))
    key_ev = pk.event
    rows_k = neuron_rows(B, P, D, key_ev)
    hm = fig_heatmap(D, rows_k, key_ev, f("f08_heatmap_every_neuron.png"))
    TG = fig_groups(E, D, [key_ev, "withdrawal"] if key_ev != "withdrawal" else [key_ev, "Heat"], f("f08b_groups.png"))
    TRs, _, WRs = fig_responsive(E, STIM, f("f09_responsive_pct.png"))
    TRb, _, WRb = fig_responsive(E, [e for e in BEHEV if (E.event == e).any()], f("f09c_responsive_pct_behaviour.png"))
    TR = pd.concat([TRs, TRb], ignore_index=True); WR = pd.concat([WRs, WRb])
    ev_r = WR.d.abs().idxmax()
    fig_maps(B, E, ev_r, f("f10_maps.png"), FPc)
    EV_OK = [e for e in ALL_EV if (E.event == e).any()]
    TC, ev_c, KC = fig_resp_change(E, EV_OK, f("f11_responsive_change.png"))
    # split parts of the wide figures (10-22, legibility): Deck.slide shows them as two slides
    fig_population(E, D, BEHEV[:3], f("f07_population_behaviour_a.png"))
    fig_population(E, D, BEHEV[3:], f("f07_population_behaviour_b.png"))
    BOK = [e for e in BEHEV if (E.event == e).any()]
    if len(BOK) > 3:
        fig_responsive(E, BOK[:3], f("f09c_responsive_pct_behaviour_a.png"))
        fig_responsive(E, BOK[3:], f("f09c_responsive_pct_behaviour_b.png"))
    fig_maps(B, E, ev_r, f("f10_maps_m1.png"), FPc, mice=MO[:2])
    fig_maps(B, E, ev_r, f("f10_maps_m2.png"), FPc, mice=MO[2:])
    fig_resp_change(E, [e for e in EV_OK if e in STIM], f("f11_responsive_change_stim.png"))
    fig_resp_change(E, [e for e in EV_OK if e not in STIM], f("f11_responsive_change_beh.png"))
    FOV = {}
    for suf, mm in (("_m1", MO[:2]), ("_m2", MO[2:])):
        FOV.update(fig_fov_change(B, f(f"f05b_fov_change{suf}.png"), FPc, mm))
    TBh, pbh = fig_behaviour(S, f("f12_behaviour.png"))
    _, bst = fig_behaviour_stim(B, f("f13_behaviour_stimulus.png"))
    fig_behaviour_stim(B, f("f13_behaviour_stimulus_a.png"), only=["withdrawal", "flinch", "attending"])
    fig_behaviour_stim(B, f("f13_behaviour_stimulus_b.png"), only=["escape / rearing", "general movement", "lick / bite"])
    TM = fig_every_mouse(S, f("f14_every_mouse.png"))
    fig_affective(S, f("f15_affective_reflexive.png"))
    TMv = fig_movement(S, f("f16_movement_before_stimulus.png"))
    import make_longitudinal_deck as LD
    nd, nt = "", ""
    if both:
        _, nd = LD.fig_design(B, f("f02_design.png"), lang=lg)
        TS = LD.tracking_summary(B)
        nt = LD.fig_tracking_counts(TS, f("f03_tracking.png")) if len(TS) else ""
    else:
        import shutil as _sh
        src = os.path.join(B, "lab_meeting", "decks_Day1_Day2", "EN", f"Day{DAYS[0]}_PI_summary_figures", "s02_design.png")
        if os.path.isfile(src):
            _sh.copy(src, f("f02_design.png"))
        src = os.path.join(B, "lab_meeting", "decks_Day1_Day2", "EN", f"Day{DAYS[0]}_PI_summary_figures", "s04_neurons_per_mouse.png")
        if os.path.isfile(src):
            _sh.copy(src, f("f03_tracking.png"))
    ntr = {m: len(P[m]) for m in MO}
    nall = int(R.groupby(["mouse", "day"]).cell.nunique().sum()) if len(R) else 0
    sec = lambda en, kr: L(lg, en, kr)
    import deck_event_slides as DE
    ea = DE.EA(a.event_dir or os.path.join(B, "analysis", "event_analysis"))
    Lf = lambda en, kr: L(lg, en, kr)
    use_ea = both and bool(ea.J)
    # ------------------------------------------------------------------ 1 summary
    up = WR.loc[ev_r]
    c1 = ((f"{sum(ntr.values())}", L(lg, "neurons followed on both days", "두 날 모두 추적한 뉴런"),
           L(lg, ", ".join(f"{m} {ntr[m]}" for m in MO) + f"; {rm['n_one'] if rm else 0} more on one day only (shown blank on the other).",
             ", ".join(f"{m} {ntr[m]}" for m in MO) + f"; 하루에만 있는 뉴런 {rm['n_one'] if rm else 0}개 (다른 날은 빈칸)."))
          if both else
          (f"{nall}", L(lg, f"neurons on {dtag}", f"{dtag} 뉴런"),
           L(lg, "All mice; each neuron followed from the baseline to the injection session of the same day.",
             "모든 마우스; 같은 날 기준 세션에서 주사 세션까지 같은 뉴런.")))
    cards = [c1,
             (f"{pk['mean_SBI-553']:+.3f}", L(lg, f"{key_ev}: population response after SBI-553 (z, best window)",
                                              f"{key_ev}: SBI-553 후 집단 반응 (z, 최적 창)"),
              L(lg, f"vs {pk['mean_vehicle']:+.3f} after vehicle and {pk['mean_baseline']:+.3f} at baseline; all neurons pooled, {ptxt(pk.p)}.",
                f"vehicle 후 {pk['mean_vehicle']:+.3f}, 기준 {pk['mean_baseline']:+.3f}; 전체 뉴런, {ptxt(pk.p)}.")),
             (f"{up['SBI-553']:.0f}%", L(lg, f"{ev_r}: neurons UP after SBI-553", f"{ev_r}: SBI-553 후 UP 뉴런"),
              L(lg, f"vs {up['vehicle']:.0f}% after vehicle and {up['baseline']:.0f}% at baseline.",
                f"vehicle 후 {up['vehicle']:.0f}%, 기준 {up['baseline']:.0f}%.")),
             (f"{int(pbh.n_lower)}/{int(pbh.n_pairs)}" if both else f"{pbh.sbi:.2f}",
              L(lg, f"mice: {pbh.behaviour} lower after SBI-553 than after vehicle" if both else f"{pbh.behaviour} per delivery after SBI-553",
                f"마리: SBI-553 후 {pbh.behaviour}가 vehicle 후보다 낮음" if both else f"SBI-553 후 자극당 {pbh.behaviour}"),
              L(lg, f"{pbh.vehicle:.2f} after vehicle -> {pbh.sbi:.2f} after SBI-553 (mean of mice); 4 mice: min p = 0.125.",
                f"vehicle 후 {pbh.vehicle:.2f} -> SBI-553 후 {pbh.sbi:.2f} (마우스 평균); 4마리: 최소 p = 0.125."))]
    dk.cards(sec("SUMMARY", "요약"),
             L(lg, f"{dtag}: SBI-553 vs vehicle" + (" in the same mice" if both else ""), f"{dtag}: SBI-553 vs vehicle" + (" (같은 마우스)" if both else "")),
             L(lg, "Crossover, 4 mice: each mouse got vehicle on one day and SBI-553 on the other" if both else
               f"{dtag}: 2 mice SBI-553, 2 mice vehicle, each with its own baseline session",
               "Crossover 4마리: 각 마우스가 하루는 vehicle, 하루는 SBI-553" if both else f"{dtag}: SBI-553 2마리, vehicle 2마리, 각자 기준 세션"),
             cards, L(lg, "Ca imaging (mini1p, BLA) and pain behaviour in every session. " + dk.n_txt + "; tests on each result slide "
                       "(Wilcoxon, Mann-Whitney, McNemar, Fisher, Cochran-Mantel-Haenszel; neurons pooled = exploratory).",
                       "모든 세션에서 Ca 이미징(mini1p, BLA)과 통증 행동. " + dk.n_txt + "; 검정은 각 결과 슬라이드 (Wilcoxon, Mann-Whitney, McNemar, "
                       "Fisher, Cochran-Mantel-Haenszel; 뉴런 합침 = 탐색적)."),
             L(lg, "Every neuronal number uses the " + WINDOC + ": response = the response-window mean of z minus the "
                   "baseline-window mean, per trial, then averaged. Stats in each slide's notes. n = 4 mice: no group test across mice "
                   "can reach p < 0.05 (minimum 0.125).",
               "모든 뉴런 수치는 " + WINDOC_KR + "를 씀: 반응 = trial마다 반응 창 평균 z - 기준 창 평균 z, 그 평균. 통계는 각 슬라이드 노트. "
               "n = 4마리 (최소 p 0.125)."))
    # ------------------------------------------------------------------ 2 design + 3 neurons
    dk.slide(sec("DESIGN", "설계"), L(lg, "Crossover: vehicle on one day, SBI-553 on the other" if both else f"{dtag}: baseline, then the injection",
                                     "Crossover: 하루 vehicle, 다른 날 SBI-553" if both else f"{dtag}: 기준 세션 후 주사 세션"),
             L(lg, "Each mouse is its own control", "각 마우스가 자기 대조군"), f("f02_design.png"),
             L(lg, "Every day: a baseline session, then the injection session; the stimulus order is randomised per session.",
               "매일: 기준 세션 후 주사 세션; 자극 순서는 세션마다 무작위."), nd or "")
    if not a.short:
        dk.slide(sec("NEURONS", "뉴런"), L(lg, "The same neurons on Day 1 and Day 2" if both else "Neurons per mouse",
                                          "Day 1과 Day 2의 같은 뉴런" if both else "마우스별 뉴런"),
                 L(lg, f"{sum(ntr.values())} neurons followed across both days" if both else f"{nall} neurons in 4 mice",
                   f"두 날에 걸쳐 {sum(ntr.values())}개 뉴런 추적" if both else f"4마리에서 뉴런 {nall}개"), f("f03_tracking.png"),
                 L(lg, "Each Day-1 neuron was looked for on Day 2 and judged on Day-2 evidence." if both else
                   "Candidates proposed by EXTRACT; you selected the neurons on the max projection.",
                   "Day 1 뉴런을 Day 2에서 찾아 Day 2 증거로 판정." if both else "EXTRACT 후보 중 max projection에서 직접 고른 뉴런."),
                 (nt + "\n" if nt else "") + "Maps for your check: CONFIRM_DAY2_20261022\\<mouse>_tracking_map.png")
    if use_ea:                                                  # your basic plots 1-2: whole sessions
        DE.whole(dk, ea, Lf, a.short)
    # ------------------------------------------------------------------ 4 activity + 5 every neuron
    dk.slide(sec("ALL NEURONS", "전체 뉴런"), L(lg, "Activity of all neurons: baseline vs vehicle vs SBI-553", "전체 뉴런 활동: 기준 vs vehicle vs SBI-553"),
             L(lg, f"Median transients/min: baseline {rr['med']['baseline']:.1f}, vehicle {rr['med']['vehicle']:.1f}, SBI-553 {rr['med']['SBI-553']:.1f} ({ptxt(rr['p'])})",
               f"분당 transient 중앙값: 기준 {rr['med']['baseline']:.1f}, vehicle {rr['med']['vehicle']:.1f}, SBI-553 {rr['med']['SBI-553']:.1f} ({ptxt(rr['p'])})"),
             f("f04_rate_pooled.png"),
             L(lg, "All neurons of every session pooled (colour = mouse); right: one line per mouse.",
               "모든 세션의 뉴런을 합침 (색 = 마우스); 오른쪽: 마우스당 한 선."),
             L(lg, "Rate = Ca transients per minute of still frames inside the 4 stimulus blocks. A: box = median and quartiles over all "
                   f"neurons of that condition (n = {rr['n']['baseline']} / {rr['n']['vehicle']} / {rr['n']['SBI-553']}); dots = neurons coloured "
                   "by mouse. Each bracket shows the pair tested: black (veh-SBI) = Mann-Whitney U SBI-553 vs vehicle over neurons "
                   "(exploratory: neurons are nested in 4 mice); blue (base-veh) and red (base-SBI) = that injection vs the same "
                   f"neurons' baseline of the same day (Wilcoxon signed-rank: vehicle {ptxt(rr['p_within']['vehicle'])}, SBI-553 "
                   f"{ptxt(rr['p_within']['SBI-553'])}). * p < 0.05, ** p < 0.01, *** p < 0.001, ns = not significant. Over the mice "
                   "(n = 4) no test can go below p = 0.125. B: per-mouse medians.",
               "빈도 = 4개 자극 블록 안 움직임 없는 프레임의 분당 Ca transient. A: 상자 = 조건별 전체 뉴런의 중앙값·사분위. "
               "괄호 = 비교한 두 조건: 검정(veh-SBI) = Mann-Whitney U SBI-553 vs vehicle (뉴런 단위, 탐색적); 파랑(base-veh)·빨강(base-SBI) "
               "= 같은 날 같은 뉴런의 기준 대비 (Wilcoxon). * p < 0.05, ** p < 0.01, *** p < 0.001, ns = 유의하지 않음. 마우스 단위(n = 4)로는 "
               "p < 0.125가 불가능. B: 마우스별 중앙값."))
    if rm and not a.short:
        dk.slide(sec("ALL NEURONS", "전체 뉴런"), L(lg, "Every neuron across the three conditions", "모든 뉴런, 세 조건"),
                 L(lg, f"{rm['n']} neurons: {rm['n_both']} on both days, {rm['n_one']} on one day only (grey = not there)" if both else
                   f"{rm['n']} neurons, each from its baseline to its injection session",
                   f"뉴런 {rm['n']}개: 두 날 {rm['n_both']}개, 하루만 {rm['n_one']}개 (회색 = 그날 없음)" if both else f"뉴런 {rm['n']}개, 각자 기준에서 주사 세션까지"),
                 f("f05_rate_every_neuron.png"),
                 L(lg, "Left: one row per neuron, grouped by mouse; right: the change of each neuron after vehicle (blue) and SBI-553 (red).",
                   "왼쪽: 뉴런당 한 줄 (마우스별); 오른쪽: vehicle(파랑), SBI-553(빨강) 후 뉴런별 변화."),
                 L(lg, "Colour = transient rate in the stimulus blocks (still frames). Baseline = mean of the neuron's baseline sessions. The strip "
                       "on the left: green = followed on both days, magenta = Day 1 only, cyan = Day 2 only. Right: log2 of "
                       "(injection + 0.1) / (baseline + 0.1) per neuron, same day only; box = median and quartiles; the star over 'vs 0' "
                       "above a box = that change differs from 0 (Wilcoxon signed-rank over the mouse's neurons).",
                   "색 = 자극 블록 transient 빈도. 기준 = 뉴런의 기준 세션 평균. 왼쪽 띠: 초록 = 두 날, 자홍 = Day 1만, 청록 = Day 2만. "
                   "상자 위 별('vs 0') = 그 변화가 0과 다름 (마우스 뉴런에 대한 Wilcoxon)."))
    if FOV:                                                     # the old A5, now per drug day (10-22)
        def _fv(m, dr):
            v = FOV.get((m, dr))
            return f"{v['median']:+.2f}" if v and np.isfinite(v["median"]) else "n.a."
        dk.slide(sec("ALL NEURONS", "전체 뉴런"),
                 L(lg, "Where in the field of view activity changed" + (": vehicle day vs SBI-553 day" if both else ""),
                   "시야에서 활동이 바뀐 곳" + (": vehicle 날 vs SBI-553 날" if both else "")),
                 L(lg, "Median log2 change, " + ("vehicle day / SBI-553 day: " if both else "") + "; ".join(
                     f"{m} {_fv(m, 'vehicle')} / {_fv(m, 'SBI-553')}" if both else f"{m} {_fv(m, ARMS[DAYS[0]][m])}" for m in MO),
                   "log2 변화 중앙값, " + ("vehicle 날 / SBI-553 날: " if both else "") + "; ".join(
                     f"{m} {_fv(m, 'vehicle')} / {_fv(m, 'SBI-553')}" if both else f"{m} {_fv(m, ARMS[DAYS[0]][m])}" for m in MO)),
                 f("f05b_fov_change.png"),
                 L(lg, "We coloured each neuron's footprint by its whole-session rate change after the injection (red = higher, blue = "
                       "lower), relative to the baseline session of the same day. n = 4 mice; descriptive.",
                   "각 뉴런 footprint를 주사 후 세션 전체 transient 빈도 변화로 색칠 (빨강 = 증가, 파랑 = 감소), 같은 날 기준 세션 대비. 마우스 4; "
                   "기술 통계."),
                 L(lg, "log2 ((injection + 0.1) / (baseline + 0.1)) transients per min over the whole session, per neuron, on that day's "
                       "max dF/F projection; 'higher / lower' = sign of the change. Same measure as the Day-1 and Day-2 A5 slides.",
                   "세션 전체 분당 transient의 log2 ((주사 + 0.1) / (기준 + 0.1)), 뉴런별, 그날 max dF/F 위. 'higher / lower' = 변화의 부호."))
    # ------------------------------------------------------------------ population
    for fn, pick, ttl, ttl_kr, T_ in (("f06_population_stimuli.png", pk, "Population response to each stimulus", "자극별 집단 반응", TP),
                                      ("f07_population_behaviour.png", pkb, "Population response to each behaviour", "행동별 집단 반응", TB)):
        if a.short and fn.startswith("f07"):
            continue
        d_ = pick["mean_SBI-553"] - pick["mean_vehicle"]
        dk.slide(sec("ALL NEURONS", "전체 뉴런"), L(lg, ttl, ttl_kr),
                 L(lg, f"{pick.event}: {'smaller' if d_ < 0 else 'larger'} after SBI-553 than after vehicle "
                       f"(response {pick['mean_SBI-553']:+.3f} vs {pick['mean_vehicle']:+.3f} z, {ptxt(pick.p)})",
                   f"{pick.event}: SBI-553 후가 vehicle 후보다 {'작음' if d_ < 0 else '큼'} (반응 {pick['mean_SBI-553']:+.3f} vs {pick['mean_vehicle']:+.3f} z, {ptxt(pick.p)})"),
                 f(fn), L(lg, "We measured every neuron's response in the best window (title of each panel). Top: mean of all neurons "
                              "(markers = mice); bottom: the average trace. n = 4 mice; brackets = the pair tested (neurons).",
                          "모든 뉴런의 반응을 최적 창(각 패널 제목)으로 잼. 위: 전체 뉴런 평균 (표식 = 마우스); 아래: 평균 trace. 마우스 4; 괄호 = "
                          "비교한 두 조건 (뉴런 단위)."),
                 L(lg, "z = dF/F / robust SD (MAD) of the session. Response = per trial, the mean z in the response window minus the mean "
                       "in the baseline window (" + WINDOC + "), averaged over trials (event analysis). Traces: z minus the 3 s before "
                       "the event (display only). Top: bar = mean +- SEM over all neurons "
                       "of the condition (sessions pooled), markers = mouse means (5941 circle, 5939 square, 5937 triangle, 1191 diamond). "
                       "Each bracket shows the pair tested: black (veh-SBI) = Mann-Whitney U SBI-553 vs vehicle over neurons (exploratory, "
                       "uncorrected); blue (base-veh) / red (base-SBI) = that injection vs the same neurons' baseline of the same day "
                       "(Wilcoxon signed-rank). * p < 0.05, ** p < 0.01, *** p < 0.001, ns = not significant. Bottom: mean +- SEM; "
                       "grey = the response window, blue = the baseline window. "
                       "n: " + "; ".join(f"{r.event} {r.n_baseline}/{r.n_vehicle}/{r['n_SBI-553']}" for _, r in T_.iterrows()),
                   "z = 세션의 dF/F / robust SD (MAD). 반응 = trial마다 반응 창 평균 z - 기준 창 평균 z (" + WINDOC_KR + "), trial 평균. trace는 사건 전 "
                   "3 s 기준 (표시용). 위: 막대 = 전체 뉴런 평균 +- SEM, 표식 = 마우스 평균. "
                   "괄호 = 비교한 두 조건: 검정(veh-SBI) = Mann-Whitney U (뉴런 단위, 보정 없음); 파랑(base-veh)·빨강(base-SBI) = 같은 날 같은 "
                   "뉴런의 기준 대비 (Wilcoxon). * p < 0.05, ** p < 0.01, *** p < 0.001, ns = 유의하지 않음. 아래: 평균 +- SEM; 회색 = 반응 창, 파랑 = 기준 창."))
    if both:                                    # 10-23 (user): the same neurons on both days, each with itself, baseline-normalized
        import deck_tracked_slides as TSL
        TSL.add_tracked_slides(dk, lg, short=a.short)
        TSL.add_p7_condition_slide(dk, lg)                     # 10-23 (user): brain-behaviour by condition (Q4, via STORY)
        import deck_pi_extra as PXS                            # 10-23 evening: the new figure sets (analysis\pi_extra)
        PXS.add_extra(dk, lg, short=a.short)
    # ------------------------------------------------------------------ heatmap every neuron
    emean = lambda ev: {c: float(E[(E.event == ev) & (E.condition == c)].auc_post.mean()) for c in COND}
    if hm:
        em_ = emean(key_ev)
        dk.slide(sec("EVERY NEURON", "모든 뉴런"), L(lg, f"{key_ev}: every neuron in the three conditions", f"{key_ev}: 모든 뉴런, 세 조건"),
                 L(lg, f"Mean response ({wtxt(key_ev)}): baseline {zz(em_['baseline'], False)}, vehicle {zz(em_['vehicle'], False)}, SBI-553 {zz(em_['SBI-553'], False)}",
                   f"평균 반응 ({wtxt(key_ev)}): 기준 {zz(em_['baseline'], True)}, vehicle {zz(em_['vehicle'], True)}, SBI-553 {zz(em_['SBI-553'], True)}"),
                 f("f08_heatmap_every_neuron.png"),
                 L(lg, f"One row = one neuron ({hm['n']}), grouped by mouse; red = activity rose after the event; grey = the neuron is not in that condition.",
                   f"한 줄 = 뉴런 하나 ({hm['n']}개), 마우스별; 빨강 = 사건 후 상승; 회색 = 그 조건에 없는 뉴런."),
                 L(lg, "Inside each mouse: neurons followed on both days first (sorted by the baseline response), then the neurons of one day only. "
                       "Colour = trial-averaged z minus the 3 s before the event (display), 0.2-s smoothing; numbers in the key sentence = the "
                       "event-analysis response in the best window. Baseline = mean of the neuron's baseline sessions. "
                       "The event shown has the largest SBI-553 vs vehicle difference on the population slide.",
                   "마우스 안에서 두 날 추적 뉴런 먼저(기준 반응 순), 그다음 하루만 있는 뉴런. 색 = trial 평균 z (사건 전 3 s 기준, 표시용); 핵심 "
                   "문장 수치 = 최적 창의 반응."))
    if not a.short:
        dk.slide(sec("RESPONSIVE NEURONS", "반응 뉴런"), L(lg, "UP, DOWN and the other neurons", "UP, DOWN, 나머지 뉴런"),
                 L(lg, "Compare the red (UP) and blue (DOWN) traces across the three columns", "세 열에서 빨강(UP)과 파랑(DOWN) trace를 비교"),
                 f("f08b_groups.png"),
                 L(lg, "Ntsr1-deck layout: mean trace of the neurons that rose (UP), fell (DOWN) or did neither, per condition.",
                   "Ntsr1 덱 방식: 조건별로 상승(UP), 하강(DOWN), 나머지 뉴런의 평균 trace."),
                 L(lg, "UP / DOWN = time-shift test per neuron in the best window, BH q <= 0.20 within the session; mean +- SEM over "
                       "neurons; grey = the response window, blue = the baseline window; traces minus the 3 s before (display).",
                   "UP / DOWN = 최적 창의 뉴런별 시간 이동 검정, 세션 안 BH q <= 0.20; 뉴런 평균 +- SEM; 회색 = 반응 창, 파랑 = 기준 창."))
    # ------------------------------------------------------------------ responsive
    evs_ = WRs.d.abs().idxmax()
    dk.slide(sec("RESPONSIVE NEURONS", "반응 뉴런"), L(lg, "How many neurons respond to each stimulus (best window)",
                                                      "자극마다 반응하는 뉴런 수 (최적 창)"),
             L(lg, f"{evs_}: {WR.loc[evs_, 'SBI-553']:.0f}% UP after SBI-553 vs {WR.loc[evs_, 'vehicle']:.0f}% after vehicle ({WR.loc[evs_, 'baseline']:.0f}% at baseline)",
               f"{evs_}: SBI-553 후 UP {WR.loc[evs_, 'SBI-553']:.0f}% vs vehicle 후 {WR.loc[evs_, 'vehicle']:.0f}% (기준 {WR.loc[evs_, 'baseline']:.0f}%)"),
             f("f09_responsive_pct.png"),
             L(lg, "Solid = % of neurons that rose after the event (UP), hatched = % that fell (DOWN); markers = mice.",
               "실선 = 사건 후 상승한 뉴런 %(UP), 빗금 = 하강 %(DOWN); 표식 = 마우스."),
             L(lg, "Responsive = time-shift test per neuron (events shifted inside their own block, 1000 shuffles), BH q <= 0.20 in the "
                   "session (exploratory, as in the Ntsr1 deck). Blue (base-veh) / red (base-SBI) bracket = the % UP changed from the same "
                   "neurons' baseline of the same day (McNemar); black (veh-SBI) bracket = Fisher exact SBI-553 vs vehicle on the UP counts: "
                   + "; ".join(f"{r.event} p = {r.p_up:.3f}" for _, r in TR.dropna(subset=["p_up"]).iterrows()),
               "반응 = 뉴런별 시간 이동 검정 (블록 안 이동, 1000회), 세션 안 BH q <= 0.20 (탐색적). 파랑(base-veh)·빨강(base-SBI) 괄호 = 같은 날 "
               "같은 뉴런의 기준 대비 UP % 변화 (McNemar); 검정(veh-SBI) 괄호 = UP 수 Fisher exact SBI-553 vs vehicle."))
    if not a.short and len(WRb):
        evb_ = WRb.d.abs().idxmax()
        dk.slide(sec("RESPONSIVE NEURONS", "반응 뉴런"), L(lg, "How many neurons respond to each behaviour (best window)",
                                                          "행동마다 반응하는 뉴런 수 (최적 창)"),
                 L(lg, f"{evb_}: {WR.loc[evb_, 'SBI-553']:.0f}% UP after SBI-553 vs {WR.loc[evb_, 'vehicle']:.0f}% after vehicle ({WR.loc[evb_, 'baseline']:.0f}% at baseline)",
                   f"{evb_}: SBI-553 후 UP {WR.loc[evb_, 'SBI-553']:.0f}% vs vehicle 후 {WR.loc[evb_, 'vehicle']:.0f}% (기준 {WR.loc[evb_, 'baseline']:.0f}%)"),
                 f("f09c_responsive_pct_behaviour.png"),
                 L(lg, "We counted the neurons that rose (UP, solid) or fell (DOWN, hatched) after each behaviour; markers = mice. n = 4 mice; "
                       "brackets = McNemar (same neurons, same day) and Fisher (vehicle vs SBI-553).",
                   "각 행동 후 상승(UP, 실선)·하강(DOWN, 빗금) 뉴런을 셈; 표식 = 마우스. 마우스 4; 괄호 = McNemar (같은 날 같은 뉴런), Fisher "
                   "(vehicle vs SBI-553)."),
                 L(lg, "Responsive = time-shift test per neuron in the best window (behaviours 0-0.5 s vs the 1 s before; events shifted "
                       "inside their own block, 1000 shuffles), BH q <= 0.20 in the session. Attending onsets: 0-8 per session, only "
                       "sessions with >= 5 are tested. Fisher SBI-553 vs vehicle on the UP counts: "
                       + "; ".join(f"{r.event} p = {r.p_up:.3f}" for _, r in TRb.dropna(subset=["p_up"]).iterrows()),
                   "반응 = 최적 창(행동: 후 0-0.5 s vs 직전 1 s)의 뉴런별 시간 이동 검정 (블록 안, 1000회), 세션 안 BH q <= 0.20. attending은 "
                   "세션당 0-8번이라 5번 이상인 세션만 검정."))
    # summary table (Ntsr1 'Summary'), drawn as a figure
    if not a.short:
        fig_table(TR, EV_OK, f("f09b_summary_table.png"))
        dk.slide(sec("RESPONSIVE NEURONS", "반응 뉴런"), L(lg, "Summary: responsive neurons per event (best window)", "요약: 사건별 반응 뉴런 (최적 창)"),
                 L(lg, f"Most UP neurons: {TR.loc[TR.pct_up.idxmax(), 'event']} after {TR.loc[TR.pct_up.idxmax(), 'condition']} "
                       f"({TR.pct_up.max():.0f}%)", f"UP 뉴런이 가장 많음: {TR.loc[TR.pct_up.idxmax(), 'condition']} 후 "
                       f"{TR.loc[TR.pct_up.idxmax(), 'event']} ({TR.pct_up.max():.0f}%)"),
                 f("f09b_summary_table.png"),
                 L(lg, "All neurons pooled over the mice; darker red = more UP neurons.", "마우스를 합친 전체 뉴런; 진한 빨강 = UP 뉴런이 많음."),
                 L(lg, "UP = rose after the event, DOWN = fell (time-shift test in the best window, BH q <= 0.20 within each session). 'of n' = all neurons of "
                       "that condition (baseline counts every baseline session).",
                   "UP = 사건 후 상승, DOWN = 하강 (시간 이동 검정, 세션 안 q <= 0.20). 'of n' = 그 조건의 전체 뉴런 수."))
    dk.slide(sec("RESPONSIVE NEURONS", "반응 뉴런"), L(lg, f"Where the {ev_r} neurons are, session by session", f"{ev_r} 반응 뉴런의 위치, 세션별"),
             L(lg, "Red = UP, blue = DOWN; compare the injection panels (vehicle vs SBI-553)", "빨강 = UP, 파랑 = DOWN; 주사 패널(vehicle vs SBI-553)을 비교"),
             f("f10_maps.png"),
             L(lg, "Each mouse: its sessions left to right (Day 1 baseline, injection, Day 2 baseline, injection)." if both else
               "Each mouse: baseline, then the injection session.",
               "마우스마다 왼쪽부터 세션 (Day 1 기준, 주사, Day 2 기준, 주사)." if both else "마우스마다 기준, 주사 세션."),
             L(lg, "Outlines on that day's max dF/F projection; thin grey = no response. UP / DOWN = time-shift test in the best window "
                   "(BH q <= 0.20), as on the previous slide.",
               "그날 max dF/F 위 윤곽; 얇은 회색 = 반응 없음. UP / DOWN = 최적 창의 시간 이동 검정 (BH q <= 0.20), 앞 슬라이드와 같음."))
    if not a.short:
        kc = KC.loc[ev_c]
        dk.slide(sec("RESPONSIVE NEURONS", "반응 뉴런"), L(lg, "Do the same neurons keep responding after the injection? (best window)",
                                                          "같은 뉴런이 주사 후에도 계속 반응하는가? (최적 창)"),
                 L(lg, f"{ev_c}: {kc['SBI-553']:.0f}% kept their response after SBI-553 vs {kc['vehicle']:.0f}% after vehicle",
                   f"{ev_c}: SBI-553 후 {kc['SBI-553']:.0f}%가 반응 유지 vs vehicle 후 {kc['vehicle']:.0f}%"),
                 f("f11_responsive_change.png"),
                 L(lg, "Every neuron within its day: baseline session -> injection session of the same day.",
                   "모든 뉴런, 같은 날 안에서: 기준 세션 -> 주사 세션."),
                 L(lg, "A: neurons UP or DOWN at the baseline of their day; bar = % with the SAME class after the injection (k/n on top). "
                       "B: neurons without a response at baseline; bar = % responsive after the injection. Bracket with a star = SBI-553 vs "
                       "vehicle differ (Fisher exact on k/n).",
                   "A: 그날 기준에서 UP/DOWN 뉴런 중 주사 후 같은 반응 %. B: 기준에 반응 없던 뉴런 중 주사 후 반응 %. 별 = SBI-553 vs vehicle "
                   "차이 (k/n Fisher exact)."))
    if not a.short:                                             # every stimulus and every behaviour, the same 3 slides (user 10-22)
        TPB = pd.concat([TP, TB], ignore_index=True).set_index("event")
        for ev in EV_OK:
            tag = ev.replace(" ", "_").replace("/", "_")
            em_ = emean(ev)
            nn = {c: int(((E.event == ev) & (E.condition == c)).sum()) for c in COND}
            pv_ = TPB.p.get(ev, np.nan) if ev in TPB.index else np.nan
            ps_ = L(lg, f"; SBI-553 vs vehicle {ptxt(pv_)}", f"; SBI-553 vs vehicle {ptxt(pv_)}") if np.isfinite(pv_) else ""
            r_ = fig_heatmap(D, neuron_rows(B, P, D, ev), ev, f(f"e_heatmap_{tag}.png"))
            # 10-25: the mice that really have data (sessions with >= MIN_TRIALS events) and the test that was really done
            mc_ = {c: sorted(set(E.mouse[(E.event == ev) & (E.condition == c)])) for c in COND}
            mu_ = [m for m in MO if any(m in mc_[c] for c in COND)]
            miss_ = [m for m in MO if m not in mu_]
            k_ = [len(mc_[c]) for c in COND]
            xe_, xk_ = [], []
            if len(set(k_)) == 1:
                mt_en, mt_kr = (f"{k_[0]} mouse" if k_[0] == 1 else f"{k_[0]} mice"), f"마우스 {k_[0]}"
            else:
                mt_en, mt_kr = f"{k_[0]} / {k_[1]} / {k_[2]} mice per condition", f"조건별 마우스 {k_[0]} / {k_[1]} / {k_[2]}"
                xe_.append(f"{len(mu_)} in all"); xk_.append(f"전체 {len(mu_)}")
            if miss_ and mu_:
                xe_.append(f"{', '.join(miss_)}: < {CA.MIN_TRIALS} events" if len(mu_) > 1 else f"{mu_[0]} only")
                xk_.append(f"{', '.join(miss_)}: 사건 {CA.MIN_TRIALS}회 미만" if len(mu_) > 1 else f"{mu_[0]}만")
            if xe_:
                mt_en += f" ({'; '.join(xe_)})"; mt_kr += f" ({'; '.join(xk_)})"
            no_ = [c for c in ("SBI-553", "vehicle") if nn[c] == 0]
            if no_:
                ts_en = f"no {no_[0]} session with >= {CA.MIN_TRIALS} events: no drug test"
                ts_kr = f"{no_[0]} 후 사건 {CA.MIN_TRIALS}회 이상 세션 없음: 약물 검정 없음"
            elif np.isfinite(pv_):
                ts_en = "SBI-553 vs vehicle: Mann-Whitney U (neurons pooled, exploratory)"
                ts_kr = "SBI-553 vs vehicle: Mann-Whitney U (뉴런 합침, 탐색적)"
            else:
                ts_en, ts_kr = "descriptive (no test)", "기술 통계 (검정 없음)"
            if r_:
                dk.slide(sec(f"EVERY EVENT: {ev}", f"사건별: {ev}"), L(lg, f"{ev}: every neuron in the three conditions", f"{ev}: 모든 뉴런, 세 조건"),
                         L(lg, f"Mean response ({wtxt(ev)}): baseline {zz(em_['baseline'], False)}, vehicle {zz(em_['vehicle'], False)}, SBI-553 {zz(em_['SBI-553'], False)}{ps_}",
                           f"평균 반응 ({wtxt(ev)}): 기준 {zz(em_['baseline'], True)}, vehicle {zz(em_['vehicle'], True)}, SBI-553 {zz(em_['SBI-553'], True)}{ps_}"),
                         f(f"e_heatmap_{tag}.png"),
                         L(lg, f"We plotted every neuron's trial-averaged activity around {ev} (one row = one neuron, grouped by mouse; grey "
                               f"= not in that condition). n = {nn['baseline']} / {nn['vehicle']} / {nn['SBI-553']} neuron-sessions, {mt_en}; "
                               f"{ts_en}.",
                           f"{ev} 주변의 모든 뉴런 trial 평균 활동 (한 줄 = 뉴런, 마우스별; 회색 = 그 조건에 없음). 뉴런-세션 {nn['baseline']} / "
                           f"{nn['vehicle']} / {nn['SBI-553']}, {mt_kr}; {ts_kr}."),
                         L(lg, "Colour = trial-averaged z minus the 3 s before the event (display), 0.2-s smoothing; dotted line = end of "
                               f"the response window ({wtxt(ev)}). Numbers = the event-analysis response in that window. Only sessions with "
                               ">= 5 events are shown.",
                           "색 = trial 평균 z (사건 전 3 s 기준, 표시용), 0.2 s 평활; 점선 = 반응 창 끝. 수치 = 그 창의 반응. 사건 5번 이상인 세션만."))
            fig_groups(E, D, [ev], f(f"e_groups_{tag}.png"))
            cnt = {c: (int(((E.event == ev) & (E.condition == c) & (E.cls_shift == "UP")).sum()),
                       int(((E.event == ev) & (E.condition == c) & (E.cls_shift == "DOWN")).sum()), max(nn[c], 1)) for c in COND}
            dk.slide(sec(f"EVERY EVENT: {ev}", f"사건별: {ev}"), L(lg, f"{ev}: UP, DOWN and the other neurons", f"{ev}: UP, DOWN, 나머지 뉴런"),
                     L(lg, "UP / DOWN neurons: " + ", ".join(f"{c} {100 * cnt[c][0] / cnt[c][2]:.0f}% / {100 * cnt[c][1] / cnt[c][2]:.0f}%" for c in COND),
                       "UP / DOWN 뉴런: " + ", ".join(f"{ {'baseline': '기준'}.get(c, c)} {100 * cnt[c][0] / cnt[c][2]:.0f}% / {100 * cnt[c][1] / cnt[c][2]:.0f}%" for c in COND)),
                     f(f"e_groups_{tag}.png"),
                     L(lg, f"We split the neurons by their test result (UP = rose, DOWN = fell, rest) and averaged each group's trace. n = "
                           f"{nn['baseline']} / {nn['vehicle']} / {nn['SBI-553']} neuron-sessions, 4 mice; time-shift test, BH q <= 0.20.",
                       f"검정 결과로 뉴런을 나누고 (UP = 상승, DOWN = 하강, 나머지) 그룹별 trace를 평균. 뉴런-세션 {nn['baseline']} / "
                       f"{nn['vehicle']} / {nn['SBI-553']}, 마우스 4; 시간 이동 검정, BH q <= 0.20."),
                     L(lg, f"UP / DOWN = within-block time-shift test per neuron in the best window ({wtxt(ev)}), 1000 shuffles, BH q <= "
                           "0.20 per session. Mean +- SEM over neurons; grey = response window, blue = baseline window; traces minus the "
                           "3 s before (display).",
                       f"UP / DOWN = 최적 창({wtxt(ev)})의 뉴런별 블록 내 시간 이동 검정, 1000회, 세션별 BH q <= 0.20. 뉴런 평균 +- SEM."))
            fig_maps(B, E, ev, f(f"e_maps_{tag}_m1.png"), FPc, mice=MO[:2])
            fig_maps(B, E, ev, f(f"e_maps_{tag}_m2.png"), FPc, mice=MO[2:])
            dk.slide(sec(f"EVERY EVENT: {ev}", f"사건별: {ev}"), L(lg, f"Where the {ev} neurons are", f"{ev} 반응 뉴런의 위치"),
                     L(lg, "Red = UP, blue = DOWN, grey = no response; compare the injection panels (vehicle vs SBI-553)",
                       "빨강 = UP, 파랑 = DOWN, 회색 = 반응 없음; 주사 패널(vehicle vs SBI-553)을 비교"),
                     f(f"e_maps_{tag}.png"),
                     L(lg, "We outlined every neuron on its day's max dF/F projection, coloured by its test result, session by session. "
                           "Counts in each panel title; 4 mice.",
                       "세션마다 그날 max dF/F 위에 모든 뉴런을 검정 결과 색으로 윤곽 표시. 패널 제목에 수; 마우스 4."),
                     L(lg, f"UP / DOWN = the time-shift test of the previous slide (best window {wtxt(ev)}, BH q <= 0.20).",
                       f"UP / DOWN = 앞 슬라이드의 시간 이동 검정 (최적 창 {wtxt(ev)}, BH q <= 0.20)."))
    if use_ea:                                                  # the response window, stimulus vs withdrawal, your plots 3-5
        DE.windows(dk, ea, Lf, a.short, lg)
        DE.overlap(dk, ea, Lf, a.short)
        DE.categories(dk, ea, Lf, a.short)
        va = DE.EA(os.path.join(ea.d, "validation"), "validation.json")               # checks + other ways (ca_event_validate.py), added 10-22
        if va.J:
            DE.checks(dk, va, Lf, a.short)
            DE.check_key(dk, va, Lf, a.short)
            DE.other_ways(dk, va, Lf, a.short)
    # ------------------------------------------------------------------ behaviour
    dk.slide(sec("BEHAVIOUR", "행동"), L(lg, "Behaviour per delivery: every mouse", "자극당 행동: 모든 마우스"),
             L(lg, f"{pbh.behaviour}: {pbh.vehicle:.2f} after vehicle -> {pbh.sbi:.2f} after SBI-553"
                   + (f" (lower in {int(pbh.n_lower)}/{int(pbh.n_pairs)} mice)" if both else ""),
               f"{pbh.behaviour}: vehicle 후 {pbh.vehicle:.2f} -> SBI-553 후 {pbh.sbi:.2f}" + (f" ({int(pbh.n_lower)}/{int(pbh.n_pairs)} 마리에서 낮음)" if both else "")),
             f("f12_behaviour.png"),
             L(lg, "Each behaviour divided by the deliveries of its session; thin line = one mouse, bold = mean +- SEM.",
               "모든 행동은 그 세션의 자극 수로 나눔; 얇은 선 = 마우스, 굵은 선 = 평균 +- SEM."),
             L(lg, "Reflexes within 2 s of a delivery, other behaviours: bout starts within 3 s. Wilcoxon signed-rank SBI-553 vs vehicle over "
                   "mice where both exist: " + "; ".join(f"{r.behaviour} {ptxt(r.p)}" for _, r in TBh.iterrows())
                   + " (4 mice: minimum possible p = 0.125). Stars: pooled over the deliveries with each mouse as its own stratum "
                   "(Mantel-Haenszel rate test) - blue (base-veh) / red (base-SBI) bracket = that injection vs the same mice's "
                   "baseline of the same day, black (veh-SBI) bracket = SBI-553 vs vehicle; * p < 0.05, ** p < 0.01, *** p < 0.001 (deliveries treated as independent inside a mouse). "
                   "Baseline = mean of the mouse's baseline sessions.",
               "반사는 자극 2 s 안, 다른 행동은 3 s 안 시작. Wilcoxon SBI-553 vs vehicle (4마리: 최소 p = 0.125). 별: 마우스를 층으로 한 "
               "자극 단위 Mantel-Haenszel 검정 - 파랑(base-veh)·빨강(base-SBI) 괄호 = 같은 날 기준 대비, 검정(veh-SBI) 괄호 = SBI-553 vs vehicle."))
    if not a.short:
        dk.slide(sec("BEHAVIOUR", "행동"), L(lg, "Every mouse, every behaviour", "모든 마우스, 모든 행동"),
                 L(lg, "Left of the dashed line = less after SBI-553" if both else "Left of the dashed line = less after the injection",
                   "점선 왼쪽 = SBI-553 후 적음" if both else "점선 왼쪽 = 주사 후 적음"),
                 f("f14_every_mouse.png"),
                 L(lg, "One dot per mouse with its 95 % CI: how much the behaviour per delivery changed.",
                   "마우스당 점 하나와 95 % CI: 자극당 행동의 변화."),
                 L(lg, "WT-deck layout. Rate ratio = (events / deliveries) of the " + ("SBI-553 injection session / the vehicle injection session"
                       if both else "injection session / the baseline session") + "; exact conditional (binomial) 95 % CI and test on the "
                       "animal's own counts; stars uncorrected. " + ("" if both else "Colour = the drug of that mouse (red SBI-553, blue vehicle)."),
                   "WT 덱 방식. 비율 = (사건 / 자극) " + ("SBI-553 주사 세션 / vehicle 주사 세션" if both else "주사 세션 / 기준 세션") + "; 정확 이항 95 % CI와 검정."))
        if True:                                                # kept in every deck (user 10-22: never remove a slide, only add)
          dk.slide(sec("BEHAVIOUR", "행동"), L(lg, "Behaviour per stimulus", "자극별 행동"),
                 (L(lg, f"{bst[2]} after {bst[4]}: {bst[5]:.0f}% of deliveries after vehicle vs {bst[6]:.0f}% after SBI-553",
                    f"{bst[4]} 후 {bst[2]}: vehicle 후 자극의 {bst[5]:.0f}% vs SBI-553 후 {bst[6]:.0f}%") if bst else ""),
                 f("f13_behaviour_stimulus.png"),
                 L(lg, "% of deliveries followed by each behaviour, per stimulus type; bar = mean +- SEM over the mice, dot = mouse.",
                   "자극 종류별로 각 행동이 뒤따른 자극 %; 막대 = 마우스 평균 +- SEM, 점 = 마우스."),
                 L(lg, "Baseline = mean of the baseline sessions of a mouse. Reflexes within 2 s, other behaviours within 3 s. Brackets "
                       "(Cochran-Mantel-Haenszel over the deliveries, each mouse its own stratum; only p < 0.05 drawn): blue (base-veh) / "
                       "red (base-SBI) = that injection vs the same mice's baseline of the same day; black (veh-SBI) = SBI-553 vs vehicle. "
                       "* p < 0.05, ** p < 0.01, *** p < 0.001.",
                   "기준 = 마우스의 기준 세션 평균. 괄호 (마우스를 층으로 한 CMH, p < 0.05만 표시): 파랑(base-veh)·빨강(base-SBI) = 같은 날 기준 "
                   "대비, 검정(veh-SBI) = SBI-553 vs vehicle."))
        dk.slide(sec("BEHAVIOUR", "행동"), L(lg, "Affective and reflexive pain together", "정서적 통증과 반사적 통증을 함께"),
                 L(lg, "A selective effect on affective pain would move the arrows straight DOWN", "정서적 통증만 줄면 화살표가 똑바로 아래로"),
                 f("f15_affective_reflexive.png"),
                 L(lg, "One arrow per mouse: " + ("from the vehicle to the SBI-553 injection session." if both else "from the baseline to the injection session."),
                   "마우스당 화살표 하나: " + ("vehicle 주사 세션에서 SBI-553 주사 세션으로." if both else "기준 세션에서 주사 세션으로.")),
                 L(lg, "Affective = attending + lick/bite + guarding per delivery, reflexive = withdrawal + flinch per delivery. Descriptive (no test).",
                   "정서 = attending + lick/bite + guarding, 반사 = withdrawal + flinch (자극당). 기술 통계."))
        mv = TMv.set_index("drug") if len(TMv) else None
        dk.slide(sec("BEHAVIOUR", "행동"), L(lg, "Movement before any stimulus", "자극 전 움직임"),
                 (L(lg, f"First 3 min moving: vehicle day {mv.loc['vehicle', 'base']:.0f} -> {mv.loc['vehicle', 'inj']:.0f}%, "
                        f"SBI-553 day {mv.loc['SBI-553', 'base']:.0f} -> {mv.loc['SBI-553', 'inj']:.0f}% (mean of mice)",
                    f"처음 3분 움직임: vehicle 날 {mv.loc['vehicle', 'base']:.0f} -> {mv.loc['vehicle', 'inj']:.0f}%, SBI-553 날 "
                    f"{mv.loc['SBI-553', 'base']:.0f} -> {mv.loc['SBI-553', 'inj']:.0f}%") if mv is not None and {"vehicle", "SBI-553"} <= set(mv.index) else ""),
                 f("f16_movement_before_stimulus.png"),
                 L(lg, "No stimulus is given in the first 3 min, so a change there is not pain relief.",
                   "처음 3분에는 자극이 없으므로 여기의 변화는 통증 완화가 아님."),
                 L(lg, "% of frames with automatic general movement in the first 3 min (no stimulus), baseline vs the injection session of the "
                       "same day; thin line = one mouse, bold = mean +- SEM. Descriptive.",
                   "처음 3분(자극 없음) 자동 general movement 프레임 %, 같은 날 기준 vs 주사 세션; 얇은 선 = 마우스. 기술 통계."))
    if use_ea:                                                  # your plots 6-7
        DE.behaviour(dk, ea, Lf, a.short)
    if both:                                                    # sex (added 10-22): 2 females + 2 males
        import make_sex_deck as SX
        SX.add_sex_slides(dk, lg, short=a.short)
    # ------------------------------------------------------------------ limits
    lim = [("4", L(lg, "mice (2 per order)", "마리 (순서마다 2마리)"),
            L(lg, "No group test across mice can reach p < 0.05 (minimum 0.125); neuron-level tests are exploratory.",
              "마우스 간 검정은 p < 0.05 불가 (최소 0.125); 뉴런 단위 검정은 탐색적."))]
    if DAYS == [1]:
        lim += [("2 + 2", L(lg, "mice per drug on Day 1", "Day 1 약물당 마우스"),
                 L(lg, "SBI-553 and vehicle are different mice on one day: the crossover (Day 1 + Day 2 deck) compares within mice.",
                   "하루 안에서는 SBI-553과 vehicle이 다른 마우스: 같은 마우스 비교는 Day 1 + Day 2 덱.")),
                ("10-21", L(lg, "Day-1 selection confirmed by you", "Day 1 선택은 확정됨"),
                 L(lg, "1191: the brightest neuron added 10-22 (#1 + #2 merged, your decision).", "1191: 가장 밝은 뉴런 10-22 추가 (#1 + #2 합침, 당신 결정).")),
                ("", "", "")]
    if DAYS != [1]:
        lim += [("5937", L(lg, "Day-2 signal weak", "Day 2 신호 약함"),
                 L(lg, "Few active neurons on Day 2 (also at baseline): its Day-2 numbers are near the floor.",
                   "Day 2에 활동 뉴런이 적음 (기준에서도): Day 2 수치는 바닥 근처.")),
                ("5939", L(lg, "tracking weakest", "추적이 가장 약함") if both else L(lg, "many recovered neurons", "복구 뉴런 많음"),
                 (L(lg, "Activity of the same neurons agrees less between days than in the other mice.",
                    "같은 뉴런의 활동이 다른 마우스보다 두 날 사이 덜 일치.") if prov == "" else
                  L(lg, "Activity of the same neurons agrees less between days; many recovered Day-1 neurons need your check.",
                    "같은 뉴런의 활동이 두 날 사이 덜 일치; 복구한 Day 1 뉴런 다수는 확인 필요."))),
                ("Day 2", L(lg, "selection provisional", "선택 잠정"),
                 L(lg, "Your confirmation of the Day-2 neurons may change every number here.", "Day 2 뉴런 확인 결과에 따라 모든 수치가 바뀔 수 있음."))]
        if prov == "":
            lim = [c for c in lim if c[0] != "Day 2"]
        if use_ea:
            lim.append(("~3 s", L(lg, "between deliveries", "자극 간격"),
                        L(lg, "Windows longer than ~3 s mix the next stimulus; fast 'inhibition' may be background (field) artefact.",
                          "~3 s보다 긴 창은 다음 자극이 섞임; 빠른 '억제'는 배경(시야) artefact일 수 있음.")))
    lim = [c for c in lim if c[0]]
    dk.cards(sec("LIMITS", "한계"), L(lg, "What the data can and cannot say yet", "지금 데이터로 말할 수 있는 것과 없는 것"), "", lim,
             L(lg, ("Neuron selection confirmed by you (10-22). Next: population dynamics (state space) on top of these basic results."
                    if prov == "" else "Next: your confirmation of the Day-2 selection, then every deck again with the final neurons.")
               if DAYS != [1] else "Next: the Day 1 + Day 2 deck compares the drugs within the same mice.",
               ("뉴런 선택 확정 (10-22). 다음: 이 기본 결과 위에 집단 동역학(state space) 분석." if prov == "" else
                "다음: Day 2 선택 확인 후 최종 뉴런으로 모든 덱을 다시 만듦.") if DAYS != [1] else "다음: Day 1 + Day 2 덱에서 같은 마우스 안 약물 비교.")
             + " " + dk.n_txt + L(lg, "; descriptive summary of the limits (no test).", "; 한계의 기술 요약 (검정 없음)."), "")
    # ------------------------------------------------------------------ supplement (FULL only)
    if not a.short and use_ea:
        DE.supplement(dk, ea, Lf)
    if False:                                                   # 10-22: replaced by the EVERY EVENT section (all 10 events)
        for ev in RESP_EV:
            if ev == key_ev:
                continue
            fn = f(f"s_heatmap_{ev.replace(' ', '_').replace('/', '_')}.png")
            r_ = fig_heatmap(D, neuron_rows(B, P, D, ev), ev, fn)
            if r_:
                dk.slide(sec("SUPPLEMENT", "부록"), L(lg, f"{ev}: every neuron in the three conditions", f"{ev}: 모든 뉴런, 세 조건"), "", fn,
                         L(lg, f"Mean 0-2 s: baseline {r_['mean']['baseline']:+.2f}, vehicle {r_['mean']['vehicle']:+.2f}, SBI-553 {r_['mean']['SBI-553']:+.2f} z.",
                           f"0-2 s 평균: 기준 {r_['mean']['baseline']:+.2f}, vehicle {r_['mean']['vehicle']:+.2f}, SBI-553 {r_['mean']['SBI-553']:+.2f} z."),
                         L(lg, "As the main heatmap slide.", "본문 히트맵 슬라이드와 같음."))
        for ev in STIM + ["withdrawal"]:
            if ev == ev_r:
                continue
            fn = f(f"s_maps_{ev.replace(' ', '_').replace('/', '_')}.png")
            fig_maps(B, E, ev, fn, FPc)
            dk.slide(sec("SUPPLEMENT", "부록"), L(lg, f"Where the {ev} neurons are", f"{ev} 반응 뉴런의 위치"), "", fn, "",
                     L(lg, "As the main map slide.", "본문 지도 슬라이드와 같음."))
    dk.prs.save(a.out)
    for nm, T_ in (("t_population_stimuli", TP), ("t_population_behaviour", TB), ("t_responsive", TR), ("t_responsive_change", TC),
                   ("t_behaviour", TBh), ("t_every_mouse", TM), ("t_groups", TG)):
        T_.round(4).to_csv(f(nm + ".csv"), index=False)
    print("->", a.out, "|", len(dk.prs.slides), "slides | figures in", od)


if __name__ == "__main__":
    main()
