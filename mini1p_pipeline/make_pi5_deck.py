"""make_pi5_deck.py - 10-26 (user): the folder-3 results (event marks AND response window chosen per event; exploratory) for the PI
in 5 compact slides + supplements (methods, numbers / tests, per mouse, reflex / movement neurons, encoding model).
Every number is computed here from analysis\\_variants\\mix4 (no hand-typed values). v2 (10-26, after an independent number check
and a PI-style audit): titles say only what the data support, the day / drug-order confound and the per-mouse support are on the
slide face, within-day tests on every slide, leave-one-mouse-out column on the summary, star legend and a footer per slide.

  python make_pi5_deck.py        -> lab_meeting\\decks_Day1_Day2\\v2_20261022\\FINAL_20261026_PI5_mixmarks\\PI5_mixmarks_EN.pptx
"""
import json
import os
import textwrap

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2, fisher_exact, mannwhitneyu, wilcoxon

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.textpath import TextPath  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from pptx import Presentation  # noqa: E402
from pptx.dml.color import RGBColor  # noqa: E402
from pptx.enum.text import PP_ALIGN  # noqa: E402
from pptx.util import Inches, Pt  # noqa: E402

B = r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553"
A = B + r"\analysis\_variants\mix4"
SCR = r"C:\Users\hsollim\AppData\Local\Temp\claude\C--Users-hsollim\e78b98bc-45eb-4eba-b6ff-ba7ac412ccce\scratchpad"
OUT = B + r"\lab_meeting\decks_Day1_Day2\v2_20261022\FINAL_20261026_PI5_mixmarks"
FIG = OUT + r"\figures"
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({"font.family": "Arial", "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "xtick.labelsize": 11,
                     "ytick.labelsize": 11, "legend.fontsize": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 200, "hatch.color": "white"})
MICE = ["5941", "5939", "5937", "1191"]
SBI = {1: {"5941", "5939"}, 2: {"5937", "1191"}}
SBIDAY = {"5941": 1, "5939": 1, "5937": 2, "1191": 2}
MC = {"5941": "#e67e22", "5939": "#8e44ad", "5937": "#16a085", "1191": "#2c3e50"}
MK = {"5941": "o", "5939": "s", "5937": "^", "1191": "D"}
SEX = {"5941": "F", "5939": "M", "5937": "M", "1191": "F"}
CC = {"vb": "#c4c4c4", "vi": "#2471a3", "sb": "#7a7a7a", "si": "#c0392b"}
CL = {"vb": "baseline, vehicle day", "vi": "after vehicle", "sb": "baseline, SBI-553 day", "si": "after SBI-553"}
STIM = ["Light touch", "Mild touch", "Pin prick", "Heat"]
REFL = ["withdrawal", "flinch", "movement onset"]
MARKS = {"Heat", "withdrawal"}
STARS = "* p < 0.05, ** p < 0.01, *** p < 0.001, ns p ≥ 0.05"
FT_NEUR = ("Exploratory: neurons pooled over 4 mice, uncorrected; event marks and response window chosen per event "
           "(heat, withdrawal = original key presses, ~0.4 s late) - S1")
FT_BEH = "Exploratory: deliveries pooled within each mouse (CMH), uncorrected; heat deliveries = original key presses - S1"
FT_RATE = "Firing does not depend on event marks or windows; the drug order is the same as the day order (S1)"


def st(p):
    return "ns" if p is None or not np.isfinite(p) or p >= 0.05 else "***" if p < 0.001 else "**" if p < 0.01 else "*"


def pf(p):
    if p is None or not np.isfinite(p):
        return "n.a."
    return "p < 0.001" if p < 0.001 else f"p = {p:.4f}" if p < 0.01 else f"p = {p:.3f}" if p < 0.1 else f"p = {p:.2f}"


def mlab(m):
    return f"{m} ({'female' if SEX[m] == 'F' else 'male'}, SBI-553 Day {SBIDAY[m]})"      # 10-26: sex spelled out


# ------------------------------------------------------------------------------------------------ data
cc = pd.read_csv(A + r"\event_analysis\cells_classes.csv")
cc["mouse"] = cc.mouse.astype(str)
cc["drug"] = [("SBI-553" if m in SBI[d] else "vehicle") for m, d in zip(cc.mouse, cc.day)]
cc["UP"] = cc.cls.eq("activated"); cc["DOWN"] = cc.cls.eq("inhibited")
cc["cond"] = [("v" if dr == "vehicle" else "s") + ("b" if ph == "pre" else "i") for dr, ph in zip(cc.drug, cc.phase)]


def win(ev):
    e = cc[cc.event == ev].iloc[0]
    return float(e.pre), float(e.a), float(e.b)


def wtxt(ev):
    pre, a, b = win(ev)
    return f"{a:g}-{b:g} s vs the {pre:g} s before"


def pairs(x):
    return x[x.phase == "pre"].set_index(["mouse", "day", "cell"]).join(
        x[x.phase == "post"].set_index(["mouse", "day", "cell"]), lsuffix="_b", rsuffix="_a", how="inner")


def mcnemar(x, col):
    j = pairs(x)
    b10 = int((j[col + "_b"] & ~j[col + "_a"]).sum()); b01 = int((~j[col + "_b"] & j[col + "_a"]).sum())
    return (binomtest(b01, b01 + b10).pvalue if b01 + b10 else 1.0), b01, b10


def single(ev, d=None):
    d = cc if d is None else d
    e = d[d.event == ev]
    R = {}
    for c in ("vb", "vi", "sb", "si"):
        x = e[e.cond == c]
        R[c] = dict(n=len(x), UP=100 * x.UP.mean() if len(x) else np.nan, DOWN=100 * x.DOWN.mean() if len(x) else np.nan,
                    kUP=int(x.UP.sum()), kDOWN=int(x.DOWN.sum()),
                    mUP={m: 100 * g.UP.mean() for m, g in x.groupby("mouse")}, mDOWN={m: 100 * g.DOWN.mean() for m, g in x.groupby("mouse")},
                    nm={m: len(g) for m, g in x.groupby("mouse")})
    T = {}
    for col in ("UP", "DOWN"):
        if R["vi"]["n"] and R["si"]["n"]:
            kv, ks = R["vi"]["k" + col], R["si"]["k" + col]
            T[col + "_after"] = fisher_exact([[kv, R["vi"]["n"] - kv], [ks, R["si"]["n"] - ks]])[1]
        T[col + "_veh"] = mcnemar(e[e.drug == "vehicle"], col)[0]
        T[col + "_sbi"] = mcnemar(e[e.drug == "SBI-553"], col)[0]
    return R, T


def wsr(x):
    j = pairs(x)
    dd = (j.resp_a - j.resp_b).dropna()
    return wilcoxon(dd)[1] if (dd != 0).sum() >= 5 else np.nan


def popul(ev, d=None):
    d = cc if d is None else d
    e = d[d.event == ev]
    R = {}
    for c in ("vb", "vi", "sb", "si"):
        x = e[e.cond == c].resp.dropna()
        R[c] = dict(n=len(x), mean=x.mean() if len(x) else np.nan, sem=x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan,
                    m={m: g.resp.mean() for m, g in e[e.cond == c].groupby("mouse")})
    T = dict(after=mannwhitneyu(e[e.cond == "si"].resp.dropna(), e[e.cond == "vi"].resp.dropna())[1],
             base=mannwhitneyu(e[e.cond == "sb"].resp.dropna(), e[e.cond == "vb"].resp.dropna())[1],
             veh=wsr(e[e.drug == "vehicle"]), sbi=wsr(e[e.drug == "SBI-553"]))
    return R, T


SG = {ev: single(ev) for ev in STIM + REFL}
PO = {ev: popul(ev) for ev in STIM + REFL}
TR = pd.read_csv(A + r"\tracked_crossover\tracked_tests.csv")
TR = TR[TR.tracking == "A+B"].set_index("event")
BT = pd.read_csv(A + r"\event_analysis\P6_behaviour_tests.csv")
PD = pd.read_csv(A + r"\event_analysis\P6_per_delivery.csv"); PD["mouse"] = PD.mouse.astype(str)
PD["drug"] = [("SBI-553" if m in SBI[d] else "vehicle") for m, d in zip(PD.mouse, PD.day)]
PD["cond"] = [("v" if dr == "vehicle" else "s") + ("b" if ph == "pre" else "i") for dr, ph in zip(PD.drug, PD.phase)]
SESS = json.load(open(A + r"\pi_extra\session\summary.json", encoding="utf-8"))
RATE = {(r["mouse"], r["drug"]): r for r in SESS["RATE_blocks.png"]["numbers"]}
RNORM = SESS["RATE_blocks_norm.png"]["numbers"]["per_group"]
BNORM = SESS["BEH_norm_delivery.png"]["numbers"]
ENC = pd.read_csv(A + r"\event_analysis\O3_encoding_model.csv")
N_VEH = int(((cc.event == "Pin prick") & (cc.cond == "vi")).sum()); N_SBI = int(((cc.event == "Pin prick") & (cc.cond == "si")).sum())
NM_DAY = {m: {d: int(((cc.event == "Pin prick") & (cc.mouse == m) & (cc.day == d) & (cc.phase == "post")).sum()) for d in (1, 2)} for m in MICE}
_nr = [r["n_cells"] for r in RATE.values()]
N_RATE = f"{min(_nr)}-{max(_nr)}"                                            # neurons per mouse-day in the firing analysis
_nd = PD.groupby(["mouse", "day", "phase"]).size()
N_DEL = f"{_nd.min()}-{_nd.max()}"                                           # deliveries per session
_nds = PD.groupby(["mouse", "day", "phase", "stim"]).size()
N_DEL_STIM = f"{_nds.min()}-{_nds.max()}"                                    # deliveries per session and stimulus
N_TR = sum(RNORM[m]["n"] for m in MICE)                                       # tracked neurons


def cmh_p(tables):
    """as ca_event_analysis.cmh_p: CMH chi-square, continuity-corrected, one 2 x 2 table per mouse"""
    num, var = 0.0, 0.0
    for t in tables:
        (a, b), (c, d) = np.asarray(t, float)
        n1, n2, m1, m2 = a + b, c + d, a + c, b + d
        N = n1 + n2
        if N <= 1 or n1 == 0 or n2 == 0 or m1 == 0 or m2 == 0:
            continue
        num += a - n1 * m1 / N
        var += n1 * n2 * m1 * m2 / (N * N * (N - 1))
    return float(chi2.sf(max(abs(num) - 0.5, 0.0) ** 2 / var, 1)) if var > 0 else np.nan


def cmh(beh, stim, which="SBI-553 vs vehicle"):
    r = BT[(BT.behaviour == beh) & (BT.stim == stim) & BT.test.str.startswith(which)]
    return float(r.p.iloc[0]) if len(r) else np.nan


def beh_pct(beh, stim, c):
    x = PD[(PD.cond == c) & (PD.stim == stim)]
    return {m: 100 * g[beh].mean() for m, g in x.groupby("mouse")}, len(x)


def did_test(beh, drop=None, stim_fe=False, stim=None):
    """10-26 (user: 'p value for last panel'): is the change after the injection (vs the same day's baseline) different on the SBI-553
    day and the vehicle day?  Deliveries, logistic regression with one intercept per mouse-day (absorbs each baseline), a post effect per
    mouse and one common post x SBI-553 term; likelihood-ratio test of that term (1 df).  Returns (p, coefficient).
    stim = one stimulus only (the per-stimulus slide)."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    d = PD if drop is None else PD[PD.mouse != drop]
    if stim is not None:
        d = d[d.stim == stim]
    if d[beh].astype(int).sum() < 3 or (~d[beh].astype(bool)).sum() < 3:
        return np.nan, np.nan
    d = pd.DataFrame(dict(y=d[beh].astype(int), md=d.mouse + "_" + d.day.astype(str), mouse=d.mouse, stim=d.stim,
                          post=(d.phase == "post").astype(int), sbi=(d.drug == "SBI-553").astype(int)))
    rhs = "C(md) + C(mouse):post" + (" + C(stim)" if stim_fe else "")
    f0 = smf.glm(f"y ~ {rhs}", d, family=sm.families.Binomial()).fit()
    f1 = smf.glm(f"y ~ {rhs} + post:sbi", d, family=sm.families.Binomial()).fit()
    return float(chi2.sf(2 * (f1.llf - f0.llf), 1)), float(f1.params["post:sbi"])


def sex_labels(a, x, vals, gap, dx=-0.13):
    """10-26 (user: 'indicate f or male'): F / M next to each mouse's point, nudged apart so they never overlap"""
    items = sorted(((vals[m], m) for m in MICE if m in vals and np.isfinite(vals[m])))
    ys = []
    for v, _ in items:
        ys.append(max(v, ys[-1] + gap) if ys else v)
    shift = np.mean([y - v for y, (v, _) in zip(ys, items)])
    for y, (v, m) in zip(ys, items):
        a.text(x + dx, y - shift, SEX[m], color=MC[m], fontsize=10, fontweight="bold", ha="right" if dx < 0 else "left", va="center",
               clip_on=False)


# ------------------------------------------------------------------------------------------------ drawing helpers
def bracket(a, x1, x2, y, txt, color="k", fs=11, h=None):
    h = h if h is not None else 0.015 * (a.get_ylim()[1] - a.get_ylim()[0])
    a.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color=color, lw=1.2, clip_on=False)
    a.text((x1 + x2) / 2, y + h, txt, ha="center", va="bottom", color=color, fontsize=fs if txt != "ns" else fs - 1,
           fontweight="bold" if txt != "ns" else "normal", clip_on=False)


OFF = {"5941": -0.21, "5939": -0.07, "5937": 0.07, "1191": 0.21}


def mice_dots(a, x, vals, s=34):
    for m in MICE:
        if m in vals and np.isfinite(vals[m]):
            a.scatter(x + OFF[m] * 0.75, vals[m], s=s, marker=MK[m], color=MC[m], edgecolor="white", lw=0.6, zorder=6, clip_on=False)


def mouse_handles():
    return [Line2D([], [], ls="", marker=MK[m], color=MC[m], markeredgecolor="white", markersize=8, label=mlab(m)) for m in MICE]


def cond_handles(hatch=False):
    h = [Patch(color=CC[c], label=CL[c]) for c in ("vb", "vi", "sb", "si")]
    if hatch:
        h.append(Patch(facecolor="0.55", hatch="///", edgecolor="white", label="hatched = DOWN"))
    return h


def legend2(fig, hatch=False, y=1.0):
    fig.legend(handles=cond_handles(hatch), ncol=5 if hatch else 4, loc="upper center", bbox_to_anchor=(0.5, y), frameon=False,
               fontsize=11, handlelength=1.3, columnspacing=1.4)
    fig.legend(handles=mouse_handles(), ncol=4, loc="upper center", bbox_to_anchor=(0.5, y - 0.065), frameon=False, fontsize=10.5,
               handlelength=1.0, columnspacing=1.4)


def day_labels(a):
    a.text(0.5, -0.20, "vehicle day", transform=a.get_xaxis_transform(), ha="center", va="top", fontsize=10.5, color=CC["vi"])
    a.text(2.5, -0.20, "SBI-553 day", transform=a.get_xaxis_transform(), ha="center", va="top", fontsize=10.5, color=CC["si"])


NOTES = {}


# ------------------------------------------------------------------------------------------------ 1 behaviour
def fig_behaviour(fn):
    fig = plt.figure(figsize=(12.3, 4.75))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.95, 2.45, 1.25], wspace=0.30, left=0.065, right=0.99, top=0.74, bottom=0.18)
    # A: any stimulus -> movement, per mouse
    a = fig.add_subplot(gs[0])
    g = {c: {m: 100 * x.movement.mean() for m, x in PD[PD.cond == c].groupby("mouse")} for c in ("vi", "si")}
    for m in MICE:
        a.plot([0, 1], [g["vi"][m], g["si"][m]], color=MC[m], lw=1.6, marker=MK[m], markersize=8, markeredgecolor="white", zorder=4,
               clip_on=False)
    pw = wilcoxon([g["si"][m] - g["vi"][m] for m in MICE])[1]
    a.set_xticks([0, 1], ["after\nvehicle", "after\nSBI-553"]); a.set_xlim(-0.4, 1.4); a.set_ylim(0, 100)
    a.set_ylabel("% of deliveries followed by\na movement bout (0-3 s)")
    a.set_title("A  any stimulus → movement", loc="left", fontsize=12.5)
    clear = [m for m in MICE if g["si"][m] - g["vi"][m] <= -10]
    a.text(0.5, 97, f"≥ 10 points lower: {len(clear)}/4 mice\n({', '.join(clear)})\nWilcoxon, 4 mice: {pf(pw)}", ha="center", va="top",
           fontsize=10.5)
    sex_labels(a, 0, g["vi"], gap=5.5)
    # B: the three behaviour x stimulus pairs that differed after SBI-553 vs after vehicle
    b = fig.add_subplot(gs[1])
    groups = [("movement", "Light touch", "Light touch → movement"), ("flinch", "Pin prick", "Pin prick → flinch"),
              ("escape/rearing", "Heat", "Heat → escape / rearing")]
    b.set_ylim(0, 134)
    info = []
    for gi, (beh, stim, lab) in enumerate(groups):
        x0 = gi * 5.0
        v = {}
        for k, c in enumerate(("vb", "vi", "sb", "si")):
            pm, n = beh_pct(beh, stim, c)
            v[c] = np.mean(list(pm.values()))
            b.bar(x0 + k, v[c], color=CC[c], width=0.8, zorder=2)
            mice_dots(b, x0 + k, pm, s=28)
        p_veh, p_sbi, p_aft = cmh(beh, stim, "vehicle vs its baseline"), cmh(beh, stim, "SBI-553 vs its baseline"), cmh(beh, stim)
        hi = lambda cs: max(max(beh_pct(beh, stim, c)[0].values()) for c in cs)
        y1, y2 = hi(("vb", "vi")) + 5, hi(("sb", "si")) + 5
        bracket(b, x0, x0 + 1, y1, st(p_veh), color=CC["vi"], h=1.5)
        bracket(b, x0 + 2, x0 + 3, y2, st(p_sbi), color=CC["si"], h=1.5)
        bracket(b, x0 + 1, x0 + 3, max(y1, y2) + 12, st(p_aft), color="k", h=1.5)
        lower = sum(beh_pct(beh, stim, "si")[0][m] < beh_pct(beh, stim, "vi")[0][m] for m in MICE)
        info.append(dict(lab=lab, beh=beh, stim=stim, p_veh=p_veh, p_sbi=p_sbi, p_aft=p_aft, v=v, lower=lower,
                         n={c: beh_pct(beh, stim, c)[1] for c in ("vb", "vi", "sb", "si")}))
    b.set_xticks([x * 5 + 1.5 for x in range(3)], [l for _, _, l in groups], fontsize=11.5)
    b.set_yticks([0, 25, 50, 75, 100]); b.set_ylabel("% of deliveries followed\nby the behaviour")
    b.set_title("B  behaviours that differed after SBI-553 vs after vehicle", loc="left", fontsize=12.5)
    # C: reflexes, change from the same day's baseline, each mouse
    c_ = fig.add_subplot(gs[2])
    for k, beh in enumerate(("withdrawal", "flinch")):
        for m in MICE:
            yv, ys = BNORM[beh]["vehicle_day"][m], BNORM[beh]["sbi_day"][m]
            c_.plot([3.2 * k, 3.2 * k + 1.3], [yv, ys], color=MC[m], marker=MK[m], markersize=7, markeredgecolor="white", lw=1.4, clip_on=False)
    c_.axhline(0, color="0.5", lw=0.8)
    c_.set_xticks([0, 1.3, 3.2, 4.5], ["vehicle\nday", "SBI-553\nday", "vehicle\nday", "SBI-553\nday"], fontsize=10); c_.set_xlim(-1.0, 5.4)
    c_.text(0.65, -0.26, "withdrawal", transform=c_.get_xaxis_transform(), ha="center", va="top", fontsize=11)
    c_.text(3.85, -0.26, "flinch", transform=c_.get_xaxis_transform(), ha="center", va="top", fontsize=11)
    c_.set_ylabel("change from the same day's\nbaseline (percentage points)")
    c_.set_title("C  reflex rise after the injection", loc="left", fontsize=12.5)
    nlw, nlf = BNORM["withdrawal"]["n_mice_sbi_lower"], BNORM["flinch"]["n_mice_sbi_lower"]
    top = max(max(BNORM[b_]["vehicle_day"].values()) for b_ in ("withdrawal", "flinch"))
    c_.set_ylim(min(0, min(min(BNORM[b_][d_].values()) for b_ in ("withdrawal", "flinch") for d_ in ("vehicle_day", "sbi_day"))) - 3,
                top * 1.75)
    c_.set_yticks(np.arange(0, top * 1.25, 10))
    pc = {}                                                     # 10-26: p values (user: 'p value for last panel')
    for k, beh in enumerate(("withdrawal", "flinch")):
        x0 = 3.2 * k
        pm = wilcoxon([BNORM[beh]["sbi_day"][m] - BNORM[beh]["vehicle_day"][m] for m in MICE])[1]
        pd_, coef = did_test(beh)
        pc[beh] = dict(p_mice=pm, p_del=pd_, coef=coef, n_lower=BNORM[beh]["n_mice_sbi_lower"])
        # 10-26 check: numbers, no stars - the smaller rise comes from higher SBI-553-day baselines (post levels equal)
        tb_ = lambda c: [[int(x[beh].sum()), int((~x[beh].astype(bool)).sum())] for x in [PD[(PD.mouse == m) & (PD.cond == c)] for m in MICE]]
        pc[beh]["p_base"] = cmh_p([[tb_("sb")[i], tb_("vb")[i]] for i in range(4)])
        pc[beh]["p_post"] = cmh_p([[tb_("si")[i], tb_("vi")[i]] for i in range(4)])
        pc[beh]["base_higher"] = sum(PD[(PD.mouse == m) & (PD.cond == "sb")][beh].mean() > PD[(PD.mouse == m) & (PD.cond == "vb")][beh].mean()
                                     for m in MICE)
        pc[beh]["p_no1191"] = did_test(beh, drop="1191")[0]
        c_.text(x0 + 0.65, top * 1.70, f"{pc[beh]['n_lower']}/4 mice smaller:\nWilcoxon p = {pm:.3f}\ndeliveries: p = {pd_:.3f}"
                + (" (ns)" if pd_ >= 0.05 else ""), ha="center", va="top", fontsize=9)
        sex_labels(c_, x0, BNORM[beh]["vehicle_day"], gap=0.075 * top)
    legend2(fig)
    fig.savefig(fn); plt.close(fig)
    allp = BT[BT.test.str.startswith("SBI-553 vs vehicle")]
    NOTES["beh"] = dict(g=g, pw=pw, clear=clear, info=info, ntest=int(allp.p.notna().sum()), nsig=int((allp.p < 0.05).sum()),
                        nall=len(allp), nlw=nlw, nlf=nlf, pc=pc)


# ------------------------------------------------------------------------------------------------ 2 firing
def fig_firing(fn):
    fig, ax = plt.subplots(1, 3, figsize=(12.3, 4.75), gridspec_kw=dict(width_ratios=[1.55, 0.95, 1.05], wspace=0.42, left=0.06,
                                                                        right=0.99, top=0.80, bottom=0.20))
    a = ax[0]
    for m in MICE:
        for k, dr in enumerate(("vehicle", "SBI-553")):
            r = RATE[(m, dr)]
            a.plot([3 * k, 3 * k + 1], [r["base_median"], r["inj_median"]], color=MC[m], marker=MK[m], markersize=8,
                   markeredgecolor="white", lw=1.6, clip_on=False, zorder=4)
    a.set_xticks([0, 1, 3, 4], ["baseline", "after\nvehicle", "baseline", "after\nSBI-553"])
    a.text(0.5, -0.25, "vehicle day", transform=a.get_xaxis_transform(), ha="center", va="top", color=CC["vi"], fontsize=11.5, fontweight="bold")
    a.text(3.5, -0.25, "SBI-553 day", transform=a.get_xaxis_transform(), ha="center", va="top", color=CC["si"], fontsize=11.5, fontweight="bold")
    a.set_xlim(-0.4, 4.6); a.set_ylim(bottom=0)
    a.set_ylabel("Ca transients / min\n(median of the mouse's neurons)")
    a.set_title("A  firing in the stimulus blocks", loc="left", fontsize=12.5)
    b = ax[1]
    for m in MICE:
        v, s_ = RATE[(m, "vehicle")]["change_pct"], RATE[(m, "SBI-553")]["change_pct"]
        b.plot([0, 1], [v, s_], color=MC[m], marker=MK[m], markersize=8, markeredgecolor="white", lw=1.6, zorder=3, clip_on=False)
    b.axhline(0, color="0.5", lw=0.8)
    b.set_xticks([0, 1], ["vehicle\nday", "SBI-553\nday"]); b.set_xlim(-0.35, 1.35); b.set_ylim(-80, 20)
    b.set_ylabel("change after the injection\n(%, median over neurons)")
    b.set_title("B  same-day change", loc="left", fontsize=12.5)
    c_ = ax[2]
    for i, m in enumerate(MICE):
        r = RNORM[m]
        c_.scatter(r["diff_median"], i, s=70, marker=MK[m], color=MC[m], edgecolor="white", zorder=4)
        c_.text(0.62, i, f"{st(r['p_wilcoxon'])}  ({r['n']})", va="center", fontsize=10.5,
                fontweight="bold" if r["p_wilcoxon"] < 0.05 else "normal")
    c_.axvline(0, color="0.5", lw=0.8)
    c_.set_yticks(range(4), MICE); c_.set_ylim(3.6, -0.6); c_.set_xlim(-1.6, 1.15)
    c_.set_xlabel("SBI-553-day change minus vehicle-day\nchange (log2 fold, same neuron)")
    c_.set_title("C  same tracked neurons", loc="left", fontsize=12.5)
    fig.legend(handles=mouse_handles(), ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=10.5)
    fig.savefig(fn); plt.close(fig)
    d1hi = sum(RATE[(m, "SBI-553" if SBIDAY[m] == 1 else "vehicle")]["base_median"] >
               RATE[(m, "vehicle" if SBIDAY[m] == 1 else "SBI-553")]["base_median"] for m in MICE)
    nl = sum(RATE[(m, "SBI-553")]["change_pct"] < RATE[(m, "vehicle")]["change_pct"] for m in MICE)
    ntr = sum(RNORM[m]["diff_median"] < 0 for m in MICE)
    trs = [m for m in MICE if RNORM[m]["diff_median"] < 0 and RNORM[m]["p_wilcoxon"] < 0.05]
    NOTES["rate"] = dict(d1hi=d1hi, nl=nl, ntr=ntr, trs=trs)


# ------------------------------------------------------------------------------------------------ 3 population
def fig_population(fn, evs=STIM):
    fig, ax = plt.subplots(1, len(evs), figsize=(12.3, 4.75), sharey=True,
                           gridspec_kw=dict(wspace=0.12, left=0.085, right=0.995, top=0.72, bottom=0.17))

    def hi_of(R, cs):
        v = []
        for c in cs:
            if R[c]["n"]:
                v += [R[c]["mean"] + (R[c]["sem"] if np.isfinite(R[c]["sem"]) else 0)] + [x for x in R[c]["m"].values() if np.isfinite(x)]
        return max(v) if v else 0.0
    allv = [x for ev in evs for c in ("vb", "vi", "sb", "si") for x in ([PO[ev][0][c]["mean"] - (PO[ev][0][c]["sem"] or 0)]
                                                                      + list(PO[ev][0][c]["m"].values())) if np.isfinite(x)]
    top = max(hi_of(PO[ev][0], ("vb", "vi", "sb", "si")) for ev in evs)
    span = top - min(min(allv), 0); pad = 0.06 * span
    lo, hi = min(min(allv), 0) - 0.04 * span, top + 5.2 * pad
    for a, ev in zip(ax, evs):
        R, T = PO[ev]
        for k, c in enumerate(("vb", "vi", "sb", "si")):
            a.bar(k, R[c]["mean"], yerr=R[c]["sem"], color=CC[c], width=0.75, capsize=3, error_kw=dict(lw=1), zorder=2)
            mice_dots(a, k, R[c]["m"], s=26)
        a.axhline(0, color="0.4", lw=0.8)
        a.set_ylim(lo, hi)
        y1, y2 = hi_of(R, ("vb", "vi")) + 0.6 * pad, hi_of(R, ("sb", "si")) + 0.6 * pad
        bracket(a, 0, 1, y1, st(T["veh"]), color=CC["vi"], h=0.25 * pad)
        bracket(a, 2, 3, y2, st(T["sbi"]), color=CC["si"], h=0.25 * pad)
        ya = max(y1, y2) + 1.5 * pad
        bracket(a, 1, 3, ya, st(T["after"]) + (f" ({pf(T['after'])})" if T["after"] < 0.05 else ""), color="k", h=0.25 * pad)
        bracket(a, 0, 2, ya + 1.5 * pad, st(T["base"]) + (f" ({pf(T['base'])})" if T["base"] < 0.05 else ""), color="0.45", h=0.25 * pad)
        a.set_xticks(range(4), ["baseline", "after", "baseline", "after"], fontsize=10.5)
        pre, a_, b_ = win(ev)
        a.set_title(f"{ev}\n{a_:g}-{b_:g} s" + (" (original marks)" if ev in MARKS else ""), fontsize=12.5)
        day_labels(a)
    ax[0].set_ylabel("response (z): response window\nminus the window just before")
    legend2(fig)
    fig.savefig(fn); plt.close(fig)


# ------------------------------------------------------------------------------------------------ 2b single neurons, baseline-normalized
TRK = pd.read_csv(A + r"\tracked_crossover\tracked_responses.csv"); TRK["mouse"] = TRK.mouse.astype(str)


def norm_cells(ev):
    """10-26 (user: 'baseline normalized ... single cell analysis, not just mouse by mouse'): each neuron's response after the injection
    minus its own response in the same day's baseline session (R after - R baseline, z), vehicle day and SBI-553 day"""
    e = cc[cc.event == ev]
    out = {}
    for dr in ("vehicle", "SBI-553"):
        j = pairs(e[e.drug == dr]).reset_index()
        j["chg"] = j.resp_a - j.resp_b
        out[dr] = j.dropna(subset=["chg"])
    t = TRK[TRK.event == ev]
    T = dict(veh=wilcoxon(out["vehicle"].chg)[1], sbi=wilcoxon(out["SBI-553"].chg)[1],
             between=mannwhitneyu(out["SBI-553"].chg, out["vehicle"].chg)[1], tracked=wilcoxon(t.d_norm)[1],
             n_trk=len(t), med_d=float(t.d_norm.median()),
             mice_lower=sum(g.d_norm.median() < 0 for _, g in t.groupby("mouse")))
    return out, t, T


NORM = {ev: norm_cells(ev) for ev in STIM}


def cls_change(ev):
    """per neuron and day: UP / DOWN class after the injection minus the class in the same day's baseline (-1, 0, +1)"""
    e = cc[cc.event == ev]
    out = {}
    for dr in ("vehicle", "SBI-553"):
        j = pairs(e[e.drug == dr]).reset_index()
        for col in ("UP", "DOWN"):
            j["d" + col] = j[col + "_a"].astype(int) - j[col + "_b"].astype(int)
        out[dr] = j
    return out


CLS = {ev: cls_change(ev) for ev in STIM}


def fig_single_norm(fn):
    """10-26 (user: 'I do more trust bar graph ... normalized by baseline'): bars of the change from the same day's baseline, vehicle day
    vs SBI-553 day. Top: each neuron's response change (z). Bottom: change in % UP / % DOWN neurons (percentage points)."""
    fig, ax = plt.subplots(2, 4, figsize=(12.3, 4.75), gridspec_kw=dict(wspace=0.16, hspace=0.62, left=0.075, right=0.995, top=0.80,
                                                                         bottom=0.09))
    S = {}
    for j, ev in enumerate(STIM):
        out, t, T = NORM[ev]
        a = ax[0, j]
        vals = []
        for x, dr in ((0, "vehicle"), (1, "SBI-553")):
            d = out[dr]
            mu, se = d.chg.mean(), d.chg.std(ddof=1) / np.sqrt(len(d))
            a.bar(x, mu, yerr=se, color=CC["vi"] if dr == "vehicle" else CC["si"], width=0.6, capsize=3, error_kw=dict(lw=1), zorder=2)
            mm = {m: g.chg.mean() for m, g in d.groupby(d.mouse.astype(str))}
            for m, v in mm.items():
                a.scatter(x + OFF[m] * 0.9, v, s=26, marker=MK[m], color=MC[m], edgecolor="white", lw=0.6, zorder=4)
            vals += [mu + se, mu - se] + list(mm.values())
        hi, lo = max(vals), min(vals)
        span = hi - min(lo, 0)
        a.set_ylim(min(lo, 0) - 0.08 * span, hi + 0.75 * span)
        for x, k in ((0, "veh"), (1, "sbi")):
            a.text(x, hi + 0.10 * span, st(T[k]), ha="center", va="bottom", fontsize=10, color=CC["vi"] if k == "veh" else CC["si"],
                   fontweight="normal" if T[k] >= 0.05 else "bold")
        bracket(a, 0, 1, hi + 0.36 * span, st(T["between"]) + (f" ({pf(T['between'])})" if T["between"] < 0.05 else ""), fs=10,
                h=0.04 * span)
        a.axhline(0, color="0.4", lw=0.8)
        a.set_xticks([0, 1], ["vehicle day", "SBI-553 day"], fontsize=10)
        pre, a_, b_ = win(ev)
        a.set_title(f"{ev}  ({a_:g}-{b_:g} s" + (", original marks)" if ev in MARKS else ")") +
                    f"\nsame {T['n_trk']} neurons: {st(T['tracked'])}" + (f" ({pf(T['tracked'])})" if T["tracked"] < 0.05 else ""),
                    fontsize=11)
        # bottom: change in % UP and % DOWN
        b = ax[1, j]
        C = CLS[ev]
        vals = []
        Sx = {}
        for k, (col, dr) in enumerate((("UP", "vehicle"), ("UP", "SBI-553"), ("DOWN", "vehicle"), ("DOWN", "SBI-553"))):
            d = C[dr]
            x = k + (0.5 if col == "DOWN" else 0)
            v = 100 * d["d" + col].mean()
            b.bar(x, v, color=CC["vi"] if dr == "vehicle" else CC["si"], width=0.8, hatch=None if col == "UP" else "///", lw=0, zorder=2)
            mm = {m: 100 * g["d" + col].mean() for m, g in d.groupby(d.mouse.astype(str))}
            for m, w in mm.items():
                b.scatter(x + OFF[m] * 0.9, w, s=22, marker=MK[m], color=MC[m], edgecolor="white", lw=0.6, zorder=4)
            p_w = SG[ev][1][col + ("_veh" if dr == "vehicle" else "_sbi")]          # McNemar, same neurons
            Sx[(col, dr)] = dict(v=v, p=p_w, x=x)
            vals += [v] + list(mm.values())
        hi, lo = max(vals + [0]), min(vals + [0])
        span = hi - lo
        b.set_ylim(lo - 0.08 * span, hi + 0.75 * span)
        for (col, dr), r in Sx.items():
            b.text(r["x"], hi + 0.08 * span, st(r["p"]), ha="center", va="bottom", fontsize=9.5,
                   color=CC["vi"] if dr == "vehicle" else CC["si"], fontweight="normal" if r["p"] >= 0.05 else "bold")
        for col, x0 in (("UP", 0), ("DOWN", 2.5)):
            pb = mannwhitneyu(C["SBI-553"]["d" + col], C["vehicle"]["d" + col])[1]
            Sx[(col, "between")] = pb
            bracket(b, x0, x0 + 1, hi + 0.34 * span, st(pb), fs=9.5, h=0.04 * span)
        b.axhline(0, color="0.4", lw=0.8)
        b.set_xticks([0.5, 3.0], ["% UP", "% DOWN"], fontsize=10.5)
        S[ev] = Sx
    ax[0, 0].set_ylabel("response change\n(z, after - baseline)", fontsize=10.5)
    ax[1, 0].set_ylabel("change in % neurons\n(after - baseline)", fontsize=10.5)
    h = [Patch(color=CC["vi"], label="vehicle day"), Patch(color=CC["si"], label="SBI-553 day"),
         Patch(facecolor="0.55", hatch="///", edgecolor="white", label="hatched = DOWN")] + mouse_handles()
    fig.legend(handles=h, ncol=7, loc="upper center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=9.5, handlelength=1.2,
               columnspacing=1.0)
    fig.savefig(fn); plt.close(fig)
    return S


# ------------------------------------------------------------------------------------------------ 3c population AUC (10-26, user: 'we need AUC')
PSTH = np.load(B + r"\analysis\new_decks\psth_cache_plainz_mix4.npz", allow_pickle=True)["D"].item()   # plain z (user 10-26)
def base_sub(Y, tt, ev):
    """10-26 (user): traces minus each neuron's mean z in the test's own baseline window (0.5 s before; 1 s for reflexes / movement)"""
    pre, a, b = win(ev)
    a0 = min(a, 0)
    m = (tt >= a0 - pre - 1e-9) & (tt < a0 - 1e-9)
    return Y - np.nanmean(Y[m], 0, keepdims=True)


CND = [("vb", CC["vb"], "baseline, vehicle day"), ("vi", CC["vi"], "after vehicle"), ("sb", CC["sb"], "baseline, SBI-553 day"),
       ("si", CC["si"], "after SBI-553")]


def auc_cells(ev):
    """per neuron-session: AUC (z*s) of its trial-mean trace in [-w, 0) and in [0, w), w = the event's response-window length"""
    pre, a, b = win(ev)
    w = b - a
    out = {c: dict(pre=[], post=[], tr=[], trb=[], auc=[]) for c, _, _ in CND}
    byid = {}
    for (m, d, ph, e), v in PSTH.items():
        if e != ev:
            continue
        c = ("s" if m in SBI[d] else "v") + ("b" if ph == "pre" else "i")
        tt, Y = v["tt"], v["mean"]
        i0, i1, i2 = (np.searchsorted(tt, x - 1e-9) for x in (-w, 0.0, w))
        a_pre = np.trapezoid(Y[i0:i1 + 1], tt[i0:i1 + 1], axis=0)
        a_post = np.trapezoid(Y[i1:i2 + 1], tt[i1:i2 + 1], axis=0)
        out[c]["pre"] += list(a_pre)
        out[c]["post"] += list(a_post)
        out[c]["tr"].append(Y)
        Yb = base_sub(Y, tt, ev)                               # 10-26 (user): minus the test's own baseline window (0.5 s before)
        k0, k1 = np.searchsorted(tt, a - 1e-9), np.searchsorted(tt, b - 1e-9)
        auc = np.trapezoid(Yb[k0:k1 + 1], tt[k0:k1 + 1], axis=0)   # AUC of the response window = R x window length
        out[c]["trb"].append(Yb)
        out[c]["auc"] += list(auc)
        byid[(m, d, ph)] = pd.Series(auc, index=v["ids"])
    for dr, k in (("veh", "v"), ("sbi", "s")):                 # within day, same neurons: baseline session vs injection session
        xs, ys = [], []
        for (m, d, ph), sr in byid.items():
            if ph == "pre" and (m in SBI[d]) == (k == "s") and (m, d, "post") in byid:
                j = pd.concat([sr, byid[(m, d, "post")]], axis=1, join="inner").dropna()
                xs += list(j.iloc[:, 0]); ys += list(j.iloc[:, 1])
        out["p_within_" + dr] = wilcoxon(ys, xs)[1] if len(xs) > 5 else np.nan
    for c, _, _ in CND:
        out[c] = dict(pre=np.array(out[c]["pre"]), post=np.array(out[c]["post"]), tr=np.hstack(out[c]["tr"]), tt=tt,
                      trb=np.hstack(out[c]["trb"]), auc=np.array(out[c]["auc"]))
        out[c]["p_prepost"] = wilcoxon(out[c]["post"], out[c]["pre"])[1]       # the kept pre / post bar version
        out[c]["p"] = wilcoxon(out[c]["auc"])[1]                                # AUC vs 0
    out["p_after"] = mannwhitneyu(out["si"]["auc"], out["vi"]["auc"])[1]
    out["p_after_prepost"] = mannwhitneyu(out["si"]["post"] - out["si"]["pre"], out["vi"]["post"] - out["vi"]["pre"])[1]
    out["w"] = w
    return out


AUC = {ev: auc_cells(ev) for ev in STIM}


def fig_auc(fn):
    fig, ax = plt.subplots(2, 4, figsize=(12.3, 4.75), gridspec_kw=dict(wspace=0.30, hspace=0.55, left=0.085, right=0.99, top=0.84,
                                                                         bottom=0.10, height_ratios=[1.1, 1.0]))
    for j, ev in enumerate(STIM):
        R = AUC[ev]
        a = ax[0, j]
        hi = []
        for k, (c, col, lab) in enumerate(CND):
            for s_, key in ((0, "pre"), (1, "post")):
                x = 3 * k + s_
                v = R[c][key]
                mu, se = v.mean(), v.std(ddof=1) / np.sqrt(len(v))
                a.bar(x, mu, yerr=se, color=col, alpha=0.35 if key == "pre" else 1.0, width=0.8, capsize=2.5, error_kw=dict(lw=0.9),
                      zorder=2)
                hi.append(mu + se); hi.append(mu - se)
        top, lo = max(hi), min(hi + [0])
        span = top - lo
        a.set_ylim(lo - 0.1 * span, top + 0.75 * span)
        for k, (c, col, lab) in enumerate(CND):
            p = R[c]["p_prepost"]
            bracket(a, 3 * k, 3 * k + 1, top + 0.12 * span, st(p), color=col if c != "vb" else "0.45", fs=9.5, h=0.03 * span)
        a.axhline(0, color="0.4", lw=0.8)
        a.set_xticks([0, 1, 3, 4, 6, 7, 9, 10], ["pre", "post"] * 4, fontsize=8)
        for k, (c, col, lab) in enumerate(CND):
            a.text(3 * k + 0.5, -0.30, ["base", "vehicle", "base", "SBI-553"][k], transform=a.get_xaxis_transform(), ha="center", va="top",
                   fontsize=8.5, color=col if c != "vb" else "0.45")
        a.set_title(f"{ev}  (±{R['w']:g} s)\nvehicle vs SBI-553: {st(R['p_after_prepost'])}"
                    + (f" {pf(R['p_after_prepost'])}" if R["p_after_prepost"] < 0.05 else ""), fontsize=10.5)
        b = ax[1, j]
        tt = R["vb"]["tt"]
        sel = (tt >= -3) & (tt <= 6)
        for c, col, lab in CND:
            Y = R[c]["tr"][sel]
            mu, se = Y.mean(1), Y.std(1, ddof=1) / np.sqrt(Y.shape[1])
            b.fill_between(tt[sel], mu - se, mu + se, color=col, alpha=0.2, lw=0)
            b.plot(tt[sel], mu, color=col, lw=1.3)
        b.axvspan(-R["w"], 0, color="#d6eaf8", lw=0, zorder=0); b.axvspan(0, R["w"], color="#e5e7e9", lw=0, zorder=0)
        b.axvline(0, color="k", lw=0.8, ls="--"); b.axhline(0, color="0.5", lw=0.7)
        b.set_xlabel("time from event (s)", fontsize=10); b.tick_params(labelsize=9)
    ax[0, 0].set_ylabel("AUC of z (z·s)\nper neuron, mean ± SEM", fontsize=10)
    ax[1, 0].set_ylabel("z (dF/F / robust SD)\nmean ± SEM", fontsize=10)
    h = [Patch(color=col, label=f"{lab} ({len(AUC['Pin prick'][c]['pre'])})") for c, col, lab in CND] + \
        [Patch(color="0.6", alpha=0.35, label="pre = window before"), Patch(color="0.6", label="post = window after")]
    fig.legend(handles=h, ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=9, handlelength=1.1,
               columnspacing=0.9)
    fig.savefig(fn); plt.close(fig)


def fig_auc_delta(fn):
    """10-26 (user: traces in plain z, AUC as bars): top = AUC after the event minus AUC before (equal windows) from the plain-z traces
    below; bottom = plain trial-mean z, mean +- SEM over neurons."""
    fig, ax = plt.subplots(2, 4, figsize=(12.3, 4.75), gridspec_kw=dict(wspace=0.30, hspace=0.55, left=0.085, right=0.99, top=0.84,
                                                                         bottom=0.10, height_ratios=[1.1, 1.0]))
    for j, ev in enumerate(STIM):
        R = AUC[ev]
        a = ax[0, j]
        hi = []
        for k, (c, col, lab) in enumerate(CND):
            dlt = R[c]["auc"]
            mu, se = dlt.mean(), dlt.std(ddof=1) / np.sqrt(len(dlt))
            a.bar(k, mu, yerr=se, color=col, width=0.72, capsize=3, error_kw=dict(lw=1), zorder=2)
            hi += [mu + se, mu - se]
        top, lo = max(hi + [0]), min(hi + [0])
        span = top - lo
        a.set_ylim(lo - 0.12 * span, top + 1.0 * span)
        for k, (c, col, lab) in enumerate(CND):
            a.text(k, top + 0.06 * span, st(R[c]["p"]), ha="center", va="bottom", fontsize=9.5, color=col if c != "vb" else "0.45",
                   fontweight="bold" if R[c]["p"] < 0.05 else "normal")
        bracket(a, 0, 1, top + 0.32 * span, st(R["p_within_veh"]), color=CC["vi"], fs=9.5, h=0.04 * span)
        bracket(a, 2, 3, top + 0.32 * span, st(R["p_within_sbi"]), color=CC["si"], fs=9.5, h=0.04 * span)
        bracket(a, 1, 3, top + 0.62 * span, st(R["p_after"]) + (f" ({pf(R['p_after'])})" if R["p_after"] < 0.05 else ""), fs=9.5,
                h=0.04 * span)
        a.axhline(0, color="0.4", lw=0.8)
        a.set_xticks(range(4), ["baseline", "after", "baseline", "after"], fontsize=9)
        day_labels(a)
        a.set_title(f"{ev}  ({win(ev)[1]:g}-{win(ev)[2]:g} s)", fontsize=11)
        b = ax[1, j]
        tt = R["vb"]["tt"]
        sel = (tt >= -3) & (tt <= 6)
        for c, col, lab in CND:
            Y = R[c]["trb"][sel]
            mu, se = Y.mean(1), Y.std(1, ddof=1) / np.sqrt(Y.shape[1])
            b.fill_between(tt[sel], mu - se, mu + se, color=col, alpha=0.2, lw=0)
            b.plot(tt[sel], mu, color=col, lw=1.3)
        pre_, a_, b_ = win(ev)
        b.axvspan(min(a_, 0) - pre_, min(a_, 0), color="#d6eaf8", lw=0, zorder=0); b.axvspan(a_, b_, color="#e5e7e9", lw=0, zorder=0)
        b.axhline(0, color="0.5", lw=0.7)
        b.axvline(0, color="k", lw=0.8, ls="--")
        b.set_xlabel("time from event (s)", fontsize=10); b.tick_params(labelsize=9)
    ax[0, 0].set_ylabel("AUC of the response\nwindow (z·s), mean ± SEM", fontsize=10)
    ax[1, 0].set_ylabel("z - mean of the\n0.5 s before", fontsize=10)
    h = [Patch(color=col, label=f"{lab} ({len(AUC['Pin prick'][c]['pre'])})") for c, col, lab in CND] + \
        [Patch(color="#d6eaf8", label="baseline (0.5 s before)"), Patch(color="#e5e7e9", label="response window = AUC")]
    fig.legend(handles=h, ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=9, handlelength=1.1,
               columnspacing=0.9)
    fig.savefig(fn); plt.close(fig)


# ------------------------------------------------------------------------------------------------ 4 single neurons
def fig_single(fn, evs=STIM):
    fig, ax = plt.subplots(2, len(evs), figsize=(12.3, 4.75), sharey="row",
                           gridspec_kw=dict(wspace=0.12, hspace=0.30, left=0.075, right=0.995, top=0.80, bottom=0.12))

    def hi_of(R, col, cs):
        v = [R[c][col] for c in cs if R[c]["n"]] + [x for c in cs if R[c]["n"] for x in R[c]["m" + col].values() if np.isfinite(x)]
        return max(v) if v else 0.0
    ymax = {col: max(hi_of(SG[ev][0], col, ("vb", "vi", "sb", "si")) for ev in evs) for col in ("UP", "DOWN")}
    for j, ev in enumerate(evs):
        R, T = SG[ev]
        for i, col in enumerate(("UP", "DOWN")):
            a = ax[i, j]
            pad = 0.075 * ymax[col]
            for k, c in enumerate(("vb", "vi", "sb", "si")):
                a.bar(k, R[c][col], color=CC[c], width=0.75, zorder=2, hatch=None if col == "UP" else "///", lw=0)
                mice_dots(a, k, R[c]["m" + col], s=22)
            a.set_ylim(0, ymax[col] + 4.3 * pad)
            y1, y2 = hi_of(R, col, ("vb", "vi")) + 0.6 * pad, hi_of(R, col, ("sb", "si")) + 0.6 * pad
            bracket(a, 0, 1, y1, st(T[col + "_veh"]), color=CC["vi"], fs=11, h=0.25 * pad)
            bracket(a, 2, 3, y2, st(T[col + "_sbi"]), color=CC["si"], fs=11, h=0.25 * pad)
            if col + "_after" in T:
                bracket(a, 1, 3, max(y1, y2) + 2.2 * pad, st(T[col + "_after"]), color="k", fs=11, h=0.25 * pad)
            a.set_xticks(range(4), ["baseline", "after", "baseline", "after"] if i == 1 else [""] * 4, fontsize=10)
            if i == 1:
                day_labels(a)
            if i == 0:
                pre, a_, b_ = win(ev)
                a.set_title(f"{ev}  ({a_:g}-{b_:g} s" + (", original marks)" if ev in MARKS else ")"), fontsize=12)
        ax[0, 0].set_ylabel("% UP neurons")
        ax[1, 0].set_ylabel("% DOWN neurons")
    legend2(fig, hatch=True)
    fig.savefig(fn); plt.close(fig)


# ------------------------------------------------------------------------------------------------ 5 robustness grid
VNAME = ["1 new marks,\nwindow per event", "2 original marks,\nwindow per event", "3 THIS DECK:\nmarks + window\nper event",
         "4 new marks,\nall 0-0.5 s", "5 original marks,\n0-1 / 0-0.5 s", "THIS DECK,\neach mouse\nleft out"]
SPEC = [
    ("Pin: fewer UP neurons after SBI-553 (Fisher)", "UP: after veh vs after SBI", "Pin prick", -1),
    ("Pin: UP neurons rise after vehicle (McNemar)", "UP: baseline -> vehicle", "Pin prick", +1),
    ("Light: more DOWN neurons after SBI-553 (Fisher)", "DOWN: after veh vs after SBI", "Light touch", +1),
    ("Mild: more DOWN neurons after SBI-553 (Fisher)", "DOWN: after veh vs after SBI", "Mild touch", +1),
    ("Light: UP neurons rise after SBI-553 (McNemar)", "UP: baseline -> SBI-553", "Light touch", +1),
    ("Heat: UP neurons rise after both injections (McNemar)", ("UP: baseline -> vehicle", "UP: baseline -> SBI-553"), "Heat", +1),
    ("Movement onset: UP rise after both injections (McNemar)", ("UP: baseline -> vehicle", "UP: baseline -> SBI-553"),
     "movement onset", +1),
    ("Light, mild: more UP neurons after SBI-553 (Fisher)", ("UP: after veh vs after SBI",), ("Light touch", "Mild touch"), +1),
    ("Heat: fewer UP neurons after SBI-553 (Fisher)", "UP: after veh vs after SBI", "Heat", -1),
    ("Withdrawal, flinch: more DOWN after SBI-553 (Fisher)", ("DOWN: after veh vs after SBI",), ("withdrawal", "flinch"), +1),
    ("Pin: smaller mean response after SBI-553 (Mann-Whitney)", "mean response: after veh vs after SBI", "Pin prick", -1),
    ("Light, mild, heat: mean response unchanged (Mann-Whitney)", ("mean response: after veh vs after SBI",),
     ("Light touch", "Mild touch", "Heat"), 0),
]
BSPEC = [("Behaviour: fewer pin flinches after SBI-553 (CMH)", "flinch", "Pin prick"),
         ("Behaviour: less light-touch movement after SBI-553 (CMH)", "movement", "Light touch"),
         ("Behaviour: less heat escape / rearing after SBI-553 (CMH)", "escape/rearing", "Heat")]


def test_on(d, test, ev):
    """one test of the grid on a cells table d: (p, direction +1 / -1)"""
    e = d[d.event == ev]
    if test.startswith("mean response"):
        v, s_ = e[e.cond == "vi"].resp.dropna(), e[e.cond == "si"].resp.dropna()
        return mannwhitneyu(s_, v)[1], (1 if s_.mean() > v.mean() else -1)
    col = test.split(":")[0]
    if "after veh vs after SBI" in test:
        v, s_ = e[e.cond == "vi"], e[e.cond == "si"]
        return fisher_exact([[v[col].sum(), len(v) - v[col].sum()], [s_[col].sum(), len(s_) - s_[col].sum()]])[1], \
            (1 if s_[col].mean() > v[col].mean() else -1)
    dr = "vehicle" if test.endswith("vehicle") else "SBI-553"
    p, b01, b10 = mcnemar(e[e.drug == dr], col)
    return p, (1 if b01 > b10 else -1)


def cell(ps, want):
    if want == 0:
        ok = all(p >= 0.05 for p, _ in ps)
        return ("ns (all)", "ns") if ok else ("sig", "opp")
    good = all(p < 0.05 and d == want for p, d in ps)
    opp = any(p < 0.05 and d != want for p, d in ps)
    return (st(max(p for p, _ in ps)) if good else ("opposite" if opp else "ns"), "yes" if good else ("opp" if opp else "ns"))


def robust_rows():
    E = pd.read_csv(SCR + r"\eval5_tests.csv")
    rows = []
    for lab, test, ev, want in SPEC:
        tests = test if isinstance(test, tuple) else (test,)
        evs = ev if isinstance(ev, tuple) else (ev,)
        cells = []
        for v in ("V1", "V2", "V3", "V4", "V5"):
            ps = []
            for t in tests:
                for e in evs:
                    r = E[(E.v == v) & (E.test == t) & (E.event == e)].iloc[0]
                    ps.append((r.p, 1 if r.b > r.a else -1))
            cells.append(cell(ps, want))
        lomo = []                                           # this deck, each mouse left out
        for m in MICE:
            dm = cc[cc.mouse != m]
            lomo.append(cell([test_on(dm, t, e) for t in tests for e in evs], want))
        k = sum(kind == ("ns" if want == 0 else "yes") for _, kind in lomo)
        fail = [m for m, (_, kind) in zip(MICE, lomo) if kind != ("ns" if want == 0 else "yes")]
        cells.append((f"{k}/4" + ("" if not fail else f" (needs {', '.join(fail)})"), "yes" if k == 4 else ("part" if k else "ns")))
        rows.append((lab, cells))
    BV = {"V1": A.replace("mix4", "mix_new"), "V2": A.replace("mix4", "mix_old"), "V3": A, "V4": A.replace("mix4", "timefix_w05"),
          "V5": B + r"\analysis\event_analysis_before_timefix_20261023"}
    for lab, beh, stim in BSPEC:
        cells = []
        for v in ("V1", "V2", "V3", "V4", "V5"):
            f = BV[v] + (r"\P6_behaviour_tests.csv" if v == "V5" else r"\event_analysis\P6_behaviour_tests.csv")
            t = pd.read_csv(f)
            r = t[(t.behaviour == beh) & (t.stim == stim) & t.test.str.startswith("SBI-553 vs vehicle")].iloc[0]
            good = r.p < 0.05 and r.sbi_pct < r.veh_pct
            cells.append((st(r.p) if good else "ns", "yes" if good else "ns"))
        lomo = []
        for m in MICE:
            tabs = []
            for mm in MICE:
                if mm == m:
                    continue
                V = PD[(PD.mouse == mm) & (PD.stim == stim) & (PD.cond == "vi")][beh]
                Sx = PD[(PD.mouse == mm) & (PD.stim == stim) & (PD.cond == "si")][beh]
                tabs.append([[Sx.sum(), (~Sx).sum()], [V.sum(), (~V).sum()]])
            p = cmh_p(tabs)
            rest = PD[(PD.mouse != m) & (PD.stim == stim)]
            lomo.append(p < 0.05 and rest[rest.cond == "si"][beh].mean() < rest[rest.cond == "vi"][beh].mean())   # and SBI-553 lower
        k = sum(lomo); fail = [m for m, ok in zip(MICE, lomo) if not ok]
        cells.append((f"{k}/4" + ("" if not fail else f" (needs {', '.join(fail)})"), "yes" if k == 4 else ("part" if k else "ns")))
        rows.append((lab, cells))
    return rows


def fig_robust(fn):
    rows = robust_rows()
    n = len(rows)
    fig = plt.figure(figsize=(12.3, 4.85))
    a = fig.add_axes([0.0, 0.0, 1.0, 1.0]); a.set_axis_off()
    a.set_xlim(0, 12.3); a.set_ylim(0, 4.85)
    x0, rh, y0 = 4.3, 0.226, 4.25
    cw = [1.1, 1.1, 1.3, 1.1, 1.1, 2.05]
    xs = [x0 + sum(cw[:j]) for j in range(len(cw))]
    COL = {"yes": "#c8e6c9", "ns": "#f2f2f2", "opp": "#ffe0b2", "part": "#fff9c4"}
    for j, nm in enumerate(VNAME):
        a.text(xs[j] + cw[j] / 2, y0 + 0.02, nm, ha="center", va="bottom", fontsize=9.5,
               fontweight="bold" if j in (2, 5) else "normal", color="#c0392b" if j == 2 else "k")
    a.plot([xs[5] - 0.04, xs[5] - 0.04], [y0 - n * rh - 0.02, y0 + 0.45], color="0.6", lw=1)
    robust = []
    for i, (lab, cells) in enumerate(rows):
        y = y0 - (i + 1) * rh
        allyes = all(k == "yes" for _, k in cells[:5]) or all(t == "ns (all)" for t, _ in cells[:5])
        robust.append(allyes)
        a.text(0.05, y + rh / 2, lab, ha="left", va="center", fontsize=9.8, fontweight="bold" if allyes else "normal")
        for j, (txt, kind) in enumerate(cells):
            a.add_patch(plt.Rectangle((xs[j] + 0.03, y + 0.02), cw[j] - 0.06, rh - 0.04, color=COL[kind], lw=0))
            a.text(xs[j] + cw[j] / 2, y + rh / 2, txt, ha="center", va="center", fontsize=9.5 if j < 5 else 9)
    yb = y0 - (n + 1) * rh - 0.02
    items = [("yes", "significant in the stated direction (stars = weakest p)"), ("ns", "not significant"),
             ("opp", "opposite direction"), ("part", "holds in some subsets")]
    xx = 0.05
    for kind, txt in items:
        a.add_patch(plt.Rectangle((xx, yb + 0.02), 0.3, 0.16, color=COL[kind], lw=0))
        a.text(xx + 0.38, yb + 0.10, txt, va="center", fontsize=9.8)
        xx += 0.55 + 0.072 * len(txt)
    for k, line in enumerate(("Bold row = same result in all five analysis versions (robust to the choice of marks and window; not a replication).",
                              "'after SBI-553' = compared with after vehicle (other day); 'rise' = compared with the same day's baseline "
                              "(same neurons).",
                              "Last column: the test repeated without each mouse in turn; 'needs' = the mice whose removal loses the result. "
                              + STARS)):
        a.text(0.05, yb - 0.19 - 0.17 * k, line, fontsize=9.6, color="0.25", va="center")
    fig.savefig(fn); plt.close(fig)
    return rows, robust


# ------------------------------------------------------------------------------------------------ supplements
def fig_design(fn):
    fig = plt.figure(figsize=(12.3, 2.1)); a = fig.add_axes([0, 0, 1, 1]); a.set_axis_off(); a.set_xlim(0, 12.3); a.set_ylim(0, 2.1)
    a.text(2.6, 1.86, "Day 1", ha="center", fontsize=13, fontweight="bold"); a.text(7.4, 1.86, "Day 2", ha="center", fontsize=13, fontweight="bold")
    for i, m in enumerate(MICE):
        y = 1.38 - i * 0.38
        a.text(0.05, y + 0.13, f"{m} ({SEX[m]})", fontsize=12, color=MC[m], fontweight="bold", va="center")
        for d, x0 in ((1, 1.3), (2, 6.1)):
            drug = "SBI-553" if m in SBI[d] else "vehicle"
            a.add_patch(plt.Rectangle((x0, y), 1.25, 0.26, color=CC["vb"] if drug == "vehicle" else CC["sb"], lw=0))
            a.text(x0 + 0.625, y + 0.13, "baseline", ha="center", va="center", fontsize=11, color="black" if drug == "vehicle" else "white")
            a.add_patch(plt.Rectangle((x0 + 1.35, y), 1.25, 0.26, color=CC["vi"] if drug == "vehicle" else CC["si"], lw=0))
            a.text(x0 + 1.975, y + 0.13, drug, ha="center", va="center", fontsize=11, color="white", fontweight="bold")
    a.text(9.2, 1.0, "each session: 4 stimulus blocks\n(light, mild, pin prick, heat),\norder randomised; within a day the\n"
           "same field of view and neurons", ha="left", va="center", fontsize=11, color="0.25")
    fig.savefig(fn); plt.close(fig)


def fig_tests_table(fn):
    rows = [("Q1 behaviour (slide 1)", f"4 mice; {N_DEL} deliveries per session", "% of deliveries followed by the behaviour (reflex 0-2 s, held 0-3 s)",
             "CMH over deliveries, mouse = stratum; mouse-level Wilcoxon (n = 4: smallest possible p = 0.125)"),
            ("Q2 firing (slide 2)", f"4 mice; {N_RATE} neurons per mouse-day; {N_TR} tracked", "Ca transients / min of still frames in the 4 stimulus blocks",
             "per mouse-day Wilcoxon over neurons; tracked: Wilcoxon on the same neurons"),
            ("Q3 population (slide 3)", f"neurons: vehicle day {N_VEH}, SBI-553 day {N_SBI}", "R = trial mean of (response-window z - baseline-window z)",
             "within day: Wilcoxon (same neurons); after vs after, baseline vs baseline: Mann-Whitney"),
            ("Q3 tracked (notes 3)", f"{N_TR} neurons found on both days", "same R, each neuron with itself", "Wilcoxon paired by neuron"),
            ("Q3 single neurons (4, S4)", f"neurons: vehicle day {N_VEH}, SBI-553 day {N_SBI}", "UP / DOWN = time-shift test, BH q ≤ 0.20",
             "within day: McNemar (same neurons); after vs after: Fisher exact"),
            ("Encoding model (S5)", f"{len(ENC)} neuron-sessions", "ridge regression, cross-validated unique variance", "100 shuffles per neuron, BH q ≤ 0.20"),
            ("Summary grid (slide 5)", "same data, 5 analysis versions", "the tests of slides 1, 3, 4", "+ this deck with each mouse left out once"),
            ("All neural p", "4 mice", "neurons pooled over mice", "exploratory, uncorrected; a rank test over 4 mice cannot reach p < 0.05")]
    fig = plt.figure(figsize=(12.3, 4.7)); a = fig.add_axes([0, 0, 1, 1]); a.set_axis_off(); a.set_xlim(0, 12.3); a.set_ylim(0, 4.7)
    xs = [0.05, 2.25, 5.0, 8.3]; hd = ["analysis", "n", "measure", "test"]; wd = {0.05: 26, 2.25: 33, 5.0: 44, 8.3: 52}
    for x, h in zip(xs, hd):
        a.text(x, 4.5, h, fontsize=12, fontweight="bold")
    a.plot([0.05, 12.25], [4.43, 4.43], color="0.3", lw=1)
    for i, r in enumerate(rows):
        y = 4.32 - i * 0.50
        for x, t in zip(xs, r):
            a.text(x, y, "\n".join(textwrap.wrap(t, wd[x])), fontsize=10.5, va="top")
        a.plot([0.05, 12.25], [y - 0.44, y - 0.44], color="0.85", lw=0.8)
    a.text(0.05, 0.12, STARS, fontsize=10.5, color="0.3")
    fig.savefig(fn); plt.close(fig)


def fig_permouse(fn):
    fig, ax = plt.subplots(1, 4, figsize=(12.3, 4.6), gridspec_kw=dict(wspace=0.33, left=0.06, right=0.99, top=0.78, bottom=0.21))
    specs = [("Pin prick", "UP", "% pin-prick UP neurons"), ("Heat", "UP", "% heat UP neurons"),
             ("Light touch", "UP", "% light-touch UP neurons"), (None, "movement", "% deliveries with movement")]
    for a, (ev, col, lab) in zip(ax, specs):
        for m in MICE:
            ys = []
            for c in ("vb", "vi", "sb", "si"):
                if ev is None:
                    x = PD[(PD.cond == c) & (PD.mouse == m)]
                    ys.append(100 * x.movement.mean() if len(x) else np.nan)
                else:
                    ys.append(SG[ev][0][c]["m" + col].get(m, np.nan))
            dx = OFF[m] * 0.35
            a.plot([0 + dx, 1 + dx], ys[:2], color=MC[m], marker=MK[m], markersize=7, markeredgecolor="white", lw=1.5, clip_on=False, zorder=4)
            a.plot([2 + dx, 3 + dx], ys[2:], color=MC[m], marker=MK[m], markersize=7, markeredgecolor="white", lw=1.5, clip_on=False, zorder=4)
        a.set_xticks(range(4), ["baseline", "after", "baseline", "after"], fontsize=10); a.set_xlim(-0.4, 3.4); a.set_ylim(bottom=0)
        a.set_title(lab, fontsize=12)
        day_labels(a)
    ax[0].set_ylabel("% (each mouse)")
    fig.legend(handles=mouse_handles(), ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=10.5)
    fig.savefig(fn); plt.close(fig)


ENC_ORDER = ["movement only", "stimulus", "stimulus + reflex", "stimulus or reflex (cannot separate)", "reflex", "none"]
ENC_COL = {"movement only": "#16a085", "stimulus": "#c0392b", "stimulus + reflex": "#e67e22",
           "stimulus or reflex (cannot separate)": "#f5b041", "reflex": "#8e44ad", "none": "#d5d8dc"}


def fig_encoding(fn):
    enc = ENC.groupby("condition").type.value_counts(normalize=True).mul(100).unstack().fillna(0)
    nn = ENC.groupby("condition").size()
    order = ["baseline", "vehicle", "SBI-553"]
    fig, a = plt.subplots(figsize=(12.3, 4.4), gridspec_kw=dict(left=0.16, right=0.70, top=0.95, bottom=0.14))
    for i, c in enumerate(order):
        left = 0
        for t in ENC_ORDER:
            v = enc.loc[c].get(t, 0)
            if v <= 0:
                continue
            a.barh(i, v, left=left, color=ENC_COL[t], height=0.62, edgecolor="white", lw=0.8)
            if v >= 4:
                a.text(left + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=10.5, color="white" if t != "none" else "0.25")
            left += v
    lab = {"baseline": "baselines (both days)", "vehicle": "after vehicle", "SBI-553": "after SBI-553"}
    a.set_yticks(range(3), [f"{lab[c]}\n(n = {nn[c]} neuron-sessions)" for c in order]); a.set_ylim(2.5, -0.5)
    a.set_xlim(0, 100); a.set_xlabel("% of neurons, by what explains their activity (unique variance, encoding model)")
    a.legend(handles=[Patch(color=ENC_COL[t], label=t) for t in ENC_ORDER], loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False,
             fontsize=11)
    fig.savefig(fn); plt.close(fig)
    return enc, nn


# ------------------------------------------------------------------------------------------------ deck
prs = Presentation(); prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
BL = prs.slide_layouts[6]


def tb(s, x, y, w, h, text, size, bold=False, color=(0x22, 0x22, 0x22), align=None, italic=False, bold_heads=()):
    t = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = t.text_frame; tf.word_wrap = True
    for i, line in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = p.add_run(); r.text = line
        r.font.size = Pt(size); r.font.bold = bold or (line in bold_heads); r.font.italic = italic; r.font.name = "Arial"
        r.font.color.rgb = RGBColor(*color)
        if align:
            p.alignment = align
    return t


def fit_pt(text, pt, bold, width, lo):
    """largest font size in [lo, pt] that keeps the text on one line of the box (else warn: shorten it)"""
    prop = FontProperties(family="Arial", weight="bold" if bold else "normal")
    while pt > lo and TextPath((0, 0), text, size=pt, prop=prop).get_extents().width / 72 > width:
        pt -= 0.5
    if TextPath((0, 0), text, size=pt, prop=prop).get_extents().width / 72 > width:
        print("TOO LONG:", text)
    return pt


def slide(section, title, key, fig, bottom, notes, num, footer):
    s = prs.slides.add_slide(BL)
    tb(s, 0.45, 0.10, 8, 0.3, section, 11, color=(0x55, 0x55, 0x55))
    tb(s, 0.45, 0.32, 12.4, 0.6, title, fit_pt(title, 24, False, 12.15, 21))
    if key:
        tb(s, 0.45, 0.92, 12.4, 0.5, key, fit_pt(key, 14.5, True, 12.15, 13), bold=True, color=(0xC0, 0x39, 0x2B))
    if fig:
        s.shapes.add_picture(fig, Inches(0.5), Inches(1.45), Inches(12.3))
    if bottom:
        tb(s, 0.45, 6.38, 12.4, 0.65, bottom, 12)
    tb(s, 0.45, 7.08, 11.3, 0.3, footer, 9.5, color=(0x88, 0x88, 0x88), italic=True)
    tb(s, 12.3, 7.08, 0.8, 0.3, str(num), 10, color=(0x88, 0x88, 0x88), align=PP_ALIGN.RIGHT)
    s.notes_slide.notes_text_frame.text = notes
    return s


def notes(**kw):
    """structured notes: one line per field"""
    return "\n".join(f"{k.replace('_', ' ').upper()}: {v}" for k, v in kw.items() if v)


def fmt_single(ev):
    R, T = SG[ev]
    return (f"{ev} ({wtxt(ev)}{', original marks' if ev in MARKS else ''}): UP vehicle day {R['vb']['UP']:.0f} → {R['vi']['UP']:.0f}% "
            f"(McNemar {pf(T['UP_veh'])}), SBI-553 day {R['sb']['UP']:.0f} → {R['si']['UP']:.0f}% ({pf(T['UP_sbi'])}), after vs after "
            f"Fisher {pf(T.get('UP_after', np.nan))}; DOWN vehicle day {R['vb']['DOWN']:.0f} → {R['vi']['DOWN']:.0f}% ({pf(T['DOWN_veh'])}), "
            f"SBI-553 day {R['sb']['DOWN']:.0f} → {R['si']['DOWN']:.0f}% ({pf(T['DOWN_sbi'])}), after vs after {pf(T.get('DOWN_after', np.nan))}")


def fmt_pop(ev):
    R, T = PO[ev]
    s_ = (f"{ev}: mean R baseline (vehicle day) {R['vb']['mean']:+.3f}, after vehicle {R['vi']['mean']:+.3f}, baseline (SBI-553 day) "
          f"{R['sb']['mean']:+.3f}, after SBI-553 {R['si']['mean']:+.3f} z; Wilcoxon within day vehicle {pf(T['veh'])}, SBI-553 {pf(T['sbi'])}; "
          f"Mann-Whitney after vs after {pf(T['after'])}, baseline vs baseline {pf(T['base'])}")
    if ev in TR.index:
        tr = TR.loc[ev]
        s_ += (f"; tracked 185: after SBI-553 - after vehicle median {tr.med_after:+.3f} z ({pf(tr.p_after)}), baseline-normalized "
               f"{tr.med_norm:+.3f} ({pf(tr.p_norm)}), baseline vs baseline {pf(tr.p_vb_sb)}")
    return s_


# --- 1 behaviour
f1 = FIG + r"\s1_behaviour.png"; fig_behaviour(f1)
NB = NOTES["beh"]; I = {d["beh"]: d for d in NB["info"]}
pc1 = NB["pc"]
mv, fl = I["movement"], I["flinch"]
lowtxt = ("all 4 mice" if mv["lower"] == 4 else f"{mv['lower']}/4 mice each") if mv["lower"] == fl["lower"] else f"{mv['lower']}/4 and {fl['lower']}/4 mice"
slide("Q1  BEHAVIOUR", "After SBI-553: fewer light-touch movements and pin flinches than after vehicle",
      f"Lower in {lowtxt}, but not below the SBI-553-day baseline (ns); smaller reflex rise on the SBI-553 day came from higher baselines",
      f1, f"n = 4 mice; deliveries after vehicle / after SBI-553: light {mv['n']['vi']} / {mv['n']['si']}, pin "
          f"{fl['n']['vi']} / {fl['n']['si']}, heat {I['escape/rearing']['n']['vi']} / {I['escape/rearing']['n']['si']}. CMH (mouse = stratum): "
          f"blue / red = vs the same day's baseline, black = after vs after; {NB['nsig']} of {NB['ntest']} after-vs-after tests p < 0.05. "
          f"C: baselines higher on the SBI-553 day (CMH withdrawal {pf(pc1['withdrawal']['p_base'])}, flinch {pf(pc1['flinch']['p_base'])}), "
          f"after-injection levels equal ({pf(pc1['withdrawal']['p_post'])}, {pf(pc1['flinch']['p_post'])}). {STARS}",
      notes(measure="% of deliveries followed by the behaviour (reflexes: from the delivery to 2 s later or the next delivery; held / "
                    "automatic behaviours such as movement and escape: a bout starting 0-3 s after the delivery). Bars = mean of the 4 mice, "
                    "markers = mice.",
            n=f"4 mice; deliveries per mouse-session {N_DEL_STIM} per stimulus; {NB['nall']} behaviour x stimulus SBI-553 vs vehicle tests, "
              f"{NB['ntest']} testable (the rest had no events), {NB['nsig']} with p < 0.05 (uncorrected).",
            tests="CMH chi-square over the 2 x 2 tables of the 4 mice (continuity-corrected): blue = vehicle vs its baseline, red = "
                  "SBI-553 vs its baseline, black = after SBI-553 vs after vehicle; panel A: Wilcoxon over the 4 mice (smallest p 0.125). "
                  "Panel C (10-26): 'deliveries' = logistic regression over deliveries with one intercept per mouse-day (absorbs each "
                  "baseline), a post-injection effect per mouse and one common post x SBI-553 term, likelihood-ratio test of that term "
                  "(is the rise after the injection different on the SBI-553 day?); 'mice' = Wilcoxon signed-rank over the 4 mice "
                  "(4/4 in the same direction gives the smallest possible p, 0.125).",
            panel_c_sensitivity=" | ".join(
                f"{b_}: deliveries p = {pc1[b_]['p_del']:.3f} (log-odds {pc1[b_]['coef']:+.2f}); with stimulus as a covariate "
                f"p = {did_test(b_, stim_fe=True)[0]:.3f}; without one mouse: "
                + ", ".join(f"no {m} p = {did_test(b_, drop=m)[0]:.3f}" for m in MICE) for b_ in ("withdrawal", "flinch"))
                + ". The smaller rise on the SBI-553 day comes from the baselines (before any injection that day): baseline higher on the "
                  "SBI-553 day in " + ", ".join(f"{b_} {pc1[b_]['base_higher']}/4 mice (CMH {pf(pc1[b_]['p_base'])})" for b_ in ("withdrawal", "flinch"))
                + "; after SBI-553 vs after vehicle " + ", ".join(f"{b_} {pf(pc1[b_]['p_post'])}" for b_ in ("withdrawal", "flinch"))
                + ". So this is not evidence that SBI-553 lowered reflexes. Without 1191 the delivery p is "
                + ", ".join(f"{b_} {pc1[b_]['p_no1191']:.2f}" for b_ in ("withdrawal", "flinch"))
                + ". Females (5941, 1191) had the larger vehicle-day rises; 2 vs 2 mice, no sex test.",
            results=" | ".join(f"{d['lab']}: baseline {d['v']['vb']:.0f} → after vehicle {d['v']['vi']:.0f}% ({pf(d['p_veh'])}); baseline "
                               f"{d['v']['sb']:.0f} → after SBI-553 {d['v']['si']:.0f}% ({pf(d['p_sbi'])}); after vs after {pf(d['p_aft'])}; "
                               f"lower after SBI-553 in {d['lower']}/4 mice" for d in NB["info"])
                    + f" | A (any stimulus → movement, after vehicle → after SBI-553): "
                    + "; ".join(f"{m} {NB['g']['vi'][m]:.0f} → {NB['g']['si'][m]:.0f}%" for m in MICE) + f", Wilcoxon {pf(NB['pw'])}"
                    + " | C (change from the same day's baseline, vehicle day vs SBI-553 day): withdrawal "
                    + ", ".join(f"{m} {BNORM['withdrawal']['vehicle_day'][m]:+.0f} vs {BNORM['withdrawal']['sbi_day'][m]:+.0f}" for m in MICE)
                    + "; flinch " + ", ".join(f"{m} {BNORM['flinch']['vehicle_day'][m]:+.0f} vs {BNORM['flinch']['sbi_day'][m]:+.0f}" for m in MICE)
                    + " percentage points.",
            caveat=f"The SBI-553 vs vehicle difference is partly a rise after vehicle (pin flinch {fl['v']['vb']:.0f} → {fl['v']['vi']:.0f}%, "
                   f"{pf(fl['p_veh'])}); movement after any stimulus is ≥ 10 points lower only in {', '.join(NB['clear'])} (SBI-553 on Day "
                   f"{'/'.join(sorted({str(SBIDAY[m]) for m in NB['clear']}))}: drug order = day). Heat escape is rare (≤ "
                   f"{max(I['escape/rearing']['v'].values()):.0f}% of deliveries) and not significant in every analysis version (slide 5). "
                   "Heat deliveries use the original key presses."), 1, FT_BEH)

# --- 2 firing
f2 = FIG + r"\s2_firing.png"; fig_firing(f2)
NR = NOTES["rate"]
slide("Q2  BLA ACTIVITY", "Firing fell more on the SBI-553 day than on the vehicle day in 3 of 4 mice",
      f"Same neurons: lower in {NR['ntr']}/4 mice (p < 0.01 in {', '.join(NR['trs'])}). Caution: Day-1 baselines higher in {NR['d1hi']}/4 "
      "(drug order = day)",
      f2, f"n = 4 mice; {N_RATE} neurons per mouse-day, {N_TR} tracked on both days (n in C). A, B: median over the mouse's neurons; per mouse-day "
          f"Wilcoxon (notes). C: Wilcoxon on the same neurons. SBI-553 on Day 1: 5941, 5939; on Day 2: 5937, 1191. {STARS}",
      notes(measure="Ca transients per minute of still frames in the 4 stimulus blocks, per neuron. A: median over the mouse's neurons per "
                    "session; B: median over neurons of the per-neuron % change after the injection vs the same day's baseline (0 = no "
                    "median change); C: per tracked neuron, log2 fold change on the SBI-553 day minus log2 fold change on the vehicle day, "
                    "median per mouse.",
            n="4 mice; neurons per session " + ", ".join(f"{m} {RATE[(m, 'vehicle')]['n_cells']} / {RATE[(m, 'SBI-553')]['n_cells']}" for m in MICE)
              + " (vehicle day / SBI-553 day); tracked " + ", ".join(f"{m} {RNORM[m]['n']}" for m in MICE) + ".",
            results="; ".join(f"{m}: vehicle day {RATE[(m, 'vehicle')]['base_median']:.2f} → {RATE[(m, 'vehicle')]['inj_median']:.2f} /min "
                              f"(median change {RATE[(m, 'vehicle')]['change_pct']:.0f}%, Wilcoxon {pf(RATE[(m, 'vehicle')]['p_wilcoxon'])}), "
                              f"SBI-553 day {RATE[(m, 'SBI-553')]['base_median']:.2f} → {RATE[(m, 'SBI-553')]['inj_median']:.2f} "
                              f"({RATE[(m, 'SBI-553')]['change_pct']:.0f}%, {pf(RATE[(m, 'SBI-553')]['p_wilcoxon'])}); tracked difference "
                              f"{RNORM[m]['diff_median']:+.2f} log2 ({pf(RNORM[m]['p_wilcoxon'])}, {RNORM[m]['n_lower_on_sbi']}/{RNORM[m]['n']} "
                              f"neurons lower on the SBI-553 day)" for m in MICE) + ".",
            caveat=f"Baseline firing was higher on Day 1 than on Day 2 in {NR['d1hi']}/4 mice, and the drug order is the day order, so a day "
                   "effect cannot be separated from the drug. 5937 on its SBI-553 day: "
                   f"{RATE[('5937', 'SBI-553')]['n_zero_both']}/{RATE[('5937', 'SBI-553')]['n_cells']} neurons silent in both sessions (floor)."),
      2, FT_RATE)

# --- 2b single neurons, baseline-normalized (10-26, user: one slide between slides 2 and 3; bar graphs normalized to baseline)
f2b = FIG + r"\s2b_single_norm.png"; S2B = fig_single_norm(f2b)
resp_sig = [ev for ev in STIM if min(NORM[ev][2][k] for k in ("veh", "sbi", "between")) < 0.05]
cnt_sig = [(ev, col) for ev in STIM for col in ("UP", "DOWN") if S2B[ev][(col, "between")] < 0.05]
SH = {"Light touch": "light", "Mild touch": "mild", "Pin prick": "pin", "Heat": "heat"}


def _dir(ev, col):
    v, s_ = S2B[ev][(col, "vehicle")]["v"], S2B[ev][(col, "SBI-553")]["v"]
    return f"{SH[ev]} {col} {v:+.0f} vs {s_:+.0f}"


slide("Q3  RESPONSES: EACH NEURON, NORMALIZED TO BASELINE",
      "Normalized to each day's baseline: responder numbers changed, response size did not"
      if cnt_sig and not resp_sig else "Each neuron, normalized to the same day's baseline: vehicle day vs SBI-553 day",
      ("Light / mild UP and DOWN rose more on the SBI-553 day, pin UP and DOWN more on the vehicle day (p < 0.05); heat ns"
       if set(cnt_sig) == {(e, c) for e in ("Light touch", "Mild touch", "Pin prick") for c in ("UP", "DOWN")} else
       "Change in % neurons differed between days (p < 0.05): " + "; ".join(_dir(ev, col) for ev, col in cnt_sig))
      if cnt_sig else "No change in % UP / DOWN differed between days",
      f2b, f"Top: mean ± SEM over neurons of (response after the injection - the same neuron's baseline response), vehicle day {N_VEH}, "
           f"SBI-553 day {N_SBI} neurons; star over a bar = Wilcoxon vs 0; bracket = Mann-Whitney between days. Bottom: change in % UP / "
           f"DOWN; star over a bar = McNemar (same neurons); bracket = Mann-Whitney on each neuron's class change. Markers = mice; "
           f"n = 4 mice. {STARS}",
      notes(measure="Top: per neuron, R in the injection session minus the same neuron's R in that day's baseline session (R = mean over "
                    "trials of the mean z in the response window minus the mean z just before; equal to the AUC of the baseline-"
                    "subtracted trace divided by the window length). Bottom: per neuron, class after minus class before (UP: +1 = became "
                    "UP, -1 = lost UP, 0 = same); the bar is the mean x 100 = change in % UP (percentage points); same for DOWN. Title "
                    "line of each top panel: the same tracked neurons on both days, Wilcoxon on (SBI-553-day change - vehicle-day change).",
            n=f"all neurons: vehicle day {N_VEH}, SBI-553 day {N_SBI} (same neurons before and after within a day); tracked {N_TR}; 4 mice.",
            results=" | ".join(
                f"{ev}: response change vehicle day {NORM[ev][0]['vehicle'].chg.mean():+.3f} z (Wilcoxon vs 0 {pf(NORM[ev][2]['veh'])}), "
                f"SBI-553 day {NORM[ev][0]['SBI-553'].chg.mean():+.3f} ({pf(NORM[ev][2]['sbi'])}), between days Mann-Whitney "
                f"{pf(NORM[ev][2]['between'])}, tracked {pf(NORM[ev][2]['tracked'])}; % UP change vehicle "
                f"{S2B[ev][('UP', 'vehicle')]['v']:+.1f} ({pf(S2B[ev][('UP', 'vehicle')]['p'])}), SBI-553 {S2B[ev][('UP', 'SBI-553')]['v']:+.1f} "
                f"({pf(S2B[ev][('UP', 'SBI-553')]['p'])}), between {pf(S2B[ev][('UP', 'between')])}; % DOWN change vehicle "
                f"{S2B[ev][('DOWN', 'vehicle')]['v']:+.1f} ({pf(S2B[ev][('DOWN', 'vehicle')]['p'])}), SBI-553 "
                f"{S2B[ev][('DOWN', 'SBI-553')]['v']:+.1f} ({pf(S2B[ev][('DOWN', 'SBI-553')]['p'])}), between {pf(S2B[ev][('DOWN', 'between')])}"
                for ev in STIM) + ".",
            caveat="The SBI-553-day baselines had fewer light-touch responders (1% vs 14% UP), so part of the light-touch rise is a return "
                   "from a low baseline; 1191's Day-1 vehicle session had no responders after the injection (lowers the vehicle-day "
                   "changes); drug order = day; neurons pooled over 4 mice, uncorrected."), "2b", FT_NEUR)

# --- 3 population
f3 = FIG + r"\s3_population.png"; fig_population(f3)
pin, Tpin = PO["Pin prick"]
_vm = np.nanmean(list(pin["vi"]["m"].values()))
pin_low = [m for m in MICE if pin["si"]["m"].get(m, np.nan) < pin["vi"]["m"].get(m, np.nan)]   # mouse: after SBI-553 < its own after vehicle
n_bl_low = sum(PO[ev][0]["sb"]["mean"] < PO[ev][0]["vb"]["mean"] for ev in STIM)
slide("Q3  RESPONSES: POPULATION", "Average stimulus responses did not change within either day",
      f"Only pin: after SBI-553 < after vehicle ({pf(Tpin['after'])}), but its baseline was already as low; same {N_TR} neurons ns "
      f"({pf(TR.loc['Pin prick'].p_after)})",
      f3, f"Bars = mean ± SEM of all neurons (vehicle day {N_VEH}, SBI-553 day {N_SBI}); markers = mouse means; n = 4 mice. Blue / red = "
          f"Wilcoxon within day (same neurons); black = Mann-Whitney after vs after; grey = Mann-Whitney baseline vs baseline. {STARS}",
      notes(measure="Response of each neuron R = mean over trials of (mean z in the response window - mean z in the window just before "
                    "the event); z = dF/F / robust SD (MAD) of the session. Windows: light, mild, heat 0-1 s; pin 0-0.5 s; baseline 0.5 s.",
            n=f"neurons: vehicle day {N_VEH}, SBI-553 day {N_SBI} (same neurons within a day); {N_TR} tracked on both days; 4 mice.",
            results=" | ".join(fmt_pop(ev) for ev in STIM) + ".",
            caveat=f"The pin difference is not in the same tracked neurons; a mouse's after-SBI-553 mean is below its own after-vehicle "
                   f"mean (other day) in {len(pin_low)}/4 mice ({', '.join(pin_low)}); the SBI-553-day baseline is lower than the vehicle-day baseline for "
                   f"{n_bl_low}/4 stimuli (day / drug-order confound). The z scale is per session (MAD includes the "
                   "neuron's activity), so sizes across sessions are approximate."), 3, FT_NEUR)

# --- 3b population AUC, pre vs post window (10-26, user: 'we need AUC right? ... sth like AUC')
f3a0 = FIG + r"\s3a_auc_pre_post_bars.png"; fig_auc(f3a0)              # kept: pre / post bars (plain z offsets dominate)
f3a = FIG + r"\s3a_auc.png"; fig_auc_delta(f3a)
sigA = {ev: [lab for c, _, lab in CND if AUC[ev][c]["p"] < 0.05] for ev in STIM}
slide("Q3  RESPONSES: POPULATION AUC",
      (lambda k: f"Response AUC ≠ 0 (p < 0.05): vehicle day {k['vb']}/4 → {k['vi']}/4 stimuli, SBI-553 day {k['sb']}/4 → "
                 f"{k['si']}/4")({c: sum(AUC[ev][c]["p"] < 0.05 for ev in STIM) for c, _, _ in CND}),
      "After vehicle vs after SBI-553 (Mann-Whitney on AUC): " + ", ".join(f"{ev.split()[0].lower()} {pf(AUC[ev]['p_after'])}" for ev in STIM),
      f3a, "Bottom: z minus each neuron's mean in the 0.5 s before the event (the test's baseline); mean ± SEM over neurons. Top: area "
           "under that curve in the response window (grey). Star over a bar = Wilcoxon vs 0; blue / red = same neurons, baseline vs "
           f"injection session (Wilcoxon); black = Mann-Whitney after vehicle vs after SBI-553. n = 4 mice. {STARS}",
      notes(measure="per neuron-session: trial-mean z trace (z = dF/F / robust SD of the session) minus its mean in the 0.5 s before the "
                    "event (the same baseline window as the UP / DOWN test); AUC = trapezoid area over the response window (pin 0-0.5 s; "
                    "light, mild, heat 0-1 s) = R x window length. The bars are areas under the plotted curves.",
            n="neurons: " + ", ".join(f"{lab} {len(AUC['Pin prick'][c]['auc'])}" for c, _, lab in CND) + "; 4 mice.",
            results=" | ".join(f"{ev}: " + "; ".join(f"{lab} AUC {AUC[ev][c]['auc'].mean():+.4f} z·s (Wilcoxon vs 0 {pf(AUC[ev][c]['p'])})"
                                                     for c, _, lab in CND)
                               + f"; within day vehicle {pf(AUC[ev]['p_within_veh'])}, SBI-553 {pf(AUC[ev]['p_within_sbi'])}; after vehicle vs "
                                 f"after SBI-553 {pf(AUC[ev]['p_after'])}" for ev in STIM) + ".",
            caveat="neurons pooled over mice, uncorrected; the SBI-553-day baseline already had weak responses, so the missing response "
                   "after SBI-553 is not by itself a drug effect; drug order = day."),
      "3b", FT_NEUR)

# --- 3c how UP / DOWN neurons are defined (10-26, user: criteria + representative neurons' traces, before the UP / DOWN slide)
f3b = FIG + r"\s3b_updown_criteria.png"                        # made by make_updown_criteria.py (needs the raw session)
if os.path.exists(f3b):
    J3 = json.load(open(f3b.replace(".png", ".json"), encoding="utf-8"))
    E3 = {e["label"]: e for e in J3["examples"]}
    slide("Q3  RESPONSES: HOW A NEURON IS CALLED UP OR DOWN",
          "UP / DOWN = the response beats 1000 time-shifted copies of the event train",
          f"Rule: R = mean z in the response window - mean z just before; p vs the shifted trains; BH q ≤ 0.20 → UP if R > 0, "
          f"DOWN if R < 0",
          f3b, f"Example session: {J3['mouse']} Day {J3['day']} after vehicle, pin prick, {J3['n_cells']} neurons → {J3['n_up']} UP, "
               f"{J3['n_down']} DOWN. Top: every trial (z minus its own 0.5 s before, as in the test); middle: mean ± SEM over trials; bottom: "
               f"grey = R at 1000 shifted event trains, line = observed R. Shifts stay inside the same stimulus block.",
          notes(rule="for every neuron, session and event: R = mean over trials of (mean z in the response window - mean z in the window "
                     "just before the event). z = dF/F / robust SD (1.4826 x MAD) of the whole session. Null: the event train is shifted "
                     "circularly inside each schedule segment (same number and spacing of events, timing destroyed) 1000 times and R is "
                     "recomputed; p = (1 + number of shifted |R| ≥ observed |R|) / 1001, two-sided. Benjamini-Hochberg over the neurons of "
                     "that session x event; q ≤ 0.20 → UP (R > 0) or DOWN (R < 0), otherwise not responsive. Trials in the first 15 s are "
                     "dropped; at least 5 trials needed.",
                examples="; ".join(f"{k}: neuron {e['cell']}, R = {e['R']:+.3f} z, p = {e['p']:.3f}, q = {e['q']:.3f}, null 95% "
                                   f"{e['null_lo']:+.2f} to {e['null_hi']:+.2f}, {e['n_trials']} trials" for k, e in E3.items()) + ".",
                why_this_test="it keeps each neuron's own noise and slow activity and the block structure of the session, so a neuron is "
                              "UP only if its activity is locked to the event times. q ≤ 0.20 is lenient (exploratory); see the "
                              "responsive-neuron criteria document for published rules (e.g. Corder 2019, rank-sum p < 0.01).",
                n="the example neurons are the clearest UP, the clearest DOWN and a typical not-responsive neuron of this one session; "
                  "the same rule is applied to every neuron and session in slides 4 and S4."), "3c", FT_NEUR)

# --- 4 single neurons
f4 = FIG + r"\s4_single.png"; fig_single(f4)
Rp, Tp = SG["Pin prick"]
d_tot = Rp["vi"]["kUP"] - Rp["vb"]["kUP"]
d_39 = round(Rp["vi"]["mUP"]["5939"] * Rp["vi"]["nm"]["5939"] / 100 - Rp["vb"]["mUP"]["5939"] * Rp["vb"]["nm"]["5939"] / 100)
Rl, Tl = SG["Light touch"]
l3 = sum(Rl["si"]["mUP"][m] > Rl["sb"]["mUP"][m] for m in MICE)
ld3 = sum(Rl["si"]["mDOWN"][m] > Rl["sb"]["mDOWN"][m] for m in MICE)
pday = {d: {dr: 100 * x.UP.mean() for dr, x in cc[(cc.event == "Pin prick") & (cc.phase == "post") & (cc.day == d)].groupby("drug")}
        for d in (1, 2)}
slide("Q3  RESPONSES: SINGLE NEURONS", "Touch responders rose on the SBI-553 day; pin-prick ones on the vehicle day",
      f"SBI-553 day: light UP {Rl['sb']['UP']:.0f} → {Rl['si']['UP']:.0f}%, DOWN {Rl['sb']['DOWN']:.0f} → {Rl['si']['DOWN']:.0f}% "
      f"({min(l3, ld3)}/4 mice); vehicle day: pin UP {Rp['vb']['UP']:.0f} → {Rp['vi']['UP']:.0f}% ({d_39} of {d_tot} extra in 5939)",
      f4, f"UP / DOWN = time-shift test per neuron, BH q ≤ 0.20; neurons pooled (vehicle day {N_VEH}, SBI-553 day {N_SBI}), markers = mice; "
          f"n = 4 mice. Blue / red = McNemar within day (same neurons); black = Fisher after vehicle vs after SBI-553. {STARS}",
      notes(measure="UP / DOWN: each neuron's R compared with 1000 copies of the event train shifted in time inside each recording block "
                    "(p); Benjamini-Hochberg within each session x event, q ≤ 0.20; UP if R > 0, DOWN if R < 0.",
            n="neurons per mouse, vehicle day / SBI-553 day: " + ", ".join(f"{m} {NM_DAY[m][3 - SBIDAY[m]]} / {NM_DAY[m][SBIDAY[m]]}" for m in MICE)
              + f" (total {N_VEH} / {N_SBI}).",
            results=" | ".join(fmt_single(ev) for ev in STIM) + ".",
            caveat=f"The pin rise rests mainly on 5939 (vehicle day {Rp['vb']['mUP']['5939']:.0f} → {Rp['vi']['mUP']['5939']:.0f}%); 5937 rose "
                   f"on its SBI-553 day ({Rp['sb']['mUP']['5937']:.0f} → {Rp['si']['mUP']['5937']:.0f}%). Pin UP after vehicle vs after SBI-553: "
                   f"Day 1: {pday[1]['vehicle']:.0f}% vs {pday[1]['SBI-553']:.0f}%; Day 2: {pday[2]['vehicle']:.0f}% vs {pday[2]['SBI-553']:.0f}%. "
                   f"The light-touch rise starts from a low SBI-553-day baseline ({Rl['sb']['UP']:.0f}% vs {Rl['vb']['UP']:.0f}% on the vehicle "
                   f"day). 1191's Day-1 vehicle session ({NM_DAY['1191'][1]} of the {N_VEH} vehicle-day neurons) had no stimulus responders "
                   "after the injection, which lowers every after-vehicle % (S3)."), 4, FT_NEUR)

# --- 5 summary
f5 = FIG + r"\s5_robust.png"; rows5, ok5 = fig_robust(f5)
lomo_ok = [lab.split(" (")[0] for (lab, cells) in rows5 if cells[5][1] == "yes"]
slide("SUMMARY", "Only the touch-neuron changes after SBI-553 survive every analysis and mouse",
      "Robust to marks / window only: pin UP rise after vehicle (needs 5939), heat and movement-onset rises, fewer pin flinches",
      f5, "Columns 1-5 = the same data analysed with other event marks / windows (S1); last column = this deck with each mouse left out "
          "once. Limits: n = 4 mice; drug order = day; pooled neurons / deliveries; marks + window chosen per event, no correction.",
      notes(versions="1 = timing-corrected marks, window per event chosen on % UP; 2 = original marks, window per event; 3 = this deck, "
                     "marks + window per event (the combination with the most p < 0.05); 4 = corrected marks, every event 0-0.5 s; 5 = "
                     "original marks, stimuli 0-1 s, behaviours 0-0.5 s. The original marks are 0.40 s (Day 1) / 0.37 s (Day 2) late "
                     "against the video.",
            leave_one_mouse_out="this deck's test repeated four times, each time without one mouse; holds in 4/4 for: " + "; ".join(lomo_ok)
                                + ". Robust to marks / window is not the same as robust to the mice (e.g. the pin gap needs 5939).",
            limits="n = 4 mice (a rank test over 4 mice cannot reach p < 0.05); neuron- or delivery-pooled p values; drug order "
                   "confounded with the day; marks and window chosen per event without correction; in 1191 (Day 1) the reflex responses "
                   "are carried by a few trials with large brain motion. Sex: 2 vs 2 mice, candidates only."), 5,
      "Exploratory; each cell = one test (named in the row); robustness across analysis choices is not a replication")

# --- S1 methods
s = prs.slides.add_slide(BL)
tb(s, 0.45, 0.10, 8, 0.3, "SUPPLEMENT S1  METHODS", 11, color=(0x55, 0x55, 0x55))
tb(s, 0.45, 0.32, 12.4, 0.6, "How the data were recorded and analysed", 24)
fd = FIG + r"\s0_design.png"; fig_design(fd)
s.shapes.add_picture(fd, Inches(0.5), Inches(0.95), Inches(12.3))
left = ("Recording\n"
        "- mini1p Ca imaging of the BLA (GCaMP, 30 Hz) + side and bottom cameras on one frame clock\n"
        f"- neurons: EXTRACT, each checked on the max dF/F projection; {N_TR} found again on Day 2 (tracked)\n"
        "Events\n"
        "- Day 1 deliveries = key marks, Day 2 = hand clicker; reflexes = key marks; movement / grooming = automatic\n"
        "- this deck: heat deliveries and withdrawal marks = the original key presses (0.40 s late on Day 1, 0.37 s on Day 2 against "
        "the video); all other marks timing-corrected\n"
        "- the marks and the window of each event were chosen as the combination with the most p < 0.05 (exploratory, no correction)\n"
        "Encoding model\n"
        "- ridge regression per neuron (stimulus, reflex, movement), cross-validated unique variance, 100 shuffles, BH q ≤ 0.20")
right = ("Response of one neuron\n"
         "- z = dF/F / robust SD (MAD) of the session\n"
         "- per trial: mean z in the response window - mean z in the window just before; R = mean over trials\n"
         "- windows: light, mild, heat 0-1 s; pin 0-0.5 s (vs the 0.5 s before); withdrawal, flinch, escape, grooming 0-1 s; "
         "movement, attending 0-0.5 s (vs the 1 s before)\n"
         "UP / DOWN\n"
         "- R vs 1000 time-shifted event trains in each block; Benjamini-Hochberg within session x event, q ≤ 0.20\n"
         "Tests\n"
         "- within day (same neurons): McNemar (% UP / DOWN), Wilcoxon (R); after vehicle vs after SBI-553: Fisher (%), Mann-Whitney (R); "
         "behaviour: CMH by mouse; tracked: paired Wilcoxon\n"
         "- neurons pooled over mice = exploratory; " + STARS)
heads = ("Recording", "Events", "Encoding model", "Response of one neuron", "UP / DOWN", "Tests")
tb(s, 0.45, 3.15, 6.1, 3.8, left, 12, bold_heads=heads)
tb(s, 6.8, 3.15, 6.1, 3.8, right, 12, bold_heads=heads)
tb(s, 0.45, 7.08, 11.3, 0.3, "Methods of this deck; numbers and tests of every analysis: S2", 9.5, color=(0x88, 0x88, 0x88), italic=True)
tb(s, 12.3, 7.08, 0.8, 0.3, "S1", 10, color=(0x88, 0x88, 0x88), align=PP_ALIGN.RIGHT)
s.notes_slide.notes_text_frame.text = notes(
    data="all numbers come from the analysis with event marks and response window chosen per event (folder 3): for every event the four "
         "combinations (original / corrected marks x 0-0.5 / 0-1 s) were scored on 9 tests (single neurons: Fisher, McNemar; population: "
         "Mann-Whitney, Wilcoxon) and the one with the most p < 0.05 was used.",
    design="crossover: SBI-553 on Day 1 for 5941 and 5939, on Day 2 for 5937 and 1191, vehicle on the other day; each day a baseline "
           "session then the injection session.",
    to_add_before_the_meeting="SBI-553 dose, route and vehicle; time from the injection to the injection session; days between Day 1 "
                              "and Day 2 (not in the analysis files).")

# --- S2 numbers and tests
f6 = FIG + r"\s6_tests.png"; fig_tests_table(f6)
slide("SUPPLEMENT S2  NUMBERS AND TESTS", "n and statistics of every analysis in this deck", "", f6, "",
      notes(behaviour="CMH = Cochran-Mantel-Haenszel chi-square over one 2 x 2 table per mouse (continuity-corrected).",
            neurons="McNemar = exact binomial test on the neurons that changed class between the baseline and the injection session of "
                    "the same day; Fisher exact = after vehicle vs after SBI-553 (different days, different session pools).",
            tracked="185 neurons matched between Day 1 and Day 2 (footprint + position), each compared with itself.",
            correction="none; every neural p pools neurons over the 4 mice and is exploratory."), "S2",
      "Exploratory; " + STARS)

# --- S3 per mouse
f7 = FIG + r"\s7_permouse.png"; fig_permouse(f7)
z1191 = all(SG[ev][0]["vi"][c].get("1191", 1) == 0 for ev in STIM for c in ("mUP", "mDOWN"))
slide("SUPPLEMENT S3  EACH MOUSE", "Per mouse: the vehicle-day pin rise is mostly 5939; 5937 rose on its SBI-553 day",
      (f"1191 had no stimulus responders after vehicle (Day 1); light-touch UP rose on the SBI-553 day in {l3}/4 mice" if z1191 else
       "Each line = one mouse, its baseline and injection session of the same day"), f7,
      "n = 4 mice; neurons per mouse and session in the notes; descriptive (no test).",
      notes(results="pin UP %: " + "; ".join(f"{m} vehicle day {SG['Pin prick'][0]['vb']['mUP'][m]:.0f} → {SG['Pin prick'][0]['vi']['mUP'][m]:.0f}, "
                                             f"SBI-553 day {SG['Pin prick'][0]['sb']['mUP'][m]:.0f} → {SG['Pin prick'][0]['si']['mUP'][m]:.0f}"
                                             for m in MICE)
                    + " | heat UP %: " + "; ".join(f"{m} {SG['Heat'][0]['vb']['mUP'][m]:.0f} → {SG['Heat'][0]['vi']['mUP'][m]:.0f}, "
                                                   f"{SG['Heat'][0]['sb']['mUP'][m]:.0f} → {SG['Heat'][0]['si']['mUP'][m]:.0f}" for m in MICE)
                    + " | light UP %: " + "; ".join(f"{m} {SG['Light touch'][0]['vb']['mUP'][m]:.0f} → {SG['Light touch'][0]['vi']['mUP'][m]:.0f}, "
                                                    f"{SG['Light touch'][0]['sb']['mUP'][m]:.0f} → {SG['Light touch'][0]['si']['mUP'][m]:.0f}" for m in MICE) + ".",
            n="neurons per mouse, vehicle day / SBI-553 day: " + ", ".join(f"{m} {NM_DAY[m][3 - SBIDAY[m]]} / {NM_DAY[m][SBIDAY[m]]}" for m in MICE) + ".",
            caveat="1191's Day-1 vehicle session has 0 responders for every stimulus and reflex (its baseline responses were carried "
                   "by a few high-motion trials)."), "S3", "Descriptive; each mouse's sessions of the same day are joined")

# --- S4 reflex / movement neurons
f8 = FIG + r"\s8_reflex.png"; fig_single(f8, evs=REFL)
z1191r = all(SG[ev][0]["vi"][c].get("1191", 1) == 0 for ev in ("withdrawal", "flinch") for c in ("mUP", "mDOWN"))
Rf, Tf = SG["flinch"]
slide("SUPPLEMENT S4  REFLEX AND MOVEMENT NEURONS",
      "Reflex and movement-onset responders rose after both injections",
      f"Flinch DOWN higher after SBI-553 than after vehicle ({Rf['si']['DOWN']:.0f} vs {Rf['vi']['DOWN']:.0f}%, Fisher {pf(Tf['DOWN_after'])}); "
      f"withdrawal UP after vs after n.s.",
      f8, f"Same format as slide 4; neurons pooled (vehicle day {N_VEH}, SBI-553 day {N_SBI}), n = 4 mice; McNemar within day, Fisher "
          f"after vs after. {STARS}",
      notes(results=" | ".join(fmt_single(ev) for ev in REFL) + ".",
            caveat=f"1191's Day-1 vehicle session has {'no' if z1191r else 'few'} withdrawal / flinch responders, which lowers the "
                   "after-vehicle %; the withdrawal / flinch DOWN difference is lost without 5941, 5939 or 1191 (slide 5). Withdrawal uses "
                   "the original key presses."), "S4", FT_NEUR)

# --- S5 encoding model
f9 = FIG + r"\s9_encoding.png"; enc, nn = fig_encoding(f9)
sday = ENC.groupby("day").type.value_counts(normalize=True).mul(100).unstack().fillna(0)
_rx = ENC[(ENC.condition == "baseline") & (ENC.type == "reflex")]
n_rx, n_rx91 = len(_rx), int(((_rx.mouse.astype(str) == "1191") & (_rx.day == 1)).sum())
slide("SUPPLEMENT S5  WHAT DO THE NEURONS FOLLOW?", "Movement explains more neurons than the stimulus",
      f"Movement only {enc['movement only'].min():.0f}-{enc['movement only'].max():.0f}% of neurons; stimulus only "
      f"{enc['stimulus'].min():.0f}-{enc['stimulus'].max():.0f}% (highest after vehicle); stimulus or reflex not separable "
      f"{enc['stimulus or reflex (cannot separate)'].min():.0f}-{enc['stimulus or reflex (cannot separate)'].max():.0f}%",
      f9, "Encoding model per neuron and session (ridge regression; unique variance of stimulus, reflex and movement; 100 shuffles, "
          "BH q ≤ 0.20); n = 4 mice, neuron-sessions in the labels.",
      notes(results="; ".join(f"{c}: " + ", ".join(f"{t} {enc.loc[c].get(t, 0):.1f}%" for t in ENC_ORDER) for c in ["baseline", "vehicle", "SBI-553"])
                    + f". By day: stimulus only {sday.loc[1, 'stimulus']:.1f}% (Day 1), {sday.loc[2, 'stimulus']:.1f}% (Day 2).",
            caveat="Stimulus and reflex come almost together (taps sit on the withdrawal), so 'stimulus only' is a lower bound; the "
                   f"baseline 'reflex' class is mostly 1191's Day-1 baseline session ({n_rx91} of {n_rx})."), "S5", FT_NEUR)

# --- A1 behaviour per stimulus, normalized to the same-day baseline, WITH tests (10-26, user: 'you missed several slides P value')
BEH6 = ["withdrawal", "flinch", "attending", "movement", "escape/rearing", "grooming"]
SHS = {"Light touch": "light", "Mild touch": "mild", "Pin prick": "pin", "Heat": "heat"}


def safe_did(beh, stim):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return did_test(beh, stim=stim)[0]
        except Exception:
            return np.nan


def fig_beh_stim_stats(fn):
    fig, axs = plt.subplots(2, 3, figsize=(12.3, 4.75), gridspec_kw=dict(wspace=0.22, hspace=0.55, left=0.07, right=0.995, top=0.86,
                                                                          bottom=0.07))
    R = {}
    for a, beh in zip(axs.ravel(), BEH6):
        vals = []
        for i, stim in enumerate(STIM):
            for dr, c_i, c_b, dx in (("vehicle", "vi", "vb", -0.19), ("SBI-553", "si", "sb", 0.19)):
                pm_i, _ = beh_pct(beh, stim, c_i)
                pm_b, _ = beh_pct(beh, stim, c_b)
                ch = {m: pm_i[m] - pm_b[m] for m in MICE if m in pm_i and m in pm_b}
                v = np.mean(list(ch.values()))
                a.bar(i + dx, v, width=0.36, color=CC[c_i], alpha=0.55, edgecolor=CC[c_i], zorder=2)
                for m, y in ch.items():
                    a.scatter(i + dx + OFF[m] * 0.35, y, marker=MK[m], s=16, color=MC[m], edgecolor="k", lw=0.4, zorder=4)
                vals += list(ch.values()) + [v]
                R[(beh, stim, dr)] = dict(v=v, p=cmh(beh, stim, f"{dr} vs its baseline"), ch=ch)
            R[(beh, stim, "did")] = safe_did(beh, stim)
        m_ = max(abs(min(vals)), abs(max(vals)), 1.0)
        a.set_ylim(-m_ * 1.15, m_ * 1.95)
        for i, stim in enumerate(STIM):
            for dr, dx in (("vehicle", -0.19), ("SBI-553", 0.19)):
                p = R[(beh, stim, dr)]["p"]
                a.text(i + dx, m_ * 1.12, st(p) if np.isfinite(p) else "", ha="center", va="bottom", fontsize=8.5,
                       color=CC["vi" if dr == "vehicle" else "si"], fontweight="bold" if np.isfinite(p) and p < 0.05 else "normal")
            pd_ = R[(beh, stim, "did")]
            if np.isfinite(pd_):
                bracket(a, i - 0.19, i + 0.19, m_ * 1.50, st(pd_), fs=8.5, h=0.04 * m_)
        a.axhline(0, color="0.3", lw=0.8)
        a.set_xticks(range(4), [SHS[s_] for s_ in STIM], fontsize=10)
        a.set_title(beh, fontsize=12)
        a.tick_params(axis="y", labelsize=9)
    for a in axs[:, 0]:
        a.set_ylabel("change in % of deliveries\n(after - same-day baseline)", fontsize=9.5)
    h = [Patch(color=CC["vi"], alpha=0.55, label="vehicle day (bar = mean of 4 mice)"), Patch(color=CC["si"], alpha=0.55,
                                                                                            label="SBI-553 day")] + mouse_handles()
    fig.legend(handles=h, ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=9, handlelength=1.1,
               columnspacing=0.9)
    fig.savefig(fn); plt.close(fig)
    return R


fA1 = FIG + r"\sA1_behaviour_per_stimulus_stats.png"; RA1 = fig_beh_stim_stats(fA1)
sigA1 = [(b_, s_) for b_ in BEH6 for s_ in STIM if np.isfinite(RA1[(b_, s_, "did")]) and RA1[(b_, s_, "did")] < 0.05]
slide("Q1  BEHAVIOUR  (replaces 'Behaviour per stimulus, normalized to the same-day baseline')",
      "Behaviour per stimulus, normalized to the same-day baseline: vehicle day vs SBI-553 day",
      ("Change differs between days (p < 0.05): " + ", ".join(f"{SHS[s_]} {b_}" for b_, s_ in sigA1)) if sigA1 else
      "No behaviour x stimulus change differed between days",
      fA1, "Bars = mean over 4 mice of (% after the injection - % in the same day's baseline), markers = mice. Star over a bar = CMH, "
           "injection vs its own baseline (mouse = stratum); bracket = day difference of that change, logistic regression over "
           f"deliveries (intercept per mouse-day). Empty = too few events. n = 4 mice. {STARS}",
      notes(tests="bar star: CMH over the 4 mice's 2 x 2 tables, injection session vs the same day's baseline session. Bracket: logistic "
                  "regression, behaviour ~ mouse-day + post per mouse + post x SBI-553 (likelihood-ratio test, 1 df), deliveries of that "
                  "stimulus only. Mouse-level Wilcoxon over 4 mice cannot go below p = 0.125 and is not drawn.",
            results=" | ".join(f"{b_} {SHS[s_]}: vehicle {RA1[(b_, s_, 'vehicle')]['v']:+.1f} pp ({pf(RA1[(b_, s_, 'vehicle')]['p'])}), "
                               f"SBI-553 {RA1[(b_, s_, 'SBI-553')]['v']:+.1f} pp ({pf(RA1[(b_, s_, 'SBI-553')]['p'])}), day difference "
                               f"{pf(RA1[(b_, s_, 'did')])}" for b_ in BEH6 for s_ in STIM) + ".",
            caveat="exploratory, uncorrected (24 behaviour x stimulus pairs); a smaller change can come from a higher baseline (see slide "
                   "1, panel C); drug order = day; heat deliveries = original key presses."), "A1", FT_BEH)

# ================================================================================================ folder-3 deck fixes (10-26, user:
# "you missed several slides P value ... check each slide"); each is a replacement slide for the folder-3 deck, exported on its own
CND4 = ("vb", "vi", "sb", "si")

XT4 = ["baseline", "after", "baseline", "after"]


def bracket_down(a, x1, x2, y, txt, color="k", fs=10, h=None):
    """a bracket that opens upward, below the bars (for the DOWN half)"""
    h = h if h is not None else 0.015 * (a.get_ylim()[1] - a.get_ylim()[0])
    a.plot([x1, x1, x2, x2], [y, y - h, y - h, y], color=color, lw=1.2, clip_on=False)
    a.text((x1 + x2) / 2, y - h, txt, ha="center", va="top", color=color, fontsize=fs if txt != "ns" else fs - 1,
           fontweight="bold" if txt != "ns" else "normal", clip_on=False)


def fisher_base(ev, col):
    R = SG[ev][0]
    kv, ks = R["vb"]["k" + col], R["sb"]["k" + col]
    return fisher_exact([[kv, R["vb"]["n"] - kv], [ks, R["sb"]["n"] - ks]])[1]


# --- F70: UP (solid, up) and DOWN (hatched, down) with every test
def fig_f70(fn):
    fig, ax = plt.subplots(1, 4, figsize=(12.3, 4.75), sharey=True, gridspec_kw=dict(wspace=0.08, left=0.07, right=0.995, top=0.80,
                                                                                      bottom=0.20))
    up_max = max(max([SG[ev][0][c]["UP"] for c in CND4] + [v for c in CND4 for v in SG[ev][0][c]["mUP"].values()]) for ev in STIM)
    dn_max = max(max([SG[ev][0][c]["DOWN"] for c in CND4] + [v for c in CND4 for v in SG[ev][0][c]["mDOWN"].values()]) for ev in STIM)
    for a, ev in zip(ax, STIM):
        R, T = SG[ev]
        for k, c in enumerate(CND4):
            a.bar(k, R[c]["UP"], color=CC[c], width=0.75, zorder=2)
            a.bar(k, -R[c]["DOWN"], color=CC[c], width=0.75, hatch="///", alpha=0.75, lw=0, zorder=2)
            for m in MICE:
                if m in R[c]["mUP"]:
                    a.scatter(k + OFF[m] * 0.75, R[c]["mUP"][m], s=20, marker=MK[m], color=MC[m], edgecolor="white", lw=0.5, zorder=4)
                    a.scatter(k + OFF[m] * 0.75, -R[c]["mDOWN"][m], s=20, marker=MK[m], color=MC[m], edgecolor="white", lw=0.5, zorder=4)
        a.set_ylim(-dn_max * 1.75, up_max * 1.75)
        u, d = up_max, dn_max
        bracket(a, 0, 1, u * 1.08, st(T["UP_veh"]), color=CC["vi"], fs=10, h=0.03 * u)
        bracket(a, 2, 3, u * 1.08, st(T["UP_sbi"]), color=CC["si"], fs=10, h=0.03 * u)
        bracket(a, 1, 3, u * 1.33, st(T["UP_after"]), fs=10, h=0.03 * u)
        bracket(a, 0, 2, u * 1.55, st(fisher_base(ev, "UP")), color="0.45", fs=9.5, h=0.03 * u)
        bracket_down(a, 0, 1, -d * 1.08, st(T["DOWN_veh"]), color=CC["vi"], fs=10, h=0.03 * d)
        bracket_down(a, 2, 3, -d * 1.08, st(T["DOWN_sbi"]), color=CC["si"], fs=10, h=0.03 * d)
        bracket_down(a, 1, 3, -d * 1.33, st(T["DOWN_after"]), fs=10, h=0.03 * d)
        bracket_down(a, 0, 2, -d * 1.55, st(fisher_base(ev, "DOWN")), color="0.45", fs=9.5, h=0.03 * d)
        a.axhline(0, color="k", lw=0.9)
        a.set_xticks(range(4), XT4, fontsize=10)
        day_labels(a)
        pre, a_, b_ = win(ev)
        a.set_title(f"{ev}  ({a_:g}-{b_:g} s" + (", original marks)" if ev in MARKS else ")"), fontsize=12)
    ax[0].set_ylabel("% of neurons\nUP (solid, up) / DOWN (hatched, down)")
    ax[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{abs(v):.0f}"))
    legend2(fig, hatch=True)
    fig.savefig(fn); plt.close(fig)


# --- F21: behaviour per delivery (any stimulus), change from the same-day baseline, with tests
def cmh_change(beh, dr):
    """CMH over mice: injection session vs the same day's baseline, all stimuli"""
    c_i, c_b = ("vi", "vb") if dr == "vehicle" else ("si", "sb")
    tabs = []
    for m in MICE:
        x, y = PD[(PD.mouse == m) & (PD.cond == c_i)][beh].astype(bool), PD[(PD.mouse == m) & (PD.cond == c_b)][beh].astype(bool)
        tabs.append([[int(x.sum()), int((~x).sum())], [int(y.sum()), int((~y).sum())]])
    return cmh_p(tabs)


def fig_f21(fn):
    fig, ax = plt.subplots(1, 6, figsize=(12.3, 4.75), gridspec_kw=dict(wspace=0.45, left=0.06, right=0.995, top=0.70, bottom=0.14))
    R = {}
    for a, beh in zip(ax, BEH6):
        ch = {}
        for dr, c_i, c_b in (("vehicle", "vi", "vb"), ("SBI-553", "si", "sb")):
            ch[dr] = {m: 100 * (PD[(PD.mouse == m) & (PD.cond == c_i)][beh].mean() - PD[(PD.mouse == m) & (PD.cond == c_b)][beh].mean())
                      for m in MICE}
        for m in MICE:
            a.plot([0, 1], [ch["vehicle"][m], ch["SBI-553"][m]], color=MC[m], marker=MK[m], ms=7, mec="white", lw=1.4, zorder=4)
        for x, dr in ((0, "vehicle"), (1, "SBI-553")):
            a.plot([x - 0.22, x + 0.22], [np.mean(list(ch[dr].values()))] * 2, color="k", lw=3, zorder=5)
        a.axhline(0, color="0.5", lw=0.8, ls=":")
        pv, ps = cmh_change(beh, "vehicle"), cmh_change(beh, "SBI-553")
        pm = wilcoxon([ch["SBI-553"][m] - ch["vehicle"][m] for m in MICE])[1] if any(ch["SBI-553"][m] != ch["vehicle"][m] for m in MICE) else np.nan
        pdd = safe_did(beh, None)
        R[beh] = dict(ch=ch, pv=pv, ps=ps, pm=pm, pd=pdd)
        vals = list(ch["vehicle"].values()) + list(ch["SBI-553"].values())
        m_ = max(abs(min(vals)), abs(max(vals)), 1.0)
        a.set_ylim(-m_ * 1.25, m_ * 2.1)
        a.text(0, m_ * 1.12, st(pv) if np.isfinite(pv) else "", ha="center", fontsize=10, color=CC["vi"],
               fontweight="bold" if np.isfinite(pv) and pv < 0.05 else "normal")
        a.text(1, m_ * 1.12, st(ps) if np.isfinite(ps) else "", ha="center", fontsize=10, color=CC["si"],
               fontweight="bold" if np.isfinite(ps) and ps < 0.05 else "normal")
        if np.isfinite(pdd):
            bracket(a, 0, 1, m_ * 1.45, st(pdd), fs=10, h=0.05 * m_)
        a.text(0.5, 1.0, f"mice: p = {pm:.3f}" if np.isfinite(pm) else "mice: n.a.", transform=a.transAxes, ha="center", va="bottom",
               fontsize=9)
        a.set_xticks([0, 1], ["vehicle\nday", "SBI-553\nday"], fontsize=9.5); a.set_xlim(-0.5, 1.5)
        a.set_title(beh, fontsize=12, pad=16)
        a.tick_params(axis="y", labelsize=9)
    ax[0].set_ylabel("change in % of deliveries\n(after - same-day baseline)", fontsize=10)
    h = mouse_handles() + [Line2D([], [], color="k", lw=3, label="mean of 4 mice")]
    fig.legend(handles=h, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=9.5)
    fig.savefig(fn); plt.close(fig)
    return R


# --- F20: seconds of each held behaviour after a delivery, baseline split by day, with tests
def van_elteren(x_by_s, y_by_s):
    """stratified Wilcoxon rank-sum (van Elteren), strata = mice; two-sided normal approximation"""
    from scipy.stats import norm as _n, rankdata as _rk
    num, var = 0.0, 0.0
    for x, y in zip(x_by_s, y_by_s):
        x, y = np.asarray(x, float), np.asarray(y, float)
        n1, n2 = len(x), len(y)
        if n1 == 0 or n2 == 0:
            continue
        r = _rk(np.r_[x, y]); N = n1 + n2
        w = 1.0 / (N + 1)
        W = r[:n1].sum()
        t = np.unique(r, return_counts=True)[1]
        v = n1 * n2 / 12 * ((N + 1) - (t ** 3 - t).sum() / (N * (N - 1)))
        num += w * (W - n1 * (N + 1) / 2); var += w * w * v
    return float(2 * _n.sf(abs(num) / np.sqrt(var))) if var > 0 else np.nan


HELD = ["attending", "lick/bite", "escape/rearing", "movement", "grooming"]


def fig_f20(fn, behs):
    fig, ax = plt.subplots(1, len(behs), figsize=(12.3, 4.75), gridspec_kw=dict(wspace=0.22, left=0.065, right=0.995, top=0.80,
                                                                                bottom=0.17))
    R = {}
    for a, beh in zip(np.atleast_1d(ax), behs):
        col = beh + "_s"
        hi = 0
        for i, stim in enumerate(STIM):
            for k, c in enumerate(CND4):
                x = i * 5 + k
                pm = {m: PD[(PD.mouse == m) & (PD.cond == c) & (PD.stim == stim)][col].mean() for m in MICE}
                v = np.nanmean(list(pm.values()))
                a.bar(x, v, color=CC[c], width=0.85, zorder=2)
                mice_dots(a, x, pm, s=12)
                hi = max([hi, v] + [y for y in pm.values() if np.isfinite(y)])
            g = lambda c: [PD[(PD.mouse == m) & (PD.cond == c) & (PD.stim == stim)][col].values for m in MICE]
            R[(beh, stim)] = dict(veh=van_elteren(g("vi"), g("vb")), sbi=van_elteren(g("si"), g("sb")), after=van_elteren(g("si"), g("vi")),
                                  mv=np.nanmean([PD[(PD.mouse == m) & (PD.cond == "vi") & (PD.stim == stim)][col].mean() for m in MICE]),
                                  ms=np.nanmean([PD[(PD.mouse == m) & (PD.cond == "si") & (PD.stim == stim)][col].mean() for m in MICE]))
        hi = max(hi, 1e-3)
        a.set_ylim(0, hi * 1.6)
        for i, stim in enumerate(STIM):
            r = R[(beh, stim)]
            if np.isfinite(r["veh"]):
                bracket(a, i * 5, i * 5 + 1, hi * 1.06, st(r["veh"]), color=CC["vi"], fs=9, h=0.02 * hi)
            if np.isfinite(r["sbi"]):
                bracket(a, i * 5 + 2, i * 5 + 3, hi * 1.06, st(r["sbi"]), color=CC["si"], fs=9, h=0.02 * hi)
            if np.isfinite(r["after"]):
                bracket(a, i * 5 + 1, i * 5 + 3, hi * 1.30, st(r["after"]), fs=9, h=0.02 * hi)
        a.set_xticks([i * 5 + 1.5 for i in range(4)], [SHS[s_] for s_ in STIM], fontsize=10.5)
        a.set_title(beh, fontsize=12)
        a.tick_params(axis="y", labelsize=9)
    np.atleast_1d(ax)[0].set_ylabel("s of the behaviour in the\n3 s after a delivery (mean of mice)", fontsize=10)
    legend2(fig)
    fig.savefig(fn); plt.close(fig)
    return R


# --- F119: UP / DOWN / other neurons, plain z traces, baseline split by day, with tests
def group_traces(ev):
    out = {}
    for c in CND4:
        G = {"UP": [], "DOWN": [], "rest": []}
        for (m, d, ph, e), v in PSTH.items():
            if e != ev or (("s" if m in SBI[d] else "v") + ("b" if ph == "pre" else "i")) != c:
                continue
            x = cc[(cc.event == ev) & (cc.mouse == m) & (cc.day == d) & (cc.phase == ph)].set_index("cell").cls
            cl = x.reindex(v["ids"]).values
            for g, lab in (("UP", "activated"), ("DOWN", "inhibited"), ("rest", "non-responsive")):
                if (cl == lab).any():
                    G[g].append(v["mean"][:, cl == lab])
            tt = v["tt"]
        out[c] = {g: (np.hstack(Y) if Y else np.zeros((len(tt), 0))) for g, Y in G.items()}
        out[c]["tt"] = tt
    return out


def fig_f119(fn, evs=("Pin prick", "withdrawal")):
    fig, ax = plt.subplots(len(evs), 4, figsize=(12.3, 4.75), sharex=True,
                           gridspec_kw=dict(wspace=0.22, hspace=0.80, left=0.075, right=0.995, top=0.80, bottom=0.10))
    GC = {"UP": "#e74c3c", "DOWN": "#2e86c1", "rest": "#95a5a6"}
    for i, ev in enumerate(evs):
        Gt = group_traces(ev)
        pre, a_, b_ = win(ev)
        for k, c in enumerate(CND4):
            a = ax[i, k]
            tt = Gt[c]["tt"]; sel = (tt >= -3) & (tt <= 6)
            for g in ("rest", "DOWN", "UP"):
                Y = base_sub(Gt[c][g], tt, ev)[sel]
                if Y.shape[1] == 0:
                    continue
                mu = Y.mean(1); se = Y.std(1, ddof=1) / np.sqrt(Y.shape[1]) if Y.shape[1] > 1 else 0 * mu
                a.fill_between(tt[sel], mu - se, mu + se, color=GC[g], alpha=0.2, lw=0); a.plot(tt[sel], mu, color=GC[g], lw=1.3)
            a.axvspan(min(a_, 0) - pre, min(a_, 0), color="#d6eaf8", lw=0, zorder=0); a.axvspan(a_, b_, color="#e5e7e9", lw=0, zorder=0)
            a.axvline(0, color="k", lw=0.7, ls="--")
            R, T = SG[ev]
            a.set_title(f"{CL[c]}\nUP {R[c]['kUP']}, DOWN {R[c]['kDOWN']}, rest {R[c]['n'] - R[c]['kUP'] - R[c]['kDOWN']}", fontsize=9.5,
                        color=CC[c] if c != "vb" else "0.4")
            a.tick_params(labelsize=8.5)
            if i == len(evs) - 1:
                a.set_xlabel("time from event (s)", fontsize=9.5)
        R, T = SG[ev]
        ax[i, 0].set_ylabel(f"{ev}\nz - mean of the {win(ev)[0]:g} s before", fontsize=9.5)
        y1 = ax[i, 0].get_position().y1
        fig.text(0.535, y1 + 0.075, f"{ev}:  % UP  vehicle day {st(T['UP_veh'])}, SBI-553 day {st(T['UP_sbi'])} (McNemar), after vs after "
                 f"{st(T['UP_after'])} (Fisher)   |   % DOWN  {st(T['DOWN_veh'])}, {st(T['DOWN_sbi'])}, {st(T['DOWN_after'])}",
                 ha="center", fontsize=9.5, fontweight="bold", va="bottom")
    h = [Line2D([], [], color=GC[g], lw=2, label=g + (" (not responsive)" if g == "rest" else "")) for g in ("UP", "DOWN", "rest")] + \
        [Patch(color="#d6eaf8", label="window before"), Patch(color="#e5e7e9", label="response window")]
    fig.legend(handles=h, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=9.5)
    fig.savefig(fn); plt.close(fig)


def up_size(ev):
    """response size R of the UP neurons after vehicle vs after SBI-553 (Mann-Whitney) and of the neurons UP at the day's baseline"""
    e = cc[cc.event == ev]
    u = e[(e.phase == "post") & e.UP]
    pu = mannwhitneyu(u[u.drug == "SBI-553"].resp, u[u.drug == "vehicle"].resp)[1]
    j = pairs(e)
    b = j[j.UP_b]
    db = {dr: (b[b.drug_b == dr].resp_a - b[b.drug_b == dr].resp_b) for dr in ("vehicle", "SBI-553")}
    return dict(mv=u[u.drug == "vehicle"].resp.mean(), ms=u[u.drug == "SBI-553"].resp.mean(), nv=int((u.drug == "vehicle").sum()),
                ns=int((u.drug == "SBI-553").sum()), p=pu, p_held=mannwhitneyu(db["SBI-553"], db["vehicle"])[1],
                hv=db["vehicle"].mean(), hs=db["SBI-553"].mean())


FT3 = "Replacement slide for the folder-3 deck (marks + window chosen per event; exploratory, uncorrected) - S1"

# slide F70
fF70 = FIG + r"\sF70_updown_all_tests.png"; fig_f70(fF70)
slide("Q3  RESPONSES: SINGLE NEURONS  (replaces folder-3 slide 70)", "Neurons UP and DOWN after each stimulus, with every test",
      "Now tested for DOWN too: light / mild DOWN rose on the SBI-553 day; pin and heat DOWN rose on the vehicle day",
      fF70, f"Solid = % UP, hatched = % DOWN (drawn downward); markers = mice. Blue / red = McNemar within day (same neurons); black = Fisher "
            f"after vehicle vs after SBI-553; grey = Fisher baseline vs baseline. Neurons pooled (vehicle day {N_VEH}, SBI-553 day {N_SBI}); "
            f"n = 4 mice. {STARS}",
      notes(results=" | ".join(fmt_single(ev) for ev in STIM) + ". Baseline vs baseline (Fisher): " + "; ".join(
          f"{ev} UP {pf(fisher_base(ev, 'UP'))}, DOWN {pf(fisher_base(ev, 'DOWN'))}" for ev in STIM) + "."), "F70", FT3)

# slide F21
fF21 = FIG + r"\sF21_beh_delivery_norm_tests.png"; RF21 = fig_f21(fF21)
slide("Q1  BEHAVIOUR  (replaces folder-3 slide 21)", "Behaviour per delivery, normalized to the same-day baseline, with tests",
      "Reflex rises were smaller on the SBI-553 day (4/4 mice, p = 0.125 = the floor for 4 mice); flinch p = 0.017 over deliveries",
      fF21, "Line = one mouse (% of all deliveries followed by the behaviour, after the injection minus the same day's baseline); bar = mean. "
            "Blue / red star = CMH injection vs its own baseline (mouse = stratum); bracket = logistic test of the day difference over "
            f"deliveries; 'mice' = Wilcoxon over 4 mice (smallest possible p 0.125). {STARS}",
      notes(results=" | ".join(f"{b_}: vehicle day {np.mean(list(r['ch']['vehicle'].values())):+.1f} pp (CMH {pf(r['pv'])}), SBI-553 day "
                               f"{np.mean(list(r['ch']['SBI-553'].values())):+.1f} ({pf(r['ps'])}), day difference deliveries {pf(r['pd'])}, "
                               f"mice {pf(r['pm'])}" for b_, r in RF21.items()) + ".",
            caveat="SBI-553-day baselines were higher for withdrawal and flinch (slide 1 panel C), so a smaller rise is not proof of a drug "
                   "effect; the delivery p depends mostly on 1191."), "F21", FT_BEH)

# slides F19 / F20
for part, behs, num in ((1, ["attending", "lick/bite"], "F19"), (2, ["escape/rearing", "movement", "grooming"], "F20")):
    fn_ = FIG + rf"\s{num}_beh_seconds_tests.png"; RF = fig_f20(fn_, behs)
    sig = [f"{SHS[s_]} {b_}" for (b_, s_), r in RF.items() if np.isfinite(r["after"]) and r["after"] < 0.05]
    slide(f"Q1  BEHAVIOUR  (replaces folder-3 slide {19 if part == 1 else 20})",
          f"Time spent in each behaviour after a delivery ({part}/2), baseline split by day, with tests",
          ("After SBI-553 vs after vehicle (p < 0.05): " + ", ".join(sig)) if sig else "No after-vs-after difference reached p < 0.05",
          fn_, "Seconds of the behaviour in the 3 s after a delivery; bars = mean of 4 mice, markers = mice. Tests over deliveries, "
               "stratified by mouse (van Elteren rank test): blue / red = injection vs its own baseline, black = after SBI-553 vs after "
               f"vehicle. n = 4 mice. {STARS}",
          notes(results=" | ".join(f"{b_} {SHS[s_]}: after vehicle {r['mv']:.2f} s, after SBI-553 {r['ms']:.2f} s; within day vehicle "
                                   f"{pf(r['veh'])}, SBI-553 {pf(r['sbi'])}; after vs after {pf(r['after'])}" for (b_, s_), r in RF.items()) + ".",
                caveat="the old slide pooled both days' baselines; here each day keeps its own baseline. Exploratory, uncorrected."),
          num, FT_BEH)

# slide F119
fF119 = FIG + r"\sF119_groups_tests.png"; fig_f119(fF119)
US = {ev: up_size(ev) for ev in ("Pin prick", "withdrawal")}
slide("Q3  RESPONSES: SINGLE NEURONS  (replaces folder-3 slide 119)", "UP, DOWN and the other neurons, with tests",
      f"Pin UP neurons: bigger response after SBI-553 ({US['Pin prick']['ms']:.2f} vs {US['Pin prick']['mv']:.2f} z, "
      f"{pf(US['Pin prick']['p'])}) but fewer of them; neurons UP at baseline changed alike ({pf(US['Pin prick']['p_held'])})",
      fF119, "Mean ± SEM of z minus its mean in the test's baseline window (0.5 s before; 1 s for withdrawal) for the neurons called UP, DOWN or neither in that session; each day keeps its "
             "own baseline. Test line above each row: % UP / % DOWN, McNemar within day, Fisher after vs after. n = 4 mice, neurons pooled. "
             f"{STARS}",
      notes(response_size=" | ".join(f"{ev}: UP after vehicle n = {u['nv']} mean R {u['mv']:+.3f} z, after SBI-553 n = {u['ns']} {u['ms']:+.3f} "
                                     f"(Mann-Whitney {pf(u['p'])}); neurons UP in the day's baseline, change after the injection vehicle "
                                     f"{u['hv']:+.3f}, SBI-553 {u['hs']:+.3f} ({pf(u['p_held'])})" for ev, u in US.items()) + ".",
            caveat="the UP group is chosen by the test itself: in a session with fewer UP neurons the BH cut-off is stricter, so the ones "
                   "that pass are the strongest (selection bias). The fair comparison is the neurons UP at baseline (held-out), which "
                   "changed alike on both days.",
            results=" | ".join(fmt_single(ev) for ev in ("Pin prick", "withdrawal")) + "."), "F119", FT_NEUR)

# --- whole-session activity with tests (replaces folder-3 slides 27, 28, 31, 32, 33)
SEGT = pd.read_csv(A + r"\pi_extra\session\segments_per_neuron.csv"); SEGT["mouse"] = SEGT.mouse.astype(str)
SEGS = ["no stimulus"] + STIM
SEGW = SEGT.pivot_table(index=["mouse", "day", "phase", "cell"], columns="seg", values="mean_z")
SEGW["blocks"] = SEGW[STIM].mean(1)
SEGW["rise"] = SEGW["blocks"] - SEGW["no stimulus"]
SEGW = SEGW.reset_index()
SEGW["cond"] = [("s" if m in SBI[d] else "v") + ("b" if ph == "pre" else "i") for m, d, ph in zip(SEGW.mouse, SEGW.day, SEGW.phase)]


def within_pairs(col, k):
    """same neurons, baseline vs injection session of the vehicle day (k='v') or SBI-553 day (k='s')"""
    x = SEGW[SEGW.cond == k + "b"].set_index(["mouse", "day", "cell"])[col]
    y = SEGW[SEGW.cond == k + "i"].set_index(["mouse", "day", "cell"])[col]
    j = pd.concat([x, y], axis=1, join="inner").dropna()
    return j.iloc[:, 0], j.iloc[:, 1]


def fig_w1(fn):
    """rise of activity from the no-stimulus start to the stimulus blocks, per neuron, 4 conditions"""
    cols = ["rise"] + [f"rise_{ev}" for ev in STIM]
    for ev in STIM:
        SEGW[f"rise_{ev}"] = SEGW[ev] - SEGW["no stimulus"]
    fig, ax = plt.subplots(1, 5, figsize=(12.3, 4.75), gridspec_kw=dict(wspace=0.30, left=0.07, right=0.995, top=0.74, bottom=0.20))
    R = {}
    for a, col in zip(ax, cols):
        hi = []
        r = {}
        for k, c in enumerate(CND4):
            v = SEGW[SEGW.cond == c][col].dropna()
            mu, se = v.mean(), v.std(ddof=1) / np.sqrt(len(v))
            a.bar(k, mu, yerr=se, color=CC[c], width=0.75, capsize=3, error_kw=dict(lw=1), zorder=2)
            mm = {m: g[col].mean() for m, g in SEGW[SEGW.cond == c].groupby("mouse")}
            mice_dots(a, k, mm, s=22)
            hi += [mu + se, mu - se] + list(mm.values())
            r[c] = dict(mean=mu, p0=wilcoxon(v)[1], n=len(v), mm=mm)
        xv, yv = within_pairs(col, "v"); xs, ys = within_pairs(col, "s")
        r["veh"], r["sbi"] = wilcoxon(yv, xv)[1], wilcoxon(ys, xs)[1]
        r["after"] = mannwhitneyu(SEGW[SEGW.cond == "si"][col].dropna(), SEGW[SEGW.cond == "vi"][col].dropna())[1]
        r["base"] = mannwhitneyu(SEGW[SEGW.cond == "sb"][col].dropna(), SEGW[SEGW.cond == "vb"][col].dropna())[1]
        R[col] = r
        top, lo = max(hi + [0]), min(hi + [0]); span = top - lo
        a.set_ylim(lo - 0.08 * span, top + 0.85 * span)
        for k, c in enumerate(CND4):
            a.text(k, top + 0.04 * span, st(r[c]["p0"]), ha="center", va="bottom", fontsize=9.5, color=CC[c] if c != "vb" else "0.45",
                   fontweight="bold" if r[c]["p0"] < 0.05 else "normal")
        bracket(a, 0, 1, top + 0.25 * span, st(r["veh"]), color=CC["vi"], fs=9.5, h=0.03 * span)
        bracket(a, 2, 3, top + 0.25 * span, st(r["sbi"]), color=CC["si"], fs=9.5, h=0.03 * span)
        bracket(a, 1, 3, top + 0.47 * span, st(r["after"]), fs=9.5, h=0.03 * span)
        bracket(a, 0, 2, top + 0.67 * span, st(r["base"]), color="0.45", fs=9.5, h=0.03 * span)
        a.axhline(0, color="0.4", lw=0.8)
        a.set_xticks(range(4), XT4, fontsize=9)
        day_labels(a)
        a.set_title("all 4 stimulus blocks" if col == "rise" else col.split("_")[1] + " block", fontsize=11.5)
    ax[0].set_ylabel("mean z in the block(s) minus\nthe no-stimulus start (15-180 s)", fontsize=10)
    legend2(fig)
    fig.savefig(fn); plt.close(fig)
    return R


def tracked_change():
    """the 185 tracked neurons: change after the injection (minus the same day's baseline) per segment, vehicle day and SBI-553 day"""
    T = TRK.drop_duplicates(["mouse", "day1_id", "day2_id"])
    W = SEGW.set_index(["mouse", "day", "phase", "cell"])
    rows = []
    for _, r in T.iterrows():
        o = dict(mouse=r.mouse)
        ok = True
        for dr, day, cid in (("veh", r.veh_day, r.veh_id), ("sbi", r.sbi_day, r.sbi_id)):
            try:
                a_, b_ = W.loc[(r.mouse, int(day), "pre", int(cid))], W.loc[(r.mouse, int(day), "post", int(cid))]
            except KeyError:
                ok = False; break
            for sg in SEGS + ["blocks"]:
                o[f"{dr}_{sg}"] = b_[sg] - a_[sg]
        if ok:
            rows.append(o)
    return pd.DataFrame(rows)


def fig_w2(fn):
    D = tracked_change()
    fig, a = plt.subplots(figsize=(12.3, 4.75), gridspec_kw=dict(left=0.07, right=0.995, top=0.78, bottom=0.14))
    R = {}
    groups = SEGS + ["blocks"]
    hi = []
    for i, sg in enumerate(groups):
        x0 = i * 3
        for k, dr, c in ((0, "veh", "vi"), (1, "sbi", "si")):
            v = D[f"{dr}_{sg}"].dropna()
            mu, se = v.mean(), v.std(ddof=1) / np.sqrt(len(v))
            a.bar(x0 + k, mu, yerr=se, color=CC[c], width=0.8, capsize=3, error_kw=dict(lw=1), zorder=2)
            mm = {m: g[f"{dr}_{sg}"].mean() for m, g in D.groupby("mouse")}
            mice_dots(a, x0 + k, mm, s=24)
            hi += [mu + se, mu - se] + list(mm.values())
            R[(sg, dr)] = dict(mean=mu, p0=wilcoxon(v)[1], mm=mm)
        R[(sg, "pair")] = wilcoxon(D[f"sbi_{sg}"], D[f"veh_{sg}"])[1]
        R[(sg, "mice")] = wilcoxon([R[(sg, "sbi")]["mm"][m] - R[(sg, "veh")]["mm"][m] for m in MICE])[1]
        R[(sg, "lower")] = sum(R[(sg, "sbi")]["mm"][m] < R[(sg, "veh")]["mm"][m] for m in MICE)
    top, lo = max(hi + [0]), min(hi + [0]); span = top - lo
    a.set_ylim(lo - 0.08 * span, top + 0.6 * span)
    for i, sg in enumerate(groups):
        x0 = i * 3
        for k, dr, c in ((0, "veh", "vi"), (1, "sbi", "si")):
            p = R[(sg, dr)]["p0"]
            a.text(x0 + k, top + 0.04 * span, st(p), ha="center", va="bottom", fontsize=10, color=CC[c],
                   fontweight="bold" if p < 0.05 else "normal")
        bracket(a, x0, x0 + 1, top + 0.22 * span, st(R[(sg, "pair")]) + (f" ({pf(R[(sg, 'pair')])})" if R[(sg, "pair")] < 0.05 else ""),
                fs=10, h=0.03 * span)
        a.text(x0 + 0.5, top + 0.42 * span, f"{R[(sg, 'lower')]}/4 mice lower\nmice p = {R[(sg, 'mice')]:.3f}", ha="center", va="bottom",
               fontsize=8.5, color="0.3")
    if len(groups) > 5:
        a.axvline(3 * 5 - 1, color="0.6", lw=0.8, ls=":")
    a.axhline(0, color="0.4", lw=0.8)
    a.set_xticks([i * 3 + 0.5 for i in range(len(groups))], ["no stimulus\n(15-180 s)"] + [f"{e}\nblock" for e in STIM] +
                 ["all 4 stimulus\nblocks"], fontsize=10.5)
    a.set_ylabel("change after the injection\n(mean z, minus the same day's baseline)", fontsize=10.5)
    h = [Patch(color=CC["vi"], label="vehicle day"), Patch(color=CC["si"], label="SBI-553 day")] + mouse_handles()
    fig.legend(handles=h, ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=10)
    fig.savefig(fn); plt.close(fig)
    return R, D


FTW = "Activity does not depend on event marks or windows; neurons pooled over 4 mice, uncorrected; drug order = day - S1"
fW1 = FIG + r"\sW1_rise_tests.png"; RW1 = fig_w1(fW1)
r_ = RW1["rise"]
slide("Q2  BLA ACTIVITY  (replaces folder-3 slides 31-33)", "Stimulus blocks raise activity above the session start: tested per neuron",
      f"Rise (blocks - start): vehicle day {r_['vb']['mean']:+.2f} → {r_['vi']['mean']:+.2f} z ({pf(r_['veh'])}), SBI-553 day "
      f"{r_['sb']['mean']:+.2f} → {r_['si']['mean']:+.2f} z ({pf(r_['sbi'])}); after vs after {pf(r_['after'])}",
      fW1, "Per neuron: mean z in the stimulus block(s) minus mean z in the no-stimulus start (15-180 s) of the same session; bars = mean ± "
           "SEM over neurons, markers = mice. Star over a bar = Wilcoxon vs 0; blue / red = same neurons, baseline vs injection session "
           "(Wilcoxon); black = Mann-Whitney after vehicle vs after SBI-553; grey = baseline vs baseline. Each day keeps its own baseline. "
           f"n = 4 mice. {STARS}",
      notes(results=" | ".join(f"{c_}: " + "; ".join(f"{CL[c]} {RW1[c_][c]['mean']:+.3f} z (vs 0 {pf(RW1[c_][c]['p0'])}, n = {RW1[c_][c]['n']})"
                                                     for c in CND4)
                               + f"; within day vehicle {pf(RW1[c_]['veh'])}, SBI-553 {pf(RW1[c_]['sbi'])}; after vs after {pf(RW1[c_]['after'])}; "
                                 f"baseline vs baseline {pf(RW1[c_]['base'])}" for c_ in RW1) + ".",
            caveat="the old slides 31-33 pooled both days' baselines and had no test; z is per session (dF/F / robust SD), so levels "
                   "between sessions are approximate; the rise is within a session."), "W1", FTW)

fW2 = FIG + r"\sW2_tracked_change_tests.png"; RW2, DW2 = fig_w2(fW2)
b_ = ("blocks",)
slide("Q2  BLA ACTIVITY  (replaces folder-3 slides 27-28)", "Same 185 neurons: activity change after the injection, vehicle day vs SBI-553 day",
      f"Stimulus blocks: vehicle day {RW2[('blocks', 'veh')]['mean']:+.2f}, SBI-553 day {RW2[('blocks', 'sbi')]['mean']:+.2f} z (paired "
      f"Wilcoxon {pf(RW2[('blocks', 'pair')])}); lower in {RW2[('blocks', 'lower')]}/4 mice (mice p = {RW2[('blocks', 'mice')]:.3f})",
      fW2, f"Each tracked neuron ({len(DW2)}): mean z after the injection minus its own mean z in the same day's baseline, per session part. "
           "Bars = mean ± SEM, markers = mice. Star over a bar = Wilcoxon vs 0; bracket = Wilcoxon paired by neuron, vehicle day vs SBI-553 "
           f"day; 'mice' = Wilcoxon over 4 mice (smallest possible p 0.125). {STARS}",
      notes(results=" | ".join(f"{sg}: vehicle day {RW2[(sg, 'veh')]['mean']:+.3f} z ({pf(RW2[(sg, 'veh')]['p0'])}), SBI-553 day "
                               f"{RW2[(sg, 'sbi')]['mean']:+.3f} ({pf(RW2[(sg, 'sbi')]['p0'])}), paired {pf(RW2[(sg, 'pair')])}, lower in "
                               f"{RW2[(sg, 'lower')]}/4 mice" for sg in SEGS + ["blocks"]) + ". Per mouse (stimulus blocks, paired Wilcoxon over "
                               "its tracked neurons): " + "; ".join(
                f"{m} {pf(wilcoxon(g.sbi_blocks, g.veh_blocks)[1])} (n = {len(g)})" for m, g in DW2.groupby("mouse")) + ".",
            caveat="the per-mouse heat maps of slides 23-26 show the same quantity per 1-s bin; their key-line changes are tested here "
                   "(per-mouse p in these notes). Drug order = day."), "W2", FTW)

# --- F34: activity of all neurons, every comparison tested (replaces folder-3 slide 34)
def load_rates():
    out = []
    for d in (1, 2):
        W = pd.read_csv(B + rf"\analysis\ca_day{d}\ca_d{d}_cells_whole_session.csv")
        W["mouse"] = W.mouse.astype(str); W["day"] = d
        num = sum(W[f"still rate | {s}"].fillna(0) * W[f"still min | {s}"].fillna(0) for s in STIM)
        den = sum(W[f"still min | {s}"].fillna(0) for s in STIM)
        W["rate"] = num / den.replace(0, np.nan)
        W["cond"] = [("s" if m in SBI[d] else "v") + ("b" if p == "pre" else "i") for m, p in zip(W.mouse, W.phase)]
        out.append(W[["mouse", "day", "phase", "cell", "cond", "rate"]])
    return pd.concat(out, ignore_index=True)


def fig_f34(fn):
    Rt = load_rates()
    fig, ax = plt.subplots(1, 2, figsize=(12.3, 4.75), gridspec_kw=dict(width_ratios=[1.5, 1], wspace=0.25, left=0.07, right=0.99,
                                                                         top=0.80, bottom=0.17))
    a = ax[0]
    rng = np.random.default_rng(0)
    top = 0
    for k, c in enumerate(CND4):
        v = Rt[Rt.cond == c].rate.dropna()
        a.boxplot([v], positions=[k], widths=0.55, showfliers=False, medianprops=dict(color="k", lw=2.5),
                  boxprops=dict(color=CC[c], lw=2), whiskerprops=dict(color=CC[c]), capprops=dict(color=CC[c]))
        for m in MICE:
            w = Rt[(Rt.cond == c) & (Rt.mouse == m)].rate.dropna()
            a.scatter(k + rng.uniform(-0.22, 0.22, len(w)), w, s=8, color=MC[m], alpha=0.45, lw=0)
        q1, q3 = np.percentile(v, [25, 75]); top = max(top, q3 + 1.5 * (q3 - q1))
    a.set_ylim(0, top * 1.55)
    pw = {}
    for k in ("v", "s"):
        x = Rt[Rt.cond == k + "b"].set_index(["mouse", "day", "cell"]).rate
        y = Rt[Rt.cond == k + "i"].set_index(["mouse", "day", "cell"]).rate
        j = pd.concat([x, y], axis=1, join="inner").dropna()
        pw[k] = wilcoxon(j.iloc[:, 1], j.iloc[:, 0])[1]
    pa = mannwhitneyu(Rt[Rt.cond == "si"].rate.dropna(), Rt[Rt.cond == "vi"].rate.dropna())[1]
    pb = mannwhitneyu(Rt[Rt.cond == "sb"].rate.dropna(), Rt[Rt.cond == "vb"].rate.dropna())[1]
    bracket(a, 0, 1, top * 1.03, st(pw["v"]), color=CC["vi"], h=0.02 * top)
    bracket(a, 2, 3, top * 1.03, st(pw["s"]), color=CC["si"], h=0.02 * top)
    bracket(a, 1, 3, top * 1.20, st(pa) + f" ({pf(pa)})", h=0.02 * top)
    bracket(a, 0, 2, top * 1.38, st(pb) + f" ({pf(pb)})", color="0.45", h=0.02 * top)
    a.set_xticks(range(4), [f"{x}\nn = {int((Rt.cond == c).sum())}" for x, c in zip(XT4, CND4)], fontsize=10)
    a.text(0.5, -0.20, "vehicle day", transform=a.get_xaxis_transform(), ha="center", va="top", fontsize=10.5, color=CC["vi"])
    a.text(2.5, -0.20, "SBI-553 day", transform=a.get_xaxis_transform(), ha="center", va="top", fontsize=10.5, color=CC["si"])
    a.set_ylabel("Ca transients / min\n(stimulus blocks, still frames)")
    a.set_title("A  all neurons pooled (dot = one neuron, box = median, IQR)", loc="left", fontsize=11.5)
    b = ax[1]
    for m in MICE:
        for k, dr in enumerate(("vehicle", "SBI-553")):
            r = RATE[(m, dr)]
            b.plot([2 * k, 2 * k + 1], [r["base_median"], r["inj_median"]], color=MC[m], marker=MK[m], ms=8, mec="white", lw=1.8)
            b.text(2 * k + 1.12, r["inj_median"], st(r["p_wilcoxon"]), color=MC[m], fontsize=9, va="center")
    b.set_xticks([0, 1, 2, 3], XT4, fontsize=10); b.set_xlim(-0.4, 3.7); b.set_ylim(bottom=0)
    b.text(0.5, -0.20, "vehicle day", transform=b.get_xaxis_transform(), ha="center", va="top", fontsize=10.5, color=CC["vi"])
    b.text(2.5, -0.20, "SBI-553 day", transform=b.get_xaxis_transform(), ha="center", va="top", fontsize=10.5, color=CC["si"])
    b.set_ylabel("median transients / min\n(of the mouse's neurons)")
    nl = sum(RATE[(m, "SBI-553")]["change_pct"] < RATE[(m, "vehicle")]["change_pct"] for m in MICE)
    pm = wilcoxon([RATE[(m, "SBI-553")]["change_pct"] - RATE[(m, "vehicle")]["change_pct"] for m in MICE])[1]
    b.set_title(f"B  each mouse (star = its own Wilcoxon)\nbigger drop on the SBI-553 day: {nl}/4 mice, mice p = {pm:.3f}", loc="left",
                fontsize=10.5)
    fig.legend(handles=cond_handles(), ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=10.5)
    fig.legend(handles=mouse_handles(), ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.94), frameon=False, fontsize=10)
    fig.savefig(fn); plt.close(fig)
    return dict(pw=pw, pa=pa, pb=pb, nl=nl, pm=pm, med={c: float(Rt[Rt.cond == c].rate.median()) for c in CND4})


fF34 = FIG + r"\sF34_rate_all_tests.png"; R34 = fig_f34(fF34)
slide("Q2  BLA ACTIVITY  (replaces folder-3 slide 34)", "Activity of all neurons: every comparison tested",
      f"Fell within both days ({pf(R34['pw']['v'])}, {pf(R34['pw']['s'])}); after SBI-553 < after vehicle ({pf(R34['pa'])}); baselines "
      + ("equal" if R34["pb"] >= 0.05 else "differ") + f" ({pf(R34['pb'])})",
      fF34, "A: blue / red = Wilcoxon, same neurons, baseline vs injection session; black = Mann-Whitney after vehicle vs after SBI-553; grey = "
            "Mann-Whitney baseline vs baseline. B: each mouse's median; star = that mouse's within-day Wilcoxon; 'mice' = Wilcoxon over 4 "
            f"mice on the % change (smallest possible p 0.125). {STARS}",
      notes(results=f"median transients / min: " + ", ".join(f"{CL[c]} {R34['med'][c]:.2f}" for c in CND4)
                    + f"; within day vehicle {pf(R34['pw']['v'])}, SBI-553 {pf(R34['pw']['s'])}; after vs after {pf(R34['pa'])}; baseline vs "
                      f"baseline {pf(R34['pb'])}. Per mouse: " + "; ".join(
                f"{m} vehicle day {RATE[(m, 'vehicle')]['change_pct']:.0f}% ({pf(RATE[(m, 'vehicle')]['p_wilcoxon'])}), SBI-553 day "
                f"{RATE[(m, 'SBI-553')]['change_pct']:.0f}% ({pf(RATE[(m, 'SBI-553')]['p_wilcoxon'])}); tracked difference "
                f"{pf(RNORM[m]['p_wilcoxon'])}" for m in MICE) + ".",
            caveat="drug order = day, so after vs after alone is not a drug effect; the same-neuron "
                   "comparison is on slide W2 / slide 2 panel C."), "F34", FT_RATE)


# --- F48: change in % UP / DOWN from baseline, 7 events, each bar tested (replaces folder-3 slide 48)
EV7 = STIM + REFL


def fig_f48(fn):
    fig, ax = plt.subplots(2, 1, figsize=(12.3, 4.75), gridspec_kw=dict(hspace=0.45, left=0.075, right=0.995, top=0.80, bottom=0.08))
    R = {}
    for i, col in enumerate(("UP", "DOWN")):
        a = ax[i]
        vals = []
        for g, ev in enumerate(EV7):
            C = cls_change(ev)
            for k, dr, c in ((0, "vehicle", "vi"), (1, "SBI-553", "si")):
                x = g * 3 + k
                v = 100 * C[dr]["d" + col].mean()
                a.bar(x, v, color=CC[c], width=0.8, hatch=None if col == "UP" else "///", lw=0, zorder=2)
                mm = {m: 100 * gg["d" + col].mean() for m, gg in C[dr].groupby(C[dr].mouse.astype(str))}
                mice_dots(a, x, mm, s=18)
                vals += [v] + list(mm.values())
                R[(col, ev, dr)] = dict(v=v, p=SG[ev][1][col + ("_veh" if dr == "vehicle" else "_sbi")])
            R[(col, ev, "between")] = mannwhitneyu(C["SBI-553"]["d" + col], C["vehicle"]["d" + col])[1]
        hi, lo = max(vals + [0]), min(vals + [0]); span = hi - lo
        a.set_ylim(lo - 0.05 * span, hi + 0.62 * span)
        for g, ev in enumerate(EV7):
            for k, dr, c in ((0, "vehicle", "vi"), (1, "SBI-553", "si")):
                p = R[(col, ev, dr)]["p"]
                a.text(g * 3 + k, hi + 0.05 * span, st(p), ha="center", va="bottom", fontsize=9, color=CC[c],
                       fontweight="bold" if p < 0.05 else "normal")
            bracket(a, g * 3, g * 3 + 1, hi + 0.30 * span, st(R[(col, ev, "between")]), fs=9.5, h=0.03 * span)
        a.axhline(0, color="0.4", lw=0.8)
        a.set_xticks([g * 3 + 0.5 for g in range(len(EV7))], EV7 if i == 1 else [""] * len(EV7), fontsize=10.5)
        a.set_ylabel(f"change in % {col}\n(after - same-day baseline)", fontsize=10)
        a.set_title("UP (activated)" if col == "UP" else "DOWN (inhibited)", loc="left", fontsize=11)
    h = [Patch(color=CC["vi"], label=f"vehicle day ({N_VEH})"), Patch(color=CC["si"], label=f"SBI-553 day ({N_SBI})")] + mouse_handles()
    fig.legend(handles=h, ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=9, columnspacing=0.8, handlelength=1.0)
    fig.savefig(fn); plt.close(fig)
    return R


fF48 = FIG + r"\sF48_updown_norm_tests.png"; R48 = fig_f48(fF48)
slide("Q3  RESPONSES: POPULATION  (replaces folder-3 slide 48)", "% UP and DOWN, baseline-normalized, every bar tested",
      "Star over a bar = did that day's % change (McNemar, same neurons); bracket = did the change differ between days",
      fF48, "Bar = change in % UP (solid) or % DOWN (hatched) from the same neurons' baseline that day; markers = mice. Star = McNemar within "
            "day; bracket = Mann-Whitney on each neuron's class change (-1 / 0 / +1), vehicle day vs SBI-553 day. This tests the change; slide "
            f"88's Fisher tests the after-injection % only, so stars can differ. n = 4 mice, neurons pooled. {STARS}",
      notes(results=" | ".join(f"{col} {ev}: vehicle {R48[(col, ev, 'vehicle')]['v']:+.1f} pp ({pf(R48[(col, ev, 'vehicle')]['p'])}), SBI-553 "
                               f"{R48[(col, ev, 'SBI-553')]['v']:+.1f} ({pf(R48[(col, ev, 'SBI-553')]['p'])}), between days "
                               f"{pf(R48[(col, ev, 'between')])}" for col in ("UP", "DOWN") for ev in EV7) + "."), "F48", FT_NEUR)

# --- annotated replacements: the original folder-3 figure + the missing tests written on the slide (10-26)
from PIL import Image as _PILImage  # noqa: E402

MX = B + r"\analysis\_variants\mix4"
NSD = B + r"\lab_meeting\decks_Day1_Day2\v2_20261022\NEW_STYLE_mix4\EN"


def ann_slide(section, title, key, png, tests, bottom, notes_txt, num, footer=FT3):
    """original figure (aspect kept) + a 'Tests added' box with one line per missing comparison"""
    s = prs.slides.add_slide(BL)
    tb(s, 0.45, 0.10, 12, 0.3, section, 11, color=(0x55, 0x55, 0x55))
    tb(s, 0.45, 0.32, 12.4, 0.6, title, fit_pt(title, 24, False, 12.15, 20))
    if key:
        tb(s, 0.45, 0.92, 12.4, 0.5, key, fit_pt(key, 14.5, True, 12.15, 12.5), bold=True, color=(0xC0, 0x39, 0x2B))
    w, h = _PILImage.open(png).size
    prop = FontProperties(family="Arial", weight="bold")
    nl = sum(int(np.ceil(max(TextPath((0, 0), t_, size=11.5, prop=prop).get_extents().width / 72, 0.1) / 11.9)) for t_ in tests)
    box_h = 0.205 * nl + 0.15
    H = 6.30 - 1.45 - box_h - 0.05
    W = min(12.3, H * w / h); H = W * h / w
    s.shapes.add_picture(png, Inches(0.5 + (12.3 - W) / 2), Inches(1.45), Inches(W))
    tb(s, 0.45, 1.45 + H + 0.03, 12.4, box_h, "\n".join(tests), 11.5, bold=True, color=(0x1B, 0x4F, 0x72))
    if bottom:
        tb(s, 0.45, 6.38, 12.4, 0.65, bottom, 11.5)
    tb(s, 0.45, 7.08, 11.3, 0.3, footer, 9.5, color=(0x88, 0x88, 0x88), italic=True)
    tb(s, 12.3, 7.08, 0.8, 0.3, str(num), 10, color=(0x88, 0x88, 0x88), align=PP_ALIGN.RIGHT)
    s.notes_slide.notes_text_frame.text = notes_txt
    return s


ANN = []                                                       # (file name, slide number in this deck)


def add_ann(fname, *a, **k):
    ann_slide(*a, **k)
    ANN.append((fname, len(prs.slides)))


def whole_rates():
    out = []
    for d in (1, 2):
        W = pd.read_csv(B + rf"\analysis\ca_day{d}\ca_d{d}_cells_whole_session.csv")
        W["mouse"] = W.mouse.astype(str); W["day"] = d
        out.append(W[["mouse", "day", "phase", "cell", "rate"]])
    X = pd.concat(out).pivot_table(index=["mouse", "day", "cell"], columns="phase", values="rate").dropna().reset_index()
    X["l2"] = np.log2((X.post + 0.1) / (X.pre + 0.1))
    X["drug"] = [("SBI-553" if m in SBI[d] else "vehicle") for m, d in zip(X.mouse, X.day)]
    return X


WR = whole_rates()
TRU = TRK.drop_duplicates(["mouse", "day1_id", "day2_id"])


def fov_line(m):
    x = WR[WR.mouse == m]
    v, s_ = x[x.drug == "vehicle"].l2, x[x.drug == "SBI-553"].l2
    t = TRU[TRU.mouse == m]
    iv = x[x.drug == "vehicle"].set_index("cell").l2; is_ = x[x.drug == "SBI-553"].set_index("cell").l2
    pv = [(iv.get(int(a)), is_.get(int(b))) for a, b in zip(t.veh_id, t.sbi_id)]
    pv = [(a, b) for a, b in pv if a is not None and b is not None and np.isfinite(a) and np.isfinite(b)]
    ptr = wilcoxon([b for a, b in pv], [a for a, b in pv])[1] if len(pv) > 5 else np.nan
    return (f"{m}: vehicle day median {v.median():+.2f} (vs 0 {pf(wilcoxon(v)[1])}), SBI-553 day {s_.median():+.2f} (vs 0 "
            f"{pf(wilcoxon(s_)[1])}); days differ: all neurons Mann-Whitney {pf(mannwhitneyu(s_, v)[1])}, same {len(pv)} tracked "
            f"neurons Wilcoxon {pf(ptr)}")


for part, mm, n_ in ((1, ("5941", "5939"), 35), (2, ("5937", "1191"), 36)):
    add_ann(f"15_FIXED_slide{n_}_FOV_change_tests", f"Q2  BLA ACTIVITY  (replaces folder-3 slide {n_})",
            f"Where in the field of view activity changed ({part}/2: {mm[0]} and {mm[1]}), with tests",
            "Tests added: each map vs 0 (Wilcoxon over its neurons) and vehicle day vs SBI-553 day (all neurons; same tracked neurons)",
            NSD + rf"\Day1+2_PI_figures\f05b_fov_change_m{part}.png", ["Tests added:"] + [fov_line(m) for m in mm],
            "Each neuron's footprint coloured by log2((injection + 0.1) / (same-day baseline + 0.1)) of its whole-session transient rate. "
            f"Neurons pooled within a mouse; exploratory, uncorrected. {STARS}",
            notes(tests="Wilcoxon signed-rank of the per-neuron log2 change vs 0; Mann-Whitney vehicle-day vs SBI-553-day neurons; paired "
                        "Wilcoxon on the tracked neurons (the same cell on both days)."), f"F{n_}", footer=FT_RATE)

tp = TR.loc["Pin prick"]
add_ann("16_FIXED_slide49_pin_four_sessions_tests", "Q3  RESPONSES: POPULATION  (replaces folder-3 slide 49)",
        "Pin prick: the same neurons in all four sessions, with tests",
        f"Same {int(tp.n)} neurons: after vehicle vs its baseline {pf(tp.p_vb_vi)}, after SBI-553 vs its baseline {pf(tp.p_sb_si)}; "
        f"the two baselines differ ({pf(tp.p_vb_sb)})",
        MX + r"\tracked_crossover\TX1_heat_pin.png",
        ["Tests added (paired Wilcoxon on the response R of the same neurons):",
         f"vehicle day baseline → after vehicle {pf(tp.p_vb_vi)};  SBI-553 day baseline → after SBI-553 {pf(tp.p_sb_si)};  "
         f"baseline vs baseline {pf(tp.p_vb_sb)};  after SBI-553 vs after vehicle {pf(tp.p_after)}"],
        f"One row = one neuron followed on both days. The colour scale of this older figure subtracts the 3 s before the event (display "
        f"only); the tests use the response window minus the 0.5 s before. n = {int(tp.n)} neurons, 4 mice. {STARS}",
        notes(tests="paired Wilcoxon over the tracked neurons on R (0-0.5 s minus the 0.5 s before)."), "F49", footer=FT_NEUR)

tpin = TRK[TRK.event == "Pin prick"]
add_ann("17_FIXED_slide56_pin_normalized_tests", "Q3  RESPONSES: POPULATION  (replaces folder-3 slide 56)",
        "Pin prick, normalized to the same-day baseline, with every test",
        f"Change vs 0: vehicle day {pf(wilcoxon(tpin.chg_veh)[1])}, SBI-553 day {pf(wilcoxon(tpin.chg_sbi)[1])}; difference "
        f"{pf(wilcoxon(tpin.d_norm)[1])}",
        MX + r"\tracked_crossover\TX4_heatnorm_pin.png",
        ["Tests added (same tracked neurons, Wilcoxon):",
         f"vehicle-day change vs 0 {pf(wilcoxon(tpin.chg_veh)[1])};  SBI-553-day change vs 0 {pf(wilcoxon(tpin.chg_sbi)[1])};  "
         f"SBI-553-day change - vehicle-day change {pf(wilcoxon(tpin.d_norm)[1])}"],
        f"n = {len(tpin)} tracked neurons, 4 mice. {STARS}", notes(tests="Wilcoxon signed-rank."), "F56", footer=FT_NEUR)

# 169: female vs male change
RT = load_rates()
J = []
for k in ("v", "s"):
    x = RT[RT.cond == k + "b"].set_index(["mouse", "day", "cell"]).rate
    y = RT[RT.cond == k + "i"].set_index(["mouse", "day", "cell"]).rate
    j = pd.concat([x, y], axis=1, join="inner").dropna().reset_index()
    j.columns = ["mouse", "day", "cell", "b", "i"]
    j["l2"] = np.log2((j.i + 0.1) / (j.b + 0.1)); j["k"] = k
    J.append(j)
J = pd.concat(J)
J["sex"] = J.mouse.map(SEX)
l169 = []
for k, lab in (("v", "vehicle day"), ("s", "SBI-553 day")):
    f_, m_ = J[(J.k == k) & (J.sex == "F")].l2, J[(J.k == k) & (J.sex == "M")].l2
    l169.append(f"{lab}: female median {f_.median():+.2f} (n = {len(f_)}) vs male {m_.median():+.2f} (n = {len(m_)}), Mann-Whitney "
                f"{pf(mannwhitneyu(f_, m_)[1])}")
add_ann("18_FIXED_slide169_sex_change_tests", "Q5  SEX  (replaces folder-3 slide 169)", "Activity of every neuron by sex, with the B test",
        "Panel B now tested female vs male (neurons pooled over 2 mice per sex: candidates only)",
        MX + r"\sex_analysis\S3_rates_by_sex.png", ["Tests added (panel B, log2 change of each neuron, female vs male):"] + l169,
        f"Exploratory: 2 vs 2 mice, so a sex difference cannot be inferred; neurons pooled. {STARS}",
        notes(tests="Mann-Whitney over neurons of log2((injection + 0.1) / (baseline + 0.1)) block transient rate."), "F169",
        footer=FT_RATE)


# 175: per-mouse key findings - responsive (UP or DOWN) change, tested
def resp_change(ev):
    e = cc[cc.event == ev].copy()
    e["R_"] = e.UP | e.DOWN
    out = {}
    for dr in ("vehicle", "SBI-553"):
        j = pairs(e[e.drug == dr])
        out[dr] = j.R__a.astype(int) - j.R__b.astype(int)
        out[dr + "_p"] = mcnemar(e[e.drug == dr].assign(UP=e.R_), "UP")[0]
        out[dr + "_m"] = {m: 100 * g.mean() for m, g in out[dr].groupby(level=0)}
    out["between"] = mannwhitneyu(out["SBI-553"], out["vehicle"])[1]
    out["mice"] = wilcoxon([out["SBI-553_m"][m] - out["vehicle_m"][m] for m in MICE])[1]
    out["lower"] = sum(out["SBI-553_m"][m] < out["vehicle_m"][m] for m in MICE)
    return out


l175 = ["Tests added (% responsive = UP or DOWN; change from the same day's baseline):"]
for ev in ("Pin prick", "Heat", "movement onset"):
    r = resp_change(ev)
    l175.append(f"{ev}: within day vehicle {pf(r['vehicle_p'])}, SBI-553 {pf(r['SBI-553_p'])} (McNemar); SBI-553-day change < vehicle-day "
                f"change in {r['lower']}/4 mice, mice Wilcoxon {pf(r['mice'])}; neurons Mann-Whitney {pf(r['between'])}")
add_ann("19_FIXED_slide175_check4_neurons_tests", "HOW SURE?  (replaces folder-3 slide 175)",
        "Check 4: do the key findings hold in every mouse? (1/2: neurons), with tests",
        "Tests added for the key-line claims (mouse level + neurons pooled)", MX + r"\event_analysis\validation\V4_key_findings_per_mouse_p1.png",
        l175, f"Mouse-level Wilcoxon over 4 mice cannot go below p = 0.125. {STARS}", notes(tests="as on the slide."), "F175", footer=FT_NEUR)

l176 = ["Tests added (after SBI-553 vs after vehicle):"]
for beh, stim in (("flinch", "Pin prick"), ("movement", "Light touch"), ("escape/rearing", "Heat")):
    pv, _ = beh_pct(beh, stim, "vi"); ps, _ = beh_pct(beh, stim, "si")
    pmw = wilcoxon([ps[m] - pv[m] for m in MICE])[1]
    l176.append(f"{stim} → {beh}: CMH over deliveries {pf(cmh(beh, stim))}; lower in {sum(ps[m] < pv[m] for m in MICE)}/4 mice, "
                f"Wilcoxon over 4 mice {pf(pmw)}")
add_ann("20_FIXED_slide176_check4_behaviour_tests", "HOW SURE?  (replaces folder-3 slide 176)",
        "Check 4: do the key findings hold in every mouse? (2/2: behaviour), with tests", "Exact mouse-level p and the delivery-level CMH",
        MX + r"\event_analysis\validation\V4_key_findings_per_mouse_p2.png", l176,
        f"4/4 mice in the same direction gives the smallest possible Wilcoxon p, 0.125. {STARS}", notes(tests="CMH by mouse; Wilcoxon."),
        "F176", footer=FT_BEH)

pm29 = wilcoxon([RATE[(m, "SBI-553")]["change_pct"] - RATE[(m, "vehicle")]["change_pct"] for m in MICE])[1]
add_ann("21_FIXED_slide29_rate_per_mouse_tests", "Q2  BLA ACTIVITY  (replaces folder-3 slide 29)",
        "Stimulus-block firing per mouse, with the day comparison tested", "The 3/4-mice claim tested at the mouse level and per mouse",
        MX + r"\pi_extra\session\RATE_blocks.png",
        ["Tests added:", f"bigger drop on the SBI-553 day in {sum(RATE[(m, 'SBI-553')]['change_pct'] < RATE[(m, 'vehicle')]['change_pct'] for m in MICE)}"
                         f"/4 mice: Wilcoxon over 4 mice {pf(pm29)};  same neurons, SBI-553-day vs vehicle-day change: " + ", ".join(
             f"{m} {pf(RNORM[m]['p_wilcoxon'])}" for m in MICE)],
        f"Mouse-level tests with 4 mice cannot reach p < 0.05. {STARS}", notes(tests="Wilcoxon."), "F29", footer=FT_RATE)

add_ann("22_FIXED_slide37_population_stimuli_tests", "Q3  RESPONSES: POPULATION  (replaces folder-3 slide 37)",
        "Population response to each stimulus, with the baseline comparison", "Baseline vs baseline added (the key line relies on it)",
        NSD + r"\Day1+2_PI_figures\f06_population_stimuli.png",
        ["Tests added (Mann-Whitney, vehicle-day baseline vs SBI-553-day baseline):", "  ".join(f"{ev} {pf(PO[ev][1]['base'])};" for ev in STIM)],
        f"Same data as PI slide 3 (grey brackets). {STARS}", notes(tests="Mann-Whitney over neurons."), "F37", footer=FT_NEUR)


def fisher_resp_base(ev):
    R = SG[ev][0]
    kv = R["vb"]["kUP"] + R["vb"]["kDOWN"]; ks = R["sb"]["kUP"] + R["sb"]["kDOWN"]
    return 100 * kv / R["vb"]["n"], 100 * ks / R["sb"]["n"], fisher_exact([[kv, R["vb"]["n"] - kv], [ks, R["sb"]["n"] - ks]])[1]


add_ann("23_FIXED_slide72_responsive_behaviours_tests", "Q3  RESPONSES: SINGLE NEURONS  (replaces folder-3 slide 72)",
        "Responsive neurons, behaviours, with the baseline comparison", "Baseline vs baseline added (Fisher)",
        MX + r"\event_analysis\P3a_responsive_pct_beh1.png",
        ["Tests added (% responsive at baseline, vehicle day vs SBI-553 day, Fisher):",
         "  ".join(f"{ev} {a_:.0f}% vs {b_:.0f}% {pf(p_)};" for ev in REFL + ["movement onset"] for a_, b_, p_ in [fisher_resp_base(ev)])],
        f"{STARS}", notes(tests="Fisher exact on responsive (UP or DOWN) neurons."), "F72", footer=FT_NEUR)

e = cc[cc.event == "Pin prick"]
jj = pairs(e)
l77 = ["Tests added (pin prick; neurons in a class at baseline, vehicle day vs SBI-553 day, Fisher):"]
for col, lab in (("DOWN", "DOWN"), ("UP", "UP")):
    seg = []
    for dr in ("vehicle", "SBI-553"):
        x = jj[(jj.drug_b == dr) & jj[col + "_b"]]
        seg.append((int(x[col + "_a"].sum()), len(x)))
    p_ = fisher_exact([[seg[0][0], seg[0][1] - seg[0][0]], [seg[1][0], seg[1][1] - seg[1][0]]])[1]
    l77.append(f"{lab} at baseline that stayed {lab}: vehicle day {seg[0][0]}/{seg[0][1]}, SBI-553 day {seg[1][0]}/{seg[1][1]}, {pf(p_)}")
add_ann("24_FIXED_slide77_pin_flow_tests", "Q3  RESPONSES: SINGLE NEURONS  (replaces folder-3 slide 77)",
        "Pin prick: where each class went, with the DOWN-origin test", "DOWN-origin transitions tested (they had none)",
        MX + r"\pi_extra\flow\FLOW_Pin_prick_pct.png", l77, f"{STARS}", notes(tests="Fisher exact."), "F77", footer=FT_NEUR)

l118 = ["Tests (McNemar within day; Fisher after vehicle vs after SBI-553) - the summary table pooled both baselines:"]
for ev in STIM:
    T_ = SG[ev][1]
    l118.append(f"{ev}: UP {st(T_['UP_veh'])} / {st(T_['UP_sbi'])} / {st(T_['UP_after'])};  DOWN {st(T_['DOWN_veh'])} / {st(T_['DOWN_sbi'])} / "
                f"{st(T_['DOWN_after'])}   (vehicle day / SBI-553 day / after vs after)")
add_ann("25_FIXED_slide118_summary_table_tests", "Q3  RESPONSES: SINGLE NEURONS  (replaces folder-3 slide 118)", "Summary table, with tests",
        "Every UP / DOWN comparison marked; see slide F70 for the bars", NSD + r"\Day1+2_FULL_figures\f09b_summary_table.png", l118,
        f"{STARS}", notes(results=" | ".join(fmt_single(ev) for ev in STIM)), "F118", footer=FT_NEUR)

add_ann("26_FIXED_slide132_pin_every_neuron_tests", "Q3  RESPONSES: SINGLE NEURONS  (replaces folder-3 slide 132)",
        "Pin prick: every neuron in the three conditions, with the within-day test", "Within-day baseline vs injection added (Wilcoxon)",
        NSD + r"\Day1+2_PI_figures\f08_heatmap_every_neuron.png",
        ["Tests added (pin prick R, same neurons within a day, Wilcoxon):",
         f"vehicle day {pf(PO['Pin prick'][1]['veh'])};  SBI-553 day {pf(PO['Pin prick'][1]['sbi'])};  after vs after (Mann-Whitney) "
         f"{pf(PO['Pin prick'][1]['after'])};  baseline vs baseline {pf(PO['Pin prick'][1]['base'])}"],
        f"The older figure pools both baselines; here each day is tested against its own baseline. {STARS}", notes(tests="Wilcoxon; Mann-Whitney."),
        "F132", footer=FT_NEUR)


def mouse_mcn(ev, m, col):
    out = []
    for dr in ("vehicle", "SBI-553"):
        x = cc[(cc.event == ev) & (cc.mouse == m) & (cc.drug == dr)]
        p_, b01, b10 = mcnemar(x, col)
        j = pairs(x)
        out.append(f"{int(j[col + '_b'].sum())}→{int(j[col + '_a'].sum())} {pf(p_)}")
    return out


for part, mm, n_ in ((1, ("5941", "5939"), 158), (2, ("5937", "1191"), 159)):
    add_ann(f"27_FIXED_slide{n_}_escape_maps_tests", f"Q4  BEHAVIOUR NEURONS  (replaces folder-3 slide {n_})",
            f"Escape / rearing neurons per mouse ({part}/2), with per-mouse tests", "Per-mouse UP count changes tested (McNemar)",
            NSD + rf"\Day1+2_PI_figures\f10_maps_m{part}.png",
            ["Tests added (escape / rearing onset UP neurons, baseline → injection, McNemar, same neurons):"] +
            [f"{m}: vehicle day {mouse_mcn('escape/rearing onset', m, 'UP')[0]};  SBI-553 day {mouse_mcn('escape/rearing onset', m, 'UP')[1]}"
             for m in mm], f"{STARS}", notes(tests="exact McNemar per mouse."), f"F{n_}", footer=FT_NEUR)

eo = ENC[ENC.condition.isin(["vehicle", "SBI-553"])]
ct = pd.crosstab(eo.condition, eo.type)
from scipy.stats import chi2_contingency as _chi2c  # noqa: E402
p163 = _chi2c(ct.values)[1]
sv = ((eo.condition == "vehicle") & (eo.type == "stimulus")).sum(); ss = ((eo.condition == "SBI-553") & (eo.type == "stimulus")).sum()
nv, ns_ = (eo.condition == "vehicle").sum(), (eo.condition == "SBI-553").sum()
add_ann("28_FIXED_slide163_encoding_tests", "Q4  ENCODING  (replaces folder-3 slide 163)",
        "What each neuron follows (C, D), with the condition test", "Panel D condition differences tested",
        MX + r"\event_analysis\O1_near_simultaneous_p2.png",
        ["Tests added (neuron-sessions, after vehicle vs after SBI-553):",
         f"all classes: chi-square {pf(p163)};  'stimulus only' {sv}/{nv} ({100 * sv / nv:.1f}%) vs {ss}/{ns_} ({100 * ss / ns_:.1f}%), Fisher "
         f"{pf(fisher_exact([[sv, nv - sv], [ss, ns_ - ss]])[1])}"],
        f"Exploratory, neuron-sessions pooled. {STARS}", notes(tests="chi-square of the class x condition table; Fisher exact."), "F163",
        footer=FT_NEUR)


def overlap_test(c, g1=("Light touch", "Mild touch"), g2=("Pin prick", "Heat")):
    """activated (UP) neurons of group 1 and group 2 in one condition: overlap vs independence (hypergeometric, upper tail)"""
    from scipy.stats import hypergeom
    x = cc[cc.cond == c].assign(R_=lambda d: d.UP.astype(float))
    g = x.pivot_table(index=["mouse", "day", "phase", "cell"], columns="event", values="R_", aggfunc="first").fillna(0) > 0
    t = g[list(g1)].any(axis=1); n_ = g[list(g2)].any(axis=1)
    N, K, nn, k = len(g), int(t.sum()), int(n_.sum()), int((t & n_).sum())
    return k, K * nn / N, float(hypergeom.sf(k - 1, N, K, nn))


add_ann("29_FIXED_slide167_overlap_tests", "Q4  TOUCH AND NOXIOUS  (replaces folder-3 slide 167)",
        "Touch / noxious overlap, tested against independence", "Observed overlap vs chance (hypergeometric, one-sided)",
        MX + r"\pi_extra\nature_style\NS3_overlap.png",
        ["Tests added (activated neurons in both groups, observed vs expected if independent, hypergeometric):",
         "A touch & noxious - " + "  ".join(f"{CL[c]}: {k} vs {e_:.1f}, {pf(p_)};" for c in CND4 for k, e_, p_ in [overlap_test(c)]),
         "B stimulus & reflex - " + "  ".join(f"{CL[c]}: {k} vs {e_:.1f}, {pf(p_)};" for c in CND4
                                              for k, e_, p_ in [overlap_test(c, tuple(STIM), ("withdrawal", "flinch"))])],
        f"{STARS}", notes(tests="hypergeometric upper tail."), "F167", footer=FT_NEUR)

for m in MICE:
    add_ann(f"30_FIXED_slide{182 + MICE.index(m)}_maps_{m}_tests", f"Q3  MAPS  (replaces folder-3 slide {182 + MICE.index(m)})",
            f"{m}: where the responsive neurons are, with per-mouse count tests", "Per-mouse UP / DOWN count changes tested (McNemar)",
            MX + rf"\pi_extra\space\MAPS_{m}.png",
            [f"Tests added ({mlab(m)}; baseline → injection, same neurons, McNemar):"] +
            [f"{ev}: UP vehicle day {mouse_mcn(ev, m, 'UP')[0]}, SBI-553 day {mouse_mcn(ev, m, 'UP')[1]};  DOWN {mouse_mcn(ev, m, 'DOWN')[0]}, "
             f"{mouse_mcn(ev, m, 'DOWN')[1]}" for ev in STIM], f"{STARS}", notes(tests="exact McNemar per mouse."),
            f"F{182 + MICE.index(m)}", footer=FT_NEUR)

json.dump(ANN, open(OUT + r"\ann_slides.json", "w"))

# --- T33: the same tracked neurons, % UP / DOWN in the 4 sessions + UP-neuron traces, z - 0.5 s (replaces slide 33 of the PI deck)
def mcn_pair(a, b):
    """exact McNemar on two boolean vectors of the same neurons"""
    b01 = int((~a & b).sum()); b10 = int((a & ~b).sum())
    return binomtest(b01, b01 + b10).pvalue if b01 + b10 else 1.0


def tracked_sessions(ev):
    t = TRK[TRK.event == ev]
    S = {"vb": [], "vi": [], "sb": [], "si": []}
    keys = {"vb": ("veh_day", "pre", "veh_id"), "vi": ("veh_day", "post", "veh_id"), "sb": ("sbi_day", "pre", "sbi_id"),
            "si": ("sbi_day", "post", "sbi_id")}
    for c, (dcol, ph, icol) in keys.items():
        for m, g in t.groupby("mouse"):
            d = int(g[dcol].iloc[0])
            v = PSTH.get((m, d, ph, ev))
            if v is None:
                continue
            pos = {int(i): k for k, i in enumerate(v["ids"])}
            up = g[g["cls_" + c] == "activated"][icol].astype(int)
            cols = [pos[i] for i in up if i in pos]
            if cols:
                S[c].append(base_sub(v["mean"][:, cols], v["tt"], ev))
            tt = v["tt"]
    return t, {c: (np.hstack(x) if x else np.zeros((len(tt), 0))) for c, x in S.items()}, tt


def fig_t33(fn):
    fig, ax = plt.subplots(2, 4, figsize=(12.3, 4.75), gridspec_kw=dict(wspace=0.22, hspace=0.62, left=0.075, right=0.995, top=0.80,
                                                                         bottom=0.09, height_ratios=[1.25, 1.0]))
    R = {}
    for j, ev in enumerate(STIM):
        t, S, tt = tracked_sessions(ev)
        a = ax[0, j]
        U = {c: t["cls_" + c].eq("activated").values for c in CND4}
        Dn = {c: t["cls_" + c].eq("inhibited").values for c in CND4}
        for k, c in enumerate(CND4):
            a.bar(k, 100 * U[c].mean(), color=CC[c], width=0.75, zorder=2)
            a.bar(k, -100 * Dn[c].mean(), color=CC[c], width=0.75, hatch="///", alpha=0.75, lw=0, zorder=2)
            for m, g in t.groupby("mouse"):
                a.scatter(k + OFF[m] * 0.75, 100 * g["cls_" + c].eq("activated").mean(), s=16, marker=MK[m], color=MC[m],
                          edgecolor="white", lw=0.5, zorder=4)
                a.scatter(k + OFF[m] * 0.75, -100 * g["cls_" + c].eq("inhibited").mean(), s=16, marker=MK[m], color=MC[m],
                          edgecolor="white", lw=0.5, zorder=4)
        r = dict(up_veh=mcn_pair(U["vb"], U["vi"]), up_sbi=mcn_pair(U["sb"], U["si"]), up_after=mcn_pair(U["vi"], U["si"]),
                 dn_veh=mcn_pair(Dn["vb"], Dn["vi"]), dn_sbi=mcn_pair(Dn["sb"], Dn["si"]), dn_after=mcn_pair(Dn["vi"], Dn["si"]),
                 pct={c: 100 * U[c].mean() for c in CND4}, n=len(t))
        R[ev] = r
        u = max(30, max(r["pct"].values()) * 1.15); d = max(20, 100 * max(Dn[c].mean() for c in CND4) * 1.15)
        a.set_ylim(-d * 1.75, u * 1.45)
        bracket(a, 0, 1, u * 0.95, st(r["up_veh"]), color=CC["vi"], fs=9.5, h=0.03 * u)
        bracket(a, 2, 3, u * 0.95, st(r["up_sbi"]), color=CC["si"], fs=9.5, h=0.03 * u)
        bracket(a, 1, 3, u * 1.20, st(r["up_after"]), fs=9.5, h=0.03 * u)
        bracket_down(a, 0, 1, -d * 1.1, st(r["dn_veh"]), color=CC["vi"], fs=9.5, h=0.03 * d)
        bracket_down(a, 2, 3, -d * 1.1, st(r["dn_sbi"]), color=CC["si"], fs=9.5, h=0.03 * d)
        bracket_down(a, 1, 3, -d * 1.38, st(r["dn_after"]), fs=9.5, h=0.03 * d)
        a.axhline(0, color="k", lw=0.8)
        a.set_xticks(range(4), XT4, fontsize=8.5)
        a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{abs(v):.0f}"))
        pre, a_, b_ = win(ev)
        a.set_title(f"{ev}  ({a_:g}-{b_:g} s)", fontsize=11.5)
        b = ax[1, j]
        sel = (tt >= -2) & (tt <= 4)
        for c in CND4:
            Y = S[c][sel]
            if Y.shape[1] < 2:
                continue
            mu, se = Y.mean(1), Y.std(1, ddof=1) / np.sqrt(Y.shape[1])
            b.fill_between(tt[sel], mu - se, mu + se, color=CC[c], alpha=0.18, lw=0)
            b.plot(tt[sel], mu, color=CC[c], lw=1.4, label=f"{CL[c]} (UP n = {Y.shape[1]})")
        b.axvspan(min(a_, 0) - pre, min(a_, 0), color="#d6eaf8", lw=0, zorder=0); b.axvspan(a_, b_, color="#e5e7e9", lw=0, zorder=0)
        b.axvline(0, color="k", lw=0.7, ls="--"); b.axhline(0, color="0.5", lw=0.6)
        b.set_xlabel("time from event (s)", fontsize=9.5); b.tick_params(labelsize=8.5)
        b.text(0.02, 0.97, "UP n: " + " / ".join(str(S[c].shape[1]) for c in CND4), transform=b.transAxes, fontsize=8, va="top", color="0.3")
    ax[0, 0].set_ylabel("% of the tracked neurons\nUP (up) / DOWN (hatched, down)", fontsize=9.5)
    ax[1, 0].set_ylabel("UP neurons: z - mean\nof the 0.5 s before", fontsize=9.5)
    legend2(fig, hatch=True)
    fig.savefig(fn); plt.close(fig)
    return R


fT33 = FIG + r"\sT33_tracked_updown.png"; R33 = fig_t33(fT33)
rp = R33["Pin prick"]
slide("4  STIMULUS RESPONSES  -  the same tracked neurons",
      f"Same {rp['n']} neurons: fewer pin-prick UP neurons after SBI-553 ({rp['pct']['si']:.0f}%) than after vehicle ({rp['pct']['vi']:.0f}%)",
      f"Pin: after vs after McNemar {pf(rp['up_after'])}; light-touch and heat UP rose after SBI-553 too",
      fT33, f"Top: % UP (up) / DOWN (hatched, down) of the {rp['n']} neurons found on both days; markers = mice. All brackets = McNemar (same "
            f"neurons): blue / red = within day, black = after vehicle vs after SBI-553. Bottom: mean trace of the UP neurons of each session. "
            f"n = 4 mice. {STARS}",
      notes(results=" | ".join(f"{ev}: % UP vb {r['pct']['vb']:.0f}, vi {r['pct']['vi']:.0f}, sb {r['pct']['sb']:.0f}, si {r['pct']['si']:.0f}; "
                               f"UP within day vehicle {pf(r['up_veh'])}, SBI-553 {pf(r['up_sbi'])}, after vs after {pf(r['up_after'])}; DOWN "
                               f"{pf(r['dn_veh'])}, {pf(r['dn_sbi'])}, {pf(r['dn_after'])}" for ev, r in R33.items()) + ".",
            measure="traces: trial-mean z of each UP neuron minus its own mean in the 0.5 s before the event (the test's baseline).",
            caveat="tracked neurons are the same cells on both days, so the after-vs-after comparison is paired (McNemar), unlike the "
                   "all-neuron slides (Fisher)."), "T33", FT_NEUR)
json.dump(dict(T33=len(prs.slides)), open(OUT + r"\t33_slide.json", "w"))

out = OUT + r"\PI5_mixmarks_EN_v3.pptx"          # 10-26: the earlier file was open in PowerPoint; earlier files are kept
prs.save(out)
print("->", out, len(prs.slides), "slides")
