"""make_pi5_deck.py - 10-26 (user): the folder-3 results (event marks AND response window chosen per event; exploratory) for the PI
in 5 compact slides + supplement slides (methods, numbers / tests, per mouse, reflex / movement neurons). Every number is computed
here from analysis\\_variants\\mix4 (no hand-typed values); methods, n and exact statistics are in the slide notes and S1-S4.

  python make_pi5_deck.py        -> lab_meeting\\decks_Day1_Day2\\v2_20261022\\FINAL_20261026_PI5_mixmarks\\PI5_mixmarks_EN.pptx
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest, fisher_exact, mannwhitneyu, wilcoxon

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
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
                     "savefig.dpi": 200})
MICE = ["5941", "5939", "5937", "1191"]
SBI = {1: {"5941", "5939"}, 2: {"5937", "1191"}}
SBIDAY = {"5941": 1, "5939": 1, "5937": 2, "1191": 2}
MC = {"5941": "#e67e22", "5939": "#8e44ad", "5937": "#16a085", "1191": "#2c3e50"}
MK = {"5941": "o", "5939": "s", "5937": "^", "1191": "D"}
SEX = {"5941": "F", "5939": "M", "5937": "M", "1191": "F"}
CC = {"vb": "#b3b3b3", "vi": "#2471a3", "sb": "#6e6e6e", "si": "#c0392b"}
CL = {"vb": "baseline\n(veh day)", "vi": "after\nvehicle", "sb": "baseline\n(SBI day)", "si": "after\nSBI-553"}
STIM = ["Light touch", "Mild touch", "Pin prick", "Heat"]
REFL = ["withdrawal", "flinch", "movement onset"]
MARKS = {"Heat": "original marks", "withdrawal": "original marks"}
TAG = ("Exploratory: event marks and response window chosen per event (heat, withdrawal = original key presses); "
       "neuron-pooled p, uncorrected (S1, S2)")


def st(p):
    return "ns" if p is None or not np.isfinite(p) or p >= 0.05 else "***" if p < 0.001 else "**" if p < 0.01 else "*"


def pf(p):
    return "n.a." if p is None or not np.isfinite(p) else "p < 0.001" if p < 0.001 else f"p = {p:.3f}" if p < 0.1 else f"p = {p:.2f}"


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
    return f"{a:g}-{b:g} s vs {pre:g} s before"


def mcnemar(ev, dr, col):
    x = cc[(cc.event == ev) & (cc.drug == dr)]
    j = x[x.phase == "pre"].set_index(["mouse", "day", "cell"]).join(
        x[x.phase == "post"].set_index(["mouse", "day", "cell"]), lsuffix="_b", rsuffix="_a", how="inner")
    b10 = int((j[col + "_b"] & ~j[col + "_a"]).sum()); b01 = int((~j[col + "_b"] & j[col + "_a"]).sum())
    return (binomtest(b01, b01 + b10).pvalue if b01 + b10 else 1.0), len(j)


def wsr(ev, dr):
    x = cc[(cc.event == ev) & (cc.drug == dr)]
    j = x[x.phase == "pre"].set_index(["mouse", "day", "cell"]).join(
        x[x.phase == "post"].set_index(["mouse", "day", "cell"]), lsuffix="_b", rsuffix="_a", how="inner")
    dd = (j.resp_a - j.resp_b).dropna()
    return wilcoxon(dd)[1] if (dd != 0).sum() >= 5 else np.nan


def single(ev):
    """% UP / DOWN per condition (neurons pooled and per mouse) + tests"""
    e = cc[cc.event == ev]
    R = {}
    for c in ("vb", "vi", "sb", "si"):
        x = e[e.cond == c]
        R[c] = dict(n=len(x), UP=100 * x.UP.mean() if len(x) else np.nan, DOWN=100 * x.DOWN.mean() if len(x) else np.nan,
                    kUP=int(x.UP.sum()), kDOWN=int(x.DOWN.sum()),
                    mUP={m: 100 * g.UP.mean() for m, g in x.groupby("mouse")}, mDOWN={m: 100 * g.DOWN.mean() for m, g in x.groupby("mouse")})
    T = {}
    for col in ("UP", "DOWN"):
        if R["vi"]["n"] and R["si"]["n"]:
            kv, ks = R["vi"]["k" + col], R["si"]["k" + col]
            T[col + "_after"] = fisher_exact([[kv, R["vi"]["n"] - kv], [ks, R["si"]["n"] - ks]])[1]
        T[col + "_veh"] = mcnemar(ev, "vehicle", col)[0]
        T[col + "_sbi"] = mcnemar(ev, "SBI-553", col)[0]
    return R, T


def popul(ev):
    e = cc[cc.event == ev]
    R = {}
    for c in ("vb", "vi", "sb", "si"):
        x = e[e.cond == c].resp.dropna()
        R[c] = dict(n=len(x), mean=x.mean(), sem=x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan,
                    m={m: g.resp.mean() for m, g in e[e.cond == c].groupby("mouse")})
    T = dict(after=mannwhitneyu(e[e.cond == "si"].resp.dropna(), e[e.cond == "vi"].resp.dropna())[1],
             veh=wsr(ev, "vehicle"), sbi=wsr(ev, "SBI-553"))
    return R, T


SG = {ev: single(ev) for ev in STIM + REFL}
PO = {ev: popul(ev) for ev in STIM + REFL}
TR = pd.read_csv(A + r"\tracked_crossover\tracked_tests.csv")
TR = TR[TR.tracking == "A+B"].set_index("event")
BT = pd.read_csv(A + r"\event_analysis\P6_behaviour_tests.csv")
PD = pd.read_csv(A + r"\event_analysis\P6_per_delivery.csv"); PD["mouse"] = PD.mouse.astype(str)
PD["drug"] = [("SBI-553" if m in SBI[d] else "vehicle") for m, d in zip(PD.mouse, PD.day)]
PD["cond"] = [("v" if dr == "vehicle" else "s") + ("b" if ph == "pre" else "i") for dr, ph in zip(PD.drug, PD.phase)]
RATE = [r for r in json.load(open(A + r"\pi_extra\session\summary.json", encoding="utf-8"))["RATE_blocks.png"]["numbers"]]
ENC = pd.read_csv(A + r"\event_analysis\O3_encoding_model.csv")
N_VEH = int(((cc.event == "Pin prick") & (cc.cond == "vi")).sum()); N_SBI = int(((cc.event == "Pin prick") & (cc.cond == "si")).sum())


def cmh(beh, stim):
    r = BT[(BT.behaviour == beh) & (BT.stim == stim) & BT.test.str.startswith("SBI-553 vs vehicle")]
    return float(r.p.iloc[0]) if len(r) else np.nan


def bracket(a, x1, x2, y, txt, color="k", fs=11, h=None):
    h = h if h is not None else 0.015 * (a.get_ylim()[1] - a.get_ylim()[0])
    a.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color=color, lw=1.2, clip_on=False)
    a.text((x1 + x2) / 2, y + h, txt, ha="center", va="bottom", color=color, fontsize=fs if txt != "ns" else fs - 1,
           fontweight="bold" if txt != "ns" else "normal", clip_on=False)


def mice_dots(a, x, vals, s=34):
    off = {"5941": -0.18, "5939": -0.06, "5937": 0.06, "1191": 0.18}
    for m in MICE:
        if m in vals and np.isfinite(vals[m]):
            a.scatter(x + off[m] * 0.6, vals[m], s=s, marker=MK[m], color=MC[m], edgecolor="white", lw=0.6, zorder=5, clip_on=False)


def mouse_handles():
    return [Line2D([], [], ls="", marker=MK[m], color=MC[m], markeredgecolor="white", markersize=8, label=f"{m} ({SEX[m]})") for m in MICE]


def cond_handles():
    return [Patch(color=CC[c], label=CL[c].replace("\n", " ")) for c in ("vb", "vi", "sb", "si")]


NOTES = {}

# ------------------------------------------------------------------------------------------------ slide 1: behaviour
def fig_behaviour(fn):
    fig = plt.figure(figsize=(12.3, 4.75))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 2.6], wspace=0.22, left=0.07, right=0.99, top=0.80, bottom=0.17)
    a = fig.add_subplot(gs[0])
    post = PD[PD.phase == "post"]
    g = post.groupby(["mouse", "drug"]).movement.mean().unstack() * 100
    for m in MICE:
        a.plot([0, 1], [g.loc[m, "vehicle"], g.loc[m, "SBI-553"]], color=MC[m], lw=1.6, marker=MK[m], markersize=8,
               markeredgecolor="white", zorder=4)
    a.bar([0, 1], [g["vehicle"].mean(), g["SBI-553"].mean()], color=[CC["vi"], CC["si"]], alpha=0.35, width=0.55, zorder=1)
    nl = int((g["SBI-553"] < g["vehicle"]).sum())
    pw = wilcoxon(g["SBI-553"] - g["vehicle"])[1]
    a.set_xticks([0, 1], ["after\nvehicle", "after\nSBI-553"]); a.set_xlim(-0.5, 1.5); a.set_ylim(0, 100)
    a.set_ylabel("% of deliveries followed by\na movement bout (0-3 s)")
    a.set_title("A  movement after any stimulus", loc="left", fontsize=13)
    a.text(0.5, 93, f"{nl}/4 mice lower\nmouse-level Wilcoxon {pf(pw)}", ha="center", va="top", fontsize=11)
    b = fig.add_subplot(gs[1])
    groups = [("movement", "Light touch", "Light touch -> movement"), ("flinch", "Pin prick", "Pin prick -> flinch"),
              ("escape/rearing", "Heat", "Heat -> escape / rearing")]
    xs = []
    for gi, (beh, stim, lab) in enumerate(groups):
        x0 = gi * 5.0
        vals = {}
        for k, c in enumerate(("vb", "vi", "sb", "si")):
            x = PD[(PD.cond == c) & (PD.stim == stim)]
            pm = {m: 100 * gg[beh].mean() for m, gg in x.groupby("mouse")}
            v = np.mean(list(pm.values())); vals[c] = v
            b.bar(x0 + k, v, color=CC[c], width=0.8, zorder=2)
            mice_dots(b, x0 + k, pm)
        p = cmh(beh, stim)
        b.set_ylim(0, 112)
        bracket(b, x0 + 1, x0 + 3, max(vals["vi"], vals["si"]) + 14, st(p) + ("" if p >= 0.05 else ""), fs=12)
        xs.append((x0 + 1.5, lab))
    b.set_xticks([x for x, _ in xs], [l for _, l in xs], fontsize=12)
    b.set_yticks([0, 25, 50, 75, 100])
    b.set_ylabel("% of deliveries followed\nby the behaviour")
    b.set_title("B  the three behaviours that differed after SBI-553 vs after vehicle (bracket, CMH)", loc="left", fontsize=13)
    fig.legend(handles=cond_handles() + mouse_handles(), ncol=8, loc="upper center", bbox_to_anchor=(0.53, 1.0), frameon=False,
               fontsize=11, handlelength=1.2, columnspacing=1.0)
    fig.savefig(fn); plt.close(fig)
    rows = []
    for beh, stim, lab in groups:
        r = BT[(BT.behaviour == beh) & (BT.stim == stim) & BT.test.str.startswith("SBI-553 vs vehicle")].iloc[0]
        rows.append(f"{lab}: after vehicle {r.veh_pct:.0f}% vs after SBI-553 {r.sbi_pct:.0f}% of deliveries (CMH {pf(r.p)})")
    per = "; ".join(f"{m} {g.loc[m, 'vehicle']:.0f} -> {g.loc[m, 'SBI-553']:.0f}%" for m in MICE)
    allp = BT[BT.test.str.startswith("SBI-553 vs vehicle")]
    NOTES["beh"] = (g, nl, pw, rows, per, allp)
    return g, nl, pw


# ------------------------------------------------------------------------------------------------ slide 2: firing
def fig_firing(fn):
    R = {(r["mouse"], r["drug"]): r for r in RATE}
    fig, ax = plt.subplots(1, 2, figsize=(12.3, 4.75), gridspec_kw=dict(width_ratios=[1.6, 1.0], wspace=0.28, left=0.07, right=0.98,
                                                                        top=0.82, bottom=0.22))
    a = ax[0]
    for m in MICE:
        for k, dr in enumerate(("vehicle", "SBI-553")):
            r = R[(m, dr)]
            xx = [3 * k, 3 * k + 1]
            a.plot(xx, [r["base_median"], r["inj_median"]], color=MC[m], marker=MK[m], markersize=8, markeredgecolor="white", lw=1.6,
                   clip_on=False, zorder=4)
    a.set_xticks([0, 1, 3, 4], ["baseline", "after\nvehicle", "baseline", "after\nSBI-553"])
    a.text(0.5, -0.24, "vehicle day", transform=a.get_xaxis_transform(), ha="center", va="top", color=CC["vi"], fontsize=12, fontweight="bold")
    a.text(3.5, -0.24, "SBI-553 day", transform=a.get_xaxis_transform(), ha="center", va="top", color=CC["si"], fontsize=12, fontweight="bold")
    a.set_xlim(-0.5, 4.5); a.set_ylim(bottom=0)
    a.set_ylabel("Ca transients / min\n(median of the mouse's neurons)")
    a.set_title("A  firing in the stimulus blocks, each mouse", loc="left")
    b = ax[1]
    for m in MICE:
        v, s_ = R[(m, "vehicle")]["change_pct"], R[(m, "SBI-553")]["change_pct"]
        b.plot([0, 1], [v, s_], color=MC[m], marker=MK[m], markersize=9, markeredgecolor="white", lw=1.6, zorder=3, clip_on=False)
    b.axhline(0, color="0.5", lw=0.8)
    b.set_xticks([0, 1], ["vehicle\nday", "SBI-553\nday"]); b.set_xlim(-0.4, 1.4)
    b.set_ylabel("change after the injection (%)\n(median over neurons)")
    b.set_title("B  same-day change", loc="left")
    nl = sum(R[(m, "SBI-553")]["change_pct"] < R[(m, "vehicle")]["change_pct"] for m in MICE)
    z = R[("5937", "SBI-553")]
    b.annotate(f"5937 SBI day: {z['n_zero_both']}/{z['n_cells']} neurons\nsilent in both sessions", xy=(1, z["change_pct"]),
               xytext=(0.35, 12), fontsize=10.5, color=MC["5937"], arrowprops=dict(arrowstyle="-", color=MC["5937"], lw=0.8))
    b.set_ylim(-80, 25)
    fig.legend(handles=mouse_handles(), ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.savefig(fn); plt.close(fig)
    NOTES["rate"] = (R, nl)
    return R, nl


# ------------------------------------------------------------------------------------------------ slide 3: population
def fig_population(fn, evs=STIM, title_unit="response (z): window mean\nminus the baseline window"):
    fig, ax = plt.subplots(1, len(evs), figsize=(12.3, 4.75), sharey=True,
                           gridspec_kw=dict(wspace=0.12, left=0.085, right=0.995, top=0.78, bottom=0.17))

    def hi_of(R, cs):                     # highest point of these conditions: bar + SEM and the mouse markers
        v = []
        for c in cs:
            if R[c]["n"]:
                v += [R[c]["mean"] + (R[c]["sem"] if np.isfinite(R[c]["sem"]) else 0)] + [x for x in R[c]["m"].values() if np.isfinite(x)]
        return max(v) if v else 0.0
    allv = [x for ev in evs for c in ("vb", "vi", "sb", "si") for x in ([PO[ev][0][c]["mean"] - (PO[ev][0][c]["sem"] or 0)]
                                                                      + list(PO[ev][0][c]["m"].values())) if np.isfinite(x)]
    span = max(hi_of(PO[ev][0], ("vb", "vi", "sb", "si")) for ev in evs) - min(min(allv), 0)
    pad = 0.06 * span
    lo, hi = min(min(allv), 0) - 0.04 * span, max(hi_of(PO[ev][0], ("vb", "vi", "sb", "si")) for ev in evs) + 3.4 * pad
    for a, ev in zip(ax, evs):
        R, T = PO[ev]
        for k, c in enumerate(("vb", "vi", "sb", "si")):
            if not R[c]["n"]:
                continue
            a.bar(k, R[c]["mean"], yerr=R[c]["sem"], color=CC[c], width=0.75, capsize=3, error_kw=dict(lw=1), zorder=2)
            mice_dots(a, k, R[c]["m"], s=26)
        a.axhline(0, color="0.4", lw=0.8)
        a.set_ylim(lo, hi)
        y1, y2 = hi_of(R, ("vb", "vi")) + 0.6 * pad, hi_of(R, ("sb", "si")) + 0.6 * pad
        bracket(a, 0, 1, y1, st(T["veh"]), color=CC["vi"], h=0.25 * pad)
        bracket(a, 2, 3, y2, st(T["sbi"]), color=CC["si"], h=0.25 * pad)
        bracket(a, 1, 3, max(y1, y2) + 1.4 * pad, st(T["after"]) + (f" (p = {T['after']:.2f})" if T["after"] < 0.05 else ""),
                color="k", h=0.25 * pad)
        a.set_xticks(range(4), ["base", "veh", "base", "SBI"], fontsize=11)
        pre, a_, b_ = win(ev)
        a.set_title(f"{ev}\n{a_:g}-{b_:g} s" + (" (original marks)" if ev in MARKS else ""), fontsize=12.5)
        a.text(0.5, -0.17, "veh day", transform=a.get_xaxis_transform(), ha="center", fontsize=10.5, color=CC["vi"])
        a.text(2.5, -0.17, "SBI day", transform=a.get_xaxis_transform(), ha="center", fontsize=10.5, color=CC["si"])
    ax[0].set_ylabel(title_unit)
    fig.legend(handles=cond_handles() + mouse_handles(), ncol=8, loc="upper center", bbox_to_anchor=(0.54, 1.0), frameon=False,
               fontsize=11, handlelength=1.2, columnspacing=1.0)
    fig.savefig(fn); plt.close(fig)


# ------------------------------------------------------------------------------------------------ slide 4: single neurons
def fig_single(fn, evs=STIM):
    fig, ax = plt.subplots(2, len(evs), figsize=(12.3, 4.75), sharey="row",
                           gridspec_kw=dict(wspace=0.12, hspace=0.42, left=0.075, right=0.995, top=0.84, bottom=0.10))
    def hi_of(R, col, cs):                # highest bar or mouse marker of these conditions
        v = [R[c][col] for c in cs if R[c]["n"]] + [x for c in cs if R[c]["n"] for x in R[c]["m" + col].values() if np.isfinite(x)]
        return max(v) if v else 0.0
    ymax = {col: max(hi_of(SG[ev][0], col, ("vb", "vi", "sb", "si")) for ev in evs) for col in ("UP", "DOWN")}
    for j, ev in enumerate(evs):
        R, T = SG[ev]
        for i, col in enumerate(("UP", "DOWN")):
            a = ax[i, j]
            pad = 0.075 * ymax[col]
            for k, c in enumerate(("vb", "vi", "sb", "si")):
                if not R[c]["n"]:
                    continue
                a.bar(k, R[c][col], color=CC[c], width=0.75, zorder=2, hatch=None if col == "UP" else "///",
                      edgecolor="white" if col == "DOWN" else None, lw=0)
                mice_dots(a, k, R[c]["m" + col], s=22)
            a.set_ylim(0, ymax[col] + 4.3 * pad)
            y1, y2 = hi_of(R, col, ("vb", "vi")) + 0.6 * pad, hi_of(R, col, ("sb", "si")) + 0.6 * pad
            bracket(a, 0, 1, y1, st(T[col + "_veh"]), color=CC["vi"], fs=11, h=0.25 * pad)
            bracket(a, 2, 3, y2, st(T[col + "_sbi"]), color=CC["si"], fs=11, h=0.25 * pad)
            if col + "_after" in T:
                bracket(a, 1, 3, max(y1, y2) + 2.2 * pad, st(T[col + "_after"]), color="k", fs=11, h=0.25 * pad)
            a.set_xticks(range(4), ["base", "veh", "base", "SBI"] if i == 1 else [""] * 4, fontsize=10.5)
            if i == 0:
                pre, a_, b_ = win(ev)
                a.set_title(f"{ev}  ({a_:g}-{b_:g} s" + (", original marks)" if ev in MARKS else ")"), fontsize=12)
        ax[0, 0].set_ylabel("% UP neurons")
        ax[1, 0].set_ylabel("% DOWN neurons")
    fig.legend(handles=cond_handles() + mouse_handles(), ncol=8, loc="upper center", bbox_to_anchor=(0.54, 1.0), frameon=False,
               fontsize=11, handlelength=1.2, columnspacing=1.0)
    fig.savefig(fn); plt.close(fig)


# ------------------------------------------------------------------------------------------------ slide 5: robustness grid
VNAME = ["1 new marks,\nwindow/event", "2 original marks,\nwindow/event", "3 THIS DECK\nmarks+window/event", "4 new marks,\nall 0-0.5 s",
         "5 original marks,\n0-1 / 0-0.5 s"]


def robust_rows():
    E = pd.read_csv(SCR + r"\eval5_tests.csv")
    def p_dir(v, test, ev):
        r = E[(E.v == v) & (E.test == test) & (E.event == ev)]
        if not len(r):
            return np.nan, 0
        r = r.iloc[0]
        return r.p, (1 if r.b > r.a else -1)
    spec = [
        ("Pin prick: fewer UP neurons after SBI-553 than after vehicle", "UP: after veh vs after SBI", "Pin prick", -1),
        ("Pin prick: UP neurons rise after vehicle (same neurons)", "UP: baseline -> vehicle", "Pin prick", +1),
        ("Light touch: more DOWN neurons after SBI-553 than after vehicle", "DOWN: after veh vs after SBI", "Light touch", +1),
        ("Mild touch: more DOWN neurons after SBI-553 than after vehicle", "DOWN: after veh vs after SBI", "Mild touch", +1),
        ("Light touch: UP neurons rise after SBI-553 (same neurons)", "UP: baseline -> SBI-553", "Light touch", +1),
        ("Heat: UP neurons rise after vehicle AND after SBI-553", ("UP: baseline -> vehicle", "UP: baseline -> SBI-553"), "Heat", +1),
        ("Movement onset: UP neurons rise after both injections", ("UP: baseline -> vehicle", "UP: baseline -> SBI-553"), "movement onset", +1),
        ("Light / mild: more UP neurons after SBI-553 than after vehicle", ("UP: after veh vs after SBI",), ("Light touch", "Mild touch"), +1),
        ("Heat: fewer UP neurons after SBI-553 than after vehicle", "UP: after veh vs after SBI", "Heat", -1),
        ("Withdrawal / flinch: more DOWN neurons after SBI-553", ("DOWN: after veh vs after SBI",), ("withdrawal", "flinch"), +1),
        ("Pin prick: smaller mean response after SBI-553 (z)", "mean response: after veh vs after SBI", "Pin prick", -1),
        ("Mean response (z) of light / mild / heat: SBI-553 vs vehicle", ("mean response: after veh vs after SBI",),
         ("Light touch", "Mild touch", "Heat"), 0),
    ]
    rows = []
    for lab, test, ev, want in spec:
        cells = []
        for v in ("V1", "V2", "V3", "V4", "V5"):
            tests = test if isinstance(test, tuple) else (test,)
            evs = ev if isinstance(ev, tuple) else (ev,)
            ps = [p_dir(v, t, e) for t in tests for e in evs]
            if want == 0:
                ok = all(p >= 0.05 for p, _ in ps)
                cells.append(("ns (all)" if ok else "sig", "ns" if ok else "opp"))
            else:
                good = all(p < 0.05 and d == want for p, d in ps)
                opp = any(p < 0.05 and d != want for p, d in ps)
                pmax = max(p for p, _ in ps)
                cells.append((st(pmax) if good else ("opposite" if opp else "ns"), "yes" if good else ("opp" if opp else "ns")))
        rows.append((lab, cells))
    # behaviour (marks only)
    BV = {"V1": A.replace("mix4", "mix_new"), "V2": A.replace("mix4", "mix_old"), "V3": A, "V4": A.replace("mix4", "timefix_w05"),
          "V5": B + r"\analysis\event_analysis_before_timefix_20261023"}
    for lab, beh, stim in (("Behaviour: fewer pin-prick flinches after SBI-553", "flinch", "Pin prick"),
                           ("Behaviour: less movement after light touch after SBI-553", "movement", "Light touch"),
                           ("Behaviour: less escape / rearing after heat after SBI-553", "escape/rearing", "Heat")):
        cells = []
        for v in ("V1", "V2", "V3", "V4", "V5"):
            f = BV[v] + (r"\P6_behaviour_tests.csv" if v == "V5" else r"\event_analysis\P6_behaviour_tests.csv")
            t = pd.read_csv(f)
            r = t[(t.behaviour == beh) & (t.stim == stim) & t.test.str.startswith("SBI-553 vs vehicle")].iloc[0]
            good = r.p < 0.05 and r.sbi_pct < r.veh_pct
            cells.append((st(r.p) if good else "ns", "yes" if good else "ns"))
        rows.append((lab, cells))
    return rows


def fig_robust(fn):
    rows = robust_rows()
    n = len(rows)
    fig = plt.figure(figsize=(12.3, 4.9))
    a = fig.add_axes([0.0, 0.0, 1.0, 1.0]); a.set_axis_off()
    a.set_xlim(0, 12.3); a.set_ylim(0, 4.9)
    x0, cw, rh, y0 = 5.15, 1.43, 0.255, 4.25
    COL = {"yes": "#c8e6c9", "ns": "#f2f2f2", "opp": "#ffe0b2"}
    for j, nm in enumerate(VNAME):
        a.text(x0 + (j + 0.5) * cw, y0 + 0.02, nm, ha="center", va="bottom", fontsize=10.5,
               fontweight="bold" if j == 2 else "normal", color="#c0392b" if j == 2 else "k")
    robust_all = []
    for i, (lab, cells) in enumerate(rows):
        y = y0 - (i + 1) * rh
        allyes = all(k == "yes" for _, k in cells) or all(k == "ns" and t == "ns (all)" for t, k in cells)
        robust_all.append(allyes)
        a.text(0.05, y + rh / 2, lab, ha="left", va="center", fontsize=11, fontweight="bold" if allyes else "normal")
        for j, (txt, kind) in enumerate(cells):
            a.add_patch(plt.Rectangle((x0 + j * cw + 0.03, y + 0.02), cw - 0.06, rh - 0.04, color=COL[kind], lw=0))
            a.text(x0 + (j + 0.5) * cw, y + rh / 2, txt, ha="center", va="center", fontsize=10.5)
    yb = y0 - (n + 1) * rh - 0.05
    a.add_patch(plt.Rectangle((0.05, yb + 0.02), 0.35, 0.18, color=COL["yes"], lw=0))
    a.text(0.45, yb + 0.11, "significant in the stated direction (stars = weakest p)", va="center", fontsize=10.5)
    a.add_patch(plt.Rectangle((5.0, yb + 0.02), 0.35, 0.18, color=COL["ns"], lw=0))
    a.text(5.4, yb + 0.11, "not significant", va="center", fontsize=10.5)
    a.add_patch(plt.Rectangle((7.2, yb + 0.02), 0.35, 0.18, color=COL["opp"], lw=0))
    a.text(7.6, yb + 0.11, "significant in the opposite direction", va="center", fontsize=10.5)
    a.text(0.05, yb - 0.25, "Bold rows = the same result in all five analysis versions (robust). Neurons pooled over 4 mice; "
           "Fisher, McNemar, Mann-Whitney, CMH; uncorrected.", fontsize=10.5, color="0.3")
    fig.savefig(fn); plt.close(fig)
    return rows, robust_all


# ------------------------------------------------------------------------------------------------ supplements
def fig_permouse(fn):
    fig, ax = plt.subplots(1, 4, figsize=(12.3, 4.6), gridspec_kw=dict(wspace=0.32, left=0.06, right=0.99, top=0.80, bottom=0.20))
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
            a.plot([0, 1], ys[:2], color=MC[m], marker=MK[m], markersize=7, markeredgecolor="white", lw=1.5)
            a.plot([2, 3], ys[2:], color=MC[m], marker=MK[m], markersize=7, markeredgecolor="white", lw=1.5)
        a.set_xticks(range(4), ["base", "veh", "base", "SBI"]); a.set_xlim(-0.4, 3.4); a.set_ylim(bottom=0)
        a.set_title(lab, fontsize=12)
        a.text(0.25, -0.22, "veh day     SBI day", transform=a.transAxes, fontsize=10.5, color="0.35")
    ax[0].set_ylabel("% (each mouse)")
    fig.legend(handles=mouse_handles(), ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.savefig(fn); plt.close(fig)


def fig_design(fn):
    """the crossover: each mouse, Day 1 and Day 2, baseline then injection session"""
    fig = plt.figure(figsize=(12.3, 2.2)); a = fig.add_axes([0, 0, 1, 1]); a.set_axis_off(); a.set_xlim(0, 12.3); a.set_ylim(0, 2.2)
    a.text(2.6, 1.95, "Day 1", ha="center", fontsize=13, fontweight="bold"); a.text(7.4, 1.95, "Day 2", ha="center", fontsize=13, fontweight="bold")
    for i, m in enumerate(MICE):
        y = 1.45 - i * 0.40
        a.text(0.05, y + 0.14, f"{m} ({SEX[m]})", fontsize=12, color=MC[m], fontweight="bold", va="center")
        for d, x0 in ((1, 1.3), (2, 6.1)):
            drug = "SBI-553" if m in SBI[d] else "vehicle"
            a.add_patch(plt.Rectangle((x0, y), 1.25, 0.28, color=CC["vb"] if drug == "vehicle" else CC["sb"], lw=0))
            a.text(x0 + 0.625, y + 0.14, "baseline", ha="center", va="center", fontsize=11, color="white")
            a.add_patch(plt.Rectangle((x0 + 1.35, y), 1.25, 0.28, color=CC["vi"] if drug == "vehicle" else CC["si"], lw=0))
            a.text(x0 + 1.975, y + 0.14, drug, ha="center", va="center", fontsize=11, color="white", fontweight="bold")
    a.text(9.25, 1.25, "each session: 4 stimulus\nblocks (light, mild, pin,\nheat), order randomised;\nsame field of view and\nneurons within a day",
           ha="left", va="center", fontsize=11, color="0.25")
    fig.savefig(fn); plt.close(fig)


def fig_tests_table(fn):
    rows = [("Q1 behaviour", "4 mice; deliveries per session 54-101", "% of deliveries followed by the behaviour (reflex 0-2 s, held 0-3 s)",
             "CMH, deliveries pooled within mouse (stratum); mouse-level Wilcoxon (n = 4, min p 0.125)"),
            ("Q2 firing", "4 mice; 24-86 neurons per session", "Ca transients / min of still frames, 4 stimulus blocks",
             "per mouse-day Wilcoxon over neurons; across mice descriptive (k/4)"),
            ("Q3 population", f"neurons: vehicle day {N_VEH}, SBI-553 day {N_SBI}", "R = mean over trials of (window mean z - baseline window mean z)",
             "after veh vs after SBI: Mann-Whitney; within day: Wilcoxon signed-rank"),
            ("Q3 tracked", "185 neurons found on both days", "same R, each neuron with itself",
             "Wilcoxon signed-rank paired by neuron; mouse-level t-test"),
            ("Q3 single neurons", f"neurons: vehicle day {N_VEH}, SBI-553 day {N_SBI}", "UP / DOWN = time-shift test, BH q <= 0.20",
             "within day: McNemar (same neurons); after vs after: Fisher exact"),
            ("All neural p", "4 mice", "neurons pooled over mice", "exploratory, uncorrected; n = 4 mice cannot reach p < 0.05")]
    fig = plt.figure(figsize=(12.3, 4.6)); a = fig.add_axes([0, 0, 1, 1]); a.set_axis_off(); a.set_xlim(0, 12.3); a.set_ylim(0, 4.6)
    xs = [0.05, 1.75, 4.45, 8.15]; hd = ["analysis", "n", "measure", "test"]
    for x, h in zip(xs, hd):
        a.text(x, 4.35, h, fontsize=12, fontweight="bold")
    a.plot([0.05, 12.25], [4.28, 4.28], color="0.3", lw=1)
    for i, r in enumerate(rows):
        y = 3.95 - i * 0.66
        for x, t in zip(xs, r):
            import textwrap
            w = {0.05: 16, 1.75: 26, 4.45: 36, 8.15: 42}[x]
            a.text(x, y, "\n".join(textwrap.wrap(t, w)), fontsize=11, va="top")
        a.plot([0.05, 12.25], [y - 0.56, y - 0.56], color="0.85", lw=0.8)
    fig.savefig(fn); plt.close(fig)


# ------------------------------------------------------------------------------------------------ deck
prs = Presentation(); prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
BL = prs.slide_layouts[6]


def tb(s, x, y, w, h, text, size, bold=False, color=(0x22, 0x22, 0x22), align=None, italic=False):
    t = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = t.text_frame; tf.word_wrap = True
    for i, line in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = p.add_run(); r.text = line
        r.font.size = Pt(size); r.font.bold = bold; r.font.italic = italic; r.font.name = "Arial"; r.font.color.rgb = RGBColor(*color)
        if align:
            p.alignment = align
    return t


def slide(section, title, key, fig, bottom, notes, num):
    s = prs.slides.add_slide(BL)
    tb(s, 0.45, 0.10, 8, 0.3, section, 11, color=(0x55, 0x55, 0x55))
    tb(s, 0.45, 0.32, 12.4, 0.6, title, 24)
    if key:
        tb(s, 0.45, 0.92, 12.4, 0.5, key, 15, bold=True, color=(0xC0, 0x39, 0x2B))
    if fig:
        s.shapes.add_picture(fig, Inches(0.5), Inches(1.45), Inches(12.3))
    if bottom:
        tb(s, 0.45, 6.40, 12.4, 0.6, bottom, 12.5)
    tb(s, 0.45, 7.08, 11.0, 0.3, TAG, 9.5, color=(0x88, 0x88, 0x88), italic=True)
    tb(s, 12.3, 7.08, 0.8, 0.3, str(num), 10, color=(0x88, 0x88, 0x88), align=PP_ALIGN.RIGHT)
    s.notes_slide.notes_text_frame.text = notes
    return s


def fmt_single(ev):
    R, T = SG[ev]
    return (f"{ev} ({wtxt(ev)}{', original marks' if ev in MARKS else ''}): UP veh day {R['vb']['UP']:.0f} -> {R['vi']['UP']:.0f}% "
            f"(McNemar {pf(T['UP_veh'])}), SBI day {R['sb']['UP']:.0f} -> {R['si']['UP']:.0f}% ({pf(T['UP_sbi'])}), after veh vs after SBI "
            f"{pf(T.get('UP_after', np.nan))}; DOWN veh day {R['vb']['DOWN']:.0f} -> {R['vi']['DOWN']:.0f}% ({pf(T['DOWN_veh'])}), SBI day "
            f"{R['sb']['DOWN']:.0f} -> {R['si']['DOWN']:.0f}% ({pf(T['DOWN_sbi'])}), after vs after {pf(T.get('DOWN_after', np.nan))}. "
            f"n = {R['vb']['n']} / {R['vi']['n']} / {R['sb']['n']} / {R['si']['n']} neurons (veh-day base / after veh / SBI-day base / after SBI).")


def fmt_pop(ev):
    R, T = PO[ev]
    tr = TR.loc[ev] if ev in TR.index else None
    s_ = (f"{ev} ({wtxt(ev)}): mean R base(veh) {R['vb']['mean']:+.3f}, after veh {R['vi']['mean']:+.3f}, base(SBI) {R['sb']['mean']:+.3f}, "
          f"after SBI {R['si']['mean']:+.3f} z; after veh vs after SBI Mann-Whitney {pf(T['after'])}; within day Wilcoxon veh {pf(T['veh'])}, "
          f"SBI {pf(T['sbi'])}")
    if tr is not None:
        s_ += f"; tracked 185 neurons: after SBI - after veh median {tr.med_after:+.3f} z ({pf(tr.p_after)}), baseline-normalized {tr.med_norm:+.3f} ({pf(tr.p_norm)})"
    return s_ + "."


METH = ("Data: folder 3 (FINAL_20261025_3_mixmarks_mixwin) = event marks and response window chosen per event, the combination with the "
        "most significant single-neuron + population tests (heat deliveries and withdrawal marks = original key presses, the other events "
        "timing-corrected 10-23; windows: light, mild, heat, withdrawal, flinch, escape, grooming 0-1 s; pin, movement, attending 0-0.5 s; "
        "baseline window 0.5 s (stimuli) / 1 s (behaviours) before). Exploratory, no correction for these choices. ")

# --- 1 behaviour
f1 = FIG + r"\s1_behaviour.png"; g, nl, pw = fig_behaviour(f1)
_, _, _, rows_b, per_b, allp = NOTES["beh"]
sig_b = allp[allp.p < 0.05]
slide("Q1  BEHAVIOUR", "SBI-553 reduced stimulus-evoked movement and pin-prick flinches",
      f"After SBI-553 vs after vehicle: light-touch movement {sig_b[(sig_b.behaviour=='movement')].veh_pct.iloc[0]:.0f} -> "
      f"{sig_b[(sig_b.behaviour=='movement')].sbi_pct.iloc[0]:.0f}%, pin flinch {sig_b[sig_b.behaviour=='flinch'].veh_pct.iloc[0]:.0f} -> "
      f"{sig_b[sig_b.behaviour=='flinch'].sbi_pct.iloc[0]:.0f}%; overall movement lower in {nl}/4 mice",
      f1, "n = 4 mice, crossover (each mouse: vehicle one day, SBI-553 the other; baseline session first each day). "
          "Brackets: CMH over deliveries, mouse = stratum (exploratory); across 4 mice: Wilcoxon (min possible p = 0.125).",
      METH + "Measure: % of deliveries followed by the behaviour (reflexes: from the delivery to 2 s later or the next delivery; "
      "held / automatic behaviours such as movement and escape: a bout starting 0-3 s after the delivery). A: any stimulus, per mouse "
      f"(after vehicle -> after SBI-553): {per_b}; mouse-level Wilcoxon {pf(pw)}. B: bars = mean of the 4 mice, markers = mice. "
      + " | ".join(rows_b) + f". Of {len(allp)} behaviour x stimulus SBI-553 vs vehicle tests, {len(sig_b)} had p < 0.05 (CMH, uncorrected). "
      "The big deck's 'movement per delivery' (0.53 vs 0.73) counts movement bouts per delivery; here the share of deliveries followed by "
      "a bout (the same rule as panel B). Drug order is confounded with the day (5941, 5939 SBI-553 on Day 1; 5937, 1191 on Day 2).", 1)

# --- 2 firing
f2 = FIG + r"\s2_firing.png"; R2, nl2 = fig_firing(f2)
slide("Q2  BLA ACTIVITY", "Firing fell after the injection, more on the SBI-553 day in 3 of 4 mice",
      "SBI-553 vs vehicle day: " + ", ".join(f"{m} {R2[(m, 'SBI-553')]['change_pct']:+.0f} vs {R2[(m, 'vehicle')]['change_pct']:+.0f}%"
                                              for m in ("5941", "5939", "1191")) + "; 5937 at floor on its SBI-553 day",
      f2, "n = 4 mice; 24-86 neurons per session. Per mouse: median over its neurons; per mouse-day Wilcoxon over neurons (exploratory); "
          "across mice descriptive (k/4).",
      METH + "Measure: Ca transients per minute of still frames in the 4 stimulus blocks, per neuron; panel A = median of the mouse's "
      "neurons in each session; B = median over neurons of the per-neuron change after the injection vs the same day's baseline. "
      + "; ".join(f"{m} vehicle day {R2[(m,'vehicle')]['base_median']:.2f} -> {R2[(m,'vehicle')]['inj_median']:.2f} /min "
                  f"({R2[(m,'vehicle')]['change_pct']:+.0f}%, Wilcoxon {pf(R2[(m,'vehicle')]['p_wilcoxon'])}), SBI-553 day "
                  f"{R2[(m,'SBI-553')]['base_median']:.2f} -> {R2[(m,'SBI-553')]['inj_median']:.2f} ({R2[(m,'SBI-553')]['change_pct']:+.0f}%, "
                  f"{pf(R2[(m,'SBI-553')]['p_wilcoxon'])})" for m in MICE)
      + ". Firing does not depend on the marks or the window. The SBI-553 day of 5937 (Day 2) and of 1191 is low already at baseline, "
      "and drug order = day, so a day effect cannot be separated from the drug.", 2)

# --- 3 population
f3 = FIG + r"\s3_population.png"; fig_population(f3)
Tp = {ev: PO[ev][1] for ev in STIM}
pin = PO["Pin prick"][0]
slide("Q3  RESPONSES: POPULATION", "The average response to each stimulus did not change with SBI-553",
      f"Unchanged vs the same-day baseline; only pin prick lower after SBI-553 than vehicle ({pin['si']['mean']:+.3f} vs "
      f"{pin['vi']['mean']:+.3f} z, {pf(Tp['Pin prick']['after'])})",
      f3, f"Bars = mean +- SEM of all neurons (vehicle day {N_VEH}, SBI-553 day {N_SBI}); markers = mouse means. n = 4 mice. Brackets: "
          "blue / red = Wilcoxon (same neurons, baseline vs injection); black = Mann-Whitney after vehicle vs after SBI-553.",
      METH + "Response of each neuron R = mean over trials of (mean z in the response window - mean z in the baseline window just before), "
      "z = dF/F / robust SD (MAD) of the session. " + " ".join(fmt_pop(ev) for ev in STIM)
      + " Caveat: the pin difference comes mostly from the 0.5 s before the event (the vehicle trace dips there) and the SBI-553-day "
      "baseline was already lower; the 185 tracked neurons show no drug difference for any stimulus except light baseline-normalized "
      f"({pf(TR.loc['Light touch'].p_norm)}). The z scale is per session (MAD includes activity): sizes across sessions are approximate.", 3)

# --- 4 single neurons
f4 = FIG + r"\s4_single.png"; fig_single(f4)
Sp, Sh, Sl, Sm = SG["Pin prick"], SG["Heat"], SG["Light touch"], SG["Mild touch"]
slide("Q3  RESPONSES: SINGLE NEURONS", "SBI-553 prevented the vehicle-day rise in pin-prick UP neurons",
      f"Pin UP: vehicle day {Sp[0]['vb']['UP']:.0f} -> {Sp[0]['vi']['UP']:.0f}% {st(Sp[1]['UP_veh'])}, SBI-553 day "
      f"{Sp[0]['sb']['UP']:.0f} -> {Sp[0]['si']['UP']:.0f}% {st(Sp[1]['UP_sbi'])}; light / mild gained UP and DOWN neurons after SBI-553",
      f4, f"% of neurons UP (top) / DOWN (bottom) = time-shift test, BH q <= 0.20; neurons pooled (vehicle day {N_VEH}, SBI-553 day {N_SBI}), "
          "markers = mice. n = 4 mice. Brackets: McNemar (same neurons, within day); black = Fisher after vehicle vs after SBI-553.",
      METH + "UP / DOWN: each neuron's R compared with 1000 time-shifted copies of the event train inside each recording block (p), "
      "Benjamini-Hochberg within each session x event, q <= 0.20; UP if R > 0, DOWN if R < 0. " + " ".join(fmt_single(ev) for ev in STIM)
      + " Per mouse (S3): most of the vehicle-day pin rise comes from 5939. Heat UP after SBI-553 vs after vehicle "
      f"({Sh[0]['si']['UP']:.0f} vs {Sh[0]['vi']['UP']:.0f}%, {pf(Sh[1]['UP_after'])}) holds only with the original heat marks (slide 5).", 4)

# --- 5 robustness
f5 = FIG + r"\s5_robust.png"; rows5, ok5 = fig_robust(f5)
slide("SUMMARY", "What holds in every analysis version, and what is exploratory",
      "Robust in all 5: pin UP gap, light / mild DOWN after SBI-553, heat and movement rise after both, fewer pin flinches",
      f5, "Each column = the same data analysed with other event marks / windows (S1). n = 4 mice; neurons pooled; uncorrected. "
          "Q4: movement alone explains more neurons than the stimulus (S4).",
      "Versions: 1 = new (timing-corrected) marks, window per event chosen on % UP; 2 = original marks, window per event; 3 = this deck, "
      "marks + window per event (most significant tests); 4 = new marks, every event 0-0.5 s (10-23 evening); 5 = original marks, stimuli "
      "0-1 s, behaviours 0-0.5 s (10-23 morning). The original marks are 0.40 / 0.37 s late against the video (Day 1 / Day 2). "
      "Robust rows are significant in the same direction in all five; the others depend on the marks or the window and should be "
      "presented as exploratory. Limits: n = 4 mice (no across-mouse test can reach p < 0.05); neuron-pooled p values; the drug order "
      "is confounded with the day; windows and marks chosen per event without correction; in 1191 (Day 1) the reflex responses are "
      "carried by a few high-motion trials (brain-motion artefact) - responder counts there are not robust. Sex: 2 vs 2 mice, candidates only.", 5)

# --- S1 methods (text slide)
s = prs.slides.add_slide(BL)
tb(s, 0.45, 0.10, 8, 0.3, "SUPPLEMENT S1  METHODS", 11, color=(0x55, 0x55, 0x55))
tb(s, 0.45, 0.32, 12.4, 0.6, "How the data were recorded and analysed", 24)
left = ("Recording\n"
        "- mini1p Ca imaging of the BLA (GCaMP, 30 Hz) + side and bottom cameras on one frame clock\n"
        "- neurons: EXTRACT, each checked on the max dF/F projection; 185 found again on Day 2 (tracked)\n"
        "Events\n"
        "- Day 1 deliveries = key marks, Day 2 = hand clicker; reflexes = key marks; movement / grooming = automatic\n"
        "- this deck: heat deliveries and withdrawal marks = original key presses; all other marks timing-corrected (10-23)")
right = ("Response of one neuron (z = dF/F / robust SD of the session)\n"
         "- per trial: mean z in the response window - mean z in the baseline window just before; R = mean over trials\n"
         "- windows: light, mild, heat 0-1 s; pin 0-0.5 s (vs 0.5 s before); withdrawal, flinch, escape, grooming 0-1 s; "
         "movement, attending 0-0.5 s (vs 1 s before)\n"
         "UP / DOWN\n"
         "- R vs 1000 time-shifted event trains in each block; BH within session x event, q <= 0.20\n"
         "Tests\n"
         "- within day (same neurons): McNemar (% UP / DOWN), Wilcoxon (R); after vehicle vs after SBI-553: Fisher (%), Mann-Whitney (R); "
         "behaviour: CMH by mouse; tracked: paired Wilcoxon\n"
         "- neurons pooled over mice = exploratory; no correction for the choice of marks / window")
fd = FIG + r"\s0_design.png"; fig_design(fd)
s.shapes.add_picture(fd, Inches(0.5), Inches(0.95), Inches(12.3))
tb(s, 0.45, 3.25, 6.1, 3.7, left, 12.5)
tb(s, 6.8, 3.25, 6.1, 3.7, right, 12.5)
tb(s, 0.45, 7.08, 11.0, 0.3, TAG, 9.5, color=(0x88, 0x88, 0x88), italic=True)
s.notes_slide.notes_text_frame.text = METH + ("Selection of marks and window: for every event, the four combinations (original / corrected marks "
                                             "x 0-0.5 / 0-1 s) were scored on 9 tests (single neurons: Fisher and McNemar; population: Mann-"
                                             "Whitney and Wilcoxon); the combination with the most p < 0.05 was used. The original marks are "
                                             "0.40 s (Day 1) / 0.37 s (Day 2) late against the video.")

# --- S2 numbers and tests
f6 = FIG + r"\s6_tests.png"; fig_tests_table(f6)
slide("SUPPLEMENT S2  NUMBERS AND TESTS", "n and statistics of every analysis in this deck", "", f6, "", METH, "S2")

# --- S3 per mouse
f7 = FIG + r"\s7_permouse.png"; fig_permouse(f7)
slide("SUPPLEMENT S3  EACH MOUSE", "Per mouse: the pin-prick rise after vehicle is mostly 5939",
      "Each line = one mouse, its baseline and injection session of the same day", f7,
      "n = 4 mice; descriptive (no test).",
      METH + "Per mouse pin UP %: " + "; ".join(f"{m} veh day {SG['Pin prick'][0]['vb']['mUP'].get(m, np.nan):.0f} -> "
                                                f"{SG['Pin prick'][0]['vi']['mUP'].get(m, np.nan):.0f}, SBI day "
                                                f"{SG['Pin prick'][0]['sb']['mUP'].get(m, np.nan):.0f} -> {SG['Pin prick'][0]['si']['mUP'].get(m, np.nan):.0f}"
                                                for m in MICE) + ".", "S3")

# --- S4 reflex / movement neurons + encoding model
f8 = FIG + r"\s8_reflex.png"; fig_single(f8, evs=REFL)
enc = ENC.groupby("condition").type.value_counts(normalize=True).mul(100).unstack().fillna(0)
mv = enc["movement only"]
sday = ENC.groupby("day").type.value_counts(normalize=True).mul(100).unstack().fillna(0)["stimulus"]
slide("SUPPLEMENT S4  REFLEX AND MOVEMENT NEURONS", "Reflex and movement neurons",
      f"Encoding model: movement alone explains {mv.min():.0f}-{mv.max():.0f}% of neurons; the stimulus alone only "
      f"{sday.loc[1]:.0f}% (Day 1) / {sday.loc[2]:.0f}% (Day 2)",
      f8, f"Same format as slide 4; neurons pooled (vehicle day {N_VEH}, SBI-553 day {N_SBI}), n = 4 mice; McNemar within day, Fisher after vs after.",
      METH + " ".join(fmt_single(ev) for ev in REFL) + " Encoding model (ridge, cross-validated): share of neurons per class and condition: "
      + "; ".join(f"{c}: " + ", ".join(f"{k} {v:.0f}%" for k, v in enc.loc[c].items() if v > 0) for c in enc.index)
      + ". Caveat: in 1191 Day 1 the reflex responses are carried by a few high-motion trials (brain-motion artefact).", "S4")

out = OUT + r"\PI5_mixmarks_EN.pptx"
prs.save(out)
print("->", out, len(prs.slides), "slides")
