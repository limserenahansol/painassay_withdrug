"""ca_sex_analysis.py - is there a pattern by SEX? (2026-10-22). Females 5941 + 1191, males 5939 + 5937; each sex has one mouse
per drug order (5941 F / 5939 M: SBI-553 on Day 1; 1191 F / 5937 M: SBI-553 on Day 2), so sex and order are not confounded.
n = 2 mice per sex: everything here is descriptive; neuron-level tests are exploratory (neurons are nested in 2 mice).

  S1  behaviour per delivery by sex (all stimuli pooled), baseline / vehicle / SBI-553, per mouse
  S2  the SBI-553 effect on behaviour (SBI-553 - vehicle injection session, same mouse) by sex, per stimulus
  S3  Ca transient rate per neuron by sex and condition; the within-day change of each neuron
  S4  % responsive neurons (chosen window) by sex, event and condition
  S5  event-locked mean response of all neurons by sex (pin prick, heat, withdrawal, movement onset)
  S6  whole-session population activity by sex
  S7  the movement-onset neurons after the injection, by sex (held-out baseline trials)
Writes analysis\\sex_analysis\\S*.png and sex_summary.json (all axes and colour scales labelled with units).

  python ca_sex_analysis.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import safe_io  # noqa: E402,F401
import ca_event_analysis as E  # noqa: E402
import pairstars as PS  # noqa: E402  (every star shows the pair it compares)
import winwords as WW  # noqa: E402  (10-23: key numbers over the event's own response window)

B = E.BASE
OUT = os.path.join(B, "analysis", "sex_analysis")
SEX = {"5941": "F", "1191": "F", "5939": "M", "5937": "M"}
SC = {"F": "#d81b60", "M": "#00838f"}
SUB, SUF = None, ""                               # split parts (10-22, legibility): behaviours / events of one part, suffix
SN = {"F": "female (5941, 1191)", "M": "male (5939, 5937)"}
BEHS = ["withdrawal", "flinch", "attending", "lick/bite", "escape/rearing", "movement"]   # guarding / grooming ~never
EVS = ["Pin prick", "Heat", "withdrawal", "movement onset"]
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11,
                     "legend.fontsize": 10.5, "axes.spines.top": False, "axes.spines.right": False})


def star(p):
    return E.star(p) or "ns"


def legend_mice(fig, y=-0.02):
    h = [Line2D([], [], color=SC[SEX[m]], marker=E.MK[m], lw=1.8, ms=8, label=f"{m} ({SEX[m]})") for m in E.MO]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, y))


def s1_behaviour(D):
    BB = [b for b in BEHS if SUB is None or b in SUB]
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.2)) if SUB is None else plt.subplots(1, len(BB), figsize=(4.6 * len(BB), 4.8), squeeze=False)
    out = {}
    for k, b in enumerate(BB):
        a = ax.ravel()[k]
        P = D.groupby(["mouse", "condition"])[b].mean().unstack().reindex(columns=E.COND) * 100
        for m in E.MO:
            if m in P.index:
                a.plot(range(3), P.loc[m].values, "-", color=SC[SEX[m]], marker=E.MK[m], lw=1.8, ms=8)
        for sx in ("F", "M"):
            ms = [m for m in E.MO if SEX[m] == sx and m in P.index]
            out[f"{b} | {sx}"] = {c: float(P.loc[ms, c].mean()) for c in E.COND}
        for m in P.index:                                                 # each mouse (for the slide sentences)
            out[f"{b} | {m}"] = {c: float(P.loc[m, c]) for c in E.COND}
        a.set_xticks(range(3), ["baseline", "vehicle", "SBI-553"])
        a.set_title(b, fontsize=13)
        a.set_ylim(bottom=0)
        if k % 3 == 0:
            a.set_ylabel("% of deliveries followed\nby the behaviour")
    legend_mice(fig)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(OUT, f"S1_behaviour_by_sex{SUF}.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def s2_drug_effect(D):
    """SBI-553 - vehicle (injection sessions of the same mouse), % of deliveries, per stimulus and behaviour, by sex"""
    behs = [b for b in ["withdrawal", "flinch", "attending", "escape/rearing", "movement"] if SUB is None or b in SUB]
    fig, ax = plt.subplots(1, len(behs), figsize=(18, 4.8) if SUB is None else (4.4 * len(behs), 4.8), sharey=False, squeeze=False)
    ax = ax[0]
    out = {}
    for k, b in enumerate(behs):
        a = ax[k]
        for i, st in enumerate(E.STIM):
            P = D[D.stim == st].groupby(["mouse", "condition"])[b].mean().unstack() * 100
            for j, sx in enumerate(("F", "M")):
                ms = [m for m in E.MO if SEX[m] == sx and m in P.index]
                d = (P.loc[ms, "SBI-553"] - P.loc[ms, "vehicle"]).values
                x = i + (j - 0.5) * 0.36
                a.bar(x, d.mean(), width=0.33, color=SC[sx], alpha=0.35)
                for dd, m in zip(d, ms):
                    a.scatter(x, dd, color=SC[sx], marker=E.MK[m], s=40, zorder=3, edgecolor="w", lw=0.5)
                out[f"{b} | {st} | {sx}"] = [float(v) for v in d]
        a.axhline(0, color="0.4", lw=0.8)
        a.set_xticks(range(4), ["light", "mild", "pin", "heat"])
        a.set_title(b, fontsize=13)
        if k == 0:
            a.set_ylabel("SBI-553 - vehicle\n(% of deliveries, same mouse)")
    fig.legend(handles=[Patch(color=SC[s], alpha=0.5, label=f"{SN[s]}: bar = mean, marker = mouse") for s in ("F", "M")],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(os.path.join(OUT, f"S2_drug_effect_by_sex{SUF}.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def load_rates():
    rows = []
    for d in (1, 2):
        W = pd.read_csv(os.path.join(B, "analysis", f"ca_day{d}", f"ca_d{d}_cells_whole_session.csv"))
        W["mouse"] = W.mouse.astype(str)
        W["day"] = d
        num = sum(W[f"still rate | {s}"].fillna(0) * W[f"still min | {s}"].fillna(0) for s in E.STIM)
        den = sum(W[f"still min | {s}"].fillna(0) for s in E.STIM)
        W["rate_blocks"] = num / den.replace(0, np.nan)
        W["condition"] = [E.cond_of(m, d, p) for m, p in zip(W.mouse, W.phase)]
        rows.append(W[["mouse", "day", "phase", "cell", "condition", "rate_blocks"]])
    R = pd.concat(rows, ignore_index=True)
    R["sex"] = R.mouse.map(SEX)
    return R


def s3_rates(R):
    fig, ax = plt.subplots(1, 2, figsize=(12.6, 5.8), gridspec_kw=dict(width_ratios=[1.3, 1]))   # slide width (10-22)
    out = {}
    a = ax[0]
    rng = np.random.default_rng(0)
    for i, c in enumerate(E.COND):
        for j, sx in enumerate(("F", "M")):
            v = R[(R.condition == c) & (R.sex == sx)].rate_blocks.dropna().values
            x = i + (j - 0.5) * 0.38
            if len(v):
                a.boxplot(v, positions=[x], widths=0.32, showfliers=False, medianprops=dict(color="k", lw=2),
                          boxprops=dict(color=SC[sx], lw=2), whiskerprops=dict(color=SC[sx]), capprops=dict(color=SC[sx]))
                a.scatter(x + rng.uniform(-0.1, 0.1, len(v)), v, s=6, color=SC[sx], alpha=0.35, lw=0)
            out[f"{c} | {sx}"] = dict(median=float(np.median(v)) if len(v) else np.nan, n=int(len(v)))
        vf = R[(R.condition == c) & (R.sex == "F")].rate_blocks.dropna()
        vm = R[(R.condition == c) & (R.sex == "M")].rate_blocks.dropna()
        p = mannwhitneyu(vf, vm).pvalue if min(len(vf), len(vm)) > 3 else np.nan
        out[f"{c} | F vs M p"] = float(p)
        top = np.nanpercentile(np.r_[vf, vm], 97)
        PS.bracket(a, i - 0.19, i + 0.19, top * 1.05, p, "k", fs=12, inset=0, h=0.025 * top)   # female vs male neurons
    a.set_xticks(range(3), E.COND)
    a.set_ylabel("Ca transients / min\n(stimulus blocks, still frames)")
    a.set_title("A  every neuron: female (pink) vs male (teal)\nbracket = female vs male (Mann-Whitney over neurons)", loc="left")
    # 10-24 figure review: the mouse labels sat on the boxes and on each other (5937 / 5939 stacked) and panel B had no colour
    # key -> each mouse's median marker goes OUTSIDE its sex's box (females left of the pink box, males right of the teal box),
    # labels pushed apart with a leader line when two medians are close; one colour key for both panels under the figure
    b = ax[1]
    labs_b = []                                                            # (x marker, y, side, mouse)
    for k, drug in enumerate(("vehicle", "SBI-553")):
        pre = R[(R.phase == "pre") & np.array([E.drug_of(m, d) == drug for m, d in zip(R.mouse, R.day)], bool)]
        post = R[(R.phase == "post") & (R.condition == drug)]
        M = pre.merge(post, on=["mouse", "day", "cell", "sex"], suffixes=("_b", "_i"))
        M["lc"] = np.log2((M.rate_blocks_i + 0.1) / (M.rate_blocks_b + 0.1))
        for j, sx in enumerate(("F", "M")):
            v = M[M.sex == sx].lc.dropna().values
            x = 1.5 * k + (j - 0.5) * 0.38                 # days 1.5 apart: room for the mouse labels
            if len(v):
                b.boxplot(v, positions=[x], widths=0.32, showfliers=False, medianprops=dict(color="k", lw=2),
                          boxprops=dict(color=SC[sx], lw=2), whiskerprops=dict(color=SC[sx]), capprops=dict(color=SC[sx]))
                p = wilcoxon(v).pvalue if len(v) >= 6 and np.any(v != 0) else np.nan
                PS.vs_ref(b, x, 0.975, p, "vs 0", SC[sx], fs=11, axes_y=True, va="top")
                out[f"change {drug} | {sx}"] = dict(median_log2=float(np.median(v)), n=int(len(v)), p=float(p))
                side = -1 if sx == "F" else 1                                    # F: left of its box, M: right of its box
                for m in [m for m in E.MO if SEX[m] == sx]:                      # each mouse's median (black marker)
                    vm = M[(M.sex == sx) & (M.mouse == m)].lc.dropna().values
                    if len(vm):
                        xm = x + side * 0.215
                        b.scatter(xm, np.median(vm), marker=E.MK[m], s=60, color="k", zorder=4, clip_on=False)
                        labs_b.append((xm, float(np.median(vm)), side, m))
                        out[f"change {drug} | {m}"] = dict(median_log2=float(np.median(vm)), n=int(len(vm)))
    b.axhline(0, color="0.4", lw=0.8)
    b.set_xticks([0, 1.5], ["vehicle day", "SBI-553 day"])
    b.set_xlim(-0.8, 2.3)
    b.set_ylabel("log2 (injection / baseline)\nper neuron, same day")
    lo_, hi_ = b.get_ylim(); b.set_ylim(lo_, hi_ + 0.22 * (hi_ - lo_))
    lo_, hi_ = b.get_ylim()
    gap = 0.065 * (hi_ - lo_)                                              # min. vertical distance between two labels
    groups = {}
    for xm, ym, side, m in labs_b:
        groups.setdefault((round(xm, 3), side), []).append([ym, ym, m])
    for (xm, side), L in groups.items():
        L.sort(key=lambda r: r[0])
        for i in range(1, len(L)):                                         # push labels apart (upwards), then re-centre
            L[i][1] = max(L[i][1], L[i - 1][1] + gap)
        sh = (np.mean([r[0] for r in L]) - np.mean([r[1] for r in L])) if len(L) > 1 and any(r[1] != r[0] for r in L) else 0.0
        for ym, yl, m in L:
            yl += sh
            xl = xm + side * 0.08
            if abs(yl - ym) > 0.25 * gap:                                  # moved: a leader line to its marker
                b.plot([xm + side * 0.03, xl - side * 0.01], [ym, yl], color="0.35", lw=0.8, clip_on=False)
            b.text(xl, yl, m, fontsize=10.5, va="center", ha="left" if side > 0 else "right")
    b.set_title("B  change of each neuron after the injection\nstar over 'vs 0' = change differs from 0 (Wilcoxon)", loc="left")
    # 10-25 figure review: the box / whisker rule was only in the notes -> on the figure (matplotlib default whis = 1.5,
    # showfliers=False above: whiskers end at the last neuron within 1.5 x IQR of the box, points beyond are not marked)
    fig.legend(handles=[Patch(color=SC[s], label=SN[s]) for s in ("F", "M")]
               + [Line2D([], [], ls="", marker="D", color="k", ms=7, label="black marker = one mouse's median (B)"),
                  Patch(facecolor="none", edgecolor="0.3", lw=1.5, label="box = 25th-75th percentile, black line = median"),
                  Line2D([], [], color="0.3", lw=1.2, label="whiskers = last neuron within 1.5 x IQR of the box"),
                  Line2D([], [], ls="", marker="o", ms=4, color="0.5", label="dot = one neuron (A)")],
               loc="lower center", ncol=3, frameon=False, fontsize=11.5, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(os.path.join(OUT, "S3_rates_by_sex.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def s4_responsive(T):
    evs = [e for e in ["Light touch", "Mild touch", "Pin prick", "Heat", "withdrawal", "flinch", "movement onset"] if SUB is None or e in SUB]
    fig, ax = plt.subplots(1, len(evs), figsize=(19, 5.0) if SUB is None else (3.3 * len(evs) + 1.2, 5.0), sharey=True, squeeze=False)
    ax = ax[0]
    out = {}
    T = T.copy()
    T["resp"] = T.cls.isin(["activated", "inhibited"])
    T["sex"] = T.mouse.map(SEX)
    for k, ev in enumerate(evs):
        a = ax[k]
        X = T[T.event == ev]
        for i, c in enumerate(E.COND):
            for j, sx in enumerate(("F", "M")):
                Y = X[(X.condition == c) & (X.sex == sx)]
                v = 100 * Y.resp.mean() if len(Y) else np.nan
                x = i + (j - 0.5) * 0.38
                a.bar(x, v, width=0.34, color=SC[sx], alpha=0.75)
                for m in [m for m in E.MO if SEX[m] == sx]:
                    Z = Y[Y.mouse == m]
                    if len(Z):
                        a.scatter(x, 100 * Z.resp.mean(), marker=E.MK[m], color="k", s=20, zorder=3)
                        out[f"{ev} | {c} | {m}"] = float(100 * Z.resp.mean())
                out[f"{ev} | {c} | {sx}"] = float(v)
        a.set_xticks(range(3), ["base", "veh", "SBI"])
        a.set_title(E.SHORT[ev], fontsize=12.5)
    ax[0].set_ylabel("% of neurons responsive\n(activated + inhibited)")
    fig.legend(handles=[Patch(color=SC[s], label=SN[s]) for s in ("F", "M")]
               + [Line2D([], [], marker=E.MK[m], ls="", color="k", label=m) for m in E.MO],
               loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(os.path.join(OUT, f"S4_responsive_by_sex{SUF}.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def s5_traces(S):
    EE = [e for e in EVS if SUB is None or e in SUB]
    fig, ax = plt.subplots(2, len(EE), figsize=(18, 7.6) if SUB is None else (3.6 * len(EE) + 1.0, 6.2), sharex=True, squeeze=False)
    out = {}
    tt = None
    for k, ev in enumerate(EE):
        for r, sx in enumerate(("F", "M")):
            a = ax[r, k]
            for c in E.COND:
                Ys = []
                for (m, d, ph), s in S.items():
                    if SEX[m] != sx or E.cond_of(m, d, ph) != c:
                        continue
                    X, tt_, used = E.CA.peri(s, s["B"]["ev"].get(ev, []))
                    if X is None or len(used) < E.CA.MIN_TRIALS:
                        continue
                    tt = tt_
                    Xb = np.nanmean(X - np.nanmean(X[:, (tt >= -3) & (tt < 0), :], 1, keepdims=True), 0)
                    Ys.append(Xb)
                if not Ys:
                    continue
                Y = np.hstack(Ys)
                mu = E.CA.sm(np.nanmean(Y, 1)[:, None], 6)[:, 0]
                se = E.CA.sm((np.nanstd(Y, 1) / np.sqrt(Y.shape[1]))[:, None], 6)[:, 0]
                a.plot(tt, mu, color=E.CC[c], lw=2, label=f"{c} (n = {Y.shape[1]})")
                a.fill_between(tt, mu - se, mu + se, color=E.CC[c], alpha=0.2, lw=0)
                r_ = (tt >= WW.of(ev)[2]) & (tt < WW.of(ev)[3])
                out[f"{ev} | {sx} | {c}"] = float(np.nanmean(mu[r_]))
            a.axvline(0, color="0.4", lw=0.8, ls="--")
            a.axhline(0, color="0.6", lw=0.6)
            a.set_xlim(-3, 6)
            a.legend(frameon=False, fontsize=8.5, loc="upper left")
            if r == 0:
                a.set_title(E.SHORT[ev], fontsize=13)
            if k == 0:
                a.set_ylabel(f"{SN[sx]}\nz (trial-baselined, -3..0 s)", color=SC[sx])
            if r == 1:
                a.set_xlabel(f"time from {E.SHORT[ev]} (s)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"S5_traces_by_sex{SUF}.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def s6_population(S):
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8), sharey=True)
    out = {}
    for k, c in enumerate(E.COND):
        a = ax[k]
        for m in E.MO:
            ys = []
            for (mm, d, ph), s in S.items():
                if mm == m and E.cond_of(mm, d, ph) == c:
                    Zb, _ = E.binned(s, 5.0)
                    ys.append(np.nanmean(Zb, 1)[:200])
            if not ys:
                continue
            L_ = min(len(y) for y in ys)
            y = np.nanmean(np.vstack([y[:L_] for y in ys]), 0)
            x = np.arange(L_) * 5 / 60
            a.plot(x, y, color=SC[SEX[m]], lw=1.4, alpha=0.85)
            a.plot(x[::24], y[::24], ls="", marker=E.MK[m], color=SC[SEX[m]], ms=6)
            out[f"{c} | {m}"] = dict(first3=float(np.nanmean(y[(x > 0.25) & (x < 3)])), blocks=float(np.nanmean(y[x >= 3])))
        for b0, b1 in E.BLOCK_S:
            a.axvspan(b0 / 60, b1 / 60, color="#f5b041", alpha=0.12, lw=0)
        a.set_title(c, color=E.CC[c], fontsize=13.5, fontweight="bold")
        a.set_xlabel("time in session (min)")
        if k == 0:
            a.set_ylabel("population mean z\n(dF/F / robust SD), 5-s bins")
    legend_mice(fig, y=-0.06)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(os.path.join(OUT, "S6_population_by_sex.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def s7_fate():
    F = pd.read_csv(os.path.join(E.OD, "P5_fate.csv"))
    F["mouse"] = F.mouse.astype(str)
    F["sex"] = F.mouse.map(SEX)
    X = F[(F.event == "movement onset") & (F.group == "activated")].copy()
    X["change"] = X.inj - X.base_heldout
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    out = {}
    rng = np.random.default_rng(1)
    for k, drug in enumerate(("vehicle", "SBI-553")):
        for j, sx in enumerate(("F", "M")):
            v = X[(X.drug == drug) & (X.sex == sx)].change.values
            x = k + (j - 0.5) * 0.38
            if len(v):
                ax.boxplot(v, positions=[x], widths=0.32, showfliers=False, medianprops=dict(color="k", lw=2),
                           boxprops=dict(color=SC[sx], lw=2), whiskerprops=dict(color=SC[sx]), capprops=dict(color=SC[sx]))
                ax.scatter(x + rng.uniform(-0.08, 0.08, len(v)), v, s=14, color=SC[sx], alpha=0.6, lw=0)
                for m in [m for m in E.MO if SEX[m] == sx]:
                    vm = X[(X.drug == drug) & (X.mouse == m)].change.values
                    if len(vm):
                        ax.scatter(x + 0.2, np.median(vm), marker=E.MK[m], s=70, color="k", zorder=4)
                        ax.text(x + 0.25, np.median(vm), f"{m} (n={len(vm)})", fontsize=8.5, va="center")
                        out[f"{drug} | {m}"] = dict(median=float(np.median(vm)), n=int(len(vm)))
                p = wilcoxon(v).pvalue if len(v) >= 6 else np.nan
                ax.text(x, 0.97, f"{PS.star(p)}\nvs 0\nn={len(v)}", ha="center", va="top", fontsize=10, color=SC[sx],
                        transform=ax.get_xaxis_transform())
                out[f"{drug} | {sx}"] = dict(median=float(np.median(v)), n=int(len(v)), p=float(p))
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xticks([0, 1], ["vehicle day", "SBI-553 day"])
    lo_, hi_ = ax.get_ylim(); ax.set_ylim(lo_, hi_ + 0.25 * (hi_ - lo_))
    ax.set_ylabel("change of the response to movement onset\n(z: injection - held-out baseline)")
    ax.set_title("star over 'vs 0' = the change of that sex's neurons differs from 0 (Wilcoxon); black = one mouse's median",
                 fontsize=10.5)
    ax.legend(handles=[Patch(color=SC[s], label=SN[s]) for s in ("F", "M")], frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "S7_movement_neurons_by_sex.png"), dpi=140)
    plt.close(fig)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    D = pd.read_csv(os.path.join(E.OD, "P6_per_delivery.csv"))
    D["mouse"] = D.mouse.astype(str)
    T = pd.read_csv(os.path.join(E.OD, "cells_classes.csv"))
    T["mouse"] = T.mouse.astype(str)
    S = E.load_all()
    J = dict(s1=s1_behaviour(D), s2=s2_drug_effect(D), s3=s3_rates(load_rates()), s4=s4_responsive(T), s5=s5_traces(S),
             s6=s6_population(S), s7=s7_fate())
    E.jdump(J, os.path.join(OUT, "sex_summary.json"))
    global SUB, SUF
    for suf, sub, fns in (("_a", ["withdrawal", "flinch", "attending"], (s1_behaviour, s2_drug_effect)),
                          ("_b", ["lick/bite", "escape/rearing", "movement"], (s1_behaviour, s2_drug_effect))):
        SUB, SUF = sub, suf
        for fn in fns:
            fn(D)
    for suf, sub in (("_stim", ["Light touch", "Mild touch", "Pin prick", "Heat"]), ("_beh", ["withdrawal", "flinch", "movement onset"])):
        SUB, SUF = sub, suf
        s4_responsive(T)
        s5_traces(S)
    SUB, SUF = None, ""
    print("->", OUT)


if __name__ == "__main__":
    main()
