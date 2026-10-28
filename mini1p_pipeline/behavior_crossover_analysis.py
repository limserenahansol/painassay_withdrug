"""behavior_crossover_analysis.py - behaviour of Day 1 + Day 2 together: SBI-553 vs vehicle WITHIN each mouse.

Crossover: every mouse got SBI-553 on one day and vehicle on the other (Day 1: 5941, 5939 SBI-553; Day 2: 5937, 1191).
Per mouse 4 sessions: baseline and injection on each day. Every count is divided by the deliveries of its session
(deliveries differ between sessions): events per delivery = events in the response windows / deliveries
(reflexes: delivery to 2 s or the next delivery; held / automatic behaviours: bout starts 0-3 s after the delivery).
  per mouse   SBI-553 injection vs vehicle injection (exact Poisson rate test); the within-day change
              (injection / that day's baseline) on the SBI-553 day vs the vehicle day: Poisson GLM
              events ~ phase * drug + offset(log deliveries), the interaction term (z test)
  across mice n = 4, each its own control: the 4 changes (SBI-553 minus vehicle) - signs and a Wilcoxon signed-rank
              (with 4 mice it cannot go below p = 0.125; shown, not a claim)
Also spontaneous movement (% time, first 3 min) and grooming (s per session).
Writes analysis\\behavior_crossover\\crossover_per_session.csv, crossover_tests.csv, crossover_*.png

  python behavior_crossover_analysis.py
"""
import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pairstars as PS  # noqa: E402  (user rule 10-22: every star shows the pair it compares)
import behavior_day_analysis as BA  # noqa: E402

BASE = r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553"
OUT = os.path.join(BASE, "analysis", "behavior_crossover")
FPS = 30.0
ARMS = {1: {"5941": "SBI-553", "5939": "SBI-553", "5937": "vehicle", "1191": "vehicle"},
        2: {"5937": "SBI-553", "1191": "SBI-553", "5941": "vehicle", "5939": "vehicle"}}
# 10-23: the timing-corrected copies of the final scoring (scoring_timefix.py: withdrawal / flinch -0.40 s Day 1, -0.37 s
# Day 2; deliveries moved only where their own reflex came first; 3 Day-2 clicker fixes). SCORING_SUFFIX="" = the originals.
SCORING_SFX = os.environ.get("SCORING_SUFFIX", "_timefix")
FINAL = {1: "20261019_final" + SCORING_SFX, 2: "20261021_final" + SCORING_SFX}
MOUSE_ORDER = ["5941", "5939", "5937", "1191"]
CC = {"baseline": "#7f8c8d", "vehicle": "#2471a3", "SBI-553": "#c0392b"}
BEH = ["withdrawal", "flinch", "attending", "lick/bite", "guarding", "escape/rearing", "general movement", "grooming"]
STIM = ["Light touch", "Mild touch", "Pin prick", "Heat"]


def counts(d):
    rows = []
    held_on = {nm: BA.spans(v)[0] for nm, v in d["held"].items()}
    for i, (f, k) in enumerate(zip(d["dF"], d["dT"])):
        if not 1 <= k <= 4:
            continue
        nxt = d["dF"][i + 1] if i + 1 < len(d["dF"]) else d["n"]
        hi = min(f + int(2 * FPS), nxt - 1)
        r = dict(stim=d["stim"][k - 1], withdrawal=int(np.sum((d["W"] >= f) & (d["W"] <= hi))),
                 flinch=int(np.sum((d["F"] >= f) & (d["F"] <= hi))))
        for nm, on in held_on.items():
            r[nm] = int(np.sum((on >= f) & (on <= f + 3 * FPS)))
        rows.append(r)
    return pd.DataFrame(rows)


def rate_p(k1, n1, k2, n2):
    if k1 + k2 == 0:
        return np.nan
    return binomtest(int(k2), int(k1 + k2), n2 / (n1 + n2)).pvalue


def glm_interaction(rows):
    """rows: (events, deliveries, post 0/1, sbi_day 0/1) x 4 -> (exp(interaction), p)"""
    try:
        import statsmodels.api as sm
    except Exception:                                                     # noqa: BLE001
        return np.nan, np.nan
    y = np.array([r[0] for r in rows], float); off = np.log([r[1] for r in rows])
    X = np.array([[1, r[2], r[3], r[2] * r[3]] for r in rows], float)
    if y.sum() == 0 or np.any(y == 0) and y.sum() < 3:
        return np.nan, np.nan
    try:
        res = sm.GLM(y, X, family=sm.families.Poisson(), offset=off).fit()
        return float(np.exp(res.params[3])), float(res.pvalues[3])
    except Exception:                                                     # noqa: BLE001
        return np.nan, np.nan


def main():
    warnings.filterwarnings("ignore")          # statsmodels: separation when a behaviour is 0 in a session
    os.makedirs(OUT, exist_ok=True)
    sess = []
    for day in (1, 2):
        for p in sorted(glob.glob(os.path.join(BASE, "scoring_mini1p", FINAL[day], "ScoringAB_*.mat"))):
            d = BA.load(p)
            m = d["mouse"].strip()
            C = counts(d)
            cond = "baseline" if d["phase"] == "baseline" else ARMS[day][m]
            mv = d["held"]["general movement"]
            gr = d["held"]["grooming"]
            r = dict(mouse=m, day=day, phase=d["phase"], condition=cond, drug_day=ARMS[day][m], deliveries=len(C),
                     movement_pct_nostim=100 * float(np.mean(mv[:int(180 * FPS)])), grooming_s=float(gr.sum() / FPS))
            for b in BEH:
                r[b] = int(C[b].sum()) if b in C else 0
            for s in STIM:
                cs = C[C.stim == s]
                r[f"n | {s}"] = len(cs)
                for b in ("withdrawal", "flinch"):
                    r[f"{b} | {s}"] = int(cs[b].sum()) if len(cs) else 0
            sess.append(r)
    S = pd.DataFrame(sess)
    S.to_csv(os.path.join(OUT, "crossover_per_session.csv"), index=False)
    tests = []
    for m in MOUSE_ORDER:
        X = S[S.mouse == m]
        get = lambda drug, ph: X[(X.drug_day == drug) & (X.phase == ph)].iloc[0]
        sp, si, vp, vi = get("SBI-553", "baseline"), get("SBI-553", "injection"), get("vehicle", "baseline"), get("vehicle", "injection")
        for b in BEH + [f"{b} | {s}" for s in STIM for b in ("withdrawal", "flinch")]:
            nb = (lambda r: r[f"n | {b.split(' | ')[1]}"]) if " | " in b else (lambda r: r["deliveries"])
            rows = [(sp[b], nb(sp), 0, 1), (si[b], nb(si), 1, 1), (vp[b], nb(vp), 0, 0), (vi[b], nb(vi), 1, 0)]
            if min(r[1] for r in rows) == 0:
                continue
            ratio_int, p_int = glm_interaction(rows)
            tests.append(dict(mouse=m, behaviour=b, sbi_base=sp[b] / nb(sp), sbi_inj=si[b] / nb(si), veh_base=vp[b] / nb(vp),
                              veh_inj=vi[b] / nb(vi), sbi_change=si[b] / nb(si) - sp[b] / nb(sp),
                              veh_change=vi[b] / nb(vi) - vp[b] / nb(vp),
                              p_sbi_vs_veh_injection=rate_p(vi[b], nb(vi), si[b], nb(si)),
                              change_ratio_sbi_over_veh=ratio_int, p_change_interaction=p_int,
                              n_deliveries=f"{nb(sp)}/{nb(si)}/{nb(vp)}/{nb(vi)}"))
    T = pd.DataFrame(tests)
    # across mice: SBI change - vehicle change, per behaviour
    acr = []
    for b, g in T.groupby("behaviour"):
        dlt = (g.sbi_change - g.veh_change).values
        p = wilcoxon(dlt).pvalue if len(dlt) == 4 and np.any(dlt != 0) else np.nan
        acr.append(dict(behaviour=b, n_mice=len(dlt), n_sbi_lower=int((dlt < 0).sum()), n_sbi_higher=int((dlt > 0).sum()),
                        mean_difference=float(np.mean(dlt)), p_wilcoxon_mice=p))
    Aac = pd.DataFrame(acr)
    T.to_csv(os.path.join(OUT, "crossover_tests.csv"), index=False)
    Aac.to_csv(os.path.join(OUT, "crossover_across_mice.csv"), index=False)
    # 10-24 (figure clarity): 15 x 6.3 in figures (the deck box ratio), 2 x 4 panels with the legend in the 8th, fonts >= 11.5 pt
    # (>= 9 pt on the slide), markers at 0 not clipped, no in-figure title that repeats the slide title. Data and tests unchanged.
    FS = 11.5
    plt.rcParams.update({"font.size": FS, "axes.titlesize": 13, "axes.labelsize": 12, "xtick.labelsize": FS, "ytick.labelsize": FS,
                         "legend.fontsize": FS, "axes.spines.top": False, "axes.spines.right": False})
    MC = {"5941": "#e67e22", "5939": "#8e44ad", "5937": "#16a085", "1191": "#34495e"}
    MK = {"5941": "o", "5939": "s", "5937": "^", "1191": "D"}
    # figure 1: events per delivery, the 4 sessions per mouse
    panels = ["withdrawal", "flinch", "attending", "lick/bite", "escape/rearing", "general movement", "grooming"]
    fig, ax = plt.subplots(2, 4, figsize=(15, 6.3))
    fig.subplots_adjust(left=0.065, right=0.99, top=0.94, bottom=0.09, wspace=0.32, hspace=0.62)
    ax = ax.ravel()
    for k, b in enumerate(panels):
        a = ax[k]
        for i, m in enumerate(MOUSE_ORDER):
            g = T[(T.mouse == m) & (T.behaviour == b)]
            if not len(g):
                continue
            g = g.iloc[0]
            x0 = i
            a.plot([x0 - 0.27, x0 - 0.09], [g.veh_base, g.veh_inj], "-", color=CC["vehicle"], lw=1.5, clip_on=False, zorder=4)
            a.plot([x0 + 0.09, x0 + 0.27], [g.sbi_base, g.sbi_inj], "-", color=CC["SBI-553"], lw=1.5, clip_on=False, zorder=4)
            a.plot([x0 - 0.27, x0 + 0.09], [g.veh_base, g.sbi_base], "o", mfc="white", mec=CC["baseline"], mew=1.3, ms=6.5,
                   clip_on=False, zorder=5)
            a.plot(x0 - 0.09, g.veh_inj, "o", color=CC["vehicle"], mec="white", mew=0.7, ms=7.5, clip_on=False, zorder=5)
            a.plot(x0 + 0.27, g.sbi_inj, "o", color=CC["SBI-553"], mec="white", mew=0.7, ms=7.5, clip_on=False, zorder=5)
            if np.isfinite(g.p_change_interaction) and g.p_change_interaction < 0.05:   # vehicle-day change vs SBI-553-day change
                PS.bracket(a, x0 - 0.18, x0 + 0.18, max(g.veh_inj, g.sbi_inj, g.veh_base, g.sbi_base) * 1.06, g.p_change_interaction, "k",
                           fs=13, inset=0, h=0.02 * max(g.veh_inj, g.sbi_inj, g.veh_base, g.sbi_base, 0.05))
        a.set_xticks(range(4), MOUSE_ORDER)
        a.set_xlim(-0.5, 3.5)
        a.set_xlabel("mouse", labelpad=2)
        a.set_title(b, loc="left")
        a.set_ylim(bottom=0)
        a.yaxis.set_major_locator(plt.MaxNLocator(5))
        if k % 4 == 0:
            a.set_ylabel("events per delivery")
    ax[7].set_axis_off()
    ax[7].legend(handles=[Line2D([], [], marker="o", ls="", mfc="white", mec=CC["baseline"], mew=1.3, ms=7, label="baseline of that day"),
                          Line2D([], [], marker="o", ls="-", color=CC["vehicle"], ms=7, label="vehicle day:\nbaseline -> injection"),
                          Line2D([], [], marker="o", ls="-", color=CC["SBI-553"], ms=7, label="SBI-553 day:\nbaseline -> injection"),
                          Line2D([], [], color="k", ls="", marker="$*$", ms=10,
                                 label="bracket: vehicle-day change\nvs SBI-553-day change\n(Poisson GLM)\n* p < 0.05, ** p < 0.01,\n*** p < 0.001")],
                 loc="center left", bbox_to_anchor=(-0.05, 0.5), frameon=False, labelspacing=0.9, handlelength=1.8)
    fig.savefig(os.path.join(OUT, "crossover_per_delivery.png"), dpi=150); plt.close(fig)
    # figure 2: SBI-553 change vs vehicle change per mouse (paired); a colour + marker per mouse instead of overlapping ID labels
    fig, ax = plt.subplots(2, 4, figsize=(15, 6.3))
    fig.subplots_adjust(left=0.085, right=0.99, top=0.9, bottom=0.11, wspace=0.42, hspace=0.75)
    ax = ax.ravel()
    for k, b in enumerate(panels):
        a = ax[k]
        g = T[T.behaviour == b].set_index("mouse")
        for m in MOUSE_ORDER:
            if m not in g.index:
                continue
            a.plot([0, 1], [g.at[m, "veh_change"], g.at[m, "sbi_change"]], "-", marker=MK[m], color=MC[m], mec="k", mew=0.5, ms=7,
                   lw=1.4, clip_on=False, zorder=4)
        a.axhline(0, color="0.5", lw=0.9, ls=":")
        a.set_xticks([0, 1], ["vehicle\nday", "SBI-553\nday"])
        for t, c in zip(a.get_xticklabels(), (CC["vehicle"], CC["SBI-553"])):
            t.set_color(c)
            t.set_fontweight("bold")
        a.set_xlim(-0.35, 1.35)
        a.yaxis.set_major_locator(plt.MaxNLocator(5))
        r = Aac[Aac.behaviour == b]
        a.set_title(f"{b}\nSBI lower in {int(r.n_sbi_lower.iloc[0])}/4 mice" if len(r) else b, loc="left", fontsize=12)
    ax[7].set_axis_off()
    ax[7].legend(handles=[Line2D([], [], marker=MK[m], color=MC[m], mec="k", mew=0.5, ms=7, lw=1.4, label=f"mouse {m}") for m in MOUSE_ORDER]
                 + [Line2D([], [], color="0.5", lw=0.9, ls=":", label="0 = same as that\nday's baseline")],
                 loc="center left", bbox_to_anchor=(-0.05, 0.5), frameon=False, labelspacing=0.7, handlelength=1.8)
    fig.text(0.014, 0.5, "change after the injection, events per delivery\n(injection session - same-day baseline session)",
             rotation=90, ha="center", va="center", fontsize=12)
    fig.savefig(os.path.join(OUT, "crossover_change_paired.png"), dpi=150); plt.close(fig)
    # figure 3: spontaneous movement + grooming per session
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.subplots_adjust(left=0.075, right=0.99, top=0.86, bottom=0.14, wspace=0.25)
    for k, (col, lab) in enumerate((("movement_pct_nostim", "time moving, first 3 min (%)"), ("grooming_s", "grooming (s per session)"))):
        a = ax[k]
        for i, m in enumerate(MOUSE_ORDER):
            X = S[S.mouse == m]
            for drug, dx in (("vehicle", -0.18), ("SBI-553", 0.18)):
                b0 = X[(X.drug_day == drug) & (X.phase == "baseline")][col].iloc[0]
                b1 = X[(X.drug_day == drug) & (X.phase == "injection")][col].iloc[0]
                a.plot([i + dx - 0.07, i + dx + 0.07], [b0, b1], "-", color=CC[drug], lw=1.5, clip_on=False, zorder=4)
                a.plot(i + dx - 0.07, b0, "o", mfc="white", mec=CC["baseline"], mew=1.3, ms=6.5, clip_on=False, zorder=5)
                a.plot(i + dx + 0.07, b1, "o", color=CC[drug], mec="white", mew=0.7, ms=7.5, clip_on=False, zorder=5)
        a.set_xticks(range(4), MOUSE_ORDER); a.set_xlabel("mouse"); a.set_ylabel(lab); a.set_ylim(bottom=0)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", mfc="white", mec=CC["baseline"], mew=1.3, label="baseline (same day)"),
                        Line2D([], [], marker="o", color=CC["vehicle"], label="vehicle day: baseline -> injection"),
                        Line2D([], [], marker="o", color=CC["SBI-553"], label="SBI-553 day: baseline -> injection")],
               loc="upper center", ncol=3, frameon=False)
    fig.savefig(os.path.join(OUT, "crossover_movement_grooming.png"), dpi=150); plt.close(fig)
    pd.set_option("display.width", 220)
    print(Aac.round(3).to_string(index=False))
    print(T[T.behaviour.isin(["withdrawal", "flinch"])][["mouse", "behaviour", "veh_base", "veh_inj", "sbi_base", "sbi_inj",
                                                          "p_sbi_vs_veh_injection", "p_change_interaction"]].round(3).to_string(index=False))
    print("->", OUT)


if __name__ == "__main__":
    main()
