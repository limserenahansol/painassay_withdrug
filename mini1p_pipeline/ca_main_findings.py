"""One figure with the result of one imaging day (CA_DAY = 1 | 2) and how much it can be trusted (numbers from the analysis files).

A  stimulus blocks, still frames: per mouse the median over cells of log2(post / pre) transient rate
   (dot, 95 % bootstrap CI over cells; Wilcoxon post vs pre over cells) and the range of that median
   over every analysis choice of R1 (grey bar)
B  the same for the no-stimulus start (first 3 min)
C  the SBI-553 - vehicle difference of A without each mouse in turn (R2)
D  population response per mouse and session (R7): RMS over cells of the mean event response, as a
   multiple of the same events shifted inside their blocks (1 = no response beyond ongoing activity)

  python ca_main_findings.py --base C:/Users/hsollim/Documents/HEAL_mini1p_SBI553
"""
import argparse
import os

import sys
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import safe_io  # noqa: E402,F401  (a figure save retries on a transient Windows write error)
import ca_day1_analysis as _d1  # noqa: E402
import deck_cfg as C  # noqa: E402
MICE = [(m, a) for m, a, _ in _d1.MICE]   # by day (CA_DAY)
ARMC = {"SBI-553": "#c0392b", "vehicle": "#2471a3"}
MAIN = "main: OASIS transients, still frames"


def pstar(p):
    if not np.isfinite(p):
        return "n.a."
    return "p<0.001" if p < 0.001 else f"p={p:.2g}"


def panel_ratio(a, R1, wn, title, lo, hi):
    V = R1[R1.window == wn]
    a.set_ylim(lo, hi)
    for i, (m, arm) in enumerate(MICE):
        X = V[V.mouse.astype(str) == m]
        if not len(X):
            continue
        a.plot([i, i], [X.log2_ratio_median.min(), X.log2_ratio_median.max()], color="0.75", lw=9, solid_capstyle="butt", zorder=1)
        r = X[X.variant == MAIN].iloc[0]
        a.errorbar(i, r.log2_ratio_median, yerr=[[r.log2_ratio_median - r.ci_lo], [r.ci_hi - r.log2_ratio_median]],
                   fmt="o", color=ARMC[arm], ms=7, capsize=3, zorder=3)
        a.text(i, hi - 0.04 * (hi - lo), f"n={int(r.n_cells)}\n{pstar(r.p_wilcoxon)}", ha="center", va="top", fontsize=7)
    a.axhline(0, color="0.4", lw=0.8)
    a.set_xticks(range(len(MICE)), [f"{m}\n{arm}" for m, arm in MICE], fontsize=8)
    for t, (m, arm) in zip(a.get_xticklabels(), MICE):
        t.set_color(ARMC[arm])
    a.set_ylabel("log2(post / pre) transient rate\n(median over cells)", fontsize=8)
    a.set_title(title, fontsize=9)
    a.set_xlim(-0.6, len(MICE) - 0.4)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    B = a.base
    rb = C.A(B, "ca_robustness")
    R1 = pd.read_csv(os.path.join(rb, "R1_robustness_per_mouse.csv"))
    RL = pd.read_csv(os.path.join(rb, "R2_leave_one_mouse_out.csv"))
    RA = pd.read_csv(os.path.join(rb, "R1_robustness_arm.csv"))
    R7 = pd.read_csv(C.A(B, "ca_responsive_change", "R7_population_response_test.csv"))
    fig, ax = plt.subplots(1, 4, figsize=(17, 5.2), gridspec_kw=dict(width_ratios=[1.2, 1.2, 1.1, 1.5]))
    V = R1[R1.window.isin(["stimulus blocks", "no-stimulus start"])]
    lo = min(V.log2_ratio_median.min(), V.ci_lo.min()) - 0.15
    hi = max(V.log2_ratio_median.max(), V.ci_hi.max(), 0.2) + 0.55
    panel_ratio(ax[0], R1, "stimulus blocks", "A  stimulus blocks (still frames)", lo, hi)
    panel_ratio(ax[1], R1, "no-stimulus start", "B  first 3 min, no stimulus (still frames)", lo, hi)
    Q = RL[RL.window == "stimulus blocks"]
    lab = ["none" if d == "none" else f"- {d}" for d in Q.left_out]
    col = ["k"] + [ARMC[dict(MICE)[d]] for d in Q.left_out[1:]]
    ax[2].bar(range(len(Q)), Q.sbi_minus_vehicle, color=col, alpha=0.8)
    for i, v in enumerate(Q.sbi_minus_vehicle):
        ax[2].text(i, v + (0.03 if v >= 0 else -0.03), f"{v:+.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=7.5)
    ax[2].axhline(0, color="0.4", lw=0.8)
    yl = ax[2].get_ylim(); ax[2].set_ylim(yl[0] - 0.1, max(yl[1], 0) + 0.15)
    ax[2].set_xticks(range(len(Q)), lab, fontsize=8)
    ax[2].set_xlabel("mouse left out", fontsize=8)
    ax[2].set_ylabel("SBI-553 - vehicle, log2 ratio\n(mean of mouse medians)", fontsize=8)
    ax[2].set_title("C  arm difference of A without each mouse", fontsize=9)
    evs = ["Light touch", "Pin prick", "Heat", "withdrawal", "movement onset"]
    R7["mouse"] = R7.mouse.astype(str)
    for i, (m, arm) in enumerate(MICE):
        for k, ev in enumerate(evs):
            y = []
            for ph in ("pre", "post"):
                r = R7[(R7.mouse == m) & (R7.event == ev) & (R7.phase == ph)]
                y.append(float(r.energy.iloc[0] / r.energy_null_median.iloc[0]) if len(r) else np.nan)
                if len(r) and r.q_energy.iloc[0] <= 0.05:
                    ax[3].text(k + (i - 1.5) * 0.18 + (0.05 if ph == "post" else -0.05), y[-1] + 0.03, "*", ha="center", fontsize=9)
            x = k + (i - 1.5) * 0.18
            ax[3].plot([x - 0.05, x + 0.05], y, color=ARMC[arm], lw=1, alpha=0.7)
            ax[3].plot(x - 0.05, y[0], "o", mfc="w", mec=ARMC[arm], ms=5, label=f"{m} ({arm}) baseline" if k == 0 else None)
            ax[3].plot(x + 0.05, y[1], "o", color=ARMC[arm], ms=5, label=f"{m} injection" if k == 0 else None)
    ax[3].axhline(1, color="0.4", lw=0.8)
    ax[3].set_xticks(range(len(evs)), evs, fontsize=8, rotation=15)
    ax[3].set_ylabel("population response / same events shifted in their block\n(RMS over cells of the mean response; 1 = none)", fontsize=8)
    ax[3].set_title("D  is there a population response? (R7)\n* = that point vs its shifted-time null (BH q <= 0.05)", fontsize=9)
    ax[3].legend(fontsize=6, frameon=False, ncol=2, loc="upper left")
    q = RA[(RA.window == "stimulus blocks")]
    neg = int((q.sbi_minus_vehicle < 0).sum())
    loo = Q[Q.left_out != "none"]
    flip = [f"{d} ({v:+.2f})" for d, v in zip(loo.left_out, loo.sbi_minus_vehicle) if np.sign(v) != np.sign(Q.sbi_minus_vehicle.iloc[0])]
    main_r = R1[(R1.window == "stimulus blocks") & (R1.variant == MAIN)]
    fall = [f"{m} ({arm})" for m, arm in MICE if (main_r[main_r.mouse.astype(str) == m].log2_ratio_median < 0).any()]
    fig.suptitle(f"Day {C.DAY}: activity change from the baseline to the injection session, per mouse (SBI-553 red, vehicle blue)",
                 fontsize=11, y=0.995)
    fig.text(0.01, 0.01,
             f"Grey bar = range of the mouse median over {q.variant.nunique()} analysis choices (R1: detector, threshold, traces, frames, cells, eps). "
             f"Stats: Wilcoxon signed-rank post vs pre over cells within each mouse (A, B); D: 2000 within-block shifts, BH over all tests.\n"
             f"SBI-553 - vehicle < 0 in {neg}/{len(q)} choices; leaving out one mouse changes its sign for: {', '.join(flip) or 'none'} (C). "
             f"Median falls in: {', '.join(fall) or 'none'}. "
             "2 mice per arm, so no arm-level test (minimum permutation p = 0.33).",
             fontsize=7.5)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    out = a.out or os.path.join(rb, f"MAIN_day{C.DAY}_findings.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print("->", out)


if __name__ == "__main__":
    main()
