"""make_locomotion_figure.py  -  the video-measured activity result.

WHY THIS IS THE DECISIVE PANEL
    Every scored behaviour fell on the drug day, which analgesia and sedation
    both predict. The 5 min BASELINE separates them: no stimulus is delivered
    during it, so there is no pain to relieve. If movement is already halved
    there, the reduction cannot be analgesia.

    This measure comes from the video pixels, not from the manual scoring, so
    it is independent of how the behaviours were labelled.

WHAT IS PLOTTED
    speed_px_s    displacement of the silhouette centroid. Geometric, so an
                  illumination difference between days does not move it.
    frac_moving   fraction of samples above 12 px/s.
    area_px       silhouette area - a hunched, still animal presents a
                  smaller blob.

    motion_index is deliberately NOT the headline: it is a pixel-difference
    measure and the Day 2 recordings are dimmer (backdrop 175 vs 200 grey
    levels), which lowers it whether or not the animal moved less.

USAGE
    python measure_locomotion.py --day1 <side> --day2 <side> --out <folder>
    python make_locomotion_figure.py --summary <folder>/Locomotion_summary.csv \\
                                     --out <figures folder>

Hansol Lim - HEAL mini1p / SBI-553
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

C1, C2 = "#444444", "#C0483B"
PERIODS = ["baseline", "block1", "block2", "block3", "block4"]
PNICE = {"baseline": "Baseline\n(NO stimulus)", "block1": "Block 1",
         "block2": "Block 2", "block3": "Block 3", "block4": "Block 4"}
plt.rcParams.update({"font.size": 11, "figure.dpi": 130})


def paired_p(x, y):
    if len(x) < 2 or np.allclose(np.asarray(y) - np.asarray(x), 0):
        return np.nan
    try:
        return float(stats.wilcoxon(x, y, zero_method="wilcox",
                                    method="exact").pvalue)
    except ValueError:
        return np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    S = pd.read_csv(a.summary)
    days = sorted(S.day.unique())
    if len(days) < 2:
        raise SystemExit("need two days in the summary")
    d1, d2 = days[0], days[1]
    mice = sorted(S.mouse.unique())

    # Short axis labels. The full day strings wrapped over four lines under
    # the first panel and pushed everything sideways.
    SHORT = {d1: "Day 1\nno drug", d2: "Day 2\nSBI-553"}
    fig = plt.figure(figsize=(15.2, 5.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.25, 1.0], wspace=.30)

    # ---- panel 1: baseline only, the decisive comparison ----------------
    ax = fig.add_subplot(gs[0, 0])
    b = S[S.period == "baseline"]
    w = b.pivot_table(index="mouse", columns="day", values="speed_px_s")
    w = w.reindex(mice).dropna()
    x, y = w[d1].to_numpy(), w[d2].to_numpy()
    for u, v, m in zip(x, y, w.index):
        ax.plot([1, 2], [u, v], "-o", color="#AAA", lw=1.3, ms=5, zorder=3)
        ax.annotate(m, (1, u), xytext=(-24, -4), textcoords="offset points",
                    fontsize=8.5, color="#555")
    for i, (v, c) in enumerate(((x, C1), (y, C2)), start=1):
        m = v.mean()
        se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0
        ax.errorbar([i], [m], yerr=[se], fmt="s", color=c, ms=13, capsize=7,
                    lw=2.4, mec="white", mew=1.6, zorder=6)
    p = paired_p(x, y)
    ax.set_xlim(.55, 2.45)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([SHORT[d1], SHORT[d2]], fontsize=10)
    ax.set_ylabel("silhouette centroid speed  (px/s)", fontsize=10.5)
    ax.set_ylim(bottom=0)
    ax.set_title(f"BASELINE only — no stimulus given\n"
                 f"×{y.mean() / x.mean():.2f}, {int((y < x).sum())}/{len(x)} "
                 f"mice down, paired p = {p:.3f}",
                 fontsize=11, fontweight="bold")
    ax.grid(alpha=.2, axis="y")

    # ---- panel 2: every period, so it is clearly not stimulus-specific --
    ax = fig.add_subplot(gs[0, 1])
    for day, col, mk in ((d1, C1, "-o"), (d2, C2, "--s")):
        m, e = [], []
        for per in PERIODS:
            v = S[(S.day == day) & (S.period == per)].speed_px_s.dropna()
            m.append(v.mean() if len(v) else np.nan)
            e.append(v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0)
        ax.errorbar(range(len(PERIODS)), m, yerr=e, fmt=mk, color=col,
                    lw=2.3, ms=7, capsize=5, label=day)
    ax.set_xticks(range(len(PERIODS)))
    ax.set_xticklabels([PNICE[p] for p in PERIODS], fontsize=8.5)
    ax.set_ylabel("centroid speed  (px/s)", fontsize=10.5)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9.5, frameon=False)
    ax.set_title("The gap is there from the start\n"
                 "not only during the stimulus blocks",
                 fontsize=11, fontweight="bold")
    ax.grid(alpha=.2)

    # ---- panel 3: the three measures, whole session ---------------------
    ax = fig.add_subplot(gs[0, 2])
    meas = [("speed_px_s", "centroid\nspeed"),
            ("frac_moving", "fraction of\ntime moving"),
            ("area_px", "silhouette\narea (posture)")]
    for i, (col, lab) in enumerate(meas):
        w = (S.groupby(["day", "mouse"])[col].mean().reset_index()
             .pivot_table(index="mouse", columns="day", values=col)
             .reindex(mice).dropna())
        rr = (w[d2] / w[d1]).to_numpy()
        pp = paired_p(w[d1].to_numpy(), w[d2].to_numpy())
        ax.plot(np.full(len(rr), i) + np.linspace(-.14, .14, len(rr)), rr,
                "o", ms=6, color=C2, alpha=.85, zorder=4)
        ax.plot([i - .27, i + .27], [np.median(rr)] * 2, "-", color=C2,
                lw=3.2, zorder=5)
        ax.text(i, 1.12, f"p={pp:.3f}", ha="center", fontsize=8.5,
                color="#333")
    ax.axhline(1, color="k", ls="--", lw=1.2)
    ax.set_xticks(range(len(meas)))
    ax.set_xticklabels([m[1] for m in meas], fontsize=9)
    ax.set_ylabel("Day 2 / Day 1", fontsize=10.5)
    ax.set_ylim(0, 1.25)
    ax.set_title("All three, whole session\none dot per mouse",
                 fontsize=11, fontweight="bold")
    ax.grid(alpha=.2, axis="y")

    fig.suptitle(
        "Activity measured from the VIDEO, independent of the manual scoring "
        "  ·   silhouette centroid, 5 Hz, all 12 sessions\n"
        "No stimulus is given during the baseline, so there is no pain to "
        "relieve there - a halving of movement cannot be analgesia.",
        fontsize=12, y=.99)
    # tight_layout fights the two-line panel titles and the suptitle; reserve
    # the top strip explicitly instead
    fig.subplots_adjust(top=.78, bottom=.16, left=.055, right=.985)
    p1 = os.path.join(a.out, "V1_locomotion.png")
    fig.savefig(p1, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  {p1}")

    rows = []
    for col, lab in [("speed_px_s", "centroid speed px/s"),
                     ("frac_moving", "fraction moving"),
                     ("area_px", "silhouette area px"),
                     ("motion_index", "pixel motion index")]:
        for per in ["ALL"] + PERIODS:
            sub = S if per == "ALL" else S[S.period == per]
            w = (sub.groupby(["day", "mouse"])[col].mean().reset_index()
                 .pivot_table(index="mouse", columns="day", values=col)
                 .reindex(mice).dropna())
            if d1 not in w.columns or d2 not in w.columns or not len(w):
                continue
            xx, yy = w[d1].to_numpy(), w[d2].to_numpy()
            rows.append(dict(measure=col, period=per, n=len(xx),
                             day1=xx.mean(), day2=yy.mean(),
                             ratio=yy.mean() / xx.mean() if xx.mean() else np.nan,
                             n_down=int((yy < xx).sum()),
                             p_wilcoxon=paired_p(xx, yy)))
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(a.out, "V1_locomotion_stats.csv"), index=False)
    print(f"  {os.path.join(a.out, 'V1_locomotion_stats.csv')}")
    print("\n  baseline only:")
    for r in T[(T.period == "baseline")].itertuples():
        print(f"    {r.measure:14s} x{r.ratio:.2f}  {r.n_down}/{r.n} down  "
              f"p={r.p_wilcoxon:.3f}")


if __name__ == "__main__":
    main()
