"""compare_block_windows.py  -  does the rest minute actually carry behaviour?

THE QUESTION
    The protocol is 5 min of stimulus followed by 1 min of rest. Behaviour does
    not stop when the stimulus does: attending, licking/biting and guarding
    continue into the rest minute. A 300 s window cuts that tail off.

    Attributing the rest minute to the block that preceded it is the defensible
    reading - nothing else caused it, and the next block has not started.

WHAT IS AND IS NOT COMPARABLE BETWEEN THE TWO VERSIONS
    n_per_delivery   COMPARABLE. The rest minute contains no deliveries, so
                     the denominator is identical in both versions and any
                     increase is real extra behaviour.
    n_bouts          comparable as a raw count within a version, and the
                     difference between versions is exactly the rest-minute
                     contribution.
    rate_per_min     NOT comparable across versions. The divisor changes
                     5 -> 6 min, so it drops about 17 % mechanically.

USAGE
    python step1_block_measures.py <scored> --block-s 300 --out <dir300>
    python step1_block_measures.py <scored> --block-s 360 --out <dir360>
    python compare_block_windows.py --dir300 <dir300> --dir360 <dir360> \\
                                    --out <figures dir>

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

ORDER = ["withdrawal", "flinch", "attending", "lickbite", "guarding", "escape"]
NICE = {"withdrawal": "Paw withdrawal", "flinch": "Flinch",
        "attending": "Paw attending", "lickbite": "Licking / biting",
        "guarding": "Guarding", "escape": "Escape / rearing"}
STIM = ["Light touch", "Mild touch", "Heat", "Pin prick"]
C5, C6 = "#444444", "#1C6E8C"
plt.rcParams.update({"font.size": 11, "figure.dpi": 130})


def load(d, label):
    f = os.path.join(d, "BlockMeasures_long.csv")
    if not os.path.exists(f):
        raise SystemExit(f"missing {f} - run step1 with the matching --block-s")
    L = pd.read_csv(f)
    L = L[L.kind == "stimulus"].copy()
    L["window"] = label
    L["n_per_del"] = np.where(L.n_del > 0, L.n_bouts / L.n_del, np.nan)
    return L[["window", "mouse", "stimulus", "behaviour", "n_bouts", "n_del",
              "n_per_del", "dur_min"]]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir300", required=True)
    ap.add_argument("--dir360", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    A = load(a.dir300, "5 min block")
    B = load(a.dir360, "6 min block + rest")
    D = pd.concat([A, B], ignore_index=True)
    D.to_csv(os.path.join(a.out, "BlockWindow_comparison_long.csv"),
             index=False)

    # sanity: the denominator must be identical, or the comparison is void
    dm = D.pivot_table(index=["mouse", "stimulus", "behaviour"],
                       columns="window", values="n_del")
    bad = int((dm["5 min block"] != dm["6 min block + rest"]).sum())
    print(f"denominator check: {len(dm)} cells, {bad} where n_del differs")
    if bad:
        print("  WARNING: the rest window picked up deliveries. That should "
              "not happen\n  and it breaks the per-delivery comparison - "
              "check the block clock.")
    else:
        print("  identical, as expected - the rest minute has no deliveries, "
              "so\n  n_per_delivery is directly comparable between the two "
              "versions.")

    rows = []
    for b in ORDER:
        for stim in ["ALL"] + STIM:
            g = D[D.behaviour == b]
            if stim != "ALL":
                g = g[g.stimulus == stim]
            w = g.pivot_table(index="mouse", columns="window",
                              values="n_per_del")
            if w.shape[1] < 2:
                continue
            w = w.dropna()
            if not len(w):
                continue
            x = w["5 min block"].to_numpy()
            y = w["6 min block + rest"].to_numpy()
            p = np.nan
            if len(x) >= 2 and not np.allclose(y - x, 0):
                try:
                    p = float(stats.wilcoxon(x, y, zero_method="wilcox",
                                             method="exact").pvalue)
                except ValueError:
                    pass
            cnt = g.pivot_table(index="mouse", columns="window",
                                values="n_bouts").dropna()
            extra = (cnt["6 min block + rest"] - cnt["5 min block"]).mean() \
                if len(cnt) else np.nan
            rows.append(dict(
                behaviour=b, stimulus=stim, n_mice=len(x),
                per_del_5min=x.mean(), per_del_6min=y.mean(),
                pct_increase=100 * (y.mean() - x.mean()) / x.mean()
                if x.mean() else np.nan,
                extra_events_per_block=extra, p_wilcoxon=p))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(a.out, "BlockWindow_comparison_stats.csv"),
             index=False)

    print("\n=== how much does the rest minute add? "
          "(total events / total stimulus delivery) ===")
    sh = R[R.stimulus == "ALL"]
    print(f"  {'behaviour':11s} {'5 min':>7s} {'6 min':>7s} {'+%':>7s} "
          f"{'extra events/block':>19s} {'p':>7s}")
    for r in sh.itertuples():
        print(f"  {r.behaviour:11s} {r.per_del_5min:7.3f} "
              f"{r.per_del_6min:7.3f} {r.pct_increase:+7.1f} "
              f"{r.extra_events_per_block:19.2f} {r.p_wilcoxon:7.3f}")

    # ---- figure: the two windows side by side, per behaviour ----
    behs = [x for x in ORDER if x in set(D.behaviour)]
    fig, axes = plt.subplots(1, len(behs), figsize=(2.7 * len(behs), 4.8))
    for ax, b in zip(axes, behs):
        w = (D[D.behaviour == b].pivot_table(index="mouse", columns="window",
                                             values="n_per_del").dropna())
        if not len(w):
            continue
        x = w["5 min block"].to_numpy()
        y = w["6 min block + rest"].to_numpy()
        for u, v in zip(x, y):
            ax.plot([1, 2], [u, v], "-o", color="#AAA", lw=1.1, ms=4,
                    zorder=3)
        for i, (v, c) in enumerate(((x, C5), (y, C6)), start=1):
            m = v.mean()
            se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0
            ax.errorbar([i], [m], yerr=[se], fmt="s", color=c, ms=12,
                        capsize=7, lw=2.3, mec="white", mew=1.5, zorder=6)
        r = R[(R.behaviour == b) & (R.stimulus == "ALL")]
        ttl = NICE[b]
        if len(r):
            ttl += f"\n{r.pct_increase.iloc[0]:+.0f} %"
        ax.set_title(ttl, fontsize=10.5, fontweight="bold")
        ax.set_xlim(.6, 2.4)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["5 min\nstimulus", "6 min\n+ rest"], fontsize=9)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.2, axis="y")
    axes[0].set_ylabel("total event number /\ntotal stimulus delivery",
                       fontsize=10.5)
    fig.suptitle(
        "Does the rest minute carry behaviour?   one grey line per mouse, "
        "square = mean ± SEM\n"
        "the denominator is identical in both windows (the rest minute has no "
        "deliveries), so any rise is real extra behaviour", fontsize=11.5)
    fig.tight_layout()
    p = os.path.join(a.out, "W1_block_window_comparison.png")
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  {p}")

    # ---- figure: per stimulus, the extra events contributed by the rest ----
    fig, axes = plt.subplots(1, len(behs), figsize=(2.7 * len(behs), 4.4))
    for ax, b in zip(axes, behs):
        r = R[(R.behaviour == b) & (R.stimulus != "ALL")]
        if not len(r):
            continue
        ax.bar(range(len(r)), r.extra_events_per_block, color=C6, width=.66)
        ax.set_xticks(range(len(r)))
        ax.set_xticklabels([s.replace(" ", "\n") for s in r.stimulus],
                           fontsize=8.5)
        ax.set_title(NICE[b], fontsize=10.5, fontweight="bold")
        ax.grid(alpha=.2, axis="y")
        ax.axhline(0, color="k", lw=1)
    axes[0].set_ylabel("extra events in the rest minute\n(per block, mean "
                       "over mice)", fontsize=10)
    fig.suptitle("What the 1 min rest adds, by stimulus   ·   "
                 "this is behaviour the 5 min window discards", fontsize=11.5)
    fig.tight_layout()
    p2 = os.path.join(a.out, "W2_rest_minute_contribution.png")
    fig.savefig(p2, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  {p2}")

    print("\n  Report BOTH windows. The 5 min version is the stimulus period "
          "proper;\n  the 6 min version is the stimulus plus its aftermath. "
          "They answer\n  slightly different questions and the drug may move "
          "one more than the other.")


if __name__ == "__main__":
    main()
