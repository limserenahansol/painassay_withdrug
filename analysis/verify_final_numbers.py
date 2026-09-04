"""verify_final_numbers.py  -  recompute every number in the five final
figures straight from the .mat files, and cross-check it against what the
figure scripts wrote.

WHY
    The figures were rebuilt several times while the definition of the
    numerator changed (response window on, then off) and the per-mouse test
    changed (Mann-Whitney, then exact Poisson). Anything stale would be
    invisible in the PNG. This recomputes from source with no shared code
    path and prints a PASS/FAIL against each CSV the figures were drawn from.

WHAT IT CHECKS
    1  total events and total deliveries per mouse per day per behaviour
    2  events per delivery, per mouse and per stimulus
    3  the change index, both aggregations (ratio of means, median of ratios)
    4  the exact Poisson p per mouse
    5  the paired Wilcoxon per behaviour, and its floor
    6  that the CSVs the figures used agree with all of the above

USAGE
    python verify_final_numbers.py --day1 <corrected> --day2 <corrected> \\
                                   --figs <figures folder>

Hansol Lim - HEAL mini1p / SBI-553
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy import stats
from scipy.io import loadmat

AFF = {1: "attending", 2: "lickbite", 3: "guarding", 4: "escape"}
REF = {1: "withdrawal", 2: "flinch"}
ORDER = ["withdrawal", "flinch", "attending", "lickbite", "guarding", "escape"]
STIM = ["Light touch", "Mild touch", "Heat", "Pin prick"]
TAIL = {"withdrawal": 3.0, "flinch": 3.0, "attending": 10.0,
        "lickbite": 10.0, "guarding": 10.0, "escape": 10.0}
FAILS = []


def s_(v, d=""):
    a = np.asarray(v).ravel()
    if a.size == 0:
        return d
    x = a[0]
    if isinstance(x, np.ndarray):
        x = x.ravel()[0] if x.size else d
    return str(x).strip()


def f_(v, d=np.nan):
    a = np.asarray(v, dtype=float).ravel()
    return float(a[0]) if a.size else d


def check(name, got, want, tol=1e-6):
    ok = (np.isnan(got) and np.isnan(want)) or abs(got - want) <= tol
    if not ok:
        FAILS.append(f"{name}: recomputed {got!r} vs figure {want!r}")
    print(f"  {'PASS' if ok else 'FAIL'}  {name:52s} "
          f"{got:10.4f} vs {want:10.4f}")
    return ok


def counts_from_source(folder, label):
    """Independent recomputation. Deliberately written from scratch rather
    than importing the figure code, so a bug in that code cannot hide."""
    rows = []
    for p in sorted(glob.glob(os.path.join(folder, "ScoringAB_*.mat"))):
        M = loadmat(p)
        fps = f_(M["frameRate"], 30.0)
        nU = int(f_(M.get("nUsed", 0), 0))
        sc = np.asarray(M["score"]).ravel().astype(int)
        sc = sc[:nU] if nU else sc
        N = len(sc)
        rx = np.asarray(M.get("reflexEvents", np.empty((0, 2)))).reshape(-1, 2)
        dF = np.asarray(M["dFrames"]).ravel().astype(int)
        dT = np.asarray(M["dTypes"]).ravel().astype(int)
        mouse = s_(M.get("mouseID", "")).upper()

        # bout start frames for each affective code, computed once
        starts = {}
        for code, nm in AFF.items():
            m = (sc == code).astype(np.int8)
            e = np.diff(np.concatenate(([0], m, [0])))
            starts[nm] = np.flatnonzero(e == 1) + 1
        for code, nm in REF.items():
            ff = rx[rx[:, 1] == code, 0].astype(int) if rx.size \
                else np.array([], int)
            starts[nm] = ff[(ff >= 1) & (ff <= N)]

        for ty in range(1, 5):
            sel = np.sort(dF[dT == ty])
            if not len(sel):
                continue
            nm_stim = [s_(x) for x in
                       np.asarray(M["stimNames"]).ravel()][:4][ty - 1]
            for b in ORDER:
                lo = int(sel[0])
                hi = min(int(sel[-1] + TAIL[b] * fps), N)
                k = int(np.sum((starts[b] >= lo) & (starts[b] < hi)))
                rows.append(dict(day=label, mouse=mouse, stimulus=nm_stim,
                                 behaviour=b, n_events=k,
                                 n_deliveries=len(sel)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--day1", required=True)
    ap.add_argument("--day2", required=True)
    ap.add_argument("--figs", required=True)
    ap.add_argument("--label1", default="Day 1 no drug")
    ap.add_argument("--label2", default="Day 2 drug")
    a = ap.parse_args()

    S = pd.concat([counts_from_source(a.day1, a.label1),
                   counts_from_source(a.day2, a.label2)], ignore_index=True)
    S.to_csv(os.path.join(a.figs, "VERIFY_counts_from_source.csv"),
             index=False)
    mice = sorted(S.mouse.unique())
    print(f"recomputed from source: {len(mice)} mice, "
          f"{S.day.nunique()} days\n")

    # ---- 1. totals per mouse per day ----------------------------------
    print("=== totals per mouse per day (all four blocks) ===")
    tot = (S.groupby(["day", "mouse", "behaviour"])
           [["n_events", "n_deliveries"]].sum().reset_index())
    dl = (S.groupby(["day", "mouse"]).n_deliveries.sum() / len(ORDER))
    print(f"  {'day':14s} {'mouse':5s} {'deliveries':>10s}  " +
          "  ".join(f"{b[:9]:>9s}" for b in ORDER))
    for (day, m), g in tot.groupby(["day", "mouse"]):
        gg = g.set_index("behaviour").reindex(ORDER)
        print(f"  {day:14s} {m:5s} {int(dl.loc[(day, m)]):10d}  " +
              "  ".join(f"{int(v):9d}" for v in gg.n_events))

    # ---- 2 and 3. change index, both aggregations ---------------------
    print("\n=== change index, recomputed two ways ===")
    print(f"  {'behaviour':11s} {'D1 rate':>8s} {'D2 rate':>8s} "
          f"{'ratio of':>9s} {'median of':>10s}  {'down':>5s}")
    print(f"  {'':11s} {'':>8s} {'':>8s} {'means':>9s} {'ratios':>10s}")
    summary = {}
    for b in ORDER:
        w = (tot[tot.behaviour == b]
             .assign(rate=lambda x: x.n_events / x.n_deliveries)
             .pivot_table(index="mouse", columns="day", values="rate")
             .reindex(mice))
        x, y = w[a.label1].to_numpy(), w[a.label2].to_numpy()
        rom = np.median(y / np.where(x == 0, np.nan, x))
        rm = y.mean() / x.mean()
        nd = int((y < x).sum())
        summary[b] = dict(d1=x.mean(), d2=y.mean(), ratio_of_means=rm,
                          median_of_ratios=rom, n_down=nd, x=x, y=y)
        print(f"  {b:11s} {x.mean():8.3f} {y.mean():8.3f} {rm:9.3f} "
              f"{rom:10.3f}  {nd:3d}/6")

    # ---- 4. exact Poisson p per mouse --------------------------------
    print("\n=== exact Poisson rate test, per mouse (escape shown) ===")
    esc = tot[tot.behaviour == "escape"].pivot_table(
        index="mouse", columns="day", values=["n_events", "n_deliveries"])
    print(f"  {'mouse':5s} {'n1':>4s} {'t1':>4s} {'n2':>4s} {'t2':>4s} "
          f"{'index':>7s} {'p exact':>9s}")
    for m in mice:
        n1 = esc.loc[m, ("n_events", a.label1)]
        t1 = esc.loc[m, ("n_deliveries", a.label1)]
        n2 = esc.loc[m, ("n_events", a.label2)]
        t2 = esc.loc[m, ("n_deliveries", a.label2)]
        idx = (n2 / t2) / (n1 / t1) if n1 else np.nan
        p = stats.binomtest(int(n2), int(n1 + n2),
                            t2 / (t1 + t2)).pvalue if (n1 + n2) else np.nan
        print(f"  {m:5s} {int(n1):4d} {int(t1):4d} {int(n2):4d} {int(t2):4d} "
              f"{idx:7.3f} {p:9.2e}")

    # ---- 5. paired Wilcoxon per behaviour ----------------------------
    print("\n=== paired Wilcoxon over the six mice ===")
    print(f"  {'behaviour':11s} {'p':>7s} {'floor':>7s}  interpretation")
    for b in ORDER:
        x, y = summary[b]["x"], summary[b]["y"]
        p = np.nan
        if not np.allclose(y - x, 0):
            p = stats.wilcoxon(x, y, zero_method="wilcox",
                               method="exact").pvalue
        floor = 2.0 / (2 ** len(x))
        note = ("at the floor - all six moved the same way" if
                np.isfinite(p) and abs(p - floor) < 1e-9 else
                "not significant" if np.isfinite(p) and p >= .05 else
                "significant")
        print(f"  {b:11s} {p:7.3f} {floor:7.3f}  {note}")

    # ---- 6. cross-check against what the figures used ---------------
    print("\n=== cross-check against the figure CSVs ===")
    f = os.path.join(a.figs, "Stats_per_mouse_allDeliveries.csv")
    if os.path.exists(f):
        PM = pd.read_csv(f)
        A = PM[PM.stimulus == "ALL"]
        for b in ORDER:
            for m in mice:
                r = A[(A.behaviour == b) & (A.mouse == m)]
                if not len(r):
                    continue
                r = r.iloc[0]
                w = (tot[(tot.behaviour == b) & (tot.mouse == m)]
                     .set_index("day"))
                n1 = w.at[a.label1, "n_events"]
                t1 = w.at[a.label1, "n_deliveries"]
                n2 = w.at[a.label2, "n_events"]
                t2 = w.at[a.label2, "n_deliveries"]
                mine = (n2 / t2) / (n1 / t1) if n1 else np.nan
                if not (np.isnan(mine) and np.isnan(r.change_index)):
                    if abs(mine - r.change_index) > 1e-6:
                        FAILS.append(
                            f"change index {b}/{m}: recomputed {mine:.6f} "
                            f"vs CSV {r.change_index:.6f}")
        print(f"  compared {len(A)} mouse x behaviour change indices")
    else:
        print(f"  MISSING {f}")
        FAILS.append(f"missing {f}")

    f2 = os.path.join(a.figs, "Day2_rate_table.csv")
    if os.path.exists(f2):
        R = pd.read_csv(f2)
        merged = S.merge(R, on=["day", "mouse", "stimulus", "behaviour"],
                         suffixes=("_mine", "_fig"))
        bad = merged[np.abs(merged.n_events_mine - merged.n_events_fig) > 0]
        print(f"  compared {len(merged)} per-stimulus cells against "
              f"Day2_rate_table.csv, {len(bad)} mismatch(es)")
        if len(bad):
            FAILS.append(f"{len(bad)} per-stimulus event-count mismatches "
                         f"vs Day2_rate_table.csv")
            print(bad[["day", "mouse", "stimulus", "behaviour",
                       "n_events_mine", "n_events_fig"]].head(10)
                  .to_string(index=False))
    else:
        print(f"  MISSING {f2}")
        FAILS.append(f"missing {f2}")

    # ---- 7. direction-aware counts, the numbers slides quote -------
    # "N mice significant" is not the same as "N mice decreased". On
    # guarding four animals are significant but one of them went UP
    # nine-fold, so quoting four as a decrease is wrong. Printing all three
    # columns makes that impossible to get wrong by accident.
    print("\n=== counts a slide is allowed to quote ===")
    f3 = os.path.join(a.figs, "Stats_per_mouse_allDeliveries.csv")
    if os.path.exists(f3):
        PMx = pd.read_csv(f3)
        PMx = PMx[PMx.stimulus == "ALL"]
        print(f"  {'behaviour':11s} {'decreased':>10s} {'sig down':>9s} "
              f"{'sig UP':>7s} {'q<0.05':>7s}  animals moving up")
        for b in ORDER:
            g = PMx[PMx.behaviour == b]
            dn = int((g.change_index < 1).sum())
            sd = int(((g.p_rate < .05) & (g.change_index < 1)).sum())
            su = int(((g.p_rate < .05) & (g.change_index > 1)).sum())
            sq = int((g.q_rate < .05).sum())
            ups = ", ".join(f"{r.mouse} x{r.change_index:.2f}"
                            for r in g[g.change_index > 1].itertuples())
            print(f"  {b:11s} {dn:8d}/6 {sd:9d} {su:7d} {sq:7d}  "
                  f"{ups or '-'}")
        print("  Quote 'decreased in N/6' or 'significantly decreased in N', "
              "never a bare\n  significance count for a behaviour that has an "
              "animal moving the other way.")
    else:
        FAILS.append(f"missing {f3}")

    print("\n" + "=" * 68)
    if FAILS:
        print(f"{len(FAILS)} PROBLEM(S):")
        for x in FAILS:
            print(f"  - {x}")
        raise SystemExit(1)
    print("ALL CHECKS PASSED - the figures and the source agree.")

    print("\n=== the sentence each figure is allowed to make ===")
    for b in ORDER:
        s = summary[b]
        x, y = s["x"], s["y"]
        p = np.nan
        if not np.allclose(y - x, 0):
            p = stats.wilcoxon(x, y, zero_method="wilcox",
                               method="exact").pvalue
        cls = "reflexive" if b in REF.values() else (
            "exploration, NOT pain" if b == "escape" else "affective")
        print(f"  {b:11s} ({cls}): {s['d1']:.3f} -> {s['d2']:.3f}, "
              f"index {s['ratio_of_means']:.2f}, {s['n_down']}/6 down, "
              f"paired p = {p:.3f}")


if __name__ == "__main__":
    main()
