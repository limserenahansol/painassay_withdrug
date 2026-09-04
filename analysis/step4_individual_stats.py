"""step4_individual_stats.py  -  Day 1 vs Day 2 at BOTH levels.

WHY THIS EXISTS
    With six animals, a population test has a hard floor: the exact paired
    Wilcoxon cannot return a two-tailed p below 2/2^6 = 0.031. But the
    per-delivery response is a Bernoulli trial, and each mouse gets ~88
    deliveries per day. That makes a WITHIN-MOUSE test possible and properly
    powered - Fisher's exact test on that animal's own 2x2 table.

    So two levels, answering two different questions:

      POPULATION   does the group change?
                   paired Wilcoxon (the design), group Mann-Whitney (what a
                   reader looks for), and a mixed model over all deliveries.

      INDIVIDUAL   did THIS mouse change?
                   Fisher exact on responded/not-responded x day, per mouse,
                   per stimulus, per behaviour. Reported as an odds ratio with
                   a confidence interval, and drawn as a forest plot so you can
                   see whether all six moved together or only some did.

RESPONSE WINDOW
    Reflexive behaviours are locked to contact, so 3 s over every delivery.
    Affective behaviours build over 1-8 s (see the peri-stimulus histogram),
    so 10 s - but only over deliveries with 10 s of clearance, because the
    median gap between deliveries is 2.7 s. Both choices are printed.

USAGE
    python step4_individual_stats.py --day1 <folder> --day2 <folder>

OUTPUT  ->  --out (default the day1 folder)
    Trials_per_delivery.csv        one row per delivery: responded 0/1
    Stats_population.csv           paired + group tests, per stimulus/behaviour
    Stats_per_mouse.csv            Fisher exact per mouse
    Fig_forest_per_mouse.png       odds ratio per mouse, per behaviour
    Fig_population_paired.png      group summary with the paired lines

Hansol Lim - HEAL mini1p / SBI-553
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.io import loadmat

AFF = {1: "attending", 2: "lickbite", 3: "guarding", 4: "escape"}
REF = {1: "withdrawal", 2: "flinch"}
ORDER = ["withdrawal", "flinch", "attending", "lickbite", "guarding", "escape"]
NICE = {"withdrawal": "Paw withdrawal", "flinch": "Flinch",
        "attending": "Paw attending", "lickbite": "Licking / biting",
        "guarding": "Guarding", "escape": "Escape / rearing"}
STIM = ["Light touch", "Mild touch", "Heat", "Pin prick"]
# window per class, justified by the peri-stimulus histogram
WIN = {"withdrawal": (3.0, False), "flinch": (3.0, False),
       "attending": (10.0, True), "lickbite": (10.0, True),
       "guarding": (10.0, True), "escape": (10.0, True)}
# The isolation filter exists so an affective response is not credited to the
# wrong delivery. It costs a lot of trials: 122 of 535 survive at 10 s, which
# is ~20 per mouse per day - enough for the population test, thin for a
# per-mouse one. --no-isolation keeps all 535.
#
# Dropping it is defensible FOR THE DAY COMPARISON specifically: the
# mis-attribution inflates P(response) on both days by the same amount,
# because the delivery pacing is set by the protocol and not by the drug. It
# is NOT defensible for comparing one stimulus against another, since the
# stimuli can differ in how tightly they are delivered. step4 therefore
# reports both, and marks which one each number came from.


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


def trials(folder, label, use_isolation=True):
    """One row per (delivery x behaviour): did the behaviour occur in-window?"""
    rows = []
    for p in sorted(glob.glob(os.path.join(folder, "ScoringAB_*.mat"))):
        M = loadmat(p)
        fps = f_(M["frameRate"], 30.0)
        n = int(f_(M.get("nUsed", 0), 0))
        sc = np.asarray(M["score"]).ravel().astype(int)
        sc = sc[:n] if n else sc
        n = len(sc)
        rx = np.asarray(M.get("reflexEvents", np.empty((0, 2)))).reshape(-1, 2)
        names = [s_(x) for x in np.asarray(M["stimNames"]).ravel()][:4]
        dF = np.asarray(M["dFrames"]).ravel().astype(int)
        dT = np.asarray(M["dTypes"]).ravel().astype(int)
        allf = np.sort(dF.astype(float))
        mouse, sex = s_(M.get("mouseID", "")), s_(M.get("sexID", ""))
        refhit = np.zeros((3, n + 2))
        if rx.size:
            for c in (1, 2):
                ff = rx[rx[:, 1] == c, 0].astype(int)
                ff = ff[(ff >= 1) & (ff <= n)]
                refhit[c, ff] = 1
        for f, ty in zip(dF, dT):
            if not 1 <= ty <= 4:
                continue
            nxt = allf[allf > f]
            clear = (nxt[0] - f) / fps if len(nxt) else np.inf
            for b in ORDER:
                w, iso = WIN[b]
                iso = iso and use_isolation
                if iso and clear < w:
                    continue          # cannot attribute a response to this one
                hi = min(int(f + w * fps), n)
                if hi <= f:
                    continue
                if b in REF.values():
                    c = [k for k, v in REF.items() if v == b][0]
                    nresp = int(refhit[c, f:hi].sum())
                else:
                    c = [k for k, v in AFF.items() if v == b][0]
                    seg = (sc[f - 1:hi - 1] == c).astype(int)
                    # count bouts, not frames: a rising edge is one event
                    nresp = int((np.diff(np.r_[0, seg]) == 1).sum())
                rows.append(dict(day=label, mouse=mouse, sex=sex,
                                 stimulus=names[ty - 1], behaviour=b,
                                 window_s=w, isolated=int(iso),
                                 clearance_s=round(min(clear, 999), 2),
                                 responded=int(nresp > 0),
                                 n_responses=nresp))
    return pd.DataFrame(rows)


def block_totals(folder, label, use_window=False):
    """Total events and total deliveries per mouse x stimulus x behaviour.

    This is the numerator and denominator of the change index. With
    use_window False (the default) EVERY event inside the stimulus block is
    counted, whatever its timing relative to an individual delivery, because
    the question is how much of the behaviour the animal did.

    The per-delivery trial table from trials() cannot answer that: it can only
    see events inside a response window, and for escape/rearing that window
    holds 29 % of them.
    """
    rows = []
    for p in sorted(glob.glob(os.path.join(folder, "ScoringAB_*.mat"))):
        M = loadmat(p)
        fps = f_(M["frameRate"], 30.0)
        n = int(f_(M.get("nUsed", 0), 0))
        sc = np.asarray(M["score"]).ravel().astype(int)
        sc = sc[:n] if n else sc
        n = len(sc)
        rx = np.asarray(M.get("reflexEvents", np.empty((0, 2)))).reshape(-1, 2)
        names = [s_(x) for x in np.asarray(M["stimNames"]).ravel()][:4]
        dF = np.asarray(M["dFrames"]).ravel().astype(int)
        dT = np.asarray(M["dTypes"]).ravel().astype(int)
        mouse, sex = s_(M.get("mouseID", "")), s_(M.get("sexID", ""))
        for ty in range(1, 5):
            sel = np.sort(dF[dT == ty])
            if not len(sel):
                continue
            for b in ORDER:
                w = WIN[b][0]
                spans = ([(int(f), min(int(f + w * fps), n)) for f in sel]
                         if use_window else
                         [(int(sel[0]), min(int(sel[-1] + w * fps), n))])
                tot = 0
                for f, hi in spans:
                    if hi <= f:
                        continue
                    if b in REF.values():
                        c = [k for k, v in REF.items() if v == b][0]
                        ff = rx[rx[:, 1] == c, 0] if rx.size else np.array([])
                        tot += int(np.sum((ff >= f) & (ff < hi)))
                    else:
                        c = [k for k, v in AFF.items() if v == b][0]
                        seg = (sc[f - 1:hi - 1] == c).astype(int)
                        tot += int((np.diff(np.r_[0, seg]) == 1).sum())
                rows.append(dict(day=label, mouse=mouse, sex=sex,
                                 stimulus=names[ty - 1], behaviour=b,
                                 n_events=tot, n_deliveries=len(sel)))
    return pd.DataFrame(rows)


def fisher(a, b, c, d):
    """Odds ratio with a Haldane-Anscombe 0.5 correction, and its 95 % CI."""
    from scipy import stats
    try:
        _, p = stats.fisher_exact([[a, b], [c, d]], alternative="two-sided")
    except ValueError:
        p = np.nan
    A, B, C, D = a + .5, b + .5, c + .5, d + .5
    orr = (A * D) / (B * C)
    se = np.sqrt(1 / A + 1 / B + 1 / C + 1 / D)
    return orr, np.exp(np.log(orr) - 1.96 * se), np.exp(np.log(orr) + 1.96 * se), p


def rr_ci(r1, r2):
    """95 % CI for the change index, from the Poisson variance of the totals.

    log(ratio) has approximate standard error sqrt(1/N1 + 1/N2) where N1 and
    N2 are the total event counts. A 0.5 offset keeps a zero total finite.
    """
    N1, N2 = float(np.sum(r1)) + .5, float(np.sum(r2)) + .5
    d1, d2 = max(len(r1), 1), max(len(r2), 1)
    rr = (N2 / d2) / (N1 / d1)
    se = np.sqrt(1 / N1 + 1 / N2)
    return dict(rr_lo=float(np.exp(np.log(rr) - 1.96 * se)),
                rr_hi=float(np.exp(np.log(rr) + 1.96 * se)))


def poisson_rate_p(n1, t1, n2, t2):
    """Exact test that two event rates are equal. This is THE p-value for the
    change index, and it is exact rather than approximate.

    n1 events over t1 stimuli against n2 events over t2 stimuli. Conditional
    on the total n1 + n2, the number falling on Day 2 is Binomial with
    p = t2 / (t1 + t2) if the rates are equal, so a two-sided binomial test
    on that is the exact test.

    Why not Mann-Whitney over the per-delivery counts, which is what this
    used to be: that needs each event assigned to an individual delivery, so
    it only works with a response window. Once every event in the block is
    counted - which is what "how much did the animal do" requires - the
    per-delivery trial structure is gone and this is the correct test. It also
    matches the confidence interval above, which already assumes Poisson
    counts; the two used to come from different models.
    """
    from scipy import stats
    if t1 <= 0 or t2 <= 0 or (n1 + n2) == 0:
        return np.nan
    try:
        return float(stats.binomtest(int(n2), int(n1 + n2),
                                     t2 / (t1 + t2)).pvalue)
    except ValueError:
        return np.nan


def mde(p1, n1, n2, alpha=.05, power=.80):
    """Smallest change in P(response) this mouse's trial count could detect.

    A non-significant per-mouse result is only informative if you know what
    the test COULD have seen. With ~89 reflex trials per day a shift of about
    0.15 is detectable; with ~20 affective trials it takes roughly 0.35. Both
    numbers go in the table so a null is not read as "no effect".
    """
    from scipy import stats
    if not (n1 and n2) or not np.isfinite(p1):
        return np.nan
    z = stats.norm.ppf(1 - alpha / 2)
    for d in np.arange(.01, 1.0, .01):
        for p2 in (min(p1 + d, 1.0), max(p1 - d, 0.0)):
            se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
            if se <= 0:
                continue
            if stats.norm.cdf(abs(p2 - p1) / se - z) >= power:
                return round(float(d), 2)
    return np.nan


def fdr(p):
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    pv = p[ok]
    o = np.argsort(pv)
    nn = len(pv)
    adj = np.empty(nn)
    prev = 1.0
    for i in range(nn - 1, -1, -1):
        prev = min(prev, pv[o[i]] * nn / (i + 1))
        adj[o[i]] = prev
    q[ok] = adj
    return q


def per_mouse(T, lab1, lab2, B=None):
    """Two readouts per mouse, because they fail in opposite directions.

    RATE   events per delivery. This is exactly the normalisation the design
           needs - 10 flinches from 10 taps is 1.0, 10 from 20 taps is 0.5 -
           and it keeps counting after the first event, so it stays sensitive
           when responses are dense. Tested with Mann-Whitney over that
           animal's own deliveries.

    HIT    did anything happen at all, yes or no. Robust and easy to read,
           but it saturates: if a window usually holds two or three bouts,
           removing half of them barely moves the probability. Tested with
           Fisher exact.

    Report the rate as primary and the hit rate as the sanity check.
    """
    from scipy import stats
    rows = []
    for (m, b), g in T.groupby(["mouse", "behaviour"]):
        for stim in ["ALL"] + sorted(g.stimulus.unique()):
            gg = g if stim == "ALL" else g[g.stimulus == stim]
            d1 = gg[gg.day == lab1]
            d2 = gg[gg.day == lab2]
            if not len(d1) or not len(d2):
                continue
            a, b1 = int(d1.responded.sum()), int((1 - d1.responded).sum())
            c, d0 = int(d2.responded.sum()), int((1 - d2.responded).sum())
            orr, lo, hi, p = fisher(a, b1, c, d0)
            P1 = a / max(a + b1, 1)
            # Rate numerator: block totals when available (every event), else
            # fall back to the windowed per-delivery counts.
            if B is not None:
                bb = B[(B.mouse == m) & (B.behaviour == b)]
                if stim != "ALL":
                    bb = bb[bb.stimulus == stim]
                s1 = bb[bb.day == lab1]
                s2 = bb[bb.day == lab2]
                N1, T1 = float(s1.n_events.sum()), float(s1.n_deliveries.sum())
                N2, T2 = float(s2.n_events.sum()), float(s2.n_deliveries.sum())
                r1 = np.full(int(T1) or 1, N1 / (T1 or 1))
                r2 = np.full(int(T2) or 1, N2 / (T2 or 1))
            else:
                r1 = np.asarray(d1.n_responses, float)
                r2 = np.asarray(d2.n_responses, float)
                N1, T1 = float(np.sum(r1)), float(len(r1))
                N2, T2 = float(np.sum(r2)), float(len(r2))
            # exact Poisson rate test on the totals - the p-value that belongs
            # to the change index, and consistent with its CI
            pc = poisson_rate_p(N1, T1, N2, T2)
            rows.append(dict(mouse=m, behaviour=b, stimulus=stim,
                             n_deliv_day1=a + b1, n_deliv_day2=c + d0,
                             n_events_day1=int(np.sum(r1)),
                             n_events_day2=int(np.sum(r2)),
                             # primary: events per delivery
                             rate_day1=float(r1.mean()) if len(r1) else np.nan,
                             rate_day2=float(r2.mean()) if len(r2) else np.nan,
                             rate_ratio=(float(r2.mean() / r1.mean())
                                         if len(r1) and len(r2)
                                         and r1.mean() > 0 else np.nan),
                             p_rate=pc, **rr_ci(r1, r2),
                             # secondary: any response, yes or no
                             p_day1=P1, p_day2=c / max(c + d0, 1),
                             delta=c / max(c + d0, 1) - P1,
                             odds_ratio=orr, ci_lo=lo, ci_hi=hi, p_fisher=p,
                             min_detectable_delta=mde(P1, a + b1, c + d0)))
    R = pd.DataFrame(rows)
    if not R.empty:
        R["q_fisher"] = fdr(R["p_fisher"])
        R["q_rate"] = fdr(R["p_rate"])
        # "change index" is the name used in the figures and slides. The
        # rate_ratio / rr_lo / rr_hi columns are kept so anything already
        # reading these files keeps working.
        R["change_index"] = R["rate_ratio"]
        R["change_index_lo"] = R["rr_lo"]
        R["change_index_hi"] = R["rr_hi"]
    return R


def population(T, lab1, lab2, B=None):
    """The mouse is the unit here - one number per animal per day.

    Runs on both readouts. 'rate' (events per delivery) is primary, 'hit'
    (any response) is the sanity check. The paired Wilcoxon matches the
    design; the group Mann-Whitney is what a reader expects to see; the
    tie/agreement count survives the exact test's p-value floor of 2/2^n.
    """
    from scipy import stats
    rows = []
    for metric, col in (("rate", "n_responses"), ("hit", "responded")):
        for b in [x for x in ORDER if x in set(T.behaviour)]:
            for stim in ["ALL"] + sorted(T.stimulus.unique()):
                if metric == "rate" and B is not None:
                    # same numerator as the per-mouse change index, so the
                    # population row and the forest plot cannot disagree
                    sub = B[B.behaviour == b]
                    if stim != "ALL":
                        sub = sub[sub.stimulus == stim]
                    agg = (sub.groupby(["day", "mouse"])
                           [["n_events", "n_deliveries"]].sum().reset_index())
                    agg["v"] = agg.n_events / agg.n_deliveries.replace(0,
                                                                       np.nan)
                    w = agg.pivot_table(index="mouse", columns="day",
                                        values="v")
                else:
                    sub = T[T.behaviour == b]
                    if stim != "ALL":
                        sub = sub[sub.stimulus == stim]
                    w = (sub.groupby(["day", "mouse"])[col].mean()
                         .reset_index()
                         .pivot_table(index="mouse", columns="day",
                                      values=col))
                if lab1 not in w.columns or lab2 not in w.columns:
                    continue
                w = w.dropna()
                x, y = w[lab1].to_numpy(), w[lab2].to_numpy()
                n = len(x)
                d = y - x
                r = dict(metric=metric, behaviour=b, stimulus=stim, n_mice=n,
                         day1_mean=float(np.mean(x)) if n else np.nan,
                         day2_mean=float(np.mean(y)) if n else np.nan,
                         day1_sem=(float(np.std(x, ddof=1) / np.sqrt(n))
                                   if n > 1 else np.nan),
                         day2_sem=(float(np.std(y, ddof=1) / np.sqrt(n))
                                   if n > 1 else np.nan),
                         n_up=int((d > 0).sum()), n_down=int((d < 0).sum()),
                         n_tied=int((d == 0).sum()),
                         p_wilcoxon=np.nan, p_mannwhitney=np.nan,
                         p_floor=2.0 / (2 ** n) if n else np.nan)
                if n >= 2 and not np.allclose(d, 0):
                    try:
                        r["p_wilcoxon"] = float(stats.wilcoxon(
                            x, y, zero_method="wilcox", method="exact").pvalue)
                    except ValueError:
                        pass
                    try:
                        r["p_mannwhitney"] = float(stats.mannwhitneyu(
                            x, y, alternative="two-sided").pvalue)
                    except ValueError:
                        pass
                rows.append(r)
    R = pd.DataFrame(rows)
    if not R.empty:
        R["q_wilcoxon"] = fdr(R["p_wilcoxon"])
    return R


MOCKUP = False


def _stamp_save(fig, path):
    """Fabricated Day-2 numbers must be impossible to mistake for a result."""
    import matplotlib.pyplot as plt
    if MOCKUP:
        fig.text(.5, .5, "MOCKUP\nsynthetic Day 2", ha="center", va="center",
                 fontsize=64, color="#C0483B", alpha=.16, rotation=24,
                 fontweight="bold", zorder=100)
        fig.text(.5, .003, "Day 2 numbers are SYNTHETIC - layout preview "
                           "only, not a result", ha="center", va="bottom",
                 fontsize=11, color="#C0483B", fontweight="bold")
        r, e = os.path.splitext(path)
        path = r + "_MOCKUP" + e
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


def stars(p):
    """Conventional significance marks. ns is printed rather than left blank
    so a missing star cannot be mistaken for a missing test."""
    if not np.isfinite(p):
        return ""
    return ("***" if p < .001 else "**" if p < .01 else
            "*" if p < .05 else "ns")


def forest(PM, path, lab1, lab2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    A = PM[PM.stimulus == "ALL"].copy()
    if A.empty:
        return
    behs = [b for b in ORDER if b in set(A.behaviour)]
    fig, axes = plt.subplots(1, len(behs), figsize=(3.0 * len(behs), 4.4),
                             sharey=True, squeeze=False)
    # One fixed row order, F1 at the top, identical in every panel. Sorting
    # per panel meant a mouse changed row between behaviours and no animal
    # could be followed across the figure.
    mice = sorted(A.mouse.unique())
    for ax, b in zip(axes[0], behs):
        g = (A[A.behaviour == b].set_index("mouse").reindex(mice)
             .reset_index())
        y = np.arange(len(mice))[::-1]
        for yy, (_, r) in zip(y, g.iterrows()):
            if not np.isfinite(r.rate_ratio):
                continue
            sig = np.isfinite(r.p_rate) and r.p_rate < .05
            ax.plot([r.rr_lo, r.rr_hi], [yy, yy], "-",
                    color="#C0483B" if sig else "#999", lw=2.2)
            ax.plot([r.rate_ratio], [yy], "o", ms=9,
                    color="#C0483B" if sig else "#444",
                    mec="white", mew=1.4, zorder=5)
            # Stars as well as colour: colour alone is not readable in
            # greyscale or by a colour-blind reader, and it does not
            # distinguish p = 0.04 from p = 0.0001.
            ax.annotate(stars(r.p_rate), (r.rate_ratio, yy),
                        xytext=(0, 7), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9,
                        color="#C0483B" if sig else "#777")
        ax.axvline(1, color="k", ls="--", lw=1.2)
        ax.set_xscale("log")
        # A dense ladder collided into "0.1250.250.5" once a wide CI pulled
        # the range out. Thin the ladder until at most four labels remain.
        lo = np.nanmin(np.r_[g.rr_lo.to_numpy(), g.rate_ratio.to_numpy()])
        hi = np.nanmax(np.r_[g.rr_hi.to_numpy(), g.rate_ratio.to_numpy()])
        ladder = np.array([1 / 64, 1 / 32, 1 / 16, .125, .25, .5, 1,
                           2, 4, 8, 16], float)
        keep = ladder[(ladder >= lo * .8) & (ladder <= hi * 1.25)]
        if len(keep) < 2:
            keep = np.array([.5, 1, 2], float)
        while len(keep) > 4:
            one = int(np.argmin(np.abs(keep - 1)))
            keep = keep[[i for i in range(len(keep))
                         if i == one or (i - one) % 2 == 0]]
            if len(keep) > 4:
                keep = np.r_[keep[keep < 1][-1:], [1.0],
                             keep[keep > 1][:1]]
                break
        ax.set_xticks(keep)
        ax.set_xticklabels([("1" if abs(t - 1) < 1e-9 else
                             (f"1/{int(round(1 / t))}" if t < 1
                              else f"{int(round(t))}")) for t in keep],
                           fontsize=9)
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_yticks(y)
        ax.set_yticklabels(mice, fontsize=10)
        ax.set_ylim(-.7, len(mice) - .3)
        ax.set_xlabel("change index  (Day 2 / Day 1)", fontsize=9)
        ax.set_title(NICE[b], fontsize=11, fontweight="bold")
        ax.grid(alpha=.22, axis="x")
    axes[0][0].set_ylabel("mouse")
    fig.suptitle(f"Change index per mouse  =  (total events / total stimuli) "
                 f"on {lab2}  ÷  the same on {lab1}\n"
                 "dot = the index, line = 95 % CI, dashed line = 1 (no "
                 "change)   ·   exact Poisson rate test on that animal's own "
                 "counts\n"
                 "*** p<0.001   ** p<0.01   * p<0.05   ns = not significant",
                 fontsize=11.5)
    fig.tight_layout()
    _stamp_save(fig, path)


def popfig(T, path, lab1, lab2, col="n_responses",
           ylab="total event number /\ntotal stimulus delivery"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    per = (T.groupby(["day", "mouse", "behaviour"])[col]
           .mean().reset_index().rename(columns={col: "responded"}))
    behs = [b for b in ORDER if b in set(per.behaviour)]
    days = [lab1, lab2]
    fig, axes = plt.subplots(1, len(behs), figsize=(2.7 * len(behs), 4.4))
    for ax, b in zip(axes, behs):
        w = per[per.behaviour == b].pivot_table(index="mouse", columns="day",
                                                values="responded")
        for _, r in w.iterrows():
            if all(d in w.columns and np.isfinite(r[d]) for d in days):
                ax.plot([1, 2], [r[days[0]], r[days[1]]], "-o", color="#777",
                        lw=1.2, ms=5, alpha=.85, zorder=3)
        for i, d in enumerate(days, start=1):
            if d not in w.columns:
                continue
            v = w[d].dropna().to_numpy()
            if not len(v):
                continue
            m = np.mean(v)
            sem = np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0
            ax.errorbar([i], [m], yerr=[sem], fmt="s",
                        color="#C0483B" if i == 2 else "#444", ms=12,
                        capsize=7, lw=2.4, zorder=6, mec="white", mew=1.6)
        ax.set_xlim(.55, 2.45)
        ax.set_xticks([1, 2])
        ax.set_xticklabels([d.replace(" ", "\n") for d in days], fontsize=10)
        ax.set_ylim(bottom=0)
        ax.set_title(NICE[b], fontsize=11, fontweight="bold")
        ax.grid(alpha=.22, axis="y")
    axes[0].set_ylabel(ylab, fontsize=11)
    fig.suptitle(f"{ylab.replace(chr(10), ' ')}   ·   one grey line per mouse, "
                 "square = mean ± SEM   ·   "
                 "each mouse divided by its own stimulus count", fontsize=12)
    fig.tight_layout()
    _stamp_save(fig, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--day1", required=True)
    ap.add_argument("--day2", default=None)
    ap.add_argument("--label1", default="Day1 no drug")
    ap.add_argument("--label2", default="Day2 drug")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-isolation", action="store_true",
                    help="score affective behaviour over EVERY delivery, not "
                         "only isolated ones. ~4x the trials, so a per-mouse "
                         "test becomes possible; valid for Day1-vs-Day2, not "
                         "for stimulus-vs-stimulus. See the note at the top.")
    ap.add_argument("--mockup", action="store_true",
                    help="stamp MOCKUP on every figure and append _MOCKUP to "
                         "its filename; use when --day2 is synthetic")
    ap.add_argument("--response-window", action="store_true",
                    help="count only events inside the response window for "
                         "the change index. Default counts every event in the "
                         "stimulus block, which is what 'how much did the "
                         "animal do' needs. The window keeps just 29 %% of "
                         "escape/rearing.")
    a = ap.parse_args()
    global MOCKUP
    MOCKUP = a.mockup
    outdir = a.out or a.day1
    os.makedirs(outdir, exist_ok=True)
    iso = not a.no_isolation
    tag = "" if iso else "_allDeliveries"

    T = trials(a.day1, a.label1, iso)
    if a.day2:
        T = pd.concat([T, trials(a.day2, a.label2, iso)], ignore_index=True)
    T.to_csv(os.path.join(outdir, f"Trials_per_delivery{tag}.csv"), index=False)
    print(f"{len(T)} delivery x behaviour trial(s) from "
          f"{T.mouse.nunique()} mouse/mice, {T.day.nunique()} day(s)")
    print("  windows: withdrawal/flinch 3 s over every delivery; affective "
          "10 s over\n  " + ("isolated deliveries only (median gap 2.7 s)"
                             if iso else "EVERY delivery (--no-isolation)"))
    nd = max(T.day.nunique(), 1)
    for b in ORDER:
        g = T[T.behaviour == b]
        if not len(g):
            continue
        print(f"    {b:11s} {len(g):5d} trial(s) "
              f"(~{len(g) / max(T.mouse.nunique(), 1) / nd:.0f} per mouse "
              f"per day), P(response) = {g.responded.mean():.3f}")

    if not a.day2:
        print("\n  Only Day 1. Re-run with --day2 for the statistics.")
        return

    # Block totals drive the change index; the per-delivery trials still drive
    # the hit-rate sanity check, which needs a window by definition.
    B = pd.concat([block_totals(a.day1, a.label1, a.response_window),
                   block_totals(a.day2, a.label2, a.response_window)],
                  ignore_index=True)
    B.to_csv(os.path.join(outdir, f"BlockTotals{tag}.csv"), index=False)
    print("\nchange index counts: "
          + ("only events inside the response window"
             if a.response_window else "EVERY event in the stimulus block"))
    POP = population(T, a.label1, a.label2, B)
    PM = per_mouse(T, a.label1, a.label2, B)
    POP.to_csv(os.path.join(outdir, f"Stats_population{tag}.csv"), index=False)
    PM.to_csv(os.path.join(outdir, f"Stats_per_mouse{tag}.csv"), index=False)

    A = PM[PM.stimulus == "ALL"]
    for metric, lbl in (("rate", "events per delivery  [PRIMARY]"),
                        ("hit", "any response, yes/no  [sanity check]")):
        sh = POP[(POP.stimulus == "ALL") & (POP.metric == metric)] \
            .set_index("behaviour")
        print(f"\n=== POPULATION, {lbl} ===")
        print(f"  {'behaviour':11s} {'D1':>6s} {'D2':>6s} {'ratio':>6s} "
              f"{'p pair':>7s} {'floor':>6s} {'q':>6s} {'p grp':>7s}  "
              f"direction")
        for b in [x for x in ORDER if x in sh.index]:
            r = sh.loc[b]
            rat = (r.day2_mean / r.day1_mean) if r.day1_mean else np.nan
            tie = f" +{r.n_tied} tied" if r.n_tied else ""
            print(f"  {b:11s} {r.day1_mean:6.3f} {r.day2_mean:6.3f} "
                  f"{rat:6.2f} {r.p_wilcoxon:7.3f} {r.p_floor:6.3f} "
                  f"{r.q_wilcoxon:6.3f} {r.p_mannwhitney:7.3f}  "
                  f"{max(r.n_up, r.n_down)}/{r.n_mice} "
                  f"{'up' if r.n_up >= r.n_down else 'down'}{tie}")
    print("\n  'ratio' = Day2 / Day1. 'floor' is the smallest p the exact "
          "paired test\n  can return at this n; 'direction' is how many mice "
          "moved the same way,\n  which carries information the floor hides.")

    print("\n=== INDIVIDUAL MICE, on that animal's own deliveries ===")
    print(f"  {'mouse':6s} {'behaviour':11s} {'nD1':>4s} {'nD2':>4s}  "
          f"{'rate D1':>7s} {'rate D2':>7s} {'ratio':>6s} {'p rate':>7s} "
          f"{'q':>6s} | {'P D1':>6s} {'P D2':>6s} {'OR':>6s} {'p hit':>7s} "
          f"{'MDE':>5s}")
    for r in A.sort_values(["behaviour", "mouse"]).itertuples():
        star = " *" if np.isfinite(r.p_rate) and r.p_rate < .05 else "  "
        print(f"  {r.mouse:6s} {r.behaviour:11s} {r.n_deliv_day1:4d} "
              f"{r.n_deliv_day2:4d}  {r.rate_day1:7.3f} {r.rate_day2:7.3f} "
              f"{r.rate_ratio:6.2f} {r.p_rate:7.3f} {r.q_rate:6.3f} | "
              f"{r.p_day1:6.3f} {r.p_day2:6.3f} {r.odds_ratio:6.2f} "
              f"{r.p_fisher:7.3f} {r.min_detectable_delta:5.2f}{star}")
    ns_r = int((A.p_rate < .05).sum())
    ns_h = int((A.p_fisher < .05).sum())
    print(f"\n  p < 0.05 on its own: {ns_r} of {len(A)} cells by rate, "
          f"{ns_h} by hit rate.")
    print("  nD1/nD2 are that mouse's delivery counts - the denominators. "
          "They differ\n  between animals and between days, which is exactly "
          "why the rate and not\n  the raw count is the comparable quantity.")
    print("  MDE applies to the hit rate: the smallest change in P(response) "
          "that\n  animal's trial count could detect at 80 % power. If "
          "|P D2 - P D1| < MDE\n  and p is not significant, read it as "
          "'underpowered for this animal'.")
    print("  A per-mouse test is possible at all ONLY because each delivery "
          "is a\n  trial. On block means there would be four numbers per "
          "mouse per day.")

    forest(PM, os.path.join(outdir, f"Fig_forest_per_mouse{tag}.png"),
           a.label1, a.label2)
    popfig(T, os.path.join(outdir, f"Fig_population_rate{tag}.png"),
           a.label1, a.label2, "n_responses",
           "total event number /\ntotal stimulus delivery")
    popfig(T, os.path.join(outdir, f"Fig_population_hitrate{tag}.png"),
           a.label1, a.label2, "responded", "P(any response per delivery)")


if __name__ == "__main__":
    main()
