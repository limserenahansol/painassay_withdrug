"""step3_day_comparison.py  -  Day 1 (no drug) vs Day 2 (drug), box plots.

WHAT IT COMPARES

  SESSION LEVEL   one number per mouse per session, over the whole recording
      n_events            how many times the behaviour happened
      events_per_min      the same, per minute of session
      pct_of_session      total time in the behaviour / session length
      total_dur_s         total time in the behaviour
      median_iei_s        median interval between consecutive events
      Also, for the stimulus itself: n deliveries, median ITI.

      This is the "rearing 10x (20 s / 1000 s) today, 1x (1 s / 1000 s)
      tomorrow" comparison.

  BLOCK LEVEL     one number per mouse per stimulus, baseline-subtracted
      taken from BlockMeasures_long.csv, so you can ask whether the drug
      changed the response to Pin prick specifically.

WHICH MEASURE TO BELIEVE
      COUNTS and RATES are the measures. The sessions were scored by tapping,
      so a bout's recorded length is the length of the tap, not of the
      behaviour. Duration columns are computed and plotted, but they are drawn
      in grey and labelled UNRELIABLE. Do not report % of time from tap
      scoring.

STATISTICS
      Paired within mouse, Wilcoxon signed-rank, exact. With 6 mice the
      smallest attainable two-tailed p is 2/2^6 = 0.031, so one pre-specified
      comparison can reach 0.05 and a table of 6 behaviours x 5 measures
      cannot. The floor is printed next to every p.

USAGE
      # today only - see the Day 1 distribution
      python step3_day_comparison.py --day1 ..\\videos\\output_corrected

      # tomorrow, once Day 2 is scored and corrected
      python step3_day_comparison.py --day1 ..\\videos\\output_corrected \\
                                     --day2 ..\\videos\\output_day2_corrected

OUTPUT  ->  the --day1 folder (or --out)
      SessionMeasures.csv          one row per session x behaviour
      DayComparison_stats.csv      paired tests, Day 1 vs Day 2
      Fig_day_box_counts.png       box + paired lines, counts and rates
      Fig_day_box_durations.png    the same for durations, marked unreliable
      Fig_day_by_stimulus.png      per-stimulus, baseline-subtracted

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
ORDER = ["attending", "lickbite", "guarding", "escape", "withdrawal", "flinch"]
COUNTY = ["n_events", "events_per_min", "n_per_delivery", "median_iei_s"]
DURY = ["total_dur_s", "pct_of_session"]
# n_per_delivery is the measure to trust when the two days differ in how many
# stimuli were delivered. Ten flinches from ten taps and ten flinches from
# twenty taps are the same n_events but not the same responsiveness: 1.00 vs
# 0.50. Day 1 already spans 16 to 31 pin-prick taps across the six mice, so
# n_events alone is not comparable even within a single day.
# events_per_min divides by TIME, which does not fix this - the session length
# is fixed by the protocol while the tap count is not.


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


def session_measures(folder, day_label):
    """One row per session x behaviour, over the whole recording."""
    rows = []
    for p in sorted(glob.glob(os.path.join(folder, "ScoringAB_*.mat"))):
        M = loadmat(p)
        fps = f_(M["frameRate"], 30.0)
        nUsed = int(f_(M.get("nUsed", 0), 0))
        sc = np.asarray(M["score"]).ravel().astype(int)
        sc = sc[:nUsed] if nUsed else sc
        nUsed = len(sc)
        sess_s = nUsed / fps
        sess_min = sess_s / 60.0
        rx = np.asarray(M.get("reflexEvents", np.empty((0, 2)))).reshape(-1, 2)
        dF = np.asarray(M["dFrames"]).ravel().astype(float)
        base = dict(day=day_label, file=os.path.basename(p),
                    session=s_(M.get("sessionNo", "")),
                    mouse=s_(M.get("mouseID", "")),
                    sex=s_(M.get("sexID", "")),
                    phase=s_(M.get("phase", "")),
                    treatment=s_(M.get("treatment", "")),
                    session_s=round(sess_s, 1),
                    n_deliveries=int(len(dF)),
                    stim_median_iti_s=(round(float(np.median(np.diff(np.sort(dF))))
                                             / fps, 2) if len(dF) > 1 else np.nan))
        for code, nm in AFF.items():
            b = (sc == code).astype(np.int8)
            d = np.diff(np.concatenate(([0], b, [0])))
            st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
            dur = float((en - st).sum()) / fps
            iei = np.diff(st) / fps if len(st) > 1 else np.array([])
            rows.append({**base, "cls": "affective", "behaviour": nm,
                         "n_events": int(len(st)),
                         "events_per_min": len(st) / sess_min if sess_min else np.nan,
                         "n_per_delivery": (len(st) / len(dF) if len(dF)
                                            else np.nan),
                         "total_dur_s": dur,
                         "pct_of_session": 100 * dur / sess_s if sess_s else np.nan,
                         "median_iei_s": float(np.median(iei)) if iei.size else np.nan})
        for code, nm in REF.items():
            k = int(np.sum(rx[:, 1] == code)) if rx.size else 0
            f = np.sort(rx[rx[:, 1] == code, 0]) if rx.size else np.array([])
            iei = np.diff(f) / fps if f.size > 1 else np.array([])
            rows.append({**base, "cls": "reflexive", "behaviour": nm,
                         "n_events": k,
                         "events_per_min": k / sess_min if sess_min else np.nan,
                         "n_per_delivery": (k / len(dF) if len(dF)
                                            else np.nan),
                         "total_dur_s": np.nan, "pct_of_session": np.nan,
                         "median_iei_s": float(np.median(iei)) if iei.size else np.nan})
    return pd.DataFrame(rows)


def wilcoxon_floor(n):
    return 2.0 / (2 ** n) if n > 0 else np.nan


def paired(d1, d2, col):
    """Paired test on one measure, matched by mouse."""
    from scipy import stats
    a = d1.set_index("mouse")[col]
    b = d2.set_index("mouse")[col]
    both = a.index.intersection(b.index)
    x, y = a.loc[both].astype(float), b.loc[both].astype(float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    out = dict(n_pairs=n, day1_median=np.nan, day2_median=np.nan,
               delta_median=np.nan, p_wilcoxon=np.nan, p_ttest=np.nan,
               p_floor=wilcoxon_floor(n))
    if n < 2:
        return out
    out["day1_median"] = float(np.median(x))
    out["day2_median"] = float(np.median(y))
    out["delta_median"] = float(np.median(y - x))
    if not np.allclose(y - x, 0):
        try:
            out["p_wilcoxon"] = float(stats.wilcoxon(
                x, y, zero_method="wilcox", alternative="two-sided",
                method="exact" if n <= 25 else "auto").pvalue)
        except ValueError:
            pass
        try:
            out["p_ttest"] = float(stats.ttest_rel(x, y).pvalue)
        except ValueError:
            pass
    return out


def unpaired(d1, d2, col):
    """Group comparison, ignoring the pairing. Reported next to the paired
    test because the two answer different questions: paired asks whether each
    animal changed, unpaired asks whether the two groups differ. With the same
    six mice on both days the PAIRED test is the right primary - but a reader
    will look for the group means, so both are shown."""
    from scipy import stats
    x = d1[col].astype(float).dropna().to_numpy()
    y = d2[col].astype(float).dropna().to_numpy()
    out = dict(n1=len(x), n2=len(y), day1_mean=np.nan, day2_mean=np.nan,
               day1_sem=np.nan, day2_sem=np.nan, p_mannwhitney=np.nan,
               p_welch=np.nan)
    if not len(x) or not len(y):
        return out
    out["day1_mean"], out["day2_mean"] = float(np.mean(x)), float(np.mean(y))
    if len(x) > 1:
        out["day1_sem"] = float(np.std(x, ddof=1) / np.sqrt(len(x)))
    if len(y) > 1:
        out["day2_sem"] = float(np.std(y, ddof=1) / np.sqrt(len(y)))
    if len(x) > 1 and len(y) > 1:
        try:
            out["p_mannwhitney"] = float(
                stats.mannwhitneyu(x, y, alternative="two-sided").pvalue)
        except ValueError:
            pass
        try:
            out["p_welch"] = float(
                stats.ttest_ind(x, y, equal_var=False).pvalue)
        except ValueError:
            pass
    return out


def block_counts(folders):
    """Per-stimulus-block event counts, raw AND per delivery.

    The block is a fixed 300 s, but the number of stimuli delivered inside it
    is not fixed - on Day 1 it runs from 16 to 31 taps for pin prick. So the
    fixed block length does NOT make the raw count comparable:

        n_bouts        how many times the behaviour happened. Confounded by
                       how many stimuli that mouse happened to receive.
        n_per_del      n_bouts / n_del. The responsiveness. Ten flinches from
                       ten taps is 1.00; ten from twenty taps is 0.50.

    Both are returned and both are tested. n_per_del is the primary when the
    delivery counts differ, which they do.
    """
    parts = []
    for lab, fo in folders.items():
        f = os.path.join(fo, "BlockMeasures_long.csv")
        if not os.path.exists(f):
            continue
        L = pd.read_csv(f)
        L = L[L.kind == "stimulus"].copy()
        L["day"] = lab
        L["n_per_del"] = np.where(L.n_del > 0, L.n_bouts / L.n_del, np.nan)
        parts.append(L[["day", "mouse", "stimulus", "behaviour",
                        "n_bouts", "n_del", "n_per_del", "dur_min"]])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def fdr(p):
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    pv = p[ok]
    o = np.argsort(pv)
    n = len(pv)
    adj = np.empty(n)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        prev = min(prev, pv[o[i]] * n / (i + 1))
        adj[o[i]] = prev
    q[ok] = adj
    return q


def box_panel(S, measures, title, path, grey=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    behs = [b for b in ORDER if b in set(S.behaviour)]
    days = sorted(S.day.unique())
    nrow, ncol = len(measures), len(behs)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.5 * ncol, 3.0 * nrow),
                             squeeze=False)
    face = "#BFBFBF" if grey else "#8FB8D0"
    face2 = "#8F8F8F" if grey else "#D08F8F"
    for r, meas in enumerate(measures):
        for c, b in enumerate(behs):
            ax = axes[r][c]
            data, labels = [], []
            for d in days:
                v = S[(S.behaviour == b) & (S.day == d)][meas].astype(float)
                v = v[np.isfinite(v)]
                data.append(v.to_numpy())
                labels.append(d)
            if all(len(v) == 0 for v in data):
                ax.axis("off")
                continue
            bp = ax.boxplot(data, labels=labels, widths=.55,
                            patch_artist=True, showfliers=False)
            for i, p in enumerate(bp["boxes"]):
                p.set_facecolor(face if i == 0 else face2)
                p.set_alpha(.75)
            for m in bp["medians"]:
                m.set_color("k")
            # paired lines, one per mouse
            if len(days) == 2:
                w = S[(S.behaviour == b)].pivot_table(
                    index="mouse", columns="day", values=meas, aggfunc="mean")
                if set(days) <= set(w.columns):
                    for _, rr in w.iterrows():
                        if np.isfinite(rr[days[0]]) and np.isfinite(rr[days[1]]):
                            ax.plot([1, 2], [rr[days[0]], rr[days[1]]],
                                    "-o", color="#444", lw=.9, ms=3, alpha=.65)
            else:
                for i, v in enumerate(data):
                    ax.plot(np.full(len(v), i + 1)
                            + np.linspace(-.12, .12, max(len(v), 1)),
                            v, "o", color="#444", ms=3.5, alpha=.7)
            if r == 0:
                ax.set_title(b, fontsize=11)
            if c == 0:
                ax.set_ylabel(meas, fontsize=9)
            ax.tick_params(labelsize=8)
            ax.grid(alpha=.25, axis="y")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  wrote {path}")


def by_stimulus(folders, outdir):
    """Per-stimulus, baseline-subtracted, from BlockMeasures_long.csv."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    parts = []
    for lab, fo in folders.items():
        p = os.path.join(fo, "BlockMeasures_long.csv")
        if not os.path.exists(p):
            print(f"  {lab}: no BlockMeasures_long.csv - run step1 first")
            continue
        L = pd.read_csv(p)
        L["day"] = lab
        parts.append(L)
    if not parts:
        return None
    L = pd.concat(parts, ignore_index=True)
    # step1's long table keys sessions by mouse/session, not by filename
    key = ["day", "mouse", "behaviour"]
    base = (L[L["kind"] == "baseline"]
            .set_index(key)["rate_per_min"].rename("base"))
    base = base[~base.index.duplicated()]
    B = L[L["kind"] == "stimulus"].join(base, on=key)
    B["delta"] = B["rate_per_min"] - B["base"]
    B.to_csv(os.path.join(outdir, "Delta_by_stimulus.csv"), index=False)

    behs = [b for b in ORDER if b in set(B.behaviour)]
    stims = sorted(B.stimulus.dropna().unique())
    days = sorted(B.day.unique())
    fig, axes = plt.subplots(1, len(behs), figsize=(3.0 * len(behs), 3.6),
                             squeeze=False)
    for c, b in enumerate(behs):
        ax = axes[0][c]
        pos, data, cols, ticks = [], [], [], []
        for j, s in enumerate(stims):
            for i, d in enumerate(days):
                v = B[(B.behaviour == b) & (B.stimulus == s) & (B.day == d)
                      ]["delta"].astype(float)
                v = v[np.isfinite(v)]
                pos.append(j * (len(days) + .8) + i)
                data.append(v.to_numpy())
                cols.append("#8FB8D0" if i == 0 else "#D08F8F")
            ticks.append(j * (len(days) + .8) + (len(days) - 1) / 2)
        keep = [i for i, v in enumerate(data) if len(v)]
        if not keep:
            ax.axis("off")
            continue
        bp = ax.boxplot([data[i] for i in keep],
                        positions=[pos[i] for i in keep],
                        widths=.7, patch_artist=True, showfliers=False)
        for k, i in enumerate(keep):
            bp["boxes"][k].set_facecolor(cols[i])
            bp["boxes"][k].set_alpha(.75)
        for m in bp["medians"]:
            m.set_color("k")
        ax.axhline(0, color="k", ls="--", lw=1, alpha=.6)
        ax.set_xticks(ticks)
        ax.set_xticklabels(stims, rotation=30, ha="right", fontsize=8)
        ax.set_title(b, fontsize=11)
        if c == 0:
            ax.set_ylabel("delta events/min\n(block - baseline)", fontsize=9)
        ax.grid(alpha=.25, axis="y")
    lab = "  ".join(f"{'blue' if i == 0 else 'red'} = {d}"
                    for i, d in enumerate(days))
    fig.suptitle(f"Per stimulus, baseline-subtracted    ({lab})", fontsize=12)
    fig.tight_layout()
    p = os.path.join(outdir, "Fig_day_by_stimulus.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"  wrote {p}")
    return B


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--day1", required=True, help="Day 1 corrected folder")
    ap.add_argument("--day2", default=None, help="Day 2 corrected folder")
    ap.add_argument("--label1", default="Day1 no drug")
    ap.add_argument("--label2", default="Day2 drug")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    outdir = a.out or a.day1
    os.makedirs(outdir, exist_ok=True)

    folders = {a.label1: a.day1}
    S = session_measures(a.day1, a.label1)
    if a.day2:
        if not glob.glob(os.path.join(a.day2, "ScoringAB_*.mat")):
            raise SystemExit(f"no ScoringAB_*.mat in {a.day2}")
        folders[a.label2] = a.day2
        S = pd.concat([S, session_measures(a.day2, a.label2)], ignore_index=True)

    S.to_csv(os.path.join(outdir, "SessionMeasures.csv"), index=False)
    print(f"{S.session.nunique()} session(s), {S.mouse.nunique()} mouse/mice, "
          f"{len(folders)} day(s)")
    print(f"  wrote {os.path.join(outdir, 'SessionMeasures.csv')}\n")

    # ---- the session-level table, which is the headline comparison ----
    piv = S.pivot_table(index=["day", "behaviour"],
                        values=["n_events", "events_per_min", "pct_of_session",
                                "median_iei_s"], aggfunc="median")
    print("=== session-level medians ===")
    print(piv.round(3).to_string())

    print("\n=== stimulus delivery ===")
    d = S.drop_duplicates(["day", "mouse"])
    print(d.groupby("day")[["n_deliveries", "stim_median_iti_s",
                            "session_s"]].describe().round(1)
          .loc[:, (slice(None), ["mean", "min", "max"])].to_string())

    # ---- paired tests, only possible with two days ----
    if a.day2:
        rows = []
        for b in [x for x in ORDER if x in set(S.behaviour)]:
            for meas in COUNTY + DURY:
                d1 = S[(S.day == a.label1) & (S.behaviour == b)]
                d2 = S[(S.day == a.label2) & (S.behaviour == b)]
                r = paired(d1, d2, meas)
                rows.append(dict(behaviour=b, measure=meas,
                                 reliable=("yes" if meas in COUNTY
                                           else "NO - tap scoring"), **r))
        R = pd.DataFrame(rows)
        R["q_wilcoxon"] = fdr(R["p_wilcoxon"])
        R.to_csv(os.path.join(outdir, "DayComparison_stats.csv"), index=False)
        print("\n=== paired Day1 vs Day2 (counts and rates first) ===")
        sh = R[R.reliable == "yes"].sort_values("p_wilcoxon", na_position="last")
        print(f"  {'behaviour':11s} {'measure':16s} {'n':>2s} {'D1':>8s} "
              f"{'D2':>8s} {'delta':>8s} {'p':>6s} {'floor':>6s} {'q':>6s}")
        for r in sh.itertuples():
            print(f"  {r.behaviour:11s} {r.measure:16s} {r.n_pairs:2d} "
                  f"{r.day1_median:8.2f} {r.day2_median:8.2f} "
                  f"{r.delta_median:8.2f} {r.p_wilcoxon:6.3f} "
                  f"{r.p_floor:6.3f} {r.q_wilcoxon:6.3f}")
        print(f"\n  Wilcoxon floor at n={sh.n_pairs.max()} is "
              f"p={wilcoxon_floor(int(sh.n_pairs.max())):.3f}. "
              f"{len(R)} tests were run;")
        print("  pick the primary outcome before reading this table.")
    else:
        print("\n  Only one day present, so no paired test yet.")
        print("  Tomorrow: python step3_day_comparison.py "
              f'--day1 "{a.day1}" --day2 "<day2 corrected folder>"')

    # ---- raw event COUNTS per stimulus block ----
    BC = block_counts(folders)
    if not BC.empty:
        BC.to_csv(os.path.join(outdir, "BlockCounts.csv"), index=False)
        print("\n=== per 5 min stimulus block (median over mice) ===")
        for val, lbl in (("n_del", "stimuli DELIVERED (the denominator)"),
                         ("n_bouts", "raw event count"),
                         ("n_per_del", "events per delivery  [PRIMARY]")):
            piv = BC.pivot_table(index=["day", "behaviour"],
                                 columns="stimulus", values=val,
                                 aggfunc="median")
            print(f"\n  -- {lbl} --")
            print(piv.round(2).to_string())
        nd = BC.groupby("stimulus").n_del.agg(["min", "max"])
        print("\n  the block is a fixed 300 s but the delivery count is not:")
        for s, r in nd.iterrows():
            print(f"    {s:12s} {int(r['min']):3d} to {int(r['max']):3d} "
                  f"taps per block")
        print("  so the raw count is NOT comparable on its own - divide by "
              "n_del.")
        print(f"  wrote {os.path.join(outdir, 'BlockCounts.csv')}")

        if a.day2:
            rows = []
            for meas in ("n_per_del", "n_bouts", "n_del"):
                for b in [x for x in ORDER if x in set(BC.behaviour)]:
                    for s in sorted(BC.stimulus.dropna().unique()):
                        d1 = BC[(BC.day == a.label1) & (BC.behaviour == b)
                                & (BC.stimulus == s)]
                        d2 = BC[(BC.day == a.label2) & (BC.behaviour == b)
                                & (BC.stimulus == s)]
                        rows.append(dict(measure=meas, behaviour=b, stimulus=s,
                                         **paired(d1, d2, meas),
                                         **unpaired(d1, d2, meas)))
            R2 = pd.DataFrame(rows)
            R2["q_wilcoxon"] = fdr(R2["p_wilcoxon"])
            R2.to_csv(os.path.join(outdir, "BlockCounts_stats.csv"), index=False)
            print("\n=== block measures, Day1 vs Day2 ===")
            print("  PAIRED is the primary (same six mice twice); the unpaired "
                  "group test is\n  shown beside it because a reader looks for "
                  "the means first.")
            print("  n_del is tested too: if it moved between days, the raw "
                  "count is\n  confounded and only events-per-delivery can be "
                  "read.")
            for meas in ("n_per_del", "n_bouts"):
                print(f"\n  -- {meas} --")
                sh = (R2[R2.measure == meas]
                      .sort_values("p_wilcoxon", na_position="last").head(10))
                print(f"  {'behaviour':11s} {'stimulus':12s} {'n':>2s} "
                      f"{'D1 med':>7s} {'D2 med':>7s} {'delta':>7s} "
                      f"{'p pair':>7s} {'q':>6s} {'D1 mean':>8s} "
                      f"{'D2 mean':>8s} {'p group':>8s}")
                for r in sh.itertuples():
                    print(f"  {r.behaviour:11s} {r.stimulus:12s} "
                          f"{r.n_pairs:2d} {r.day1_median:7.2f} "
                          f"{r.day2_median:7.2f} {r.delta_median:+7.2f} "
                          f"{r.p_wilcoxon:7.3f} {r.q_wilcoxon:6.3f} "
                          f"{r.day1_mean:8.2f} {r.day2_mean:8.2f} "
                          f"{r.p_mannwhitney:8.3f}")
            dl = R2[R2.measure == "n_del"]
            worst = dl.loc[dl.p_wilcoxon.idxmin()] if dl.p_wilcoxon.notna().any() \
                else None
            if worst is not None:
                print(f"\n  delivery count itself: most different block is "
                      f"{worst.stimulus} "
                      f"({worst.day1_median:.0f} vs {worst.day2_median:.0f} "
                      f"taps, p = {worst.p_wilcoxon:.3f})")
            print(f"  wrote {os.path.join(outdir, 'BlockCounts_stats.csv')}")

    # ---- session-level: add the unpaired view next to the paired one ----
    if a.day2:
        rows = []
        for b in [x for x in ORDER if x in set(S.behaviour)]:
            for meas in COUNTY:
                d1 = S[(S.day == a.label1) & (S.behaviour == b)]
                d2 = S[(S.day == a.label2) & (S.behaviour == b)]
                rows.append(dict(behaviour=b, measure=meas,
                                 **paired(d1, d2, meas),
                                 **unpaired(d1, d2, meas)))
        RS = pd.DataFrame(rows)
        RS["q_wilcoxon"] = fdr(RS["p_wilcoxon"])
        RS.to_csv(os.path.join(outdir, "SessionComparison_paired_and_group.csv"),
                  index=False)
        print("\n=== whole-session totals, paired AND group ===")
        print(f"  {'behaviour':11s} {'measure':15s} {'D1 med':>7s} "
              f"{'D2 med':>7s} {'p pair':>7s} {'D1 mean':>8s}±{'sem':<5s} "
              f"{'D2 mean':>8s}±{'sem':<5s} {'p group':>8s}")
        for r in RS[RS.measure == "n_events"].itertuples():
            print(f"  {r.behaviour:11s} {r.measure:15s} {r.day1_median:7.1f} "
                  f"{r.day2_median:7.1f} {r.p_wilcoxon:7.3f} "
                  f"{r.day1_mean:8.1f}±{r.day1_sem:<5.1f} "
                  f"{r.day2_mean:8.1f}±{r.day2_sem:<5.1f} "
                  f"{r.p_mannwhitney:8.3f}")

    # ---- figures ----
    print()
    box_panel(S, COUNTY, "Counts and rates  (the reliable measures)",
              os.path.join(outdir, "Fig_day_box_counts.png"))
    box_panel(S, DURY, "Durations  -  UNRELIABLE with tap scoring, "
                       "shown for completeness only",
              os.path.join(outdir, "Fig_day_box_durations.png"), grey=True)
    by_stimulus(folders, outdir)


if __name__ == "__main__":
    main()
