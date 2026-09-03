"""step2_block_stats.py  -  baseline-subtracted block measures, then the
difference-in-differences contrast for the drug.

WHAT IT COMPUTES

  1. delta          per session x stimulus x behaviour
                    delta = block rate_per_min  -  that session's own
                            5 min baseline rate_per_min
                    Rate per minute is the only measure comparable to
                    baseline, because baseline has no deliveries at all.

  2. DiD            per mouse x stimulus x behaviour
                    DiD = delta(SBI-553) - delta(Vehicle)
                    This is the primary contrast (chosen 2026-09-02). It
                    removes each session's own baseline first, so a drug
                    effect on resting behaviour cannot masquerade as a drug
                    effect on the pain response - which matters here, because
                    SBI-553 acts on NTSR1 and neurotensin affects arousal.

  3. injection      per mouse x stimulus x behaviour
     control        delta(Vehicle) - delta(None)
                    Tells you whether the injection itself changed anything,
                    independent of the compound.

  4. Mixed model    per behaviour, pooled over stimuli:
                        delta ~ C(treatment) * C(stimulus) + block_pos
                                + (1 | mouse)
                    block_pos is in there because the stimulus order is
                    randomised but the session still runs 28 min, and later
                    blocks are not equivalent to earlier ones.

READ THIS BEFORE BELIEVING ANY P VALUE
  With 6 mice the exact Wilcoxon signed-rank test cannot return a two-tailed
  p below 2/2^6 = 0.031. A single PRE-SPECIFIED test can therefore reach 0.05;
  4 stimuli x 6 behaviours = 24 tests cannot all survive FDR correction, by
  construction. Pick the primary outcome before looking, and treat the rest
  as exploratory. The script prints the floor next to every p value.

USAGE
  python step2_block_stats.py <folder with BlockMeasures_long.csv>
        [--schedule path\\Stimulus_randomisation_mini1p.xlsx]
        [--measure rate_per_min|dur_pct_time|n_per_delivery]

Hansol Lim - HEAL mini1p / SBI-553
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

DEFAULT_SCHEDULE = (r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553"
                    r"\Stimulus_randomisation_mini1p.xlsx")
BEHAV_ORDER = ["attending", "lickbite", "guarding", "escape",
               "withdrawal", "flinch"]


def load_treatment(path):
    """Session -> treatment, from the randomisation sheet."""
    if not os.path.exists(path):
        print(f"  no schedule at {path}; treatment will stay BLIND")
        return None
    # keep_default_na=False matters: the no-injection sessions are labelled
    # "None" in the sheet, and pandas parses that string as NaN by default,
    # which silently drops them and kills the injection-control contrast.
    S = pd.read_excel(path, sheet_name="Schedule", keep_default_na=False)
    S.columns = [str(c).strip() for c in S.columns]
    S["Treatment"] = S["Treatment"].astype(str).str.strip()
    S.loc[S["Treatment"].isin(["", "nan", "NaN"]), "Treatment"] = "None"
    keep = S[["Session", "Day", "Mouse ID", "Phase", "Treatment"]].copy()
    keep["Session"] = keep["Session"].astype(str).str.strip()
    print(f"  schedule: {len(keep)} sessions, treatments "
          f"{sorted(keep['Treatment'].dropna().unique())}")
    return keep


def wilcoxon_floor(n):
    """Smallest attainable two-tailed p for the exact signed-rank test."""
    return 2.0 / (2 ** n) if n > 0 else np.nan


def paired_test(x):
    """Exact Wilcoxon signed-rank + paired t on a vector of within-mouse diffs."""
    from scipy import stats
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    out = dict(n=n, median=np.nan, mean=np.nan, p_wilcoxon=np.nan,
               p_ttest=np.nan, p_floor=wilcoxon_floor(n))
    if n < 2:
        return out
    out["median"] = float(np.median(x))
    out["mean"] = float(np.mean(x))
    if np.allclose(x, 0):
        return out
    try:
        out["p_wilcoxon"] = float(
            stats.wilcoxon(x, zero_method="wilcox", alternative="two-sided",
                           method="exact" if n <= 25 else "auto").pvalue)
    except ValueError:
        pass
    try:
        out["p_ttest"] = float(stats.ttest_1samp(x, 0).pvalue)
    except ValueError:
        pass
    return out


def fdr(p):
    """Benjamini-Hochberg, NaN-safe."""
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


def mixed_per_behaviour(D, measure, ref_stim=None, ref_treat="Vehicle"):
    """One mixed model per behaviour, pooled over stimuli.

        delta ~ C(treatment)*C(stimulus) + pos + (1|mouse)

    IMPORTANT ON READING THE COEFFICIENTS
        Treatment and stimulus are treated as factors, so the plain
        C(treatment)[T.X] term is the effect of X **at the reference
        stimulus**, not an average over stimuli. The reference is whichever
        level sorts first unless it is set explicitly, which is very easy to
        misread. This function therefore forces the references and prints
        them, and returns one row per model term so nothing is hidden behind
        a single "smallest p".
    """
    import statsmodels.formula.api as smf
    rows = []
    for b in [x for x in BEHAV_ORDER if x in set(D.behaviour)]:
        d = D[(D.behaviour == b) & np.isfinite(D["delta"])].copy()
        if d.mouse.nunique() < 3 or len(d) < 12:
            rows.append(dict(behaviour=b, term="", n_obs=len(d),
                             n_mouse=d.mouse.nunique(), coef=np.nan, p=np.nan,
                             note="too few data"))
            continue
        rs = ref_stim if ref_stim in set(d.stimulus) else sorted(d.stimulus)[0]
        rt = ref_treat if ref_treat in set(d.treatment) else sorted(d.treatment)[0]
        d["treatment"] = pd.Categorical(d["treatment"].astype(str))
        d["stimulus"] = pd.Categorical(d["stimulus"].astype(str))
        f = (f'delta ~ C(treatment, Treatment(reference="{rt}")) '
             f'* C(stimulus, Treatment(reference="{rs}")) + pos')
        try:
            m = smf.mixedlm(f, d, groups=d["mouse"]).fit(reml=True)
            for k in m.params.index:
                if "treatment" not in k:
                    continue
                rows.append(dict(behaviour=b, term=tidy_term(k),
                                 n_obs=len(d), n_mouse=d.mouse.nunique(),
                                 coef=float(m.params[k]),
                                 p=float(m.pvalues[k]),
                                 ref_stimulus=rs, ref_treatment=rt, note="ok"))
        except Exception as e:                                  # noqa: BLE001
            rows.append(dict(behaviour=b, term="", n_obs=len(d),
                             n_mouse=d.mouse.nunique(), coef=np.nan, p=np.nan,
                             note=f"{type(e).__name__}: {e}"))
    R = pd.DataFrame(rows)
    if not R.empty and "p" in R:
        R["q"] = fdr(R["p"])          # FDR across every treatment term shown
    return R


def tidy_term(k):
    """Turn statsmodels' patsy names into something readable."""
    out = (k.replace('C(treatment, Treatment(reference="', "trt:")
            .replace('C(stimulus, Treatment(reference="', "stim:")
            .replace('"))', "").replace("[T.", "=").replace("]", ""))
    return out.replace(":stim:", " x stim:")


def figures(D, C, measure, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    behs = [b for b in BEHAV_ORDER if b in set(D.behaviour)]
    stims = sorted(D.stimulus.unique())
    treats = [t for t in ["None", "Vehicle", "SBI-553"]
              if t in set(D.treatment)] or sorted(D.treatment.unique())
    cols = {"None": "#9CA3AF", "Vehicle": "#1C6E8C", "SBI-553": "#C0483B"}

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, b in zip(axes.ravel(), behs):
        d = D[D.behaviour == b]
        for i, t in enumerate(treats):
            xs, ys = [], []
            for j, s in enumerate(stims):
                v = d[(d.treatment == t) & (d.stimulus == s)]["delta"].dropna()
                if len(v):
                    xs.append(j + (i - (len(treats) - 1) / 2) * 0.22)
                    ys.append(v.mean())
                    ax.scatter([xs[-1]] * len(v), v, s=14, alpha=.55,
                               color=cols.get(t, "#333"), zorder=3)
            if xs:
                ax.plot(xs, ys, "o-", color=cols.get(t, "#333"), lw=1.8,
                        ms=7, mfc="white", mew=1.6, label=t, zorder=4)
        ax.axhline(0, color="k", lw=1, ls="--", alpha=.6)
        ax.set_xticks(range(len(stims)))
        ax.set_xticklabels(stims, rotation=25, ha="right", fontsize=8)
        ax.set_title(b, fontsize=11)
        ax.set_ylabel(f"delta {measure}\n(block - baseline)", fontsize=8)
        ax.grid(alpha=.25)
    for ax in axes.ravel()[len(behs):]:
        ax.axis("off")
    axes.ravel()[0].legend(fontsize=8, loc="best")
    fig.suptitle(f"Baseline-subtracted block measures  ({measure})", fontsize=13)
    fig.tight_layout()
    p1 = os.path.join(outdir, "Fig_delta_by_stimulus.png")
    fig.savefig(p1, dpi=140)
    plt.close(fig)

    # DiD forest
    if not C.empty:
        fig, ax = plt.subplots(figsize=(9, max(4, 0.30 * len(C))))
        C = C.sort_values(["behaviour", "stimulus"]).reset_index(drop=True)
        y = np.arange(len(C))
        sem = C["mean"] / np.where(C["n"] > 0, np.sqrt(C["n"]), np.nan)
        ax.errorbar(C["mean"], y, xerr=np.abs(sem), fmt="o", color="#1F2937",
                    ecolor="#9CA3AF", capsize=3, ms=5)
        ax.axvline(0, color="#C0483B", lw=1.2, ls="--")
        ax.set_yticks(y)
        ax.set_yticklabels([f"{r.behaviour} / {r.stimulus}"
                            for r in C.itertuples()], fontsize=8)
        ax.set_xlabel(f"DiD  =  delta(SBI-553) - delta(Vehicle)   [{measure}]")
        ax.set_title("Difference-in-differences, paired within mouse")
        ax.grid(alpha=.25, axis="x")
        fig.tight_layout()
        p2 = os.path.join(outdir, "Fig_DiD_forest.png")
        fig.savefig(p2, dpi=140)
        plt.close(fig)
        return p1, p2
    return p1, None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--schedule", default=DEFAULT_SCHEDULE)
    ap.add_argument("--measure", default="rate_per_min",
                    choices=["rate_per_min", "dur_pct_time", "n_per_delivery"])
    ap.add_argument("--ref-stimulus", default=None,
                    help="stimulus used as the model reference level; the plain "
                         "treatment coefficient is its effect AT this stimulus")
    a = ap.parse_args()

    lp = os.path.join(a.folder, "BlockMeasures_long.csv")
    if not os.path.exists(lp):
        raise SystemExit(f"{lp} not found - run step1_block_measures.py first")
    L = pd.read_csv(lp)
    L["session"] = L["session"].astype(str).str.strip()
    print(f"loaded {len(L)} rows, {L.mouse.nunique()} mouse/mice, "
          f"{L.session.nunique()} session(s)")

    T = load_treatment(a.schedule)
    if T is not None:
        L = L.merge(T[["Session", "Treatment"]].rename(
            columns={"Session": "session", "Treatment": "treatment"}),
            on="session", how="left")
        miss = L["treatment"].isna().sum()
        if miss:
            print(f"  WARNING: {miss} row(s) had no schedule match; "
                  "check the Session numbers you typed while scoring")
        L["treatment"] = L["treatment"].fillna("UNKNOWN")
    else:
        L["treatment"] = "UNKNOWN"

    m = a.measure
    if m == "n_per_delivery":
        print("  NOTE: n_per_delivery is undefined for baseline (no deliveries),"
              "\n  so baseline subtraction is not possible with this measure. "
              "It is\n  reported for between-stimulus comparison only.")

    # ---- 1. delta vs each session's own baseline ----
    base = (L[L["kind"] == "baseline"]
            .set_index(["session", "behaviour"])[m].rename("baseline_val"))
    B = L[L["kind"] == "stimulus"].copy()
    B = B.join(base, on=["session", "behaviour"])
    B["delta"] = B[m] - B["baseline_val"]
    dp = os.path.join(a.folder, f"Delta_vs_baseline_{m}.csv")
    B.to_csv(dp, index=False)
    print(f"\n  wrote {dp}")

    # ---- 2. DiD and the injection control ----
    def contrast(hi, lo, label):
        rows = []
        piv = B.pivot_table(index=["mouse", "stimulus", "behaviour"],
                            columns="treatment", values="delta",
                            aggfunc="mean")
        if hi not in piv.columns or lo not in piv.columns:
            print(f"  {label}: needs both '{hi}' and '{lo}' sessions - skipped")
            return pd.DataFrame(), pd.DataFrame()
        piv = piv.reset_index()
        piv["diff"] = piv[hi] - piv[lo]
        for (s, b), g in piv.groupby(["stimulus", "behaviour"]):
            r = paired_test(g["diff"].to_numpy())
            rows.append(dict(contrast=label, stimulus=s, behaviour=b, **r))
        C = pd.DataFrame(rows)
        if not C.empty:
            C["q_wilcoxon"] = fdr(C["p_wilcoxon"])
            C["q_ttest"] = fdr(C["p_ttest"])
        return C, piv

    print("\n  ======== PRIMARY: DiD = delta(SBI-553) - delta(Vehicle) ========")
    C, piv = contrast("SBI-553", "Vehicle", "SBI-553 vs Vehicle")
    if not C.empty:
        cp = os.path.join(a.folder, f"DiD_SBI_vs_Vehicle_{m}.csv")
        C.to_csv(cp, index=False)
        piv.to_csv(os.path.join(a.folder, f"DiD_per_mouse_{m}.csv"), index=False)
        show = C.sort_values("p_wilcoxon", na_position="last")
        print(f"  {'behaviour':11s} {'stimulus':12s} {'n':>2s} {'median':>8s} "
              f"{'p_wilc':>7s} {'p_floor':>8s} {'q_wilc':>7s} {'p_t':>7s}")
        for r in show.itertuples():
            print(f"  {r.behaviour:11s} {r.stimulus:12s} {r.n:2d} "
                  f"{r.median:8.3f} {r.p_wilcoxon:7.3f} {r.p_floor:8.3f} "
                  f"{r.q_wilcoxon:7.3f} {r.p_ttest:7.3f}")
        print(f"  wrote {cp}")

    print("\n  ---- injection control: delta(Vehicle) - delta(None) ----")
    C2, _ = contrast("Vehicle", "None", "Vehicle vs None")
    if not C2.empty:
        C2.to_csv(os.path.join(a.folder, f"Injection_control_{m}.csv"),
                  index=False)
        sig = C2[C2.p_wilcoxon < 0.05]
        if sig.empty:
            print("  nothing reaches p<0.05: the injection alone did not "
                  "detectably change any behaviour.")
        else:
            for r in sig.itertuples():
                print(f"  {r.behaviour}/{r.stimulus}: median {r.median:.3f}, "
                      f"p={r.p_wilcoxon:.3f}  <- injection effect, not the drug")

    # ---- 3. mixed model per behaviour ----
    print("\n  ---- mixed model  delta ~ treatment*stimulus + pos + (1|mouse) ----")
    MM = mixed_per_behaviour(B, m, ref_stim=a.ref_stimulus)
    MM.to_csv(os.path.join(a.folder, f"MixedModel_{m}.csv"), index=False)
    if not MM.empty and "ref_stimulus" in MM:
        refs = MM.dropna(subset=["coef"])
        if not refs.empty:
            print(f"  reference levels: stimulus = '{refs.ref_stimulus.iloc[0]}', "
                  f"treatment = '{refs.ref_treatment.iloc[0]}'")
            print("  so a plain 'trt:=X' term is the effect of X AT THAT "
                  "STIMULUS, not an average.")
    keep = MM[np.isfinite(pd.to_numeric(MM.get("p"), errors="coerce"))]
    if keep.empty:
        for rec in MM.to_dict("records"):
            print(f"  {rec['behaviour']:11s} {rec.get('note', '')}")
    else:
        keep = keep.sort_values("p")
        print(f"  {'behaviour':11s} {'term':44s} {'coef':>8s} {'p':>7s} {'q':>7s}")
        for r in keep.itertuples():
            star = "  *" if r.q < 0.05 else ""
            print(f"  {r.behaviour:11s} {r.term[:44]:44s} {r.coef:8.3f} "
                  f"{r.p:7.4f} {r.q:7.4f}{star}")
        print("  q = Benjamini-Hochberg across every treatment term above.")

    # ---- 4. figures ----
    p1, p2 = figures(B, C, m, a.folder)
    print(f"\n  wrote {p1}")
    if p2:
        print(f"  wrote {p2}")

    n = B.mouse.nunique()
    print(f"\n  ---- how much to trust this ----")
    print(f"  {n} mouse/mice. Exact Wilcoxon floor at n={n} is "
          f"p = {wilcoxon_floor(n):.3f} two-tailed.")
    if not C.empty:
        print(f"  {len(C)} stimulus x behaviour cells were tested. A single "
              f"pre-specified\n  cell can reach 0.05; {len(C)} cells cannot all "
              f"survive FDR by construction.\n  Decide the primary outcome "
              f"before reading the table above.")


if __name__ == "__main__":
    main()
