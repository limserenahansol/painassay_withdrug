"""Day-1 Ca analysis, extended (on top of ca_day1_analysis.py): controls and deeper analyses.

E1  SIGNAL-QUALITY CONTROL  Is a pre -> post change a change in the cells or in the recording?
    Per cell and session: noise of the local dF/F (PSD), transient amplitude, decay tau, the
    EXTRACT trace's slow baseline (bleaching), fraction of dropped frames. If these are equal
    in both sessions, a rate change is not an SNR artefact.
E2  MIXED-EFFECTS DRUG MODEL  cells nested in mice (statsmodels MixedLM, random intercept per
    mouse + variance component per cell): y ~ session * arm. The session x arm interaction is
    the change after SBI-553 beyond the change after vehicle (both arms re-tested). Outcomes:
    log2 transient rate (whole session, no-stimulus start, still frames) and the event
    responses. n = 2 mice per arm: the model uses every cell but the mouse-level uncertainty
    stays large (shown as 95% CI).
E3  ENCODING MODEL (ridge GLM, 10 Hz)  z(t) = fast brain motion (|change of the MC shift|; not the
    slow shift level or the frame/template r, both shared with real activity) + stimulus kernels +
    withdrawal / flinch kernels (0-3 s) + movement (state + onset kernel) + grooming + escape +
    slow drift. Cross-validated (5 folds of interleaved 20-s chunks) unique variance of each group = R2(full)
    - R2(without it). And the stimulus kernels WITH movement in the model: the stimulus
    response that movement cannot explain, baseline vs injection.
E4  SAME CELLS?  Are the cells responsive in the baseline also responsive after the injection
    (overlap vs hypergeometric chance), and does each cell keep its response (correlation of
    the response across cells, pre vs post)?
E5  ADAPTATION  population response vs delivery number within the block, slope pre vs post.
E6  RELIABILITY + LATENCY  trial-to-trial correlation of each cell's response (0-3 s) and the
    latency of its peak, pre vs post.
E7  PAIRWISE CORRELATIONS  in the no-stimulus start of each session (0.5-s bins), pre vs post
    on the same pairs, and against the distance between the cells.
E8  POPULATION TRAJECTORIES  PCA of the trial-mean responses (axes from the baseline session),
    every stimulus, baseline and injection on the same axes.
E9  SPATIAL CLUSTERING  are responsive cells closer to each other than random sets of cells?
E10 INTENSITY CODING  does the response grow from light touch to heat (WT deck slide 10), and
    does the injection flatten the ordering? (Spearman per cell, pre vs post)
E11 PREDICTING THE WITHDRAWAL  from the population state before the delivery / its response,
    beyond what the stimulus type alone predicts (CV ROC AUC, within-stimulus shuffles)
E12 RESPONSE TYPES  k-means on the baseline response profiles; the same cells after injection
E15 BLOCK POSITION  the blocks run in a different order per session: mixed model of the still-frame
    block activity with block position and stimulus as covariates (drug effect held at equal time)
E13 'NOXIOUS' AXIS  decoder noxious (pin, heat) vs innocuous (light, mild) trained on the baseline,
    applied to the injection session (generalisation, d' along the axis), with the drift control

  python ca_day1_extended.py --out-root <mini1p_out/20261019> --scoring-dir <final> --day1 <analysis/ca_day1>
      --out <analysis/ca_day1_extended>
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import hypergeom, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import safe_io  # noqa: E402,F401  (a figure save retries on a transient Windows write error)
import ca_day1_analysis as d1  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FPS = d1.FPS
MICE = d1.MICE
ARMC, PHC, EVC, STIM, EVENTS = d1.ARMC, d1.PHC, d1.EVC, d1.STIM, d1.EVENTS
PH = {"pre": "baseline", "post": "injection"}
warnings.filterwarnings("ignore", category=RuntimeWarning)


def paired_panel(a, P, color, ylab, title, log=False):
    for _, r in P.iterrows():
        a.plot([0, 1], [r.pre, r.post], color="0.8", lw=0.5)
    a.plot([0, 1], [P.pre.median(), P.post.median()], color=color, lw=2.5, marker="o")
    a.set_xticks([0, 1], ["baseline", "injection"], fontsize=8); a.set_xlim(-0.3, 1.3)
    if log:
        a.set_yscale("log")
    a.set_ylabel(ylab, fontsize=8)
    a.set_title(f"{title}\nn = {len(P)}, Wilcoxon {d1.pstr(d1.wsr(P.post, P.pre))}", fontsize=8)


# ------------------------------------------------------------------------------ E1 quality
def e1_quality(sess, out):
    rows = []
    for s in sess:
        T, E = s["T"], s["E"]
        ok = T["valid"].astype(bool) & (T["t_s"] >= d1.SKIP_S)
        for j, c in enumerate(T["cell_ids"].astype(int)):
            e = E[E.cell == c]
            b = T["baseline"][:, j]
            okb = ok & np.isfinite(b)
            q = len(b[okb]) // 10
            rows.append(dict(mouse=s["mouse"], phase=s["phase"], cell=int(c), noise=float(T["sn"][j]),
                             tau_s=float(T["tau_s"][j]), amp=float(e.amplitude_dff.median()) if len(e) else np.nan,
                             peak_z=float(e.peak_z.median()) if len(e) else np.nan,
                             drift=float(np.nanmean(b[okb][-q:]) - np.nanmean(b[okb][:q])) if q > 0 else np.nan,
                             dropped=float(1 - T["valid"].mean())))
    Q = pd.DataFrame(rows)
    mice = [m for m in MICE if m[0] in set(Q.mouse)]
    meas = [("noise", "noise of local dF/F (SD)", True), ("amp", "median transient amplitude (dF/F)", True),
            ("tau_s", "decay tau (s)", False), ("drift", "slow baseline change over the session\n(EXTRACT dF/F units)", False)]
    fig, ax = plt.subplots(len(meas), len(mice), figsize=(3.6 * len(mice), 3.0 * len(meas)), squeeze=False)
    tests = []
    for k, (m, arm, sex) in enumerate(mice):
        for i, (c, lab, lg) in enumerate(meas):
            P = Q[Q.mouse == m].pivot(index="cell", columns="phase", values=c).dropna()
            if not {"pre", "post"} <= set(P.columns):
                ax[i, k].axis("off"); continue
            paired_panel(ax[i, k], P, ARMC[arm], lab, f"{m} ({arm})" if i == 0 else "", log=lg and (P > 0).all().all())
            tests.append(dict(mouse=m, arm=arm, measure=c, n=len(P), pre_median=P.pre.median(), post_median=P.post.median(),
                              ratio=P.post.median() / P.pre.median() if P.pre.median() else np.nan, p=d1.wsr(P.post, P.pre)))
    fig.suptitle("E1 signal-quality control: noise, transient size, decay and bleaching per cell, baseline vs injection\n"
                 "(if these do not change, a change in transient rate is not a change in how well the cells are seen)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(out, "E1_signal_quality.png"), dpi=110); plt.close(fig)
    Q.round(6).to_csv(os.path.join(out, "E1_cells_quality.csv"), index=False)
    pd.DataFrame(tests).round(5).to_csv(os.path.join(out, "E1_quality_tests.csv"), index=False)


def fit_mixed(form, D, vc_col):
    """MixedLM (random mouse + variance component per cell) fitted with several optimizers; the
    first that converges is kept. Returns (fit, converged, method). With few mice the variance of
    the mouse intercept often sits at 0 and lbfgs stops early - powell / nm then usually finish."""
    import warnings as _w
    import statsmodels.formula.api as smf
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    md = smf.mixedlm(form, D, groups=D["mouse"], re_formula="1", vc_formula={"cell": f"0 + C({vc_col})"})
    best = None
    for meth in ("lbfgs", "powell", "nm", "cg"):
        with _w.catch_warnings(record=True) as wl:
            _w.simplefilter("always", ConvergenceWarning)
            try:
                f = md.fit(reml=True, method=meth, maxiter=2000)
            except Exception:                                                # noqa: BLE001
                continue
        ok = bool(getattr(f, "converged", True)) and not any(issubclass(w.category, ConvergenceWarning) for w in wl)
        if ok:
            return f, True, meth
        if best is None:
            best = (f, False, meth)
    return best if best else (None, False, "none")


# ------------------------------------------------------------------------------ E2 mixed model
def e2_mixed(W, R, out):
    import statsmodels.formula.api as smf
    outs = []
    W = W.copy()
    W["arm"] = W.mouse.astype(str).map({m: a for m, a, _ in MICE})
    for c, lab in (("rate", "transient rate, whole session"), ("rate | no stimulus", "rate, no-stimulus start"),
                   ("rate | still", "rate, still frames"), ("rate | moving", "rate, moving frames")):
        if c in W:
            D = W[["mouse", "cell", "phase", "arm", c]].dropna().rename(columns={c: "v"})
            D["y"] = np.log2(D.v + 0.1)
            outs.append((lab, D))
    if R is not None and len(R):
        R = R.copy()
        R["arm"] = R.mouse.astype(str).map({m: a for m, a, _ in MICE})
        for e in EVENTS:
            D = R[R.event == e][["mouse", "cell", "phase", "arm", "resp_mean"]].dropna().rename(columns={"resp_mean": "y"})
            if len(D):
                outs.append((f"response: {e}", D))
    rows = []
    for lab, D in outs:
        D = D.copy()
        D["mouse"] = D.mouse.astype(str); D["mc"] = D.mouse + "_" + D.cell.astype(str)
        D["post"] = (D.phase == "post").astype(float); D["sbi"] = (D.arm == "SBI-553").astype(float)
        if D.mouse.nunique() < 2 or D.groupby("mc").phase.nunique().max() < 2:
            continue
        both_arms = D.sbi.nunique() == 2
        form = "y ~ post * sbi" if both_arms else "y ~ post"
        try:
            f, conv, meth = fit_mixed(form, D, "mc")
            for term in (["post", "post:sbi"] if both_arms else ["post"]):
                ci = f.conf_int().loc[term]
                rows.append(dict(outcome=lab, term=term, coef=f.params[term], ci_lo=ci[0], ci_hi=ci[1],
                                 p=f.pvalues[term], n_obs=len(D), n_cells=D.mc.nunique(), n_mice=D.mouse.nunique(),
                                 converged=conv, optimizer=meth))
        except Exception as ex:                                       # noqa: BLE001
            rows.append(dict(outcome=lab, term="fit failed", coef=np.nan, ci_lo=np.nan, ci_hi=np.nan, p=np.nan,
                             n_obs=len(D), n_cells=D.mc.nunique(), n_mice=D.mouse.nunique(), note=str(ex)[:120]))
    M = pd.DataFrame(rows)
    M.round(5).to_csv(os.path.join(out, "E2_mixed_model.csv"), index=False)
    if not len(M) or "post:sbi" not in set(M.term):
        return M
    V = M[M.term == "post:sbi"].reset_index(drop=True)
    V0 = M[M.term == "post"].set_index("outcome")
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(V) + 1.8))
    y = np.arange(len(V))[::-1]
    ax.errorbar(V.coef, y, xerr=[V.coef - V.ci_lo, V.ci_hi - V.coef], fmt="o", color=ARMC["SBI-553"], capsize=3,
                label="SBI-553 beyond vehicle (session x arm)")
    vv = V0.reindex(V.outcome)
    ax.errorbar(vv.coef.values, y - 0.25, xerr=[vv.coef.values - vv.ci_lo.values, vv.ci_hi.values - vv.coef.values], fmt="s",
                color=ARMC["vehicle"], capsize=3, ms=4, label="vehicle: re-test change (session)")
    for yy, r in zip(y, V.itertuples()):
        ax.text(ax.get_xlim()[1] if False else max(V.ci_hi.max(), 0) * 1.05 + 0.02, yy, d1.pstr(r.p), va="center", fontsize=7)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y, V.outcome, fontsize=8)
    ax.set_xlabel("effect (log2 rate, or z units for responses), 95% CI")
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    ax.set_title("E2 mixed-effects model, cells nested in mice: y ~ session x arm (random mouse + cell)\n"
                 f"{V.n_mice.max()} mice, {V.n_cells.max()} cells; with 2 mice per arm the CI is wide", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "E2_mixed_model.png"), dpi=120); plt.close(fig)
    return M


# ------------------------------------------------------------------------------ E3 encoding model
def bumps(n, t1, width, dt):
    t = np.arange(0, t1 + dt / 2, dt)
    c = np.linspace(0, t1, n)
    return np.stack([np.exp(-0.5 * ((t - ci) / width) ** 2) for ci in c], 1), t


def design(s, bin_f=3):
    T, B = s["T"], s["B"]
    fr = T["frame"].astype(int)
    n = len(fr) // bin_f
    dt = bin_f / FPS
    Z = np.nanmean(T["z"][:n * bin_f].reshape(n, bin_f, -1), 1)
    tb = fr[:n * bin_f].reshape(n, bin_f)[:, 0]
    groups, cols = {}, []

    def impulse(frames):
        v = np.zeros(n)
        i = np.searchsorted(tb, np.asarray(frames, int), side="right") - 1
        i = i[(i >= 0) & (i < n)]
        np.add.at(v, i, 1.0)
        return v

    def ev(nm):
        f = B["ev"][nm]
        return f if len(f) >= 3 else []                     # < 3 events: no kernel can be estimated

    def add(group, X):
        groups.setdefault(group, []).extend(range(len(cols), len(cols) + X.shape[1]))
        cols.extend(list(X.T))
    Kst, _ = bumps(6, 5.0, 0.6, dt)
    Kre, _ = bumps(4, 3.0, 0.5, dt)
    Kmv, _ = bumps(3, 2.0, 0.5, dt)
    conv = lambda v, K: np.stack([np.convolve(v, K[:, k])[:n] for k in range(K.shape[1])], 1)
    for st in STIM:
        add("stimuli", conv(impulse(ev(st)), Kst))
    add("reflexes", conv(impulse(ev("withdrawal")), Kre))
    add("reflexes", conv(impulse(ev("flinch")), Kre))
    st = B["st"]
    state = lambda v: np.array([v[min(f, len(v) - 1)] if f < len(v) else 0 for f in tb], float)
    add("movement", np.c_[state(st["move"])])
    add("movement", conv(impulse(ev("movement onset")), Kmv))
    add("grooming", np.c_[state(st["groom"])])
    add("escape", conv(impulse(ev("escape/rearing onset")), Kre))
    # registration artefacts (if the shifts table exists): |shift|, its change, 1 - frame/template r
    mot = s.get("motion")
    if mot is not None:
        bmean = lambda v: np.nanmean(np.asarray(v[:n * bin_f], float).reshape(n, bin_f), 1)
        shb, cob = bmean(mot["shift"]), bmean(mot["corr"])
        shb = np.nan_to_num(shb - np.nanmedian(shb)); cob = np.nan_to_num(np.nanmedian(cob) - cob)
        # only the SHIFTS: the frame/template correlation is lowered by the cells' own activity (an
        # active frame differs from the mean image), so as a regressor it would remove real signal
        # (5941: it 'explained' 14% of the variance - circular)
        # and only the FAST part (|change of the shift| per bin and the next bin): the slow shift level
        # follows posture over minutes and is collinear with slow state changes of the cells, so it
        # would also take real block-level activity (5941: 6.5% 'unique' variance)
        dsh = np.abs(np.diff(np.r_[shb[0], shb]))
        dsh = dsh / (np.std(dsh) + 1e-9)
        add("motion_artefact", np.c_[dsh, np.r_[dsh[1:], 0.0]])
    # slow drift: cosines up to one cycle per 2 min
    tt = np.arange(n) * dt
    L = tt[-1] + dt
    nd = max(int(L / 120), 1)
    add("drift", np.stack([np.cos(np.pi * k * tt / L) for k in range(1, nd + 1)], 1))
    X = np.stack(cols, 1)
    ok = np.all(np.isfinite(Z), 1) & (tb / FPS >= d1.SKIP_S)
    return X[ok], Z[ok], groups, dt, Kst


def ridge_cv(X, Y, lam, folds=5, chunk=200):
    """ridge regression, cross-validated on interleaved 20-s chunks (10 Hz bins): every fold holds
    pieces of every schedule block (stimuli come in blocks, so contiguous folds would test a
    stimulus the training data never saw), and 20 s is much longer than a Ca transient"""
    n = len(X)
    fold_of = (np.arange(n) // chunk) % folds
    pred = np.zeros_like(Y)
    for k in range(folds):
        te = fold_of == k
        Xtr = np.c_[np.ones((~te).sum()), X[~te]]; Xte = np.c_[np.ones(te.sum()), X[te]]
        # regressors are all on a 0-1 (or +-1) scale: centred, NOT divided by their SD (a rare
        # event split across folds would otherwise blow up); absent in the training part -> 0
        mu = Xtr[:, 1:].mean(0)
        dead = Xtr[:, 1:].std(0) < 1e-6
        Xtr[:, 1:] = Xtr[:, 1:] - mu; Xte[:, 1:] = Xte[:, 1:] - mu
        Xtr[:, 1:][:, dead] = 0; Xte[:, 1:][:, dead] = 0
        I = np.eye(Xtr.shape[1]); I[0, 0] = 0
        Bt = np.linalg.solve(Xtr.T @ Xtr + lam * I, Xtr.T @ Y[~te])
        pred[te] = Xte @ Bt
    sst = ((Y - Y.mean(0)) ** 2).sum(0)
    return 1 - ((Y - pred) ** 2).sum(0) / np.maximum(sst, 1e-12)


def e3_encoding(sess, out):
    rows, kern = [], []
    # ONE ridge penalty per mouse (best summed CV R2 over its two sessions): with a different
    # penalty per session the kernels would shrink differently and a pre/post change would be
    # partly the penalty (the synthetic test showed this)
    D = {(s["mouse"], s["phase"]): design(s) for s in sess if s["B"] is not None}
    grid = (1.0, 10.0, 100.0, 1000.0, 1e4)
    lam_m = {}
    for m in {k[0] for k in D}:
        sc = [sum(np.nanmedian(ridge_cv(D[k][0], D[k][1], l)) for k in D if k[0] == m) for l in grid]
        lam_m[m] = grid[int(np.argmax(sc))]
    for s in sess:
        if s["B"] is None:
            continue
        X, Z, G, dt, Kst = D[(s["mouse"], s["phase"])]
        ids = s["T"]["cell_ids"].astype(int)
        lam = lam_m[s["mouse"]]
        full = ridge_cv(X, Z, lam)
        uniq = {}
        for g, idx in G.items():
            if g == "drift":
                continue
            keep = np.setdiff1d(np.arange(X.shape[1]), idx)
            uniq[g] = full - ridge_cv(X[:, keep], Z, lam)
        # kernels on all data (movement and the rest in the model)
        Xa = np.c_[np.ones(len(X)), X - X.mean(0)]
        I = np.eye(Xa.shape[1]); I[0, 0] = 0
        Bt = np.linalg.solve(Xa.T @ Xa + lam * I, Xa.T @ Z)
        for k, st in enumerate(STIM):
            cidx = G["stimuli"][6 * k:6 * k + 6]
            if not np.any(X[:, cidx]):
                continue
            w = Bt[1 + np.array(cidx)]                                 # per unit impulse
            resp = Kst @ w                                             # time x cells
            t = np.arange(Kst.shape[0]) * dt
            amp = resp[(t >= 0) & (t < 2.0)].mean(0)
            for j, c in enumerate(ids):
                kern.append(dict(mouse=s["mouse"], phase=s["phase"], stimulus=st, cell=int(c), kernel_amp_0_2s=amp[j]))
        for j, c in enumerate(ids):
            rows.append(dict(mouse=s["mouse"], phase=s["phase"], cell=int(c), lam=lam, r2_full=full[j],
                             **{f"unique_{g}": v[j] for g, v in uniq.items()}))
        print(f"E3 {s['rid']}: ridge lambda {lam:g} (one per mouse), median CV R2 {np.median(full):.3f}")
    U = pd.DataFrame(rows); K = pd.DataFrame(kern)
    U.round(6).to_csv(os.path.join(out, "E3_encoding_unique_variance.csv"), index=False)
    K.round(6).to_csv(os.path.join(out, "E3_stimulus_kernels_movement_controlled.csv"), index=False)
    if not len(U):
        return
    mice = [m for m in MICE if m[0] in set(U.mouse)]
    grp = [c for c in U.columns if c.startswith("unique_")]
    fig, ax = plt.subplots(2, len(mice), figsize=(4.3 * len(mice), 8), squeeze=False)
    for k, (m, arm, sex) in enumerate(mice):
        a = ax[0, k]
        x = np.arange(len(grp))
        for dx, ph in ((-0.2, "pre"), (0.2, "post")):
            V = U[(U.mouse == m) & (U.phase == ph)]
            if not len(V):
                continue
            mu = [100 * np.clip(V[g], 0, None).mean() for g in grp]
            se = [100 * np.clip(V[g], 0, None).std() / np.sqrt(len(V)) for g in grp]
            a.bar(x + dx, mu, 0.38, yerr=se, color=PHC[ph], capsize=2, label=PH[ph])
        a.set_xticks(x, [g.replace("unique_", "") for g in grp], rotation=30, fontsize=8)
        a.set_ylabel("unique variance explained (% , CV)\nmean +- SEM over cells", fontsize=8)
        a.set_title(f"{m} ({arm})", color=ARMC[arm], fontsize=10)
        a.legend(fontsize=7, frameon=False)
        b = ax[1, k]
        xs = np.arange(len(STIM))
        for dx, ph in ((-0.2, "pre"), (0.2, "post")):
            V = K[(K.mouse == m) & (K.phase == ph)]
            mu = [V[V.stimulus == st].kernel_amp_0_2s.mean() for st in STIM]
            se = [V[V.stimulus == st].kernel_amp_0_2s.std() / np.sqrt(max((V.stimulus == st).sum(), 1)) for st in STIM]
            b.bar(xs + dx, mu, 0.38, yerr=se, color=PHC[ph], capsize=2, label=PH[ph])
        for i, st in enumerate(STIM):
            P = K[(K.mouse == m) & (K.stimulus == st)].pivot(index="cell", columns="phase", values="kernel_amp_0_2s").dropna()
            if {"pre", "post"} <= set(P.columns) and len(P) > 3:
                p = d1.wsr(P.post, P.pre)
                b.text(i, b.get_ylim()[1] * 0.92, d1.pstr(p).replace("p = ", "p="), ha="center", fontsize=6)
        b.axhline(0, color="k", lw=0.6)
        b.set_xticks(xs, STIM, fontsize=8)
        b.set_ylabel("stimulus kernel, mean z over 0-2 s\n(movement, reflexes, grooming in the model)", fontsize=8)
        b.legend(fontsize=7, frameon=False)
    fig.suptitle("E3 encoding model (ridge GLM, 10 Hz, 5-fold CV on interleaved 20-s chunks): what each cell's activity is explained by, and the "
                 "stimulus response that movement cannot explain\n(Wilcoxon pre vs post over cells)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(out, "E3_encoding_model.png"), dpi=110); plt.close(fig)


# ------------------------------------------------------------------------------ E4 same cells
def e4_identity(R, out):
    rows = []
    mice = [m for m in MICE if m[0] in set(R.mouse)]
    evs = [e for e in EVENTS if e in set(R.event)]
    fig, ax = plt.subplots(len(mice), len(evs), figsize=(2.4 * len(evs), 2.6 * len(mice)), squeeze=False)
    for i, (m, arm, sex) in enumerate(mice):
        for k, e in enumerate(evs):
            a = ax[i, k]
            V = R[(R.mouse == m) & (R.event == e)]
            P = V.pivot(index="cell", columns="phase", values="resp_mean").dropna()
            C = V.pivot(index="cell", columns="phase", values="cls")
            if not {"pre", "post"} <= set(P.columns) or len(P) < 5:
                a.axis("off"); continue
            C = C.loc[P.index]
            rp = C.pre != "no response"; rq = C.post != "no response"
            N, n1, n2, ov = len(C), int(rp.sum()), int(rq.sum()), int((rp & rq).sum())
            p_ov = hypergeom.sf(ov - 1, N, n1, n2) if ov > 0 else 1.0
            rho, prho = spearmanr(P.pre, P.post)
            col = np.where(rp & rq, "#8e44ad", np.where(rp, PHC["pre"], np.where(rq, PHC["post"], "0.8")))
            a.scatter(P.pre, P.post, s=8, c=col)
            lim = np.nanmax(np.abs(P.values)) * 1.05 + 1e-6
            a.plot([-lim, lim], [-lim, lim], color="k", lw=0.5, ls=":")
            a.axhline(0, color="0.7", lw=0.4); a.axvline(0, color="0.7", lw=0.4)
            a.set_title(f"{e}\nrho {rho:.2f} ({d1.pstr(prho)})\nresp. both {ov} (pre {n1}, post {n2}) p={p_ov:.2g}",
                        fontsize=6.5, color=EVC[e])
            a.tick_params(labelsize=6)
            if k == 0:
                a.set_ylabel(f"{m} ({arm})\ninjection response", fontsize=7, color=ARMC[arm])
            if i == len(mice) - 1:
                a.set_xlabel("baseline response", fontsize=7)
            rows.append(dict(mouse=m, arm=arm, event=e, n_cells=N, resp_pre=n1, resp_post=n2, resp_both=ov,
                             p_overlap_hypergeom=p_ov, spearman_rho=rho, p_rho=prho))
    fig.suptitle("E4 do the same cells respond after the injection? each dot = a cell's response [0,2]-[-2,0] s "
                 "(purple = responsive in both, grey/red = only baseline/injection); overlap vs hypergeometric chance",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(out, "E4_same_cells.png"), dpi=115); plt.close(fig)
    pd.DataFrame(rows).round(5).to_csv(os.path.join(out, "E4_same_cells.csv"), index=False)


# ------------------------------------------------------------------------------ E5 adaptation
def e5_adaptation(PS, out):
    rows = []
    mice = [m for m in MICE if any((m[0], ph, st) in PS for ph in ("pre", "post") for st in STIM)]
    fig, ax = plt.subplots(len(mice), len(STIM), figsize=(3.2 * len(STIM), 2.6 * len(mice)), squeeze=False)
    for i, (m, arm, sex) in enumerate(mice):
        for k, st in enumerate(STIM):
            a = ax[i, k]
            for ph in ("pre", "post"):
                d = PS.get((m, ph, st))
                if d is None:
                    continue
                r = np.nanmean(np.abs(d["r_trials"]), 1)              # population |response| per trial
                x = np.arange(1, len(r) + 1)
                a.plot(x, r, "o-", ms=2.5, lw=0.8, color=PHC[ph], label=PH[ph])
                if len(r) >= 5:
                    sl = np.polyfit(x, r, 1)[0]
                    rho, p = spearmanr(x, r)
                    rows.append(dict(mouse=m, arm=arm, phase=ph, stimulus=st, n_trials=len(r), slope_per_trial=sl,
                                     spearman_rho=rho, p=p))
                    a.text(0.98, 0.92 if ph == "pre" else 0.80, f"{PH[ph][:4]} rho {rho:.2f} p {p:.2g}",
                           transform=a.transAxes, ha="right", fontsize=6, color=PHC[ph])
            a.set_title(f"{m} ({arm}) {st}" if True else st, fontsize=7.5, color=EVC[st])
            a.tick_params(labelsize=6)
            if k == 0:
                a.set_ylabel("mean |response| over cells", fontsize=7)
            if i == len(mice) - 1:
                a.set_xlabel("delivery number in the block", fontsize=7)
    fig.suptitle("E5 adaptation / sensitisation within the block: population response size per delivery "
                 "(Spearman rho over deliveries)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(out, "E5_adaptation.png"), dpi=110); plt.close(fig)
    pd.DataFrame(rows).round(5).to_csv(os.path.join(out, "E5_adaptation.csv"), index=False)


# ------------------------------------------------------------------------------ E6 reliability
def e6_reliability(PS, R, out):
    rows = []
    for (m, ph, e), d in PS.items():
        if d.get("Xb") is None or d["n"] < 8:
            continue
        tt = d["tt"]
        w = (tt >= 0) & (tt < 3)
        X = d["Xb"][:, w, :]                                           # trials x time x cells
        ids = None
        for j in range(X.shape[2]):
            Y = X[:, :, j]
            Y = Y[np.all(np.isfinite(Y), 1)]
            if len(Y) < 5:
                continue
            C = np.corrcoef(Y)
            rel = np.nanmean(C[np.triu_indices(len(Y), 1)])
            mtr = d1.sm(np.nanmean(Y, 0)[:, None])[:, 0]
            lat = tt[w][np.argmax(np.abs(mtr))]
            rows.append(dict(mouse=m, phase=ph, event=e, col=j, reliability=rel, latency_s=lat))
    D = pd.DataFrame(rows)
    if not len(D):
        return
    # attach cell ids and class from R (same column order as cell_ids of the session)
    Rr = R.copy()
    Rr["col"] = Rr.groupby(["mouse", "phase", "event"]).cumcount()
    D = D.merge(Rr[["mouse", "phase", "event", "col", "cell", "cls"]], on=["mouse", "phase", "event", "col"], how="left")
    D.round(5).to_csv(os.path.join(out, "E6_reliability_latency.csv"), index=False)
    mice = [m for m in MICE if m[0] in set(D.mouse)]
    fig, ax = plt.subplots(2, len(mice), figsize=(4.4 * len(mice), 7.5), squeeze=False)
    for k, (m, arm, sex) in enumerate(mice):
        for i, (c, lab) in enumerate((("reliability", "trial-to-trial correlation, 0-3 s"),
                                      ("latency_s", "latency of the peak (s), responsive cells"))):
            a = ax[i, k]
            x = np.arange(len(STIM))
            for dx, ph in ((-0.2, "pre"), (0.2, "post")):
                V = D[(D.mouse == m) & (D.phase == ph) & D.event.isin(STIM)]
                if c == "latency_s":
                    V = V[V.cls != "no response"]
                mu = [V[V.event == st][c].mean() for st in STIM]
                se = [V[V.event == st][c].std() / np.sqrt(max((V.event == st).sum(), 1)) for st in STIM]
                nn = [(V.event == st).sum() for st in STIM]
                a.bar(x + dx, mu, 0.38, yerr=se, color=PHC[ph], capsize=2, label=PH[ph])
                for xi, n_ in zip(x, nn):
                    a.text(xi + dx, 0, str(n_), ha="center", va="bottom", fontsize=6, color="w")
            a.set_xticks(x, STIM, fontsize=8)
            a.set_ylabel(lab, fontsize=8)
            a.set_title(f"{m} ({arm})", color=ARMC[arm], fontsize=10)
            a.legend(fontsize=7, frameon=False)
    fig.suptitle("E6 response reliability (mean pairwise correlation of single-trial z, 0-3 s, every cell) and peak latency "
                 "(responsive cells; n in the bars)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(out, "E6_reliability_latency.png"), dpi=110); plt.close(fig)


# ------------------------------------------------------------------------------ E7 correlations
def centroids(out_root, m, ids):
    f = os.path.join(out_root, f"{m}_d{d1.DAY}", "3_extract", f"{m}_d{d1.DAY}_extract.mat")
    if not os.path.isfile(f):
        return None
    M = sio.loadmat(f, squeeze_me=True, struct_as_record=False, variable_names=["cell_qc"])
    cxy = np.asarray(M["cell_qc"].centroid_xy, float)
    return cxy[np.asarray(ids) - 1]


def e7_correlations(sess, out_root, out):
    by = {}
    for s in sess:
        T = s["T"]
        t = T["t_s"]
        sel = (t >= d1.SKIP_S) & (t < 180) & T["valid"].astype(bool)
        Z = T["z"][sel]
        nb = len(Z) // 15
        Zb = np.nanmean(Z[:nb * 15].reshape(nb, 15, -1), 1)
        ok = np.all(np.isfinite(Zb), 1)
        by[(s["mouse"], s["phase"])] = (np.corrcoef(Zb[ok].T), T["cell_ids"].astype(int))
    rows = []
    mice = [m for m in MICE if (m[0], "pre") in by and (m[0], "post") in by]
    fig, ax = plt.subplots(2, len(mice), figsize=(4.2 * len(mice), 7.4), squeeze=False)
    for k, (m, arm, sex) in enumerate(mice):
        C1, ids = by[(m, "pre")]
        C2, _ = by[(m, "post")]
        iu = np.triu_indices(len(ids), 1)
        r1, r2 = C1[iu], C2[iu]
        a = ax[0, k]
        a.hist(r1, bins=40, alpha=0.6, color=PHC["pre"], label=PH["pre"], density=True)
        a.hist(r2, bins=40, alpha=0.6, color=PHC["post"], label=PH["post"], density=True)
        p = d1.wsr(r2, r1)
        a.set_title(f"{m} ({arm}): {len(r1)} pairs\nmedian r {np.nanmedian(r1):.3f} -> {np.nanmedian(r2):.3f}, "
                    f"Wilcoxon {d1.pstr(p)}", fontsize=8, color=ARMC[arm])
        a.set_xlabel("pairwise correlation (0.5-s bins, no-stimulus start)", fontsize=8)
        a.legend(fontsize=7, frameon=False)
        cxy = centroids(out_root, m, ids)
        b = ax[1, k]
        if cxy is not None:
            dd = np.hypot(*(cxy[iu[0]] - cxy[iu[1]]).T)
            bins = np.percentile(dd, np.linspace(0, 100, 9))
            for rr, ph in ((r1, "pre"), (r2, "post")):
                mu = [np.nanmean(rr[(dd >= bins[i]) & (dd < bins[i + 1])]) for i in range(8)]
                b.plot((bins[:-1] + bins[1:]) / 2, mu, "o-", color=PHC[ph], label=PH[ph])
            rho, pr = spearmanr(dd, r1)
            b.set_title(f"correlation vs distance (baseline rho {rho:.2f}, {d1.pstr(pr)})", fontsize=8)
            b.set_xlabel("distance between the cells (px)", fontsize=8)
            b.set_ylabel("mean pairwise r", fontsize=8)
            b.legend(fontsize=7, frameon=False)
        rows.append(dict(mouse=m, arm=arm, n_pairs=len(r1), median_r_pre=np.nanmedian(r1), median_r_post=np.nanmedian(r2), p=p))
    fig.suptitle("E7 pairwise correlations of the cells before any stimulus (first 3 min), the same pairs in both sessions; "
                 "bottom: neighbours share neuropil / light, so r falls with distance", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(out, "E7_pairwise_correlations.png"), dpi=110); plt.close(fig)
    pd.DataFrame(rows).round(5).to_csv(os.path.join(out, "E7_pairwise_correlations.csv"), index=False)


# ------------------------------------------------------------------------------ E8 trajectories
def e8_trajectories(PS, out):
    mice = [m for m in MICE if any((m[0], "pre", st) in PS for st in STIM)]
    if not mice:
        return
    fig, ax = plt.subplots(1, len(mice), figsize=(4.6 * len(mice), 4.6), squeeze=False)
    evrows = []
    for k, (m, arm, sex) in enumerate(mice):
        a = ax[0, k]
        sts = [st for st in STIM if (m, "pre", st) in PS and (m, "post", st) in PS]
        if not sts:
            a.axis("off"); continue
        tt = PS[(m, "pre", sts[0])]["tt"]
        w = (tt >= -1) & (tt <= 4)
        A = np.vstack([np.nan_to_num(PS[(m, "pre", st)]["mean"][w]) for st in sts])   # (stim*time) x cells
        mu = A.mean(0)
        U, Sv, Vt = np.linalg.svd(A - mu, full_matrices=False)
        ev = Sv[:2] ** 2 / (Sv ** 2).sum()
        evrows.append(dict(mouse=m, arm=arm, pc1_pct=round(100 * ev[0], 1), pc2_pct=round(100 * ev[1], 1)))
        for st in sts:
            for ph, ls in (("pre", "-"), ("post", "--")):
                Y = (np.nan_to_num(PS[(m, ph, st)]["mean"][w]) - mu) @ Vt[:2].T
                a.plot(Y[:, 0], Y[:, 1], ls, color=EVC[st], lw=1.4 if ph == "pre" else 1.0,
                       label=f"{st} ({PH[ph]})")
                a.plot(Y[np.argmin(np.abs(tt[w])), 0], Y[np.argmin(np.abs(tt[w])), 1], "o", color=EVC[st], ms=3)
        a.set_xlabel(f"PC1 ({100 * ev[0]:.0f}% of the baseline responses)", fontsize=8)
        a.set_ylabel(f"PC2 ({100 * ev[1]:.0f}%)", fontsize=8)
        a.set_title(f"{m} ({arm}): -1 to +4 s, dot = delivery", fontsize=9, color=ARMC[arm])
        if k == 0:
            a.legend(fontsize=6, frameon=False)
    fig.suptitle("E8 population trajectories: every stimulus's mean response in the space of the baseline session's two "
                 "main patterns (solid baseline, dashed injection; same axes)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(out, "E8_population_trajectories.png"), dpi=115); plt.close(fig)
    pd.DataFrame(evrows).to_csv(os.path.join(out, "E8_explained_variance.csv"), index=False)


# ------------------------------------------------------------------------------ E9 spatial clustering
def e9_spatial(R, out_root, out, n_perm=2000):
    rows = []
    rng = np.random.default_rng(3)
    for m, arm, sex in MICE:
        V = R[R.mouse == m]
        if not len(V):
            continue
        ids = np.sort(V.cell.unique())
        cxy = centroids(out_root, m, ids)
        if cxy is None:
            continue
        pos = dict(zip(ids, cxy))
        for (ph, e), W in V.groupby(["phase", "event"]):
            sel = W[W.cls != "no response"].cell.values
            if len(sel) < 3:
                continue
            P = np.array([pos[c] for c in sel])
            nn = lambda Q: np.mean(np.sort(np.hypot(*(Q[:, None] - Q[None]).transpose(2, 0, 1)), 1)[:, 1])
            obs = nn(P)
            null = [nn(cxy[rng.choice(len(ids), len(sel), replace=False)]) for _ in range(n_perm)]
            rows.append(dict(mouse=m, arm=arm, phase=ph, event=e, n_responsive=len(sel), mean_nn_px=obs,
                             null_mean_px=np.mean(null), p_clustered=(1 + np.sum(np.array(null) <= obs)) / (1 + n_perm)))
    D = pd.DataFrame(rows)
    D.round(5).to_csv(os.path.join(out, "E9_spatial_clustering.csv"), index=False)
    if not len(D):
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for (m, arm, sex) in MICE:
        V = D[D.mouse == m]
        if not len(V):
            continue
        ax.scatter(V.null_mean_px, V.mean_nn_px, color=ARMC[arm], marker="o" if sex == "F" else "s", s=20 + 3 * V.n_responsive,
                   label=f"{m} ({arm})")
    lim = [0, max(D.null_mean_px.max(), D.mean_nn_px.max()) * 1.1]
    ax.plot(lim, lim, "k:", lw=0.8)
    ax.set_xlabel("mean nearest-neighbour distance, random cells (px)")
    ax.set_ylabel("... responsive cells (px)")
    ax.set_title(f"E9 spatial clustering of responsive cells (below the line = closer than chance; {n_perm} random sets); "
                 f"{(D.p_clustered < 0.05).sum()} of {len(D)} sets p < 0.05", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "E9_spatial_clustering.png"), dpi=120); plt.close(fig)


# ------------------------------------------------------------------------------ E10 intensity
RANK = {"Light touch": 1, "Mild touch": 2, "Pin prick": 3, "Heat": 4}


def e10_intensity(PS, out):
    """WT deck slide 10, in Ca: does the response grow from light touch to heat, and does the
    injection flatten that ordering? Per cell: Spearman rho of the single-trial response with the
    stimulus rank (light 1, mild 2, pin 3, heat 4), BH over cells; population: mean response per
    stimulus, baseline vs injection"""
    from scipy.stats import spearmanr as sp
    rows = []
    mice = [m for m in MICE if any((m[0], ph, st) in PS for ph in ("pre", "post") for st in STIM)]
    fig, ax = plt.subplots(2, len(mice), figsize=(4.3 * len(mice), 8), squeeze=False)
    for k, (m, arm, sex) in enumerate(mice):
        rho_ph = {}
        for ph in ("pre", "post"):
            parts = [(RANK[st], PS[(m, ph, st)]["r_trials"]) for st in STIM if (m, ph, st) in PS]
            if len(parts) < 3:
                continue
            x = np.concatenate([np.full(len(r), rk) for rk, r in parts])
            Y = np.vstack([r for _, r in parts])
            rr, pp = [], []
            for j in range(Y.shape[1]):
                ok = np.isfinite(Y[:, j])
                q = sp(x[ok], Y[ok, j]) if ok.sum() > 10 else (np.nan, np.nan)
                rr.append(q[0]); pp.append(q[1])
            rr, pp = np.array(rr), np.array(pp)
            qq = d1.bh(pp)
            rho_ph[ph] = rr
            mu = [np.nanmean(r) for _, r in parts]
            se = [np.nanstd(np.nanmean(r, 0)) / np.sqrt(r.shape[1]) for _, r in parts]
            xs = [rk for rk, _ in parts]
            ax[0, k].errorbar(xs, mu, yerr=se, color=PHC[ph], marker="o", capsize=3, lw=2, label=PH[ph])
            for j in range(Y.shape[1]):
                ax[0, k].plot(xs, [np.nanmean(r[:, j]) for _, r in parts], color=PHC[ph], lw=0.3, alpha=0.25)
            rows.append(dict(mouse=m, arm=arm, phase=ph, n_cells=Y.shape[1], n_graded_up=int(np.sum((qq <= 0.2) & (rr > 0))),
                             n_graded_down=int(np.sum((qq <= 0.2) & (rr < 0))), median_rho=np.nanmedian(rr),
                             mean_resp=dict(zip([st for st in STIM if (m, ph, st) in PS], np.round(mu, 4)))))
        a = ax[0, k]
        a.set_xticks([1, 2, 3, 4], ["light", "mild", "pin", "heat"])
        a.axhline(0, color="k", lw=0.5)
        a.set_ylabel("response z[0,2]-z[-2,0]\n(thin = each cell, bold = mean +- SEM)", fontsize=8)
        a.set_title(f"{m} ({arm})", color=ARMC[arm], fontsize=10)
        a.legend(fontsize=7, frameon=False)
        b = ax[1, k]
        if "pre" in rho_ph and "post" in rho_ph:
            r1, r2 = rho_ph["pre"], rho_ph["post"]
            ok = np.isfinite(r1) & np.isfinite(r2)
            for u, v in zip(r1[ok], r2[ok]):
                b.plot([0, 1], [u, v], color="0.8", lw=0.5)
            b.plot([0, 1], [np.median(r1[ok]), np.median(r2[ok])], color=ARMC[arm], lw=2.5, marker="o")
            b.set_title(f"intensity coding per cell, n = {ok.sum()}, Wilcoxon {d1.pstr(d1.wsr(r2[ok], r1[ok]))}", fontsize=8)
        b.axhline(0, color="k", lw=0.5)
        b.set_xticks([0, 1], ["baseline", "injection"]); b.set_xlim(-0.3, 1.3)
        b.set_ylabel("Spearman rho, response vs stimulus rank", fontsize=8)
    fig.suptitle("E10 intensity coding: response from light touch to heat (WT deck slide 10, in Ca); "
                 "bottom: each cell's ordering, baseline vs injection", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(out, "E10_intensity_coding.png"), dpi=110); plt.close(fig)
    pd.DataFrame(rows).to_csv(os.path.join(out, "E10_intensity_coding.csv"), index=False)


# ------------------------------------------------------------------------------ E11 predict withdrawal
def e11_predict_withdrawal(sess, PS, out, n_perm=200):
    """does the population state BEFORE a delivery, or its response, predict whether the mouse
    withdraws? logistic regression (L2, 5-fold stratified CV, ROC AUC); the stimulus type alone is
    the reference model, so only what the cells add beyond the stimulus counts"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rows = []
    for s in sess:
        m, ph = s["mouse"], s["phase"]
        allD = np.sort(np.concatenate([s["B"]["ev"][st] for st in STIM]))
        Wd = s["B"]["ev"]["withdrawal"]
        F_pre, F_rsp, S1, y = [], [], [], []
        for k, st in enumerate(STIM):
            d = PS.get((m, ph, st))
            if d is None:
                continue
            for i, f in enumerate(d["frames"]):
                nxt = allD[allD > f]
                hi = min(f + int(2 * FPS), (nxt[0] - 1) if len(nxt) else f + int(2 * FPS))
                y.append(int(np.any((Wd >= f) & (Wd <= hi))))
                F_pre.append(d["pre_trials"][i]); F_rsp.append(d["r_trials"][i])
                S1.append(np.eye(4)[k])
        y = np.array(y)
        if len(y) < 20 or y.sum() < 6 or (1 - y).sum() < 6:
            continue
        Xs = np.array(S1); Xp = np.nan_to_num(np.array(F_pre)); Xr = np.nan_to_num(np.array(F_rsp))
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000))
        auc = lambda X: roc_auc_score(y, cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1])
        a_s, a_p, a_r = auc(Xs), auc(np.c_[Xs, Xp]), auc(np.c_[Xs, Xr])
        rng = np.random.default_rng(2)
        null = []
        for _ in range(n_perm):                       # cells shuffled across trials WITHIN each stimulus
            Xq = Xp.copy()
            for k in range(4):
                idx = np.where(Xs[:, k] == 1)[0]
                Xq[idx] = Xq[rng.permutation(idx)]
            null.append(auc(np.c_[Xs, Xq]))
        rows.append(dict(mouse=m, arm=dict((x[0], x[1]) for x in MICE)[m], phase=ph, n_deliveries=len(y),
                         n_withdrawal=int(y.sum()), auc_stimulus_only=a_s, auc_stim_plus_prestate=a_p,
                         auc_stim_plus_response=a_r, null_p95_prestate=float(np.percentile(null, 95)),
                         p_prestate=(1 + np.sum(np.array(null) >= a_p)) / (1 + n_perm)))
    D = pd.DataFrame(rows)
    D.round(4).to_csv(os.path.join(out, "E11_predict_withdrawal.csv"), index=False)
    if not len(D):
        return
    fig, ax = plt.subplots(figsize=(2.6 * len(D) + 2, 4.6))
    x = np.arange(len(D))
    ax.bar(x - 0.27, D.auc_stimulus_only, 0.26, color="0.7", label="stimulus type only")
    ax.bar(x, D.auc_stim_plus_prestate, 0.26, color="#2e86c1", label="+ cells BEFORE the delivery [-2,0] s")
    ax.bar(x + 0.27, D.auc_stim_plus_response, 0.26, color="#c0392b", label="+ cells' response [0,2] s")
    for i, r in D.iterrows():
        ax.plot([i - 0.13, i + 0.13], [r.null_p95_prestate] * 2, "k:", lw=1)
        ax.text(i, r.auc_stim_plus_prestate + 0.02, f"p={r.p_prestate:.3f}", ha="center", fontsize=6)
    ax.axhline(0.5, color="k", lw=0.6)
    ax.set_xticks(x, [f"{r.mouse} {PH[r.phase]}\n{r.n_withdrawal}/{r.n_deliveries} wd" for r in D.itertuples()], fontsize=7)
    ax.set_ylabel("cross-validated ROC AUC (withdrawal yes / no)")
    ax.set_ylim(0.3, 1.0)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.set_title("E11 can the cells predict the withdrawal? dotted = 95th pct of 200 shuffles of the pre-delivery state "
                 "within each stimulus", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "E11_predict_withdrawal.png"), dpi=120); plt.close(fig)


# ------------------------------------------------------------------------------ E12 response types
def e12_types(PS, out, kmax=6):
    """functional response types: k-means on every cell's baseline-session response profile
    (trial-mean z, -1..+4 s at 5 Hz, four stimuli + withdrawal, z-scored per cell), k by
    silhouette; the injection profile of the same cell is assigned to the nearest type, so the
    table says how many cells changed type"""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    evs = STIM + ["withdrawal"]
    feats, meta, post = [], [], []
    for m, arm, sex in MICE:
        if not all((m, "pre", e) in PS for e in evs):
            continue
        tt = PS[(m, "pre", evs[0])]["tt"]
        w = (tt >= -1) & (tt <= 4)
        idx = np.where(w)[0][::6]
        A = np.hstack([np.nan_to_num(PS[(m, "pre", e)]["mean"][idx]).T for e in evs])
        Bp = np.hstack([np.nan_to_num(PS[(m, "post", e)]["mean"][idx]).T for e in evs]) if all(
            (m, "post", e) in PS for e in evs) else None
        for j in range(A.shape[0]):
            feats.append(A[j]); meta.append((m, arm, j)); post.append(Bp[j] if Bp is not None else None)
    if len(feats) < 20:
        return
    F = np.array(feats)
    nz = lambda X: (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-9)
    Fz = nz(F)
    best, km = -1, None
    for k in range(2, kmax + 1):
        kk = KMeans(k, n_init=20, random_state=0).fit(Fz)
        sc = silhouette_score(Fz, kk.labels_)
        if sc > best:
            best, km = sc, kk
    lab = km.labels_
    K = km.n_clusters
    has_post = np.array([p is not None for p in post])
    lab_post = np.full(len(lab), -1)
    if has_post.any():
        P = nz(np.array([p for p in post if p is not None]))
        lab_post[has_post] = km.predict(P)
    nt = len(np.where((PS[(meta[0][0], "pre", evs[0])]["tt"] >= -1) & (PS[(meta[0][0], "pre", evs[0])]["tt"] <= 4))[0][::6])
    tx = np.linspace(-1, 4, nt)
    fig, ax = plt.subplots(K, len(evs) + 1, figsize=(2.4 * (len(evs) + 1), 1.9 * K), squeeze=False)
    rows = []
    for c in range(K):
        sel = lab == c
        for e_i, e in enumerate(evs):
            a = ax[c, e_i]
            seg = slice(e_i * nt, (e_i + 1) * nt)
            a.plot(tx, F[sel][:, seg].mean(0), color=PHC["pre"], lw=1.5, label="baseline")
            sp_ = sel & has_post
            if sp_.any():
                a.plot(tx, np.array([post[i] for i in np.where(sp_)[0]])[:, seg].mean(0), color=PHC["post"], lw=1.2,
                       ls="--", label="injection (same cells)")
            a.axvline(0, color="k", lw=0.5); a.axhline(0, color="0.7", lw=0.4)
            a.tick_params(labelsize=6)
            if c == 0:
                a.set_title(e, fontsize=8, color=EVC[e])
            if e_i == 0:
                a.set_ylabel(f"type {c + 1}\nn = {sel.sum()}", fontsize=8)
        b = ax[c, -1]
        cnt = [np.sum(sel & np.array([mm[0] == m for mm in meta])) for m, _, _ in MICE]
        b.bar(range(len(MICE)), cnt, color=[ARMC[a_] for _, a_, _ in MICE])
        b.set_xticks(range(len(MICE)), [m for m, _, _ in MICE], fontsize=6)
        b.tick_params(labelsize=6)
        if c == 0:
            b.set_title("cells per mouse", fontsize=8)
        stay = np.sum(sel & (lab_post == c)); moved = np.sum(sel & has_post & (lab_post != c))
        rows.append(dict(type=c + 1, n_cells=int(sel.sum()), stay_same_type_after_injection=int(stay), changed_type=int(moved),
                         **{f"n_{m}": int(x) for (m, _, _), x in zip(MICE, cnt)}))
    ax[0, 0].legend(fontsize=6, frameon=False)
    fig.suptitle(f"E12 response types (k-means on the baseline profiles of all {len(F)} cells, k = {K} by silhouette "
                 f"{best:.2f}); dashed = the same cells after the injection", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(out, "E12_response_types.png"), dpi=110); plt.close(fig)
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(out, "E12_response_types.csv"), index=False)
    pd.DataFrame([dict(mouse=mm[0], arm=mm[1], col=mm[2], type_pre=int(l) + 1, type_post=int(lp) + 1 if lp >= 0 else np.nan)
                  for mm, l, lp in zip(meta, lab, lab_post)]).to_csv(os.path.join(out, "E12_cell_types.csv"), index=False)


# ------------------------------------------------------------------------------ E13 pain axis
def e13_pain_axis(PS, out, n_perm=200):
    """a 'nociceptive' direction in population space: logistic regression separating noxious (pin,
    heat) from innocuous (light, mild) deliveries on the baseline session's single-trial responses.
    (1) baseline accuracy, cross-validated; (2) the SAME decoder applied to the injection session:
    if the drug changes how noxious input is coded, accuracy and the separation along the axis drop;
    (3) the reverse (train injection, test baseline) and the injection's own CV accuracy. Control:
    the sham response z[-2,0] - z[-4,-2] (blocks are in time order, so raw pre-event activity
    separates them by drift alone). d' on the baseline axis is out of sample in both sessions."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    NOX = {"Pin prick": 1, "Heat": 1, "Light touch": 0, "Mild touch": 0}
    rows = []
    proj = {}
    for m, arm, sex in MICE:
        data = {}
        for ph in ("pre", "post"):
            X, P0, y = [], [], []
            for st, lab in NOX.items():
                d = PS.get((m, ph, st))
                if d is None:
                    continue
                X.append(d["r_trials"]); P0.append(d["sham_trials"]); y += [lab] * d["n"]
            if len(X) < 2:
                continue
            y = np.array(y)
            if y.sum() < 8 or (1 - y).sum() < 8:
                continue
            data[ph] = (np.nan_to_num(np.vstack(X)), np.nan_to_num(np.vstack(P0)), y)
        if set(data) != {"pre", "post"}:
            continue
        clf = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000))
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        r = dict(mouse=m, arm=arm)
        for ph in ("pre", "post"):
            X, P0, y = data[ph]
            r[f"cv_acc_{ph}"] = cross_val_score(clf(), X, y, cv=cv, scoring="balanced_accuracy").mean()
            r[f"cv_acc_{ph}_prewindow"] = cross_val_score(clf(), P0, y, cv=cv, scoring="balanced_accuracy").mean()
        from sklearn.metrics import balanced_accuracy_score
        f1 = clf().fit(data["pre"][0], data["pre"][2])
        f2 = clf().fit(data["post"][0], data["post"][2])
        r["pre_to_post"] = balanced_accuracy_score(data["post"][2], f1.predict(data["post"][0]))
        r["post_to_pre"] = balanced_accuracy_score(data["pre"][2], f2.predict(data["pre"][0]))
        # separation along the baseline axis (d'), both sessions OUT OF SAMPLE: baseline by
        # cross-validation, injection with the decoder fitted on the whole baseline
        from sklearn.model_selection import cross_val_predict
        for ph in ("pre", "post"):
            X, _, y = data[ph]
            sc = (cross_val_predict(clf(), X, y, cv=cv, method="decision_function") if ph == "pre"
                  else f1.decision_function(X))
            d_ = (sc[y == 1].mean() - sc[y == 0].mean()) / np.sqrt(0.5 * (sc[y == 1].var() + sc[y == 0].var()) + 1e-12)
            r[f"dprime_{ph}_on_baseline_axis"] = d_
            proj[(m, ph)] = (sc, y)
        rng = np.random.default_rng(4)
        X, _, y = data["pre"]
        null = [cross_val_score(clf(), X, rng.permutation(y), cv=cv, scoring="balanced_accuracy").mean() for _ in range(n_perm)]
        r["null_p95"] = float(np.percentile(null, 95))
        rows.append(r)
    D = pd.DataFrame(rows)
    D.round(4).to_csv(os.path.join(out, "E13_pain_axis.csv"), index=False)
    if not len(D):
        return
    fig, ax = plt.subplots(1, len(D) + 1, figsize=(3.6 * (len(D) + 1), 4.2), squeeze=False)
    for k, r in D.reset_index(drop=True).iterrows():
        a = ax[0, k]
        for ph, off in (("pre", 0), ("post", 1)):
            sc, y = proj[(r.mouse, ph)]
            for lab, col in ((0, "#5dade2"), (1, "#c0392b")):
                v = sc[y == lab]
                a.scatter(np.full(len(v), off) + (lab - 0.5) * 0.25 + np.random.default_rng(0).uniform(-0.06, 0.06, len(v)), v,
                          s=6, color=col, alpha=0.6, label=("noxious (pin, heat)" if lab else "innocuous (light, mild)")
                          if off == 0 else None)
        a.axhline(0, color="k", lw=0.5)
        a.set_xticks([0, 1], [f"baseline\nd' {r.dprime_pre_on_baseline_axis:.2f}", f"injection\nd' {r.dprime_post_on_baseline_axis:.2f}"],
                     fontsize=8)
        a.set_ylabel("position on the baseline 'noxious' axis", fontsize=8)
        a.set_title(f"{r.mouse} ({r.arm})", color=ARMC[r.arm], fontsize=10)
        if k == 0:
            a.legend(fontsize=6, frameon=False)
    b = ax[0, -1]
    x = np.arange(len(D))
    b.bar(x - 0.3, D.cv_acc_pre, 0.2, color=PHC["pre"], label="baseline (CV)")
    b.bar(x - 0.1, D.pre_to_post, 0.2, color="#e67e22", label="baseline decoder on injection")
    b.bar(x + 0.1, D.cv_acc_post, 0.2, color=PHC["post"], label="injection (CV)")
    b.bar(x + 0.3, D.cv_acc_pre_prewindow, 0.2, color="0.8", label="baseline, sham response (control)")
    b.scatter(x - 0.3, D.null_p95, marker="_", color="k", s=80, label="95% of shuffles")
    b.axhline(0.5, color="k", lw=0.5)
    b.set_xticks(x, D.mouse, fontsize=8)
    b.set_ylim(0.3, 1.0)
    b.set_ylabel("balanced accuracy, noxious vs innocuous", fontsize=8)
    b.legend(fontsize=6, frameon=False)
    fig.suptitle("E13 a population 'noxious' axis from the baseline session, and the injection session on the same axis "
                 "(does the drug change how noxious input is coded?)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(out, "E13_pain_axis.png"), dpi=115); plt.close(fig)


# ------------------------------------------------------------------------------ E14 who changes
def e14_who_changes(R, out):
    """are the cells that respond most at baseline the ones the drug changes? x = baseline
    response, y = injection - baseline, per cell. Regression to the mean makes the slope negative
    even with no drug (a strong baseline response is partly noise), so the vehicle mice are the
    reference: only an SBI-553 slope steeper than vehicle's says the drug acts on responsive cells."""
    from scipy.stats import linregress
    evs = ["Heat", "Pin prick", "Mild touch", "withdrawal"]
    mice = [m for m in MICE if m[0] in set(R.mouse)]
    fig, ax = plt.subplots(1, len(evs), figsize=(3.8 * len(evs), 3.9), squeeze=False)
    rows = []
    for k, e in enumerate(evs):
        a = ax[0, k]
        for m, arm, sex in mice:
            P = R[(R.mouse == m) & (R.event == e)].pivot(index="cell", columns="phase", values="resp_mean").dropna()
            if not {"pre", "post"} <= set(P.columns) or len(P) < 5:
                continue
            x, y = P.pre.values, (P.post - P.pre).values
            lr = linregress(x, y)
            a.scatter(x, y, s=6, color=ARMC[arm], alpha=0.5, marker="o" if sex == "F" else "s")
            xx = np.linspace(x.min(), x.max(), 10)
            a.plot(xx, lr.intercept + lr.slope * xx, color=ARMC[arm], lw=1.3, label=f"{m}: slope {lr.slope:.2f}")
            rows.append(dict(mouse=m, arm=arm, event=e, n_cells=len(P), slope=lr.slope, intercept=lr.intercept, r=lr.rvalue,
                             p=lr.pvalue))
        a.axhline(0, color="k", lw=0.5); a.axvline(0, color="k", lw=0.5)
        a.set_xlabel("baseline response (z)", fontsize=8)
        a.set_ylabel("injection - baseline", fontsize=8)
        a.set_title(e, fontsize=9, color=EVC[e])
        a.legend(fontsize=6, frameon=False)
    fig.suptitle("E14 do the most responsive cells change most? (regression to the mean gives a negative slope by itself: "
                 "compare SBI-553 (red) with vehicle (blue))", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(out, "E14_who_changes.png"), dpi=115); plt.close(fig)
    pd.DataFrame(rows).round(5).to_csv(os.path.join(out, "E14_who_changes.csv"), index=False)


# ------------------------------------------------------------------------------ E15 block position
def e15_block_position(W, scoring_dir, out):
    """the stimulus blocks run in a different order in every session (5941: Heat was the last block
    of the baseline and the second of the injection session), so a per-stimulus pre/post change is
    partly a change of time in session. Mixed model on every cell x session x stimulus block, still
    frames only: log2(rate + 0.1) ~ session * arm + block position (1-4) + stimulus, random mouse +
    cell. The session x arm term is the drug effect with position and stimulus held fixed."""
    import statsmodels.formula.api as smf
    rows = []
    for m, arm, sex in MICE:
        for ph in ("pre", "post"):
            S, _ = d1.scoring_for(scoring_dir, m, ph)
            if S is None:
                continue
            order = [str(x) for x in np.atleast_1d(S["stimNames"])]
            V = W[(W.mouse.astype(str) == m) & (W.phase == ph)]
            for pos, st in enumerate(order, 1):
                c = f"still rate | {st}"
                if c not in V:
                    continue
                for _, r in V.iterrows():
                    if np.isfinite(r[c]):
                        rows.append(dict(mouse=m, arm=arm, phase=ph, cell=f"{m}_{int(r.cell)}", stimulus=st, position=pos,
                                         y=np.log2(r[c] + 0.1)))
            c = "still rate | no stimulus"
            if c in V:
                for _, r in V.iterrows():
                    if np.isfinite(r[c]):
                        rows.append(dict(mouse=m, arm=arm, phase=ph, cell=f"{m}_{int(r.cell)}", stimulus="none", position=0,
                                         y=np.log2(r[c] + 0.1)))
    D = pd.DataFrame(rows)
    if not len(D) or D.mouse.nunique() < 2:
        return
    D["post"] = (D.phase == "post").astype(float); D["sbi"] = (D.arm == "SBI-553").astype(float)
    out_rows = []
    for lab, sub, form in (("stimulus blocks, position + stimulus adjusted", D[D.position > 0],
                            "y ~ post * sbi + C(position) + C(stimulus)"),
                           ("stimulus blocks, unadjusted", D[D.position > 0], "y ~ post * sbi"),
                           ("no-stimulus start", D[D.position == 0], "y ~ post * sbi")):
        if sub.sbi.nunique() < 2:
            form = form.replace(" * sbi", "")
        try:
            f, conv, meth = fit_mixed(form, sub, "cell")
            for term in [t for t in ("post", "post:sbi") if t in f.params]:
                ci = f.conf_int().loc[term]
                out_rows.append(dict(model=lab, term=term, coef=f.params[term], ci_lo=ci[0], ci_hi=ci[1], p=f.pvalues[term],
                                     n_obs=len(sub), n_mice=sub.mouse.nunique(), converged=conv, optimizer=meth))
            if "C(position)" in form:
                for t in [t for t in f.params.index if t.startswith("C(position)")]:
                    out_rows.append(dict(model=lab, term=t, coef=f.params[t], ci_lo=np.nan, ci_hi=np.nan, p=f.pvalues[t],
                                         n_obs=len(sub), n_mice=sub.mouse.nunique()))
        except Exception as ex:                                          # noqa: BLE001
            out_rows.append(dict(model=lab, term="fit failed: " + str(ex)[:80], coef=np.nan, ci_lo=np.nan, ci_hi=np.nan,
                                 p=np.nan, n_obs=len(sub), n_mice=sub.mouse.nunique()))
    pd.DataFrame(out_rows).round(5).to_csv(os.path.join(out, "E15_block_position_model.csv"), index=False)


# ------------------------------------------------------------------------------ E2b mouse level
def e2b_mouse_level(W, out):
    """the honest unit is the MOUSE: per mouse, the median over cells of log2((post + 0.1) /
    (pre + 0.1)); arms compared by the exact permutation of the 4 mice (6 ways to pick 2 'SBI-553'
    mice): with 2 vs 2 the smallest possible two-sided p is 0.33, so no mouse-level test can be
    significant yet - the cell-level mixed model (E2) shows the size, this shows the uncertainty"""
    from itertools import combinations
    conds = [("transient rate, whole session", "rate"), ("no-stimulus start", "rate | no stimulus"),
             ("still frames", "rate | still")] + [(f"{st} block, still", f"still rate | {st}") for st in STIM]
    rows = []
    for lab, c in conds:
        if c not in W:
            continue
        per = {}
        for m, arm, sex in MICE:
            P = W[W.mouse.astype(str) == m].pivot(index="cell", columns="phase", values=c).dropna()
            if {"pre", "post"} <= set(P.columns) and len(P) >= 3:
                per[m] = (arm, float(np.median(np.log2((P.post + 0.1) / (P.pre + 0.1)))))
        if len(per) < 4:
            continue
        ms = list(per)
        y = np.array([per[m][1] for m in ms])
        sbi = np.array([per[m][0] == "SBI-553" for m in ms])
        obs = y[sbi].mean() - y[~sbi].mean()
        null = []
        for comb in combinations(range(len(ms)), int(sbi.sum())):
            g = np.zeros(len(ms), bool); g[list(comb)] = True
            null.append(y[g].mean() - y[~g].mean())
        p = np.mean(np.abs(null) >= abs(obs) - 1e-12)
        rows.append(dict(outcome=lab, **{f"{m} ({per[m][0]})": round(per[m][1], 3) for m in ms},
                         sbi_minus_vehicle=round(obs, 3), p_permutation=round(p, 3), n_labelings=len(null)))
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(out, "E2b_mouse_level.csv"), index=False)
    return D


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--scoring-dir", required=True)
    ap.add_argument("--day1", required=True, help="output folder of ca_day1_analysis.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--traces", default="traces", help="traces (default) or traces_bgcorr")
    ap.add_argument("--z-norm", default="mad", choices=["mad", "noise"], help="see ca_day1_analysis.Z_NORM")
    a = ap.parse_args(argv)
    d1.TRACES = a.traces
    d1.Z_NORM = a.z_norm
    os.makedirs(a.out, exist_ok=True)
    sess, PS = [], {}
    for m, arm, sex in MICE:
        for ph in ("pre", "post"):
            s = d1.load_session(a.out_root, m, ph, a.scoring_dir)
            if s is None or s["B"] is None:
                continue
            sess.append(s)
            _, ps = d1.event_tests(s, null=False)
            for e, d in ps.items():
                PS[(m, ph, e)] = d
    if not sess:
        print("nothing yet"); return 1
    W = pd.read_csv(os.path.join(a.day1, f"ca_d{d1.DAY}_cells_whole_session.csv"))
    R = pd.read_csv(os.path.join(a.day1, f"ca_d{d1.DAY}_cells_event_tests.csv"))
    W["mouse"] = W.mouse.astype(str); R["mouse"] = R.mouse.astype(str)
    steps = [("E1", lambda: e1_quality(sess, a.out)), ("E2", lambda: e2_mixed(W, R, a.out)),
             ("E2b", lambda: e2b_mouse_level(W, a.out)),
             ("E3", lambda: e3_encoding(sess, a.out)), ("E4", lambda: e4_identity(R, a.out)),
             ("E5", lambda: e5_adaptation(PS, a.out)), ("E6", lambda: e6_reliability(PS, R, a.out)),
             ("E7", lambda: e7_correlations(sess, a.out_root, a.out)), ("E8", lambda: e8_trajectories(PS, a.out)),
             ("E9", lambda: e9_spatial(R, a.out_root, a.out)), ("E10", lambda: e10_intensity(PS, a.out)),
             ("E11", lambda: e11_predict_withdrawal(sess, PS, a.out)), ("E12", lambda: e12_types(PS, a.out)),
             ("E13", lambda: e13_pain_axis(PS, a.out)), ("E14", lambda: e14_who_changes(R, a.out)),
             ("E15", lambda: e15_block_position(W, a.scoring_dir, a.out))]
    for nm, f in steps:
        try:
            f()
            print(nm, "done")
        except Exception as ex:                                       # noqa: BLE001
            import traceback
            print(nm, "FAILED:", ex); traceback.print_exc()
    print("ca_day1_extended done ->", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
