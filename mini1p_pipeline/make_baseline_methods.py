"""make_baseline_methods.py - 10-27 (user: 'normally other papers are also using zscore - mean of 0.5 s? or only z-score? ... try to plot
event-locked responses with all the different ways'): the same trials, events (mix4 marks), response windows and neurons as the deck,
only the baseline / normalisation changes:

  M1  z - mean 0.5 s before        (the deck; withdrawal / flinch: 1 s before)   z = dF/F / robust SD of the whole session
  M2  z only                        (no pre-event subtraction)
  M3  baseline z (-2..0 s)          per trial (dF/F - its mean in -2..0 s) / that neuron's SD of the -2..0 s frames over all trials
  M4  dF/F - mean -2..0 s           per trial, dF/F of the pipeline (running F0; raw F is neuropil-subtracted and often <= 0, so
                                    (F - F0_pre) / F0_pre is undefined for many neurons) minus its mean in -2..0 s, in %
  M5  z - mean -5..-3 s             Corder 2019 style gap before the event

Per neuron: trial mean, then AUC = trapezoid over the event's response window. Tests as in the deck: within day = Wilcoxon signed-rank
(same neurons, baseline vs after session); between days = Mann-Whitney (after vs after, baseline vs baseline). Exploratory, uncorrected.

  python make_baseline_methods.py  -> analysis\\baseline_methods\\*.png, summary.csv
"""
import os
import sys

os.environ.setdefault("SCORING_SUFFIX", "_mix4")
sys.argv = sys.argv[:1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import mannwhitneyu, wilcoxon  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import ca_day1_analysis as CA  # noqa: E402
import ca_event_analysis as EA  # noqa: E402

B = r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553"
OUT = os.path.join(B, "analysis", "baseline_methods")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "Arial", "font.size": 12, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 170})
SBI = {1: {"5941", "5939"}, 2: {"5937", "1191"}}
EVS = ["Light touch", "Mild touch", "Pin prick", "Heat", "withdrawal", "flinch"]
CC = {"vb": "#c4c4c4", "vi": "#2471a3", "sb": "#7a7a7a", "si": "#c0392b"}
CL = {"vb": "baseline, vehicle day", "vi": "after vehicle", "sb": "baseline, SBI-553 day", "si": "after SBI-553"}
cc = pd.read_csv(os.path.join(B, r"analysis\_variants\mix4\event_analysis\cells_classes.csv"))
cc["mouse"] = cc.mouse.astype(str)
FPS = EA.FPS
lag = np.arange(int(-5 * FPS), int(8 * FPS) + 1)
tt = lag / FPS


def win(ev):
    e = cc[cc.event == ev].iloc[0]
    return float(e.pre), float(e.a), float(e.b)


def m(t0, t1):
    return (tt >= t0 - 1e-9) & (tt < t1 - 1e-9)


METHODS = [("M1", "z − mean 0.5 s before (deck)", "z·s"), ("M2", "z only (no subtraction)", "z·s"),
           ("M3", "baseline z (−2..0 s)", "baseline-z·s"), ("M4", "ΔF/F − mean −2..0 s", "%·s"),
           ("M5", "z − mean −5..−3 s (Corder gap)", "z·s")]


def methods(ev, Z, X):
    """Z, X: trials x time x cells (session z, raw trace F). Returns {method: trial-mean trace, time x cells}"""
    pre, a, b = win(ev)
    a0 = min(a, 0)
    out = {}
    out["M1"] = np.nanmean(Z - np.nanmean(Z[:, m(a0 - pre, a0)], 1, keepdims=True), 0)
    out["M2"] = np.nanmean(Z, 0)
    w = m(-2, 0)
    base = Z[:, w]
    mu = np.nanmean(base, 1, keepdims=True)
    sd = np.nanstd(base - mu, axis=(0, 1), ddof=1)                 # neuron's SD of the pre-event frames over all trials
    sd = np.where(sd > 0, sd, np.nan)
    out["M3"] = np.nanmean((Z - mu) / sd[None, None, :], 0)
    out["M4"] = 100 * np.nanmean(X - np.nanmean(X[:, w], 1, keepdims=True), 0)    # X = dF/F here
    out["M5"] = np.nanmean(Z - np.nanmean(Z[:, m(-5, -3)], 1, keepdims=True), 0)
    return out


# --------------------------------------------------------------------------------------------- per neuron x session x event x method
CACHE = os.path.join(OUT, "cache.npz")
if os.path.exists(CACHE):
    R = np.load(CACHE, allow_pickle=True)["R"].item()
else:
    R = {}
    F0neg = []
    for mo in EA.MO:
        for d in (1, 2):
            for ph in ("pre", "post"):
                s = CA.load_session(EA.ROOT[d], mo, ph, EA.SCORE[d], day=d)
                if s is None or s["B"] is None:
                    continue
                Zs = np.asarray(s["T"]["z"], float); Xs = np.asarray(s["T"]["dff"], float)
                fr = s["T"]["frame"].astype(int); s["fr"] = fr
                for ev in EVS:
                    idx, f = EA.ev_rows(s, s["B"]["ev"].get(ev, []))
                    keep = (f / FPS >= CA.SKIP_S) & (idx + lag[0] >= 0) & (idx + lag[-1] < len(fr))
                    idx = idx[keep]
                    if len(idx) < CA.MIN_TRIALS:
                        continue
                    sl = idx[:, None] + lag[None, :]
                    R[(mo, d, ph, ev)] = dict(tr=methods(ev, Zs[sl], Xs[sl]), ids=s["T"]["cell_ids"].astype(int), n=len(idx))
                print("done", mo, d, ph, flush=True)
    np.savez_compressed(CACHE, R=np.array(R, dtype=object))


def cond(mo, d, ph):
    return ("s" if mo in SBI[d] else "v") + ("b" if ph == "pre" else "i")


def auc_table(ev, M):
    pre, a, b = win(ev)
    k = (tt >= a - 1e-9) & (tt <= b + 1e-9)
    rows = []
    for (mo, d, ph, e), v in R.items():
        if e != ev:
            continue
        Y = v["tr"][M]
        A = np.trapezoid(np.nan_to_num(Y[k], nan=0.0), tt[k], axis=0)
        A[np.all(np.isnan(Y[k]), 0)] = np.nan
        for cid, x in zip(v["ids"], A):
            rows.append(dict(mouse=mo, day=d, phase=ph, cond=cond(mo, d, ph), cell=int(cid), auc=x))
    return pd.DataFrame(rows)


def tests(df):
    T = {}
    for dr, b_, i_ in (("veh", "vb", "vi"), ("sbi", "sb", "si")):
        x = df[df.cond == b_].set_index(["mouse", "day", "cell"]).auc
        y = df[df.cond == i_].set_index(["mouse", "day", "cell"]).auc
        j = pd.concat([x, y], axis=1, join="inner").dropna()
        T[dr] = wilcoxon(j.iloc[:, 1], j.iloc[:, 0])[1] if len(j) > 5 else np.nan
    g = lambda c: df[df.cond == c].auc.dropna()                       # noqa: E731
    T["after"] = mannwhitneyu(g("si"), g("vi"))[1]
    T["base"] = mannwhitneyu(g("sb"), g("vb"))[1]
    return T


def st(p):
    return "ns" if not np.isfinite(p) or p >= 0.05 else "***" if p < 0.001 else "**" if p < 0.01 else "*"


def pf(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}" if p < 0.1 else f"p = {p:.2f}"


def bracket(a, x1, x2, y, txt, color="k", h=None):
    h = h or 0.02 * (a.get_ylim()[1] - a.get_ylim()[0])
    a.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color=color, lw=1.1, clip_on=False)
    a.text((x1 + x2) / 2, y + h, txt, ha="center", va="bottom", color=color, fontsize=10.5, fontweight="bold" if txt != "ns" else "normal",
           clip_on=False)


SUM = []
for M, mname, unit in METHODS:
    # 10-27 (user: 'always try to match Y axis in same label'): one y range per row (all events share the bar scale and the trace scale)
    fig, ax = plt.subplots(2, 6, figsize=(17, 6.2), sharey="row",
                           gridspec_kw=dict(wspace=0.12, hspace=0.55, left=0.055, right=0.995, top=0.82, bottom=0.09))
    P = {}
    for ev in EVS:
        df = auc_table(ev, M)
        mu = {c: df[df.cond == c].auc.mean() for c in CC}
        se = {c: df[df.cond == c].auc.std(ddof=1) / np.sqrt(df[df.cond == c].auc.notna().sum()) for c in CC}
        P[ev] = dict(T=tests(df), mu=mu, se=se)
    hi = max(P[ev]["mu"][c] + P[ev]["se"][c] for ev in EVS for c in CC)
    ZOOM = M == "M2"        # 10-27 (user: 'just zoom, too much empty value'): no baseline subtraction -> all values ~0.7-1.1, so do not
    lo = min(P[ev]["mu"][c] - P[ev]["se"][c] for ev in EVS for c in CC)    # force 0 in; mean +- SEM as points (bars must start at 0)
    if not ZOOM:
        lo = min(lo, 0)
    span = hi - lo if hi > lo else 1
    ax[0, 0].set_ylim(lo - (0.15 if ZOOM else 0.05) * span, hi + 0.62 * span)
    for j, ev in enumerate(EVS):
        T, mu, se = P[ev]["T"], P[ev]["mu"], P[ev]["se"]
        pre, a_, b_ = win(ev)
        a = ax[0, j]
        for k_, c in enumerate(CC):
            if ZOOM:
                a.errorbar(k_, mu[c], yerr=se[c], fmt="o", ms=10, color=CC[c], mec="k", mew=0.6, capsize=4, elinewidth=1.3, ecolor="k")
            else:
                a.bar(k_, mu[c], yerr=se[c], color=CC[c], width=0.75, capsize=3, error_kw=dict(lw=1))
        if not ZOOM:
            a.axhline(0, color="0.4", lw=0.8)
        a.set_xlim(-0.6, 3.6)
        hj = max(mu[c] + se[c] for c in CC)
        bracket(a, 0, 1, hj + 0.08 * span, st(T["veh"]), CC["vi"], h=0.02 * span)
        bracket(a, 2, 3, hj + 0.08 * span, st(T["sbi"]), CC["si"], h=0.02 * span)
        bracket(a, 1, 3, hj + 0.30 * span, st(T["after"]) + (f" ({pf(T['after'])})" if T["after"] < 0.05 else ""), h=0.02 * span)
        a.set_xticks(range(4), ["base", "after", "base", "after"], fontsize=10)
        a.text(0.5, -0.20, "vehicle", transform=a.get_xaxis_transform(), ha="center", va="top", color=CC["vi"], fontsize=10.5)
        a.text(2.5, -0.20, "SBI-553", transform=a.get_xaxis_transform(), ha="center", va="top", color=CC["si"], fontsize=10.5)
        a.set_title(f"{ev}  ({a_:g}-{b_:g} s)", fontsize=12.5)
        a.tick_params(labelsize=10)
        b = ax[1, j]
        sel = (tt >= -5) & (tt <= 6)
        for c in CC:
            Ys = [v["tr"][M] for (mo, d, ph, e), v in R.items() if e == ev and cond(mo, d, ph) == c]
            Y = np.hstack(Ys)[sel]
            mu_t = np.nanmean(Y, 1); se_t = np.nanstd(Y, 1, ddof=1) / np.sqrt(np.sum(np.isfinite(Y), 1))
            b.fill_between(tt[sel], mu_t - se_t, mu_t + se_t, color=CC[c], alpha=0.2, lw=0); b.plot(tt[sel], mu_t, color=CC[c], lw=1.1)
        bw = {"M1": (min(a_, 0) - pre, min(a_, 0)), "M2": None, "M3": (-2, 0), "M4": (-2, 0), "M5": (-5, -3)}[M]
        if bw:
            b.axvspan(*bw, color="#d6eaf8", lw=0, zorder=0)
        b.axvspan(a_, b_, color="#e5e7e9", lw=0, zorder=0)
        b.axvline(0, color="k", lw=0.7, ls="--")
        if not ZOOM:
            b.axhline(0, color="0.5", lw=0.6)
        b.set_xlabel("time from event (s)", fontsize=11); b.tick_params(labelsize=10)
        SUM.append(dict(method=M, name=mname, event=ev, **{f"mean_{c}": mu[c] for c in CC},
                        p_vehicle_day=T["veh"], p_sbi_day=T["sbi"], p_after_vs_after=T["after"], p_base_vs_base=T["base"]))
    ax[0, 0].set_ylabel(f"AUC of the response\nwindow ({unit}), mean ± SEM" + ("\n(zoomed: axis not from 0)" if M == "M2" else ""),
                        fontsize=11)
    ax[1, 0].set_ylabel({"M1": "z − mean before", "M2": "z", "M3": "baseline z", "M4": "ΔF/F − mean before (%)", "M5": "z − mean −5..−3 s"}[M], fontsize=11)
    h = [Patch(color=CC[c], label=CL[c]) for c in CC] +         ([Patch(color="#d6eaf8", label="baseline window (subtracted)")] if M != "M2" else []) + [Patch(color="#e5e7e9", label="response window = AUC")]
    fig.legend(handles=h, ncol=6, loc="upper center", bbox_to_anchor=(0.5, 0.995), frameon=False, fontsize=11)
    fig.suptitle(f"{M}  {mname}  ·  all neurons (vehicle day 216, SBI-553 day 246), 4 mice  ·  brackets: blue / red = Wilcoxon within day, "
                 "black = Mann-Whitney after vs after", y=0.925, fontsize=12, fontweight="bold")
    fig.savefig(os.path.join(OUT, f"{M}_population.png")); plt.close(fig)
    print("->", M)

S = pd.DataFrame(SUM)
S.to_csv(os.path.join(OUT, "summary.csv"), index=False)

# summary: which tests are significant under each method
fig, ax = plt.subplots(1, 3, figsize=(16, 4.4), gridspec_kw=dict(wspace=0.45, left=0.14, right=0.99, top=0.82, bottom=0.12))
for a, col, ttl in zip(ax, ("p_vehicle_day", "p_sbi_day", "p_after_vs_after"),
                       ("vehicle day: baseline vs after", "SBI-553 day: baseline vs after", "after vehicle vs after SBI-553")):
    P = S.pivot(index="method", columns="event", values=col)[EVS]
    a.imshow(-np.log10(P.values.astype(float)), cmap="Reds", vmin=0, vmax=4, aspect="auto")
    for i in range(P.shape[0]):
        for k in range(P.shape[1]):
            p = P.values[i, k]
            a.text(k, i, st(p), ha="center", va="center", fontsize=11, color="k" if p > 0.01 else "w")
    a.set_xticks(range(len(EVS)), [e.replace(" ", "\n") for e in EVS], fontsize=10)
    a.set_yticks(range(len(METHODS)), [f"{M} {n}" for M, n, _ in METHODS] if a is ax[0] else [M for M, _, _ in METHODS], fontsize=10)
    a.set_title(ttl, fontsize=12.5)
fig.suptitle("Same neurons and windows, 5 baseline definitions: which tests are significant (colour = −log10 p, star = p)", fontsize=13,
             fontweight="bold")
fig.savefig(os.path.join(OUT, "summary_tests.png")); plt.close(fig)

# pin prick UP / DOWN / rest, each method (classes as in the deck)
GC = {"UP": "#e74c3c", "DOWN": "#2e86c1", "rest": "#95a5a6"}
LAB = {"UP": "activated", "DOWN": "inhibited", "rest": "non-responsive"}
fig, ax = plt.subplots(len(METHODS), 4, figsize=(15, 13.5), sharex=True, sharey="row", gridspec_kw=dict(wspace=0.10, hspace=0.45, left=0.09, right=0.995,
                                                                                           top=0.93, bottom=0.05))
for i, (M, mname, unit) in enumerate(METHODS):
    for k, c in enumerate(CC):
        a = ax[i, k]
        for g in ("rest", "DOWN", "UP"):
            Ys = []
            for (mo, d, ph, e), v in R.items():
                if e != "Pin prick" or cond(mo, d, ph) != c:
                    continue
                x = cc[(cc.event == "Pin prick") & (cc.mouse == mo) & (cc.day == d) & (cc.phase == ph)].set_index("cell").cls
                cl = x.reindex(v["ids"]).values
                if (cl == LAB[g]).any():
                    Ys.append(v["tr"][M][:, cl == LAB[g]])
            if not Ys:
                continue
            Y = np.hstack(Ys); sel = (tt >= -3) & (tt <= 4)
            mu_t = np.nanmean(Y[sel], 1); se_t = np.nanstd(Y[sel], 1, ddof=1) / np.sqrt(Y.shape[1])
            a.fill_between(tt[sel], mu_t - se_t, mu_t + se_t, color=GC[g], alpha=0.2, lw=0); a.plot(tt[sel], mu_t, color=GC[g], lw=1.3)
        a.axvspan(0, 0.5, color="#e5e7e9", lw=0, zorder=0); a.axvline(0, color="k", lw=0.7, ls="--")
        if M != "M2":
            a.axhline(0, color="0.6", lw=0.6)
        if i == 0:
            a.set_title(CL[c], fontsize=12.5, color=CC[c] if c != "vb" else "0.4")
        if k == 0:
            a.set_ylabel(f"{M}\n{mname}", fontsize=10.5)
        if i == len(METHODS) - 1:
            a.set_xlabel("time from pin prick (s)", fontsize=11)
        a.tick_params(labelsize=10)
fig.legend(handles=[Line2D([], [], color=GC[g], lw=2.5, label=g) for g in GC] + [Patch(color="#e5e7e9", label="0-0.5 s response window")],
           ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.995), frameon=False, fontsize=12)
fig.savefig(os.path.join(OUT, "pin_UP_DOWN_rest_by_method.png")); plt.close(fig)
print(S[["method", "event", "p_vehicle_day", "p_sbi_day", "p_after_vs_after", "p_base_vs_base"]].round(4).to_string())
