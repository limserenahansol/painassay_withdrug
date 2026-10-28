"""make_baseline_methods_norm.py - 10-27 (user: 'just show me this figure [change from each day's baseline session] in M1-5 versions again
and with same y axis'): the 'normalized to each day's baseline session' slide (make_auc_basenorm.py) for each of the 5 baseline
definitions of make_baseline_methods.py, all 6 events, one y range per row.

Per neuron and day: dAUC = AUC(after the injection) - AUC(same day's baseline session), same ROI; change trace = after - baseline trace.
Inside each session the trace is first normalised with the method (M1 minus 0.5 s before, M2 nothing, M3 baseline z -2..0 s, M4 dF/F
minus -2..0 s, M5 minus -5..-3 s). Tests: vs 0 = Wilcoxon; vehicle vs SBI-553 day = Mann-Whitney (all neurons); title = paired Wilcoxon
over the 185 neurons tracked on both days. Exploratory, uncorrected.

  python make_baseline_methods_norm.py  (needs analysis\\baseline_methods\\cache.npz from make_baseline_methods.py)
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

B = r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553"
OUT = os.path.join(B, "analysis", "baseline_methods")
plt.rcParams.update({"font.family": "Arial", "font.size": 12, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 170})
R = np.load(os.path.join(OUT, "cache.npz"), allow_pickle=True)["R"].item()
cc = pd.read_csv(os.path.join(B, r"analysis\_variants\mix4\event_analysis\cells_classes.csv"))
TRK = pd.read_csv(os.path.join(B, r"analysis\_variants\mix4\tracked_crossover\tracked_responses.csv")); TRK["mouse"] = TRK.mouse.astype(str)
MICE = ["5941", "5939", "5937", "1191"]
SBI = {1: {"5941", "5939"}, 2: {"5937", "1191"}}
SBIDAY = {"5941": 1, "5939": 1, "5937": 2, "1191": 2}
SEX = {"5941": "F", "5939": "M", "5937": "M", "1191": "F"}
MC = {"5941": "#e67e22", "5939": "#8e44ad", "5937": "#16a085", "1191": "#2c3e50"}
MK = {"5941": "o", "5939": "s", "5937": "^", "1191": "D"}
OFF = {"5941": -0.21, "5939": -0.07, "5937": 0.07, "1191": 0.21}
CV, CS = "#2471a3", "#c0392b"
EVS = ["Light touch", "Mild touch", "Pin prick", "Heat", "withdrawal", "flinch"]
METHODS = [("M1", "z − mean 0.5 s before (deck)", "z·s", "Δz"), ("M2", "z only (no subtraction)", "z·s", "Δz"),
           ("M3", "baseline z (−2..0 s)", "baseline-z·s", "Δ baseline z"), ("M4", "ΔF/F − mean −2..0 s", "%·s", "Δ(ΔF/F) (%)"),
           ("M5", "z − mean −5..−3 s (Corder gap)", "z·s", "Δz")]
tt = np.arange(int(-5 * 30), int(8 * 30) + 1) / 30.0


def win(ev):
    e = cc[cc.event == ev].iloc[0]
    return float(e.pre), float(e.a), float(e.b)


def st(p):
    return "ns" if not np.isfinite(p) or p >= 0.05 else "***" if p < 0.001 else "**" if p < 0.01 else "*"


def pf(p):
    return "n.a." if not np.isfinite(p) else "p < 0.001" if p < 0.001 else f"p = {p:.3f}" if p < 0.1 else f"p = {p:.2f}"


def analyse(ev, M):
    pre, a, b = win(ev)
    k = (tt >= a - 1e-9) & (tt <= b + 1e-9)
    rows, tr = [], {"veh": [], "sbi": []}
    for mo in MICE:
        for d in (1, 2):
            if (mo, d, "pre", ev) not in R or (mo, d, "post", ev) not in R:
                continue
            dr = "sbi" if mo in SBI[d] else "veh"
            v0, v1 = R[(mo, d, "pre", ev)], R[(mo, d, "post", ev)]
            i0 = {int(c): j for j, c in enumerate(v0["ids"])}; i1 = {int(c): j for j, c in enumerate(v1["ids"])}
            ids = [c for c in i0 if c in i1]
            Y0 = v0["tr"][M][:, [i0[c] for c in ids]]; Y1 = v1["tr"][M][:, [i1[c] for c in ids]]
            A0 = np.trapezoid(Y0[k], tt[k], axis=0); A1 = np.trapezoid(Y1[k], tt[k], axis=0)
            for c, x0, x1 in zip(ids, A0, A1):
                rows.append(dict(mouse=mo, day=d, drug=dr, cell=c, d=x1 - x0))
            tr[dr].append(Y1 - Y0)
    df = pd.DataFrame(rows).dropna()
    V, S = df[df.drug == "veh"], df[df.drug == "sbi"]
    lut = {(r.mouse, r.day, r.cell): r.d for r in df.itertuples()}
    t = TRK[TRK.event == ev]
    tk = pd.DataFrame([dict(dv=lut.get((r.mouse, int(r.veh_day), int(r.veh_id))), ds=lut.get((r.mouse, int(r.sbi_day), int(r.sbi_id))))
                       for r in t.itertuples()]).dropna()
    return dict(V=V, S=S, p_veh=wilcoxon(V.d)[1], p_sbi=wilcoxon(S.d)[1], p_between=mannwhitneyu(S.d, V.d)[1],
                p_trk=wilcoxon(tk.ds, tk.dv)[1] if len(tk) > 5 else np.nan, n_trk=len(tk),
                mv=V.groupby("mouse").d.mean().to_dict(), ms=S.groupby("mouse").d.mean().to_dict(),
                trace={k_: np.hstack(x) for k_, x in tr.items()})


ROWS = []
for M, mname, unit, tlab in METHODS:
    A = {ev: analyse(ev, M) for ev in EVS}
    fig, ax = plt.subplots(2, 6, figsize=(17, 6.4), sharey="row",
                           gridspec_kw=dict(wspace=0.12, hspace=0.62, left=0.06, right=0.995, top=0.80, bottom=0.09))
    vals = []
    for ev in EVS:
        r = A[ev]
        for D_, mm in ((r["V"], r["mv"]), (r["S"], r["ms"])):
            mu, se = D_.d.mean(), D_.d.std(ddof=1) / np.sqrt(len(D_))
            vals += [mu + se, mu - se] + list(mm.values())
    hi, lo = max(vals + [0]), min(vals + [0])
    span = hi - lo
    ax[0, 0].set_ylim(lo - 0.08 * span, hi + 0.55 * span)
    for j, ev in enumerate(EVS):
        r = A[ev]
        a = ax[0, j]
        for x, D_, mm, col in ((0, r["V"], r["mv"], CV), (1, r["S"], r["ms"], CS)):
            mu, se = D_.d.mean(), D_.d.std(ddof=1) / np.sqrt(len(D_))
            a.bar(x, mu, yerr=se, color=col, width=0.6, capsize=3, error_kw=dict(lw=1), zorder=2)
            for m_ in MICE:
                if m_ in mm:
                    a.scatter(x + OFF[m_] * 0.9, mm[m_], s=24, marker=MK[m_], color=MC[m_], edgecolor="white", lw=0.6, zorder=4)
        a.axhline(0, color="0.4", lw=0.8)
        for x, p, col in ((0, r["p_veh"], CV), (1, r["p_sbi"], CS)):
            a.text(x, hi + 0.08 * span, "vs 0: " + st(p), ha="center", va="bottom", fontsize=10, color=col,
                   fontweight="bold" if p < 0.05 else "normal")
        y = hi + 0.30 * span
        a.plot([0, 0, 1, 1], [y, y + 0.03 * span, y + 0.03 * span, y], color="k", lw=1.1)
        a.text(0.5, y + 0.03 * span, f"{st(r['p_between'])} ({pf(r['p_between'])})", ha="center", va="bottom", fontsize=10.5,
               fontweight="bold" if r["p_between"] < 0.05 else "normal")
        a.set_xlim(-0.6, 1.6)
        a.set_xticks([0, 1], ["vehicle\nday", "SBI-553\nday"], fontsize=10.5)
        for lab, c in zip(a.get_xticklabels(), (CV, CS)):
            lab.set_color(c)
        a.tick_params(axis="y", labelsize=10)
        pre, a_, b_ = win(ev)
        a.set_title(f"{ev}  ({a_:g}-{b_:g} s)\n{r['n_trk']} tracked, day vs day: {st(r['p_trk'])}"
                    + (f" ({pf(r['p_trk'])})" if r["p_trk"] < 0.05 else ""), fontsize=11)
        b = ax[1, j]
        sel = (tt >= -5) & (tt <= 6)
        for k_, col in (("veh", CV), ("sbi", CS)):
            Y = r["trace"][k_][sel]
            mu, se = np.nanmean(Y, 1), np.nanstd(Y, 1, ddof=1) / np.sqrt(Y.shape[1])
            b.fill_between(tt[sel], mu - se, mu + se, color=col, alpha=0.2, lw=0); b.plot(tt[sel], mu, color=col, lw=1.1)
        bw = {"M1": (min(a_, 0) - pre, min(a_, 0)), "M2": None, "M3": (-2, 0), "M4": (-2, 0), "M5": (-5, -3)}[M]
        if bw:
            b.axvspan(*bw, color="#d6eaf8", lw=0, zorder=0)
        b.axvspan(a_, b_, color="#e5e7e9", lw=0, zorder=0)
        b.axhline(0, color="0.5", lw=0.6); b.axvline(0, color="k", lw=0.7, ls="--")
        b.set_xlabel("time from event (s)", fontsize=11); b.tick_params(labelsize=10)
        b.set_title("change trace: after − baseline session", fontsize=9.5, color="0.35")
        ROWS.append(dict(method=M, event=ev, dAUC_veh=r["V"].d.mean(), dAUC_sbi=r["S"].d.mean(), p_veh=r["p_veh"], p_sbi=r["p_sbi"],
                         p_between=r["p_between"], p_tracked=r["p_trk"]))
    ax[0, 0].set_ylabel(f"ΔAUC ({unit}):\nafter − baseline session", fontsize=11)
    ax[1, 0].set_ylabel(f"{tlab}, mean ± SEM", fontsize=11)
    h = [Patch(color=CV, label="vehicle day (216 neurons)"), Patch(color=CS, label="SBI-553 day (246)")] + \
        ([Patch(color="#d6eaf8", label="baseline window (subtracted first)")] if M != "M2" else []) + \
        [Patch(color="#e5e7e9", label="response window = AUC")] + \
        [Line2D([], [], ls="", marker=MK[m_], color=MC[m_], markeredgecolor="white", markersize=7,
                label=f"{m_} ({'F' if SEX[m_] == 'F' else 'M'}, SBI-553 Day {SBIDAY[m_]})") for m_ in MICE]
    fig.legend(handles=h, ncol=8, loc="upper center", bbox_to_anchor=(0.5, 0.995), frameon=False, fontsize=10, handlelength=1.2,
               columnspacing=1.0)
    fig.suptitle(f"{M}  {mname}  ·  change from each day's baseline session  ·  bracket = Mann-Whitney vehicle vs SBI-553 day; "
                 "vs 0 = Wilcoxon; markers = mouse means", y=0.935, fontsize=12, fontweight="bold")
    fig.savefig(os.path.join(OUT, f"{M}_change_from_baseline_session.png")); plt.close(fig)
    print("->", M)
S_ = pd.DataFrame(ROWS)
S_.to_csv(os.path.join(OUT, "summary_change_from_baseline_session.csv"), index=False)
print(S_.round(4).to_string())
