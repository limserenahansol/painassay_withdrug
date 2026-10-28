"""ca_cell_event_matrix.py - which neuron responds to which stimulus or behaviour, and how the responsive
neurons change after the injection (activated and inhibited separately). Day 1.

Responses are z-scored dF/F (not Ca transients), as in the pain-imaging papers (e.g. Corder 2019: 2 s after
vs before the event, rank test): per trial r = mean z [0, 2] s - mean z [-2, 0] s around the event, z = dF/F
divided by the neuron's noise SD (one scale for both sessions, so baseline and injection sizes compare).

Events: the 4 stimuli, withdrawal, flinch, and the ONSETS of attending, lick/bite, escape/rearing, general
movement and grooming (a bout that starts after >= 2 s without it). Events with < 5 trials are not tested.

1. Neuron x event matrix (per mouse, both sessions): the time-shift test of ca_day1_analysis.event_tests
   (each event's response vs the same events shifted inside their block; BH q <= 0.20 over the neurons).
   -> cell_event_tests_all.csv (long), cell_event_matrix_<mouse>.csv (wide), cell_event_matrix_<mouse>.png
2. Responsive neurons, baseline vs injection, ACTIVATED and INHIBITED separately (split-half, no
   regression-to-the-mean bias): the baseline trials are split in odd / even; a neuron is selected on one half
   (Wilcoxon signed-rank of r over those trials p < 0.05; activated if its mean r > 0, inhibited if < 0) and its
   baseline response is measured on the OTHER half; the injection response is the mean over all injection
   trials. A neuron selected in both splits gets the mean of its two held-out values.
   -> responsive_change_split_half.csv (per neuron), responsive_change_summary.csv, responsive_change_bars.png
3. Behaviour per delivery (every event counted in the response window / deliveries) -> behaviour_per_delivery.csv

  python ca_cell_event_matrix.py --base C:/Users/hsollim/Documents/HEAL_mini1p_SBI553
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import wilcoxon

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pairstars as PS  # noqa: E402  (user rule 10-22: every star shows the pair it compares)
import ca_day1_analysis as A                                        # noqa: E402
MICE = A.MICE                     # by day (CA_DAY)

ARMC = {"SBI-553": "#c0392b", "vehicle": "#2471a3"}
STIM = ["Light touch", "Mild touch", "Pin prick", "Heat"]
EVM = STIM + ["withdrawal", "flinch", "attending onset", "lick/bite onset", "escape/rearing onset", "movement onset",
              "grooming onset"]
SHORT = {"Light touch": "light", "Mild touch": "mild", "Pin prick": "pin", "Heat": "heat", "withdrawal": "withdr.",
         "flinch": "flinch", "attending onset": "attend", "lick/bite onset": "lick/bite", "escape/rearing onset": "escape",
         "movement onset": "move", "grooming onset": "groom"}


def wsr(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3 or np.allclose(x, 0):
        return np.nan
    try:
        return float(wilcoxon(x).pvalue)
    except ValueError:
        return np.nan


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    a = ap.parse_args(argv)
    B = a.base
    out = os.path.join(B, "analysis", "ca_cell_event_matrix" + ("" if A.DAY == 1 else f"_day{A.DAY}"))
    os.makedirs(out, exist_ok=True)
    A.Z_NORM = "noise"
    A.EVENTS = list(dict.fromkeys(list(A.EVENTS) + ["attending onset", "lick/bite onset"]))
    rows, split = [], []
    PS = {}
    for m, arm, sex in MICE:
        for ph in ("pre", "post"):
            s = A.load_session(os.path.join(B, "mini1p_out", {1: "20261019", 2: "20261021"}[A.DAY]), m, ph, os.path.join(B, "scoring_mini1p", {1: "20261019_final", 2: "20261021_final"}[A.DAY]))
            S = sio.loadmat(s["scoring"], squeeze_me=True)
            sc = np.atleast_1d(S["score"]).astype(int)
            s["B"]["ev"]["attending onset"] = A.onsets(sc == 1) + 1
            s["B"]["ev"]["lick/bite onset"] = A.onsets(sc == 2) + 1
            R, ps = A.event_tests(s)
            R["arm"] = arm
            rows.append(R)
            PS[(m, ph)] = (ps, s["T"]["cell_ids"].astype(int))
            print(f"{m} {ph}: " + ", ".join(f"{SHORT.get(e, e)} {d['n']}" for e, d in ps.items()), flush=True)
    T = pd.concat(rows)
    T.to_csv(os.path.join(out, "cell_event_tests_all.csv"), index=False)
    # 1. matrix per mouse
    for m, arm, sex in MICE:
        X = T[T["mouse"].astype(str) == m]
        ids = sorted(X.cell.unique())
        evs = [e for e in EVM if e in set(X.event)]
        W = pd.DataFrame(index=ids)
        for ph in ("pre", "post"):
            for e in evs:
                Y = X[(X.phase == ph) & (X.event == e)].set_index("cell")
                W[f"{SHORT[e]} {'base' if ph == 'pre' else 'inj'}"] = [Y.cls_shift.get(c, "n.t.") if c in Y.index else "n.t." for c in ids]
        W.index.name = "neuron"
        W.to_csv(os.path.join(out, f"cell_event_matrix_{m}.csv"))
        code = W.replace({"UP": 1, "DOWN": -1, "no response": 0, "n.t.": np.nan}).astype(float).values
        n = len(ids)
        fig_h = max(4.0, 0.105 * n + 1.6)
        fig, ax = plt.subplots(figsize=(12.4, fig_h))
        cmap = matplotlib.colors.ListedColormap(["#2471a3", "#f4f4f4", "#c0392b"])
        ax.imshow(np.ma.masked_invalid(code), aspect="auto", cmap=cmap, vmin=-1.5, vmax=1.5, interpolation="nearest")
        nan = np.isnan(code)
        if nan.any():
            yy, xx = np.where(nan)
            ax.scatter(xx, yy, marker="x", s=6, color="0.6", lw=0.5)
        ax.set_yticks(range(n), [str(c) for c in ids], fontsize=6)
        ax.set_xticks(range(W.shape[1]), list(W.columns), rotation=60, ha="right", fontsize=8)
        ax.axvline(len(evs) - 0.5, color="k", lw=1.5)
        for k in range(1, W.shape[1]):
            ax.axvline(k - 0.5, color="white", lw=0.5)
        ax.set_ylabel("neuron number (as on the field-of-view map)")
        ax.set_title(f"{m} ({arm}, {sex}): red = activated (UP), blue = inhibited (DOWN), white = no response, x = too few events",
                     fontsize=10, loc="left")
        fig.tight_layout(); fig.savefig(os.path.join(out, f"cell_event_matrix_{m}.png"), dpi=150); plt.close(fig)
    # 2. split-half baseline vs injection, activated / inhibited separately
    for m, arm, sex in MICE:
        (ps0, ids0), (ps1, ids1) = PS[(m, "pre")], PS[(m, "post")]
        if not np.array_equal(ids0, ids1):
            raise SystemExit(f"{m}: different neurons in the two sessions")
        for e in EVM:
            if e not in ps0 or e not in ps1:
                continue
            r0, r1 = ps0[e]["r_trials"], ps1[e]["r_trials"]
            post = np.nanmean(r1, 0)
            nt = r0.shape[0]
            halves = [np.arange(0, nt, 2), np.arange(1, nt, 2)]
            held = {"activated": {}, "inhibited": {}}
            for hs, hm in ((0, 1), (1, 0)):
                sel, mea = halves[hs], halves[hm]
                if len(sel) < 3 or len(mea) < 2:
                    continue
                for j, c in enumerate(ids0):
                    p = wsr(r0[sel, j])
                    if not (np.isfinite(p) and p < 0.05):
                        continue
                    k = "activated" if np.nanmean(r0[sel, j]) > 0 else "inhibited"
                    held[k].setdefault(int(c), []).append(float(np.nanmean(r0[mea, j])))
            for k, dct in held.items():
                for c, v in dct.items():
                    j = int(np.where(ids0 == c)[0][0])
                    split.append(dict(mouse=m, arm=arm, event=e, group=k, cell=c, n_trials_base=nt, n_trials_inj=r1.shape[0],
                                      base_heldout=float(np.mean(v)), inj=float(post[j]), n_splits=len(v)))
    Sp = pd.DataFrame(split)
    Sp.to_csv(os.path.join(out, "responsive_change_split_half.csv"), index=False)
    summ = []
    for (m, arm, e, k), g in Sp.groupby(["mouse", "arm", "event", "group"]):
        summ.append(dict(mouse=m, arm=arm, event=e, group=k, n_cells=len(g), base_mean=g.base_heldout.mean(),
                         inj_mean=g.inj.mean(), base_sem=g.base_heldout.sem(), inj_sem=g.inj.sem(),
                         p_paired=wsr(g.inj - g.base_heldout) if len(g) >= 5 else np.nan))
    Su = pd.DataFrame(summ)
    Su.to_csv(os.path.join(out, "responsive_change_summary.csv"), index=False)
    # figure: rows = activated / inhibited, columns = events; per mouse two bars (base, inj)
    evs = [e for e in EVM if e in set(Sp.event)]
    fig, ax = plt.subplots(2, len(evs), figsize=(12.4, 4.55), sharey="row")
    fig.subplots_adjust(left=0.06, right=0.995, top=0.86, bottom=0.16, wspace=0.12, hspace=0.45)
    for r_i, k in enumerate(("activated", "inhibited")):
        for e_i, e in enumerate(evs):
            a_ = ax[r_i, e_i]
            for i, (m, arm, _) in enumerate(MICE):
                g = Su[(Su.mouse == m) & (Su.event == e) & (Su.group == k)]
                if not len(g):
                    continue
                g = g.iloc[0]
                x = i * 2.6
                a_.bar(x, g.base_mean, yerr=g.base_sem if np.isfinite(g.base_sem) else 0, color="white", edgecolor=ARMC[arm], width=1.0, capsize=1.5, lw=1)
                a_.bar(x + 1.05, g.inj_mean, yerr=g.inj_sem if np.isfinite(g.inj_sem) else 0, color=ARMC[arm], width=1.0, capsize=1.5)
                a_.text(x + 0.5, 0, f"{int(g.n_cells)}", ha="center", va="top" if k == "activated" else "bottom", fontsize=6, color="0.3")
                if np.isfinite(g.p_paired) and g.p_paired < 0.05:
                    top = max(g.base_mean, g.inj_mean) if k == "activated" else min(g.base_mean, g.inj_mean)
                    PS.bracket(a_, x, x + 1.05, top, g.p_paired, "k", fs=9, inset=0, below=k != "activated")   # baseline vs injection bar
            a_.axhline(0, color="0.4", lw=0.6)
            a_.set_xticks([]); a_.tick_params(labelsize=7)
            if r_i == 0:
                a_.set_title(SHORT[e], fontsize=9)
            if e_i == 0:
                a_.set_ylabel(f"{k} neurons\nresponse (z)", fontsize=8.5)
            for sp in ("top", "right", "bottom"):
                a_.spines[sp].set_visible(False)
    fig.legend(handles=[Patch(fc="white", ec="k", label="baseline (held-out half)"), Patch(color="k", label="after injection"),
                        Patch(color=ARMC["SBI-553"], label="SBI-553: 5941, 5939"), Patch(color=ARMC["vehicle"], label="vehicle: 5937, 1191")],
               loc="upper center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.0))
    fig.savefig(os.path.join(out, "responsive_change_bars.png"), dpi=160); plt.close(fig)
    # 3. behaviour per delivery
    import make_day1_full_deck as FD
    D, G = FD.behaviour_counts(B)
    Tt = G.groupby(["mouse", "phase"]).sum(numeric_only=True).reset_index()
    for b in FD.BEH_ORDER:
        Tt[f"{b} per delivery"] = Tt[b] / Tt.n
    Tt.rename(columns={"n": "deliveries"}).to_csv(os.path.join(out, "behaviour_per_delivery.csv"), index=False)
    G2 = G.copy()
    for b in FD.BEH_ORDER:
        G2[f"{b} per delivery"] = G2[b] / G2.n
    G2.rename(columns={"n": "deliveries"}).to_csv(os.path.join(out, "behaviour_per_delivery_by_stimulus.csv"), index=False)
    print("->", out)


if __name__ == "__main__":
    main()
