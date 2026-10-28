"""Longitudinal (Day 1 + Day 2) Ca analysis: baseline (mean of the two days) vs vehicle vs SBI-553.

Crossover design: each mouse has a baseline and an injection session on both days, SBI-553 on one
day and vehicle on the other. Until Day 2 is recorded, Day 2 is the TOY made by ca_toy_day2.py
(known changes in toy_truth.json); every figure then says TOY in red.

1. Cross-day cell matching (Day 1 and Day 2 are separate EXTRACT runs): the two mean images are
   registered (normalised cross-correlation of the high-passed images), each Day-1 footprint is
   shifted into the Day-2 frame and the kept cells are matched one-to-one (cosine overlap >= 0.5,
   Hungarian) - the same code that carries your decisions to the whole day
   (ca_transfer_decisions.py). With the toy, the matching is scored against the truth.
2. Per matched cell, conditions: BASELINE = mean of the two baseline sessions; VEHICLE and SBI-553 =
   the injection session of that day; and the within-day change (injection - that day's baseline),
   which removes day-to-day drift of the field of view.
   Whole session: transient rate. Event-locked: response r = z[0,2] - z[-2,0] s (trial mean).
   Tests per mouse over matched cells: Friedman (baseline mean, vehicle, SBI-553) and Wilcoxon
   signed-rank SBI-553 vs vehicle (absolute and within-day change). Across mice (n = 4, each its own
   control): per-mouse medians, Wilcoxon over mice (cannot go below p = 0.125 with 4 mice: shown).

  python ca_longitudinal.py --day1-root <mini1p_out/20261019> --day1-scoring <final>
      --day2-root <TOY_day2 or real day 2> --day2-scoring <its scoring dir> --out <analysis/ca_longitudinal>
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import friedmanchisquare

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import safe_io  # noqa: E402,F401  (a figure save retries on a transient Windows write error)
import ca_day1_analysis as d1  # noqa: E402
from ca_transfer_decisions import load_extract, register, footprint_info, cosine_shifted  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# the Day-1 groups (arm1), whatever CA_DAY says: the SBI-553 day of each mouse is derived from them
MICE = [(m, d1._ARMS[1][m], d1._SEX[m]) for m in ("5941", "5939", "5937", "1191")]
CC = {"baseline": "#7f8c8d", "vehicle": "#2471a3", "SBI-553": "#c0392b"}


def toy_mark(fig, toy):
    if toy:
        fig.text(0.995, 0.003, "TOY DAY 2 - simulated, not data", color="red", fontsize=13, fontweight="bold",
                 ha="right", va="bottom")


def match_days(root1, root2, m, ids1, ids2, min_cos=0.5):
    S1, m1 = load_extract(os.path.join(root1, f"{m}_d1", "3_extract", f"{m}_d1_extract.mat"))
    S2, m2 = load_extract(os.path.join(root2, f"{m}_d2", "3_extract", f"{m}_d2_extract.mat"))
    dy, dx, peak = register(m1, m2)
    i1, i2 = footprint_info(S1), footprint_info(S2)
    C = np.zeros((len(ids1), len(ids2)))
    for a, c1 in enumerate(ids1):
        r = i1[c1 - 1]
        if r is None:
            continue
        for b, c2 in enumerate(ids2):
            q = i2[c2 - 1]
            if q is None or abs(r["cy"] - dy - q["cy"]) > 25 or abs(r["cx"] - dx - q["cx"]) > 25:
                continue
            C[a, b] = cosine_shifted(S1[:, :, c1 - 1], S2[:, :, c2 - 1], dy, dx, r, q)
    ra, rb = linear_sum_assignment(-C)
    pairs = [(int(ids1[a]), int(ids2[b]), float(C[a, b])) for a, b in zip(ra, rb) if C[a, b] >= min_cos]
    # real Day 2: the pairs of ca_crossday_match.py (shift + rotation / scale anchoring, position + shape + neighbour
    # pattern) replace the footprint-overlap pairs, when they exist for the kept neurons of both days
    pf = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(root1))), "analysis", "crossday", f"{m}_pairs.csv")
    if os.path.isfile(pf) and "TOY" not in root2.upper():
        X = pd.read_csv(pf)
        s1, s2 = set(int(x) for x in ids1), set(int(x) for x in ids2)
        pairs = [(int(r.day1_id), int(r.day2_id), float(r.cos)) for r in X.itertuples() if r.day1_id in s1 and r.day2_id in s2]
        print(f"{m}: {len(pairs)} cross-day pairs from {pf}")
    return pairs, dict(dy=dy, dx=dx, ncc=peak), (S1, S2, m1, m2, i1, i2)


def per_session(root, scoring, m, ph, day):
    s = d1.load_session(root, m, ph, scoring, day=day)
    if s is None:
        return None
    w = d1.whole_session(s).set_index("cell")
    r, ps = d1.event_tests(s)
    return dict(s=s, w=w, r=r.set_index(["event", "cell"]) if len(r) else pd.DataFrame(), ps=ps,
                ids=s["T"]["cell_ids"].astype(int))


def cond_table(sess, pairs, arm1, value):
    """per matched cell: baseline mean, vehicle, SBI-553 and the two within-day changes of `value`(sess, cell)"""
    sbi_day = 1 if arm1 == "SBI-553" else 2
    rows = []
    for c1, c2, cs in pairs:
        cell = {1: c1, 2: c2}
        v = {(d, ph): value(sess[(d, ph)], cell[d]) for d in (1, 2) for ph in ("pre", "post")}
        if any(not np.isfinite(x) for x in v.values()):
            continue
        veh_day = 3 - sbi_day
        rows.append(dict(d1_cell=c1, d2_cell=c2, cos=cs, baseline=(v[(1, "pre")] + v[(2, "pre")]) / 2,
                         **{"vehicle": v[(veh_day, "post")], "SBI-553": v[(sbi_day, "post")],
                            "d_vehicle": v[(veh_day, "post")] - v[(veh_day, "pre")],
                            "d_SBI": v[(sbi_day, "post")] - v[(sbi_day, "pre")]}))
    return pd.DataFrame(rows)


def tests(T):
    if len(T) < 5:
        return dict(n=len(T), p_friedman=np.nan, p_sbi_vs_veh=np.nan, p_dsbi_vs_dveh=np.nan)
    try:
        pf = friedmanchisquare(T.baseline, T.vehicle, T["SBI-553"]).pvalue
    except ValueError:
        pf = np.nan
    return dict(n=len(T), p_friedman=pf, p_sbi_vs_veh=d1.wsr(T["SBI-553"], T.vehicle),
                p_dsbi_vs_dveh=d1.wsr(T.d_SBI, T.d_vehicle),
                med_baseline=T.baseline.median(), med_vehicle=T.vehicle.median(), med_sbi=T["SBI-553"].median(),
                med_d_vehicle=T.d_vehicle.median(), med_d_sbi=T.d_SBI.median())


def three_panel(a, T, ylab, title):
    x = [0, 1, 2]
    for _, r in T.iterrows():
        a.plot(x, [r.baseline, r.vehicle, r["SBI-553"]], color="0.82", lw=0.5)
    for i, c in enumerate(["baseline", "vehicle", "SBI-553"]):
        a.plot([i - 0.25, i + 0.25], [T[c].median()] * 2, color=CC[c], lw=3)
    a.set_xticks(x, ["baseline\n(mean of 2)", "vehicle", "SBI-553"], fontsize=8)
    a.set_ylabel(ylab, fontsize=8)
    a.set_title(title, fontsize=8)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--day1-root", required=True)
    ap.add_argument("--day1-scoring", required=True)
    ap.add_argument("--day2-root", required=True)
    ap.add_argument("--day2-scoring", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    tf = os.path.join(a.day2_root, "toy_truth.json")
    truth = json.load(open(tf)) if os.path.isfile(tf) else None
    toy = truth is not None
    allrows, evrows, matchrows, cohort = [], [], [], []
    per_mouse = {}
    for m, arm1, sex in MICE:
        sess = {}
        for d, root, sc in ((1, a.day1_root, a.day1_scoring), (2, a.day2_root, a.day2_scoring)):
            for ph in ("pre", "post"):
                sess[(d, ph)] = per_session(root, sc, m, ph, d)
        if any(v is None for v in sess.values()):
            print(f"{m}: not all four sessions yet"); continue
        pairs, reg, geo = match_days(a.day1_root, a.day2_root, m, sess[(1, "pre")]["ids"], sess[(2, "pre")]["ids"])
        mr = dict(mouse=m, n_day1=len(sess[(1, "pre")]["ids"]), n_day2=len(sess[(2, "pre")]["ids"]), n_matched=len(pairs), **reg)
        if toy and m in truth["mice"]:
            tp = {p["d2_cell"]: p["d1_cell"] for p in truth["mice"][m]["pairs"]}
            ok = sum(1 for c1, c2, _ in pairs if tp.get(c2) == c1)
            n_true = sum(1 for v in tp.values() if v is not None)
            mr.update(toy_correct=ok, toy_precision=ok / max(len(pairs), 1), toy_recall=ok / max(n_true, 1), toy_true_pairs=n_true)
        matchrows.append(mr)
        print(f"{m}: {len(pairs)} cells matched across days (shift {reg['dy']}, {reg['dx']} px)" +
              (f", toy precision {mr['toy_precision']:.2f} recall {mr['toy_recall']:.2f}" if toy else ""))
        rate = lambda S, c: float(S["w"].loc[c, "rate"]) if c in S["w"].index else np.nan
        T = cond_table(sess, pairs, arm1, rate)
        T.insert(0, "mouse", m)
        allrows.append(T)
        tr = dict(mouse=m, arm_day1=arm1, measure="transient rate", event="", **tests(T))
        cohort.append(tr)
        ev_t = {}
        for e in d1.EVENTS:
            def resp(S, c, e=e):
                try:
                    return float(S["r"].loc[(e, c), "resp_mean"])
                except (KeyError, TypeError):
                    return np.nan
            Te = cond_table(sess, pairs, arm1, resp)
            if len(Te) >= 5:
                Te.insert(0, "mouse", m); Te.insert(1, "event", e)
                evrows.append(Te)
                ev_t[e] = Te
                cohort.append(dict(mouse=m, arm_day1=arm1, measure="event response", event=e, **tests(Te)))
        per_mouse[m] = dict(arm1=arm1, sex=sex, T=T, ev=ev_t, sess=sess, pairs=pairs, reg=reg, geo=geo)
    if not per_mouse:
        print("nothing to analyse yet"); return 1
    pd.DataFrame(matchrows).to_csv(os.path.join(a.out, "long_cell_matching.csv"), index=False)
    pd.concat(allrows).round(5).to_csv(os.path.join(a.out, "long_cells_rate.csv"), index=False)
    if evrows:
        pd.concat(evrows).round(5).to_csv(os.path.join(a.out, "long_cells_event_response.csv"), index=False)
    C = pd.DataFrame(cohort)
    C.round(5).to_csv(os.path.join(a.out, "long_tests.csv"), index=False)

    # L1: whole-session rate, 3 conditions + within-day change, per mouse
    mice = list(per_mouse)
    fig, ax = plt.subplots(2, len(mice), figsize=(4.4 * len(mice), 8), squeeze=False)
    for k, m in enumerate(mice):
        P = per_mouse[m]
        t = tests(P["T"])
        three_panel(ax[0, k], P["T"], "Ca transients per min",
                    f"{m} (SBI-553 on day {1 if P['arm1'] == 'SBI-553' else 2}), n = {t['n']} matched cells\n"
                    f"Friedman {d1.pstr(t['p_friedman'])}; SBI vs vehicle {d1.pstr(t['p_sbi_vs_veh'])}")
        b = ax[1, k]
        b.scatter(P["T"].d_vehicle, P["T"].d_SBI, s=12, color="0.3")
        lim = np.nanmax(np.abs(np.r_[P["T"].d_vehicle, P["T"].d_SBI, 0.5])) * 1.1
        b.plot([-lim, lim], [-lim, lim], color="k", lw=0.6, ls=":")
        b.axhline(0, color="0.6", lw=0.5); b.axvline(0, color="0.6", lw=0.5)
        b.set_xlim(-lim, lim); b.set_ylim(-lim, lim)
        b.set_xlabel("vehicle day: injection - baseline (per min)", fontsize=8)
        b.set_ylabel("SBI-553 day: injection - baseline", fontsize=8)
        b.set_title(f"within-day change, Wilcoxon {d1.pstr(t['p_dsbi_vs_dveh'])}", fontsize=8)
    fig.suptitle("Longitudinal: transient rate per matched cell - baseline (mean of the two days) vs vehicle vs SBI-553\n"
                 "bottom: within-day change (below the diagonal = SBI-553 lowered activity more than vehicle)", fontsize=10)
    toy_mark(fig, toy)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(a.out, "long_L1_rate_three_conditions.png"), dpi=115); plt.close(fig)

    # L2: event-locked responses, 3 conditions, per event x mouse
    evs = [e for e in d1.EVENTS if any(e in per_mouse[m]["ev"] for m in mice)]
    if evs:
        fig, ax = plt.subplots(len(mice), len(evs), figsize=(2.6 * len(evs), 2.9 * len(mice)), squeeze=False)
        for i, m in enumerate(mice):
            for k, e in enumerate(evs):
                aa = ax[i, k]
                Te = per_mouse[m]["ev"].get(e)
                if Te is None:
                    aa.axis("off"); continue
                t = tests(Te)
                three_panel(aa, Te, "r = z[0,2]-z[-2,0]" if k == 0 else "",
                            f"{e}\nn = {t['n']}, F {d1.pstr(t['p_friedman'])}\nSBI vs veh {d1.pstr(t['p_sbi_vs_veh'])}")
                aa.axhline(0, color="k", lw=0.5)
                aa.tick_params(labelsize=6)
                if k == 0:
                    aa.set_ylabel(f"{m}\ndelta z per neuron\n(z 0-2 s - z -2-0 s)", fontsize=8)
        fig.suptitle("Longitudinal event-locked responses per matched cell\n(median bars; Friedman over the 3 conditions, "
                     "Wilcoxon SBI-553 vs vehicle)", fontsize=10)
        toy_mark(fig, toy)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(os.path.join(a.out, "long_L2_event_responses.png"), dpi=110); plt.close(fig)

    # L3: population PSTH, 3 conditions (matched cells), per mouse x event
    if evs:
        fig, ax = plt.subplots(len(mice), len(evs), figsize=(2.6 * len(evs), 2.6 * len(mice)), squeeze=False)
        for i, m in enumerate(mice):
            P = per_mouse[m]
            sbi_day = 1 if P["arm1"] == "SBI-553" else 2
            for k, e in enumerate(evs):
                aa = ax[i, k]
                curves = {}
                for lab, keys in (("baseline", [(1, "pre"), (2, "pre")]), ("vehicle", [(3 - sbi_day, "post")]),
                                  ("SBI-553", [(sbi_day, "post")])):
                    cs = []
                    for d, ph in keys:
                        S = P["sess"][(d, ph)]
                        if e not in S["ps"]:
                            continue
                        idx = {c: j for j, c in enumerate(S["ids"])}
                        cols = [idx[c1 if d == 1 else c2] for c1, c2, _ in P["pairs"] if (c1 if d == 1 else c2) in idx]
                        cs.append(S["ps"][e]["mean"][:, cols])
                        tt = S["ps"][e]["tt"]
                    if cs:
                        curves[lab] = (np.nanmean(np.stack([np.nanmean(c, 1) for c in cs]), 0), tt)
                for lab, (mu, tt) in curves.items():
                    aa.plot(tt, mu, color=CC[lab], lw=1.2, label=lab)
                aa.axvline(0, color="k", lw=0.7); aa.axhline(0, color="0.6", lw=0.4)
                aa.axvspan(*d1.RESP_W, color="0.93", lw=0, zorder=0)
                aa.tick_params(labelsize=6)
                if i == 0:
                    aa.set_title(e, fontsize=8, color=d1.EVC[e])
                if k == 0:
                    aa.set_ylabel(f"{m}\nmean z (matched cells)", fontsize=7)
                if i == 0 and k == 0:
                    aa.legend(fontsize=6, frameon=False)
        fig.suptitle("Longitudinal population response (mean over matched cells):\nbaseline (mean of the two days), "
                     "vehicle, SBI-553", fontsize=10)
        toy_mark(fig, toy)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(os.path.join(a.out, "long_L3_population_psth.png"), dpi=110); plt.close(fig)

    # L4: cohort - per mouse median of SBI-553 minus vehicle (within-day changes), rate and every event
    meas = [("transient rate", "")] + [("event response", e) for e in evs]
    fig, ax = plt.subplots(figsize=(1.3 * len(meas) + 3, 4.6))
    for j, (mm, e) in enumerate(meas):
        V = C[(C.measure == mm) & (C.event == e)]
        y = (V.med_d_sbi - V.med_d_vehicle).values
        ax.scatter(np.full(len(y), j) + np.linspace(-0.15, 0.15, max(len(y), 1))[:len(y)], y,
                   c=[d1.ARMC[x] for x in V.arm_day1], s=30)
        if len(y) >= 2:
            ax.plot([j - 0.25, j + 0.25], [np.median(y)] * 2, color="k", lw=2)
            p = d1.wsr(y) if len(y) >= 3 else np.nan
            ax.text(j, np.nanmax(y) + 0.05 * (np.nanmax(np.abs(y)) + 1e-3), d1.pstr(p).replace("p = ", "p="),
                    ha="center", fontsize=6)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(len(meas)), ["transient rate"] + evs, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("per mouse: median within-day change\nSBI-553 day minus vehicle day")
    ax.set_title("Cohort: each mouse is its own control (dot colour = treatment on Day 1: red SBI-553, blue vehicle)\n"
                 "Wilcoxon over mice", fontsize=9)
    toy_mark(fig, toy)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "long_L4_cohort.png"), dpi=120); plt.close(fig)

    # L5: cross-day matching QC
    fig, ax = plt.subplots(1, len(mice), figsize=(5.5 * len(mice), 5), squeeze=False)
    for k, m in enumerate(mice):
        P = per_mouse[m]
        S1, S2, m1, m2, i1, i2 = P["geo"]
        dy, dx = P["reg"]["dy"], P["reg"]["dx"]
        lo, hi = np.percentile(m2, [1, 99.7])
        aa = ax[0, k]
        aa.imshow(m2, cmap="gray", vmin=lo, vmax=hi)
        matched1 = {c1 for c1, _, _ in P["pairs"]}
        matched2 = {c2 for _, c2, _ in P["pairs"]}
        for c in P["sess"][(1, "pre")]["ids"]:
            r = i1[c - 1]
            if r is not None:
                aa.plot(r["cx"] - dx, r["cy"] - dy, "o", mfc="none", mec="#2ecc71" if c in matched1 else "#e67e22", ms=7, mew=1)
        for c in P["sess"][(2, "pre")]["ids"]:
            q = i2[c - 1]
            if q is not None:
                aa.plot(q["cx"], q["cy"], "+", color="#3498db" if c in matched2 else "#e74c3c", ms=6, mew=1)
        mr = [r for r in matchrows if r["mouse"] == m][0]
        tt = f", toy precision {mr['toy_precision']:.2f}, recall {mr['toy_recall']:.2f}" if toy else ""
        aa.set_title(f"{m}: {mr['n_matched']} matched of {mr['n_day1']} (day 1) / {mr['n_day2']} (day 2)\n"
                     f"shift ({dy}, {dx}) px, NCC {mr['ncc']:.2f}{tt}", fontsize=8)
        aa.axis("off")
    fig.suptitle("Cross-day cell matching on the Day-2 mean image\ncircles = Day-1 cells (green matched, orange not), "
                 "+ = Day-2 cells (blue matched, red new)", fontsize=10)
    toy_mark(fig, toy)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(a.out, "long_L5_cell_matching.png"), dpi=110); plt.close(fig)
    if toy:
        ok = C[C.measure == "transient rate"]
        print("TOY check (SBI toy effect = fewer transients): median d_SBI - d_vehicle per mouse:",
              (ok.med_d_sbi - ok.med_d_vehicle).round(3).tolist())
    print("ca_longitudinal done ->", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
