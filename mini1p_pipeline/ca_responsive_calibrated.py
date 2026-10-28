"""R8  RESPONSIVE CELLS, CALIBRATED FOR THE DEPENDENCE BETWEEN CELLS. The single-cell time-shift
rule (ca_day1_analysis B1c: per-cell p against shifted copies of the event train, BH q <= 0.20)
treats cells as independent, but their null responses are correlated (one shared component per
trial, R6): applied to SHIFTED trains (no real event) it calls >= 1 cell in ~12 % of the sets and
then often tens of cells at once. Two calibrated readouts per mouse x session x event:
  count test   the observed number of q <= 0.20 cells against the numbers the same rule gives
               for 500 shifted trains (each tested against the other shifts): p_count
  max-T cells  Westfall-Young: each cell's response as a z against its own null, the null of the
               LARGEST |z| over all cells per shift -> 95th percentile threshold; cells above it
               are responsive with family-wise error 0.05 whatever the correlation (UP / DOWN by sign)
z = dF/F / the cell's noise SD over both sessions; 2000 within-segment shifts (as B1c).

  python ca_responsive_calibrated.py --out-root <mini1p_out/20261019> --scoring-dir <final> --out <analysis/ca_responsive_change>
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import safe_io  # noqa: E402,F401  (a figure save retries on a transient Windows write error)
import ca_day1_analysis as d1  # noqa: E402
from ca_population_response_test import null_mu  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EVS = d1.STIM + ["withdrawal", "flinch", "movement onset"]
N_PSEUDO = 500


def pvals(x, ref):
    return (1 + np.sum(np.abs(ref) >= np.abs(x), 0)) / (1 + len(ref))


def fig_r8(D, out, n_shuf, SUF):
    """the figure (10-22: also drawn alone from the saved table with --replot)"""
    evs = [e for e in EVS if e in set(D.event)]
    fig, ax = plt.subplots(2, 4, figsize=(17, 7), sharey="row")
    for i, (m, arm, sex) in enumerate(d1.MICE):
        for k, ph in enumerate(("pre", "post")):
            a_ = ax[k, i]
            X = D[(D.mouse == m) & (D.phase == ph)].set_index("event").reindex(evs)
            x = np.arange(len(evs))
            a_.bar(x - 0.2, X.rule_q20_cells, 0.38, color="0.65", label="single-cell rule (q<=0.20)")
            a_.errorbar(x - 0.2, X.null_rule_cells_mean, yerr=[np.zeros(len(x)), X.null_rule_cells_p95 - X.null_rule_cells_mean],
                        fmt="_", color="k", capsize=3, label="same rule on shifted trains (mean, 95th pct)")
            a_.bar(x + 0.2, X.maxT_up, 0.38, color="#c0392b", label="max-T UP (FWER 0.05)")
            a_.bar(x + 0.2, X.maxT_down, 0.38, bottom=X.maxT_up, color="#2471a3", label="max-T DOWN")
            for xi, (_, r) in zip(x, X.iterrows()):
                if np.isfinite(r.p_count) and r.p_count <= 0.05:
                    a_.text(xi - 0.2, r.rule_q20_cells + 1, "*", ha="center", fontsize=10)
            a_.set_xticks(x, evs, rotation=35, ha="right", fontsize=7)
            a_.set_title(f"{m} ({arm}) {'baseline' if ph == 'pre' else 'injection'}", fontsize=9, color=d1.ARMC[arm])
            if i == 0:
                a_.set_ylabel("cells", fontsize=9)
            if i == 0 and k == 0:
                a_.legend(fontsize=6.5, frameon=False)
    fig.suptitle("R8 responsive cells calibrated for the correlation between cells. Grey: the single-cell rule; black: what the same rule "
                 "finds when there is no event (shifted trains); * above a grey bar = that bar vs its black null (more cells than 95 % of the "
                 "shifted trains, p_count <= 0.05).\n"
                 "Red/blue: Westfall-Young max-T cells (family-wise error 0.05 under any correlation). z in noise SD units, 2000 "
                 "within-segment shifts.", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(out, f"R8_responsive_calibrated{SUF}.png"), dpi=130); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--scoring-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--traces", default="traces", help="traces | traces_bgcorr | traces_roi; outputs get a suffix if not 'traces'")
    ap.add_argument("--n-shuf", type=int, default=2000)
    ap.add_argument("--replot", action="store_true", help="only redraw the figure from the saved table (no recomputation)")
    a = ap.parse_args(argv)
    d1.TRACES = a.traces
    SUF = "" if a.traces == "traces" else "_" + a.traces
    os.makedirs(a.out, exist_ok=True)
    d1.Z_NORM = "noise"
    if a.replot:
        D = pd.read_csv(os.path.join(a.out, f"R8_responsive_calibrated{SUF}.csv")); D["mouse"] = D.mouse.astype(str)
        fig_r8(D, a.out, a.n_shuf, SUF)
        print("->", a.out, "(figure redrawn)")
        return 0
    rows, cells = [], []
    for m, arm, sex in d1.MICE:
        for ph in ("pre", "post"):
            s = d1.load_session(a.out_root, m, ph, a.scoring_dir)
            if s is None or s["B"] is None:
                continue
            ids = s["T"]["cell_ids"].astype(int)
            for ev in EVS:
                X, tt, used = d1.peri(s, s["B"]["ev"].get(ev, []))
                if X is None or len(used) < d1.MIN_TRIALS:
                    continue
                obs, N = null_mu(s, used, a.n_shuf, seed=11)
                good = np.isfinite(obs) & np.isfinite(N).all(0)
                obs, N, cid = obs[good], N[:, good], ids[good]
                # uncalibrated rule (as B1c, recomputed here on the same shifts)
                q = d1.bh(pvals(obs, N))
                n_obs = int(np.sum(q <= d1.Q_RESP))
                ref = N[N_PSEUDO:]
                cnt = np.array([np.sum(d1.bh(pvals(N[k], ref)) <= d1.Q_RESP) for k in range(N_PSEUDO)])
                p_count = (1 + np.sum(cnt >= n_obs)) / (1 + N_PSEUDO) if n_obs > 0 else 1.0
                # Westfall-Young max-T
                mu0, sd0 = N.mean(0), N.std(0)
                sd0 = np.where(sd0 > 0, sd0, np.nan)
                t_obs = (obs - mu0) / sd0
                t_null = np.nanmax(np.abs((N - mu0) / sd0), 1)
                thr = float(np.percentile(t_null, 95))
                up, dn = (t_obs > thr), (t_obs < -thr)
                rows.append(dict(mouse=m, arm=arm, phase=ph, event=ev, n_trials=len(used), n_cells=int(good.sum()),
                                 rule_q20_cells=n_obs, null_rule_cells_mean=float(cnt.mean()),
                                 null_rule_cells_p95=float(np.percentile(cnt, 95)), p_count=float(p_count),
                                 maxT_threshold=thr, maxT_up=int(up.sum()), maxT_down=int(dn.sum())))
                for j in np.where(up | dn)[0]:
                    cells.append(dict(mouse=m, arm=arm, phase=ph, event=ev, cell=int(cid[j]), cls="UP" if up[j] else "DOWN",
                                      t=float(t_obs[j]), mean_response=float(obs[j])))
                print(f"{m} {ph} {ev}: rule {n_obs} (null p95 {np.percentile(cnt, 95):.0f}, p_count {p_count:.3f}), "
                      f"max-T {int(up.sum())} up + {int(dn.sum())} down")
    D = pd.DataFrame(rows)
    D.round(4).to_csv(os.path.join(a.out, f"R8_responsive_calibrated{SUF}.csv"), index=False)
    pd.DataFrame(cells, columns=["mouse", "arm", "phase", "event", "cell", "cls", "t", "mean_response"]).round(4).to_csv(
        os.path.join(a.out, f"R8_maxT_cells{SUF}.csv"), index=False)
    fig_r8(D, a.out, a.n_shuf, SUF)
    print("->", a.out)


if __name__ == "__main__":
    main()
