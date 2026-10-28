"""R7  IS THERE A POPULATION RESPONSE AT ALL? Per mouse x session x event, the mean response
vector over cells (mu_j = mean over trials of z[0,2] - z[-2,0], noise-SD z) against 2000 copies of
the SAME event train shifted inside each schedule segment (one random circular offset per segment,
as the single-cell time-shift null of ca_day1_analysis), so spacing and block context are kept.
  energy  = RMS of mu over cells (any consistent response, up or down)
  up      = mean of mu over cells (net increase), two-sided
  n_cells_up95 = cells whose mu exceeds their own 95th null percentile (vs 5 % expected)
p = (1 + #null >= observed) / (1 + n). Why: R5b/R6 showed that the response PATTERN repeats poorly
between trial halves and that random times in the same blocks look just as 'reliable' after
removing the shared component - so the right question is whether the deliveries do anything
beyond the ongoing activity of that block.

  python ca_population_response_test.py --out-root <mini1p_out/20261019> --scoring-dir <final> --out <analysis/ca_responsive_change>
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import safe_io  # noqa: E402,F401  (a figure save retries on a transient Windows write error)
import ca_day1_analysis as d1  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FPS = d1.FPS
EVS = d1.STIM + ["withdrawal", "flinch", "movement onset"]


def null_mu(s, used, n_shuf=2000, seed=0, pre=None, rsp=None):
    """observed mu (cells) and the null matrix (n_shuf x cells), within-segment circular shifts"""
    T = s["T"]
    fr = T["frame"].astype(int)
    Z = np.asarray(T["z"], float)
    ok = np.isfinite(Z)
    Zc = np.vstack([np.zeros((1, Z.shape[1])), np.cumsum(np.where(ok, Z, 0), 0)])
    Nc = np.vstack([np.zeros((1, Z.shape[1])), np.cumsum(ok, 0)])
    n = len(fr)
    pre, rsp = (d1.PRE_W if pre is None else pre), (d1.RESP_W if rsp is None else rsp)
    wa, wb = int(round((rsp[1] - rsp[0]) * FPS)), int(round((pre[1] - pre[0]) * FPS))   # response / pre window (frames)
    w = max(wa, wb)
    st = int(np.searchsorted(fr, int(d1.SKIP_S * FPS))) + wb
    idx = np.searchsorted(fr, np.asarray(used, int))

    def resp(ii):
        ii = ii[(ii >= st) & (ii + wa < n)]
        a = (Zc[ii + wa] - Zc[ii]) / np.maximum(Nc[ii + wa] - Nc[ii], 1)
        b = (Zc[ii] - Zc[ii - wb]) / np.maximum(Nc[ii] - Nc[ii - wb], 1)
        return np.nanmean(a - b, 0)
    obs = resp(idx)
    bounds_s = sorted({d1.SKIP_S + w / FPS} | {b_ for _, b0, b1 in d1.BLOCKS for b_ in (b0, b1)} | {fr[-1] / FPS})
    bounds = np.unique(np.searchsorted(fr, (np.array(bounds_s) * FPS).astype(int)))
    seg = np.searchsorted(bounds, idx, side="right") - 1
    rng = np.random.default_rng(seed)
    N = np.empty((n_shuf, Z.shape[1]))
    for k_ in range(n_shuf):
        ii = idx.copy()
        for k in np.unique(seg):
            if k < 0 or k + 1 >= len(bounds):
                continue
            a0, a1 = bounds[k], bounds[k + 1]
            L = a1 - a0
            if L <= 2 * w:
                continue
            msk = seg == k
            ii[msk] = a0 + (idx[msk] - a0 + int(rng.integers(w, L - w))) % L
        N[k_] = resp(np.sort(ii))
    return obs, N


def fig_r7(D, out, n_shuf, SUF):
    """the figure (10-22: also drawn alone from the saved table with --replot)"""
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    evs = [e for e in EVS if e in set(D.event)]
    for k, ph in enumerate(("pre", "post")):
        a_ = ax[k]
        for i, (m, arm, sex) in enumerate(d1.MICE):
            X = D[(D.mouse == m) & (D.phase == ph)].set_index("event")
            for j, ev in enumerate(evs):
                if ev not in X.index:
                    continue
                r = X.loc[ev]
                ratio = r.energy / r.energy_null_median
                x = j + (i - 1.5) * 0.18
                a_.bar(x, ratio, 0.16, color=d1.ARMC[arm], alpha=0.95 if i % 2 == 0 else 0.55,
                       label=f"{m} ({arm})" if j == 0 else None)
                if r.q_energy <= 0.05:
                    a_.text(x, ratio + 0.02, "*", ha="center", fontsize=10)
        a_.axhline(1, color="0.4", lw=0.8)
        a_.set_xticks(range(len(evs)), evs, rotation=25, ha="right", fontsize=8)
        a_.set_ylabel("population response energy / its null median\n(RMS over cells of the mean response)", fontsize=8)
        a_.set_title(("A  baseline" if ph == "pre" else "B  injection") + " session", fontsize=9)
        a_.legend(fontsize=7, frameon=False)
    fig.suptitle("R7 does each event move the population beyond the ongoing activity of its block? 1 = like the same events shifted "
                 f"in time inside their block ({n_shuf} shifts).\n* above a bar = that bar vs its own shifted-time null (the line at 1), BH q <= 0.05 "
                 f"over all {len(D)} tests", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(out, f"R7_population_response_test{SUF}.png"), dpi=130); plt.close(fig)


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
        D = pd.read_csv(os.path.join(a.out, f"R7_population_response_test{SUF}.csv")); D["mouse"] = D.mouse.astype(str)
        fig_r7(D, a.out, a.n_shuf, SUF)
        print("->", a.out, "(figure redrawn)")
        return 0
    rows = []
    for m, arm, sex in d1.MICE:
        for ph in ("pre", "post"):
            s = d1.load_session(a.out_root, m, ph, a.scoring_dir)
            if s is None or s["B"] is None:
                continue
            for ev in EVS:
                pre_w, rsp_w = d1.windows(ev)             # CA_WIN=best: the best window (10-22)
                X, tt, used = d1.peri(s, s["B"]["ev"].get(ev, []), pre0=pre_w[0])
                if X is None or len(used) < d1.MIN_TRIALS:
                    continue
                obs, N = null_mu(s, used, a.n_shuf, pre=pre_w, rsp=rsp_w)
                good = np.isfinite(obs) & np.isfinite(N).all(0)
                obs, N = obs[good], N[:, good]
                e_obs, e_null = np.sqrt(np.mean(obs ** 2)), np.sqrt(np.mean(N ** 2, 1))
                u_obs, u_null = np.mean(obs), np.mean(N, 1)
                hi = np.percentile(N, 95, 0)
                lo = np.percentile(N, 5, 0)
                rows.append(dict(mouse=m, arm=arm, phase=ph, event=ev, n_trials=len(used), n_cells=int(good.sum()),
                                 energy=e_obs, energy_null_median=float(np.median(e_null)),
                                 p_energy=(1 + np.sum(e_null >= e_obs)) / (1 + len(e_null)),
                                 mean_response=u_obs, p_mean=(1 + np.sum(np.abs(u_null - np.median(u_null)) >= abs(u_obs - np.median(u_null)))) / (1 + len(u_null)),
                                 frac_cells_above_null95=float(np.mean(obs > hi)), frac_cells_below_null5=float(np.mean(obs < lo))))
                print(f"{m} {ph} {ev}: energy {e_obs:.3f} (null {np.median(e_null):.3f}), p {rows[-1]['p_energy']:.3f}")
    D = pd.DataFrame(rows)
    D["q_energy"] = d1.bh(D.p_energy.values)
    D.round(4).to_csv(os.path.join(a.out, f"R7_population_response_test{SUF}.csv"), index=False)
    fig_r7(D, a.out, a.n_shuf, SUF)
    print(D.pivot_table(index=["mouse", "event"], columns="phase", values="q_energy").round(3).to_string())
    print("->", a.out)


if __name__ == "__main__":
    main()
