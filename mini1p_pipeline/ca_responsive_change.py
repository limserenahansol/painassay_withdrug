"""R5  DO THE CELLS THAT RESPONDED IN THE BASELINE RESPOND LESS AFTER THE INJECTION?
Selecting responsive cells and measuring them on the same trials inflates the baseline response
(regression to the mean: the selected cells look smaller afterwards even with no change). So the
baseline trials are split: cells are selected on one half (UP: Wilcoxon of r over those trials
p < 0.05 and median r > 0) and their baseline response is measured on the OTHER half; both splits
(odd / even) are averaged. The injection response is measured on all injection trials.
r = mean z [0, 2] s - mean z [-2, 0] s per trial; z = dF/F / the cell's noise SD over both
sessions (one fixed scale per cell, so baseline and injection sizes are comparable).
R5b: reliability of the response PATTERN over cells (random split halves of the trials), per
mouse x session x event, against the number of trials - how many trials a stable map needs.
Per mouse and event: n selected cells, mean response baseline (held-out half) and injection,
log2 ratio of the means, Wilcoxon post vs pre over the selected cells. Arms are shown, not tested
(2 mice per arm).

  python ca_responsive_change.py --out-root <mini1p_out/20261019> --scoring-dir <final> --out <analysis/ca_responsive_change>
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ca_day1_analysis as d1  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EVS = d1.STIM + ["withdrawal"]


def trial_r(s, ev):
    pre_w, rsp_w = d1.windows(ev)                         # CA_WIN=best: the best window (10-22)
    X, tt, used = d1.peri(s, s["B"]["ev"].get(ev, []), pre0=pre_w[0])
    if X is None:
        return None
    pre = (tt >= pre_w[0]) & (tt < pre_w[1])
    rsp = (tt >= rsp_w[0]) & (tt < rsp_w[1])
    return np.nanmean(X[:, rsp, :], 1) - np.nanmean(X[:, pre, :], 1)        # trials x cells


def select_up(r):
    out = np.zeros(r.shape[1], bool)
    for j in range(r.shape[1]):
        x = r[:, j][np.isfinite(r[:, j])]
        if len(x) >= 4 and not np.allclose(x, 0):
            try:
                out[j] = wilcoxon(x).pvalue < 0.05 and np.median(x) > 0
            except ValueError:
                pass
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--scoring-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    d1.Z_NORM = "noise"
    rows, cells = [], []
    for m, arm, sex in d1.MICE:
        S = {ph: d1.load_session(a.out_root, m, ph, a.scoring_dir) for ph in ("pre", "post")}
        if any(v is None or v["B"] is None for v in S.values()):
            continue
        ids = S["pre"]["T"]["cell_ids"].astype(int)
        for ev in EVS:
            rp, rq = trial_r(S["pre"], ev), trial_r(S["post"], ev)
            if rp is None or rq is None or len(rp) < 2 * d1.MIN_TRIALS or len(rq) < d1.MIN_TRIALS:
                continue
            post = np.nanmean(rq, 0)
            pre_sum, post_sum, nsel, sel_any = [], [], [], np.zeros(len(ids), bool)
            per_cell_pre = np.full((2, len(ids)), np.nan)
            for k, (A, Bh) in enumerate(((slice(0, None, 2), slice(1, None, 2)), (slice(1, None, 2), slice(0, None, 2)))):
                up = select_up(rp[A])
                per_cell_pre[k, up] = np.nanmean(rp[Bh][:, up], 0)
                sel_any |= up
            pre_ho = np.nanmean(per_cell_pre, 0)                       # held-out baseline response
            ok = sel_any & np.isfinite(pre_ho) & np.isfinite(post)
            if ok.sum() >= 3:
                try:
                    p = float(wilcoxon(post[ok], pre_ho[ok]).pvalue)
                except ValueError:
                    p = np.nan
                mp, mq = float(np.mean(pre_ho[ok])), float(np.mean(post[ok]))
                rows.append(dict(mouse=m, arm=arm, event=ev, n_trials_pre=len(rp), n_trials_post=len(rq), n_cells=int(ok.sum()),
                                 n_cells_total=len(ids), pre_heldout=mp, post=mq,
                                 log2_ratio=np.log2(mq / mp) if mp > 0 and mq > 0 else np.nan, p_wilcoxon=p))
                for j in np.where(ok)[0]:
                    cells.append(dict(mouse=m, arm=arm, event=ev, cell=int(ids[j]), pre_heldout=pre_ho[j], post=post[j]))
    R = pd.DataFrame(rows)
    R.round(4).to_csv(os.path.join(a.out, "R5_baseline_responsive_cells_change.csv"), index=False)
    pd.DataFrame(cells).round(4).to_csv(os.path.join(a.out, "R5_cells.csv"), index=False)
    C = pd.DataFrame(cells)
    fig, ax = plt.subplots(1, len(EVS), figsize=(3.4 * len(EVS), 4.2), sharey=True)
    for k, ev in enumerate(EVS):
        a_ = ax[k]
        for i, (m, arm, sex) in enumerate(d1.MICE):
            X = C[(C.mouse == m) & (C.event == ev)] if len(C) else pd.DataFrame()
            r = R[(R.mouse == m) & (R.event == ev)]
            if not len(X) or not len(r):
                continue
            col = d1.ARMC[arm]
            for _, c in X.iterrows():
                a_.plot([i - 0.15, i + 0.15], [c.pre_heldout, c.post], color=col, alpha=0.25, lw=0.7)
            a_.plot([i - 0.15, i + 0.15], [r.pre_heldout.iloc[0], r.post.iloc[0]], color=col, lw=2.5, marker="o")
            pv = r.p_wilcoxon.iloc[0]
            a_.text(i, a_.get_ylim()[1] if False else max(X.pre_heldout.max(), X.post.max()) + 0.05,
                    f"n={int(r.n_cells.iloc[0])}\n{'p<0.001' if pv < 0.001 else f'p={pv:.2g}'}", ha="center", fontsize=6.5)
        a_.axhline(0, color="0.5", lw=0.6)
        a_.set_xticks(range(len(d1.MICE)), [f"{m}\n{arm}" for m, arm, _ in d1.MICE], fontsize=7)
        a_.set_title(ev, fontsize=9)
        if k == 0:
            a_.set_ylabel("response z[0,2] - z[-2,0] (noise SD units)\nleft: baseline (held-out trials), right: injection", fontsize=8)
    fig.suptitle("R5 cells that responded (UP) in the baseline: their response in held-out baseline trials vs the injection session "
                 "(thin = cells, thick = mean; Wilcoxon over cells)\nselection on one half of the baseline trials, measurement on the "
                 "other half (both splits) - no regression-to-the-mean bias; SBI-553 red, vehicle blue", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.6 / fig.get_figheight()))
    fig.savefig(os.path.join(a.out, "R5_baseline_responsive_change.png"), dpi=130); plt.close(fig)
    print(R.round(3).to_string(index=False))
    rel = split_half(a, d1.MICE)
    rel.round(3).to_csv(os.path.join(a.out, "R5_split_half_reliability.csv"), index=False)
    fig_reliability(rel, a.out)
    print(rel.groupby("event")[["n_trials", "reliability_sb"]].mean().round(2).to_string())
    print("->", a.out)


def split_half(a, mice, n_split=100, seed=0):
    """reliability of the population response PATTERN: Spearman correlation over cells of the
    mean response in two random halves of the trials (100 random splits, mean), Spearman-Brown
    corrected to the full trial count. ~0 = the pattern of which cells respond does not repeat."""
    from scipy.stats import spearmanr
    rng = np.random.default_rng(seed)
    rows = []
    for m, arm, sex in mice:
        for ph in ("pre", "post"):
            s = d1.load_session(a.out_root, m, ph, a.scoring_dir)
            if s is None or s["B"] is None:
                continue
            for ev in EVS + ["movement onset"]:
                r = trial_r(s, ev)
                if r is None or len(r) < 10:
                    continue
                rho = []
                for _ in range(n_split):
                    o = rng.permutation(len(r))
                    x, y = np.nanmean(r[o[: len(r) // 2]], 0), np.nanmean(r[o[len(r) // 2:]], 0)
                    ok = np.isfinite(x) & np.isfinite(y)
                    rho.append(spearmanr(x[ok], y[ok])[0])
                rh = float(np.nanmean(rho))
                rows.append(dict(mouse=m, arm=arm, phase=ph, event=ev, n_trials=len(r), split_half_rho=rh,
                                 reliability_sb=2 * rh / (1 + rh) if rh > -1 else np.nan))
    return pd.DataFrame(rows)


def fig_reliability(rel, out):
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    col = dict(d1.EVC)
    for ev, X in rel.groupby("event"):
        ax.scatter(X.n_trials, X.reliability_sb, s=28, color=col.get(ev, "0.3"), label=f"{ev} (n={len(X)} sessions)",
                   edgecolor="k", lw=0.3)
    ax.axhline(0, color="0.5", lw=0.6)
    ax.set_xlabel("number of trials in the session", fontsize=9)
    ax.set_ylabel("reliability of the response pattern over cells\n(random split-half Spearman, Spearman-Brown corrected)", fontsize=8)
    ax.set_title("R5b which cells respond: does the pattern repeat between halves of the trials?\n"
                 "(every mouse x session; 100 random splits each)", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "R5b_split_half_reliability.png"), dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
