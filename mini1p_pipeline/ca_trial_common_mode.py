"""R6  THE TRIAL-TO-TRIAL COMMON MODE. The response pattern over cells repeats poorly between
halves of the trials (R5b), and split halves are often ANTI-correlated - the signature of one
shared component whose size changes from trial to trial with a cell-specific sign: its two half
means have opposite signs when it averages out. Per mouse x session x event:
  r (trials x cells, noise-SD z) centred per cell (the mean response is kept aside), PCA:
  - share of the trial-to-trial variance in PC1
  - what PC1's trial score follows: the background change (pixels outside every footprint,
    [0,2] - [-2,0] s), fast brain motion (max |change of the MC shift| in [-2, 2] s), the mouse's
    movement ([0,2] - [-2,0] s moving fraction), a withdrawal within 2 s (stimulus trials)
  - split-half reliability of the response pattern before and after removing PC1 from the
    trial-to-trial part (the mean response is not touched)
  - control: the same at random times in the same span (>= 3 s from every delivery): if the PC1
    share and the 'cleaned' reliability are as large there, both are properties of the ongoing
    activity, not of the stimulus (that is what the data show)
Outputs: analysis/ca_responsive_change/R6_trial_common_mode.csv, R6_trial_common_mode.png

  python ca_trial_common_mode.py --out-root <mini1p_out/20261019> --scoring-dir <final> --out <analysis/ca_responsive_change>
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ca_day1_analysis as d1  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FPS = d1.FPS
EVS = d1.STIM + ["withdrawal", "movement onset"]


def reliability(r, rng, n_split=100):
    rho = []
    for _ in range(n_split):
        o = rng.permutation(len(r))
        x, y = np.nanmean(r[o[: len(r) // 2]], 0), np.nanmean(r[o[len(r) // 2:]], 0)
        ok = np.isfinite(x) & np.isfinite(y)
        rho.append(spearmanr(x[ok], y[ok])[0])
    rh = float(np.nanmean(rho))
    return 2 * rh / (1 + rh) if rh > -1 else np.nan


def window_change(v, idx, k_pre, k_post):
    out = []
    for i in idx:
        if i - k_pre < 0 or i + k_post >= len(v):
            out.append(np.nan); continue
        out.append(np.nanmean(v[i:i + k_post]) - np.nanmean(v[i - k_pre:i]))
    return np.array(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--scoring-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    d1.Z_NORM = "noise"
    rng = np.random.default_rng(0)
    rows = []
    for m, arm, sex in d1.MICE:
        for ph in ("pre", "post"):
            s = d1.load_session(a.out_root, m, ph, a.scoring_dir)
            if s is None or s["B"] is None:
                continue
            T = s["T"]
            fr = T["frame"].astype(int)
            fb = os.path.join(a.out_root, f"{m}_d{d1.DAY}", "5_analysis", "traces", f"{m}_d{d1.DAY}_{ph}_background.npz")
            bg = None
            if os.path.isfile(fb):
                Bz = np.load(fb)
                bg = pd.Series(Bz["b_frac"].astype(float), index=Bz["frame"].astype(int)).reindex(fr).values
            mot = s.get("motion")
            dsh = np.abs(np.diff(np.r_[mot["shift"][0], mot["shift"]])) if mot is not None else None
            mvst = s["B"]["st"]["move"]
            mv = np.array([mvst[f] if f < len(mvst) else False for f in fr], float)
            wd = np.asarray(s["B"]["ev"]["withdrawal"], int)
            for ev in EVS:
                pre_w, rsp_w = d1.windows(ev)             # CA_WIN=best: the best window (10-22)
                X, tt, used = d1.peri(s, s["B"]["ev"].get(ev, []), pre0=pre_w[0])
                if X is None or len(used) < 10:
                    continue
                pre = (tt >= pre_w[0]) & (tt < pre_w[1])
                rsp = (tt >= rsp_w[0]) & (tt < rsp_w[1])
                r = np.nanmean(X[:, rsp, :], 1) - np.nanmean(X[:, pre, :], 1)
                good = np.isfinite(r).all(0)
                r = r[:, good]
                mu = r.mean(0)
                Rc = r - mu
                U, S_, Vt = np.linalg.svd(Rc, full_matrices=False)
                share = float(S_[0] ** 2 / np.sum(S_ ** 2))
                score = U[:, 0] * S_[0]
                r_clean = mu + Rc - np.outer(score, Vt[0])
                idx = np.searchsorted(fr, np.asarray(used, int))
                cov = {}
                if bg is not None:
                    cov["background"] = window_change(bg, idx, int(2 * FPS), int(2 * FPS))
                if dsh is not None:
                    cov["fast brain motion"] = np.array([np.nanmax(dsh[max(0, i - 60):i + 60]) for i in idx])
                cov["mouse movement"] = window_change(mv, idx, int(2 * FPS), int(2 * FPS))
                if ev in d1.STIM:
                    cov["withdrawal within 2 s"] = np.array([np.any((wd >= f) & (wd < f + 2 * FPS)) for f in used], float)
                row = dict(mouse=m, arm=arm, phase=ph, event=ev, n_trials=len(used), n_cells=int(good.sum()),
                           pc1_share=share, reliability=reliability(r, rng), reliability_pc1_removed=reliability(r_clean, rng))
                if ev in d1.STIM:
                    # control: the same number of random times in the same span of the session, >= 3 s from
                    # every delivery of this stimulus (no event) - if PC1 share and the reliability after
                    # removing PC1 are as large there, they say nothing about the stimulus
                    real = np.asarray(used, int)
                    cand = np.arange(max(real.min() - 300, int((d1.SKIP_S + 5) * FPS)), min(real.max() + 300, fr[-1] - 300))
                    cand = cand[np.min(np.abs(cand[:, None] - real[None, :]), 1) > 90]
                    if len(cand) > len(real):
                        Xs, _, us = d1.peri(s, np.sort(rng.choice(cand, len(real), replace=False)))
                        if Xs is not None and len(us) >= 10:
                            rs = np.nanmean(Xs[:, rsp, :], 1) - np.nanmean(Xs[:, pre, :], 1)
                            rs = rs[:, np.isfinite(rs).all(0)]
                            mus = rs.mean(0); Rs = rs - mus
                            Us, Ss, Vs = np.linalg.svd(Rs, full_matrices=False)
                            row["sham_pc1_share"] = float(Ss[0] ** 2 / np.sum(Ss ** 2))
                            row["sham_reliability"] = reliability(rs, rng)
                            row["sham_reliability_pc1_removed"] = reliability(mus + Rs - np.outer(Us[:, 0] * Ss[0], Vs[0]), rng)
                for k, v in cov.items():
                    ok = np.isfinite(v) & np.isfinite(score)
                    row[f"|r| PC1 vs {k}"] = abs(float(np.corrcoef(score[ok], v[ok])[0, 1])) if ok.sum() >= 5 and np.std(v[ok]) > 0 else np.nan
                rows.append(row)
    D = pd.DataFrame(rows)
    D.round(3).to_csv(os.path.join(a.out, "R6_trial_common_mode.csv"), index=False)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    for ev, X in D.groupby("event"):
        c = d1.EVC.get(ev, "0.3")
        ax[0].scatter(X.n_trials, X.pc1_share, color=c, edgecolor="k", lw=0.3, s=26, label=ev)
        ax[1].scatter(X.reliability, X.reliability_pc1_removed, color=c, edgecolor="k", lw=0.3, s=26, label=ev)
    n_tr = D.n_trials.median()
    ax[0].axhline(1 / max(n_tr - 1, 1), color="0.5", ls="--", lw=0.8)
    ax[0].set_xlabel("trials", fontsize=9); ax[0].set_ylabel("share of trial-to-trial variance in PC1", fontsize=9)
    ax[0].set_title("A  one shared component per trial?\n(dashed ~ share expected without one, 1/(trials-1))", fontsize=9)
    ax[0].legend(fontsize=7, frameon=False)
    ax[1].plot([-1, 1], [-1, 1], color="0.5", lw=0.8)
    ax[1].set_xlabel("pattern reliability, as measured", fontsize=9)
    ax[1].set_ylabel("pattern reliability, PC1 removed", fontsize=9)
    ax[1].set_title("B  reliability of the response pattern over cells\n(split-half, Spearman-Brown; 100 random splits)", fontsize=9)
    cols = [c for c in D.columns if c.startswith("|r| PC1 vs")]
    M = D[cols].mean()
    ax[2].bar(range(len(cols)), M.values, color="0.5")
    for i, c in enumerate(cols):
        ax[2].scatter(np.full(D[c].notna().sum(), i) + np.random.default_rng(1).uniform(-0.2, 0.2, D[c].notna().sum()),
                      D[c].dropna(), s=8, color="k", alpha=0.5)
    ax[2].set_xticks(range(len(cols)), [c.replace("|r| PC1 vs ", "") for c in cols], fontsize=8, rotation=15)
    ax[2].set_ylabel("|correlation| of the PC1 trial score", fontsize=9)
    ax[2].set_title(f"C  what the shared component follows (bars = mean, dots = {len(D)} mouse x session x event)", fontsize=9)
    fig.suptitle("R6 trial-to-trial common mode of the event responses (all mice, both sessions; z in noise SD units)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(a.out, "R6_trial_common_mode.png"), dpi=130); plt.close(fig)
    print(D.groupby("event")[["n_trials", "pc1_share", "reliability", "reliability_pc1_removed"] + cols].mean().round(2).to_string())
    print("->", a.out)


if __name__ == "__main__":
    main()
