"""R10  THE FIELD-WIDE BRIGHTNESS CHANGE AT THE DELIVERIES: does it come with the mouse's movement?
Per delivery: the change of the background (mean of the pixels outside every footprint,
ca_background_signal; [0, 2] - [-2, 0] s, in SD of that session's background) and of fast brain
motion, split by what the mouse did: moved in [-2, 2] s or not, withdrew within 2 s or not.
If the brightening is there only with movement / withdrawal -> posture / focus (optical) change;
if it is there in still trials too -> a stimulus-locked field signal (diffuse neuropil, or light).

  python ca_field_change.py --out-root <mini1p_out/20261019> --scoring-dir <final> --out <analysis/ca_response_shape>
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ca_day1_analysis as d1  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FPS = d1.FPS


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--scoring-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    rows = []
    k2 = int(2 * FPS)
    for m, arm, sex in d1.MICE:
        for ph in ("pre", "post"):
            s = d1.load_session(a.out_root, m, ph, a.scoring_dir)
            if s is None or s["B"] is None:
                continue
            fr = s["T"]["frame"].astype(int)
            fb = os.path.join(a.out_root, f"{m}_d{d1.DAY}", "5_analysis", "traces", f"{m}_d{d1.DAY}_{ph}_background.npz")
            if not os.path.isfile(fb):
                continue
            Bz = np.load(fb)
            bg = pd.Series(Bz["b_frac"].astype(float), index=Bz["frame"].astype(int)).reindex(fr).values
            sd = np.nanstd(bg)
            mot = s.get("motion")
            dsh = np.abs(np.diff(np.r_[mot["shift"][0], mot["shift"]])) if mot is not None else np.full(len(fr), np.nan)
            mvst = s["B"]["st"]["move"]
            mv = np.array([mvst[f] if f < len(mvst) else False for f in fr], bool)
            wd = np.asarray(s["B"]["ev"]["withdrawal"], int)
            for ev in d1.STIM:
                pre_w, rsp_w = d1.windows(ev)                 # CA_WIN=best: the best window (10-22); old = 2 s / 2 s
                kb, ka = int(round(-pre_w[0] * FPS)), int(round(rsp_w[1] * FPS))
                for f in s["B"]["ev"].get(ev, []):
                    i = np.searchsorted(fr, f)
                    if i >= len(fr) or fr[i] != f or i - k2 < 0 or i + k2 >= len(fr) or fr[i] / FPS < d1.SKIP_S + 2:
                        continue
                    rows.append(dict(mouse=m, arm=arm, phase=ph, event=ev, frame=int(f),
                                     bg_change_sd=(np.nanmean(bg[i:i + ka]) - np.nanmean(bg[i - kb:i])) / sd,
                                     fast_motion_px=float(np.nanmax(dsh[i - k2:i + k2])),
                                     moved=bool(mv[i - k2:i + k2].any()),
                                     withdrawal=bool(np.any((wd >= f) & (wd < f + 2 * FPS)))))
    D = pd.DataFrame(rows)
    D.round(4).to_csv(os.path.join(a.out, "R10_field_change_trials.csv"), index=False)
    D["state"] = np.where(D.withdrawal, "withdrawal", np.where(D.moved, "moved, no withdrawal", "still, no withdrawal"))
    S = D.groupby(["event", "state"]).bg_change_sd.agg(["count", "mean", "median"]).reset_index()
    pv = []
    for (ev, st), X in D.groupby(["event", "state"]):
        x = X.bg_change_sd.dropna()
        pv.append(dict(event=ev, state=st, p_vs_0=float(wilcoxon(x).pvalue) if len(x) >= 6 and not np.allclose(x, 0) else np.nan))
    S = S.merge(pd.DataFrame(pv), on=["event", "state"])
    S.round(4).to_csv(os.path.join(a.out, "R10_field_change_by_state.csv"), index=False)
    fig, ax = plt.subplots(1, len(d1.STIM), figsize=(4 * len(d1.STIM), 4.2), sharey=True)
    states = ["still, no withdrawal", "moved, no withdrawal", "withdrawal"]
    for k, ev in enumerate(d1.STIM):
        a_ = ax[k]
        for j, st in enumerate(states):
            x = D[(D.event == ev) & (D.state == st)].bg_change_sd.dropna()
            if not len(x):
                continue
            a_.scatter(np.full(len(x), j) + np.random.default_rng(j).uniform(-0.15, 0.15, len(x)), x, s=8, alpha=0.4, color="0.3")
            a_.plot([j - 0.25, j + 0.25], [x.median()] * 2, color="#c0392b", lw=2.5)
            r = S[(S.event == ev) & (S.state == st)]
            pvv = r.p_vs_0.iloc[0] if len(r) else np.nan
            a_.text(j, a_.get_ylim()[1] if False else np.nanpercentile(D.bg_change_sd, 99), f"n={len(x)}\n" +
                    ("p<0.001" if pvv < 0.001 else f"p={pvv:.2g}" if np.isfinite(pvv) else ""), ha="center", fontsize=7, va="top")
        a_.axhline(0, color="0.5", lw=0.6)
        a_.set_xticks(range(len(states)), ["still", "moved,\nno withdr.", "withdrawal"], fontsize=8)
        a_.set_title(ev, fontsize=9)
        if k == 0:
            a_.set_ylabel(f"background change {d1.wtxt(ev).replace('z', '')}\n(SD of the session background)", fontsize=8)
    fig.suptitle("R10 field-wide brightness change at each delivery, by what the mouse did (all mice and sessions pooled; "
                 "red = median; Wilcoxon against 0)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(a.out, "R10_field_change.png"), dpi=130); plt.close(fig)
    print(S.round(3).to_string(index=False))
    # how much of the trial-to-trial field change does movement / withdrawal explain?
    import statsmodels.formula.api as smf
    X = D.dropna(subset=["bg_change_sd"]).copy()
    X["moved"] = X.moved.astype(int); X["withdrawal"] = X.withdrawal.astype(int)
    f = smf.ols("bg_change_sd ~ C(event) + moved + withdrawal + fast_motion_px + C(mouse) + C(phase)", X).fit()
    print(f.summary().tables[1])
    print("->", a.out)


if __name__ == "__main__":
    main()
