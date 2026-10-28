"""R3  WHAT 'UP' AND 'DOWN' RESPONSES LOOK LIKE. The response r = z[0, 2] - z[-2, 0] is a
difference, so 'DOWN' can be a drop after the delivery OR a rise just BEFORE it (the experimenter's
approach, the mouse moving) that returns afterwards - and 'UP' can partly start before the delivery.
For every responsive cell (time-shift rule, q <= 0.20; analysis/ca_day1) the trial-mean z in four
windows, against an earlier baseline that the response definition does not use:
  far [-5, -3] s   pre [-2, 0] s   post [0, 2] s   late [2, 4] s
  UP    'after delivery'  post - far > 0 and the pre-window rise < half of it
        'starts before'   the pre-window rise >= half of the post-window change
        'dip before'      post not above far: the pre window dipped and the delivery ends the dip
  DOWN  'drop after'      post below the far baseline (a real decrease after the delivery)
        'rise before'     pre above far and post not below it (the delivery ends a pre-delivery rise)
Also the fraction of trials in which the mouse moved in [-2, 0] s before the delivery.

  python ca_response_shape.py --out-root <mini1p_out/20261019> --scoring-dir <final>
      --day1 <analysis/ca_day1> --out <analysis/ca_response_shape>
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ca_day1_analysis as d1  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FPS = d1.FPS
EVS = d1.STIM + ["withdrawal"]
W = {"far": (-5.0, -3.0), "pre": (-2.0, 0.0), "post": (0.0, 2.0), "late": (2.0, 4.0)}
CLC = {"after delivery": "#c0392b", "starts before": "#e59866", "dip before": "#a04000",
       "drop after": "#2471a3", "rise before": "#7fb3d5"}


def classify(cls, d_pre, d_post):
    if cls == "UP":
        if d_post <= 0:
            return "dip before"          # the pre window dips below the far baseline and comes back
        return "starts before" if d_pre >= 0.5 * d_post else "after delivery"
    return "drop after" if d_post < 0 else "rise before"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--scoring-dir", required=True)
    ap.add_argument("--day1", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    ET = pd.read_csv(os.path.join(a.day1, f"ca_d{d1.DAY}_cells_event_tests.csv"))
    ET["mouse"] = ET.mouse.astype(str)
    rows, traces, mv_rows = [], {}, []
    for m, arm, sex in d1.MICE:
        for ph in ("pre", "post"):
            s = d1.load_session(a.out_root, m, ph, a.scoring_dir)
            if s is None or s["B"] is None:
                continue
            ids = s["T"]["cell_ids"].astype(int)
            move = s["B"]["st"]["move"]
            for ev in EVS:
                X, tt, used = d1.peri(s, s["B"]["ev"].get(ev, []))
                if X is None or len(used) < d1.MIN_TRIALS:
                    continue
                f = np.asarray(used, int)
                mv = np.mean([move[max(0, x - int(2 * FPS)):x].any() if x < len(move) else False for x in f])
                mv_rows.append(dict(mouse=m, arm=arm, phase=ph, event=ev, n_trials=len(f), frac_moving_before=mv))
                M = np.nanmean(X, 0)                                        # time x cells, trial mean
                win = {k: np.nanmean(M[(tt >= w0) & (tt < w1)], 0) for k, (w0, w1) in W.items()}
                far = win["far"]
                E = ET[(ET.mouse == m) & (ET.phase == ph) & (ET.event == ev)].set_index("cell")
                for j, c in enumerate(ids):
                    if c not in E.index:
                        continue
                    cl = E.at[c, "cls_shift"]
                    if cl not in ("UP", "DOWN"):
                        continue
                    sub = classify(cl, win["pre"][j] - far[j], win["post"][j] - far[j])
                    rows.append(dict(mouse=m, arm=arm, phase=ph, event=ev, cell=int(c), cls=cl, shape=sub,
                                     d_pre=win["pre"][j] - far[j], d_post=win["post"][j] - far[j],
                                     d_late=win["late"][j] - far[j]))
                    traces.setdefault((cl, ev, sub), []).append(M[:, j] - far[j])
    D = pd.DataFrame(rows)
    D.round(4).to_csv(os.path.join(a.out, "R3_response_shape_cells.csv"), index=False)
    C = D.groupby(["mouse", "arm", "phase", "event", "cls", "shape"]).size().rename("n_cells").reset_index()
    C.to_csv(os.path.join(a.out, "R3_response_shape_counts.csv"), index=False)
    MV = pd.DataFrame(mv_rows)
    MV.round(3).to_csv(os.path.join(a.out, "R3_moving_before_delivery.csv"), index=False)
    tt = np.arange(int(d1.WIN[0] * FPS), int(d1.WIN[1] * FPS) + 1) / FPS
    fig, ax = plt.subplots(2, len(EVS), figsize=(3.3 * len(EVS), 6.6), sharex=True)
    for i, cl in enumerate(("UP", "DOWN")):
        for k, ev in enumerate(EVS):
            a_ = ax[i, k]
            for sub in (("after delivery", "starts before", "dip before") if cl == "UP" else ("drop after", "rise before")):
                L = traces.get((cl, ev, sub), [])
                if not L:
                    continue
                Y = np.vstack(L)
                mu = d1.sm(np.nanmean(Y, 0)[:, None])[:, 0]
                se = np.nanstd(Y, 0) / np.sqrt(len(Y))
                a_.plot(tt, mu, color=CLC[sub], lw=1.3, label=f"{sub} (n={len(Y)})")
                a_.fill_between(tt, mu - se, mu + se, color=CLC[sub], alpha=0.25, lw=0)
            for w0, w1 in (W["far"], W["pre"]):
                a_.axvspan(w0, w1, color="0.92", lw=0, zorder=0)
            a_.axvline(0, color="k", lw=0.7)
            a_.axhline(0, color="0.5", lw=0.5)
            a_.set_title(f"{ev}: {cl}", fontsize=9)
            a_.legend(fontsize=6.5, frameon=False, loc="upper left")
            if k == 0:
                a_.set_ylabel("trial-mean z - z[-5,-3] s\n(mean +- SEM over cell-sessions)", fontsize=8)
            if i == 1:
                a_.set_xlabel("time from event (s)", fontsize=8)
    n_mv = MV.groupby("event").frac_moving_before.mean()
    fig.suptitle("R3 shape of the responses called UP / DOWN (time-shift rule, q <= 0.20), all mice and both sessions pooled. "
                 "Grey: the far baseline [-5,-3] s and the response's own baseline [-2,0] s.\n"
                 "Mouse moving in the 2 s before the delivery: "
                 + ", ".join(f"{e} {100 * n_mv.get(e, np.nan):.0f}%" for e in EVS) + " of trials (mean over sessions)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(os.path.join(a.out, "R3_response_shape.png"), dpi=130); plt.close(fig)
    tot = D.groupby(["cls", "shape"]).size()
    print(tot.to_string())
    print(MV.groupby("event").frac_moving_before.mean().round(2).to_string())
    print("response shape ->", a.out)


if __name__ == "__main__":
    main()
