"""R4  ARE THE 'DOWN' RESPONSES CALCIUM? When a neuron fires less, its GCaMP fluorescence can only
fall with the indicator decay: for a linear indicator any step change of the firing rate is
approached as 1 - exp(-t / tau), whatever its size. So the speed of a decrease has a physical
limit, while an optical change (focus / z motion, a brightness change of the background that the
demixing removes from the cell) can be a step.
  speed = drop reached at [0.1, 0.3] s / drop reached at [0.8, 1.2] s   (both against z[-1.5, -0.5])
  calcium limit with the cell's tau (~1.3 s): 0.27 if the change starts at the event, 0.44 if it
  started 0.2 s before (late key press), 0.53 for 0.3 s before; an instantaneous step gives ~1.
  'too fast for calcium' = speed > 0.7 and no drop yet in [-0.5, 0] s (so not an earlier start)
Also: the event-locked background (mean of the pixels outside every footprint, ca_background_signal)
- if the background brightens at the event, a cell trace can go negative where the background
model over-subtracts; if it darkens, the whole FOV changed (focus / z).
Outputs next to R3 (analysis/ca_response_shape): R4_down_speed_cells.csv, R4_background_psth.csv,
R4_down_check.png

  python ca_down_check.py --out-root <mini1p_out/20261019> --scoring-dir <final> --day1 <analysis/ca_day1>
      --out <analysis/ca_response_shape>
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
EVS = d1.STIM + ["withdrawal"]
TOO_FAST = 0.7


def wmean(M, tt, w0, w1):
    return np.nanmean(M[(tt >= w0) & (tt < w1)], 0)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--scoring-dir", required=True)
    ap.add_argument("--day1", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--traces", default="traces", help="traces | traces_bgcorr | traces_roi; outputs get a suffix if not 'traces'")
    a = ap.parse_args(argv)
    d1.TRACES = a.traces
    SUF = "" if a.traces == "traces" else "_" + a.traces
    os.makedirs(a.out, exist_ok=True)
    ET = pd.read_csv(os.path.join(a.day1, f"ca_d{d1.DAY}_cells_event_tests.csv"))
    ET["mouse"] = ET.mouse.astype(str)
    rows, bg_rows, keep_tr = [], [], {}
    for m, arm, sex in d1.MICE:
        for ph in ("pre", "post"):
            s = d1.load_session(a.out_root, m, ph, a.scoring_dir)
            if s is None or s["B"] is None:
                continue
            T = s["T"]
            ids = T["cell_ids"].astype(int)
            tau = dict(zip(ids, np.asarray(T["tau_s"], float)))
            fb = os.path.join(a.out_root, f"{m}_d{d1.DAY}", "5_analysis", "traces", f"{m}_d{d1.DAY}_{ph}_background.npz")
            bg = None
            if os.path.isfile(fb):
                Bz = np.load(fb)
                bmap = pd.Series(Bz["b_frac"].astype(float), index=Bz["frame"].astype(int))
                bg = bmap.reindex(T["frame"].astype(int)).values
            for ev in EVS:
                X, tt, used = d1.peri(s, s["B"]["ev"].get(ev, []))
                if X is None or len(used) < d1.MIN_TRIALS:
                    continue
                M = d1.sm(np.nanmean(X, 0), 3)                       # trial mean, 0.1-s smoothing
                base = wmean(M, tt, -1.5, -0.5)
                M = M - base
                early, late, pre = wmean(M, tt, 0.1, 0.3), wmean(M, tt, 0.8, 1.2), wmean(M, tt, -0.5, 0.0)
                E = ET[(ET.mouse == m) & (ET.phase == ph) & (ET.event == ev)].set_index("cell")
                for j, c in enumerate(ids):
                    if c not in E.index or E.at[c, "cls_shift"] not in ("UP", "DOWN"):
                        continue
                    cl = E.at[c, "cls_shift"]
                    sp = early[j] / late[j] if abs(late[j]) > 1e-6 else np.nan
                    no_pre = (pre[j] > 0.25 * late[j]) if cl == "DOWN" else (pre[j] < 0.25 * late[j])
                    rows.append(dict(mouse=m, arm=arm, phase=ph, event=ev, cell=int(c), cls=cl, tau_s=tau.get(c, np.nan),
                                     drop_early=early[j], drop_1s=late[j], pre_change=pre[j], speed=sp,
                                     too_fast=bool(cl == "DOWN" and np.isfinite(sp) and sp > TOO_FAST and no_pre and late[j] < 0)))
                    keep_tr.setdefault((cl, ev), []).append(M[:, j] / (abs(late[j]) if abs(late[j]) > 1e-6 else np.nan))
                if bg is not None:
                    fr = T["frame"].astype(int)
                    k0, k1 = int(d1.WIN[0] * FPS), int(d1.WIN[1] * FPS)
                    W = []
                    for f in used:
                        i = np.searchsorted(fr, f)
                        if i + k0 >= 0 and i + k1 < len(fr):
                            W.append(bg[i + k0:i + k1 + 1])
                    if W:
                        Bm = np.nanmean(np.vstack(W), 0)
                        Bm = Bm - np.nanmean(Bm[(tt >= -1.5) & (tt < -0.5)])
                        sd = np.nanstd(bg[np.isfinite(bg)])
                        bg_rows.append(dict(mouse=m, arm=arm, phase=ph, event=ev, n_trials=len(W),
                                            bg_change_0_2s=float(np.nanmean(Bm[(tt >= 0) & (tt < 2)])),
                                            bg_change_in_sd=float(np.nanmean(Bm[(tt >= 0) & (tt < 2)]) / sd) if sd > 0 else np.nan))
                        keep_tr.setdefault(("bg", ev), []).append(Bm / sd if sd > 0 else Bm)
    D = pd.DataFrame(rows)
    D.round(4).to_csv(os.path.join(a.out, f"R4_down_speed_cells{SUF}.csv"), index=False)
    BG = pd.DataFrame(bg_rows)
    BG.round(5).to_csv(os.path.join(a.out, f"R4_background_psth{SUF}.csv"), index=False)
    tt = np.arange(int(d1.WIN[0] * FPS), int(d1.WIN[1] * FPS) + 1) / FPS
    fig, ax = plt.subplots(2, len(EVS), figsize=(3.3 * len(EVS), 6.2), sharex=True)
    lim = 1 - np.exp(-tt.clip(0) / 1.27)
    for k, ev in enumerate(EVS):
        a_ = ax[0, k]
        for cl, col in (("UP", "#c0392b"), ("DOWN", "#2471a3")):
            L = keep_tr.get((cl, ev), [])
            if L:
                Y = np.vstack(L)
                a_.plot(tt, np.nanmedian(Y, 0), color=col, lw=1.3, label=f"{cl} (n={len(Y)})")
        ref = lim / (1 - np.exp(-1.0 / 1.27))
        a_.plot(tt, -ref, color="k", ls="--", lw=0.9, label="calcium limit for a\ndecrease (tau 1.27 s)")
        a_.axvline(0, color="k", lw=0.6); a_.axhline(0, color="0.5", lw=0.5)
        a_.set_xlim(-2, 4); a_.set_ylim(-2, 2)
        a_.set_title(ev, fontsize=9)
        if k == 0:
            a_.set_ylabel("trial-mean z / |change at 1 s|\n(median over cell-sessions)", fontsize=8)
        a_.legend(fontsize=6.5, frameon=False, loc="upper left")
        a_ = ax[1, k]
        L = keep_tr.get(("bg", ev), [])
        if L:
            Y = np.vstack(L)
            a_.plot(tt, d1.sm(Y.T, 6).T.mean(0), color="0.2", lw=1.2, label=f"background (n={len(Y)} sessions)")
            for y in d1.sm(Y.T, 6).T:
                a_.plot(tt, y, color="0.6", lw=0.5, alpha=0.6)
        a_.axvline(0, color="k", lw=0.6); a_.axhline(0, color="0.5", lw=0.5)
        a_.set_xlim(-2, 4)
        a_.set_xlabel("time from event (s)", fontsize=8)
        if k == 0:
            a_.set_ylabel("background outside footprints\n(change, SD of the session)", fontsize=8)
        a_.legend(fontsize=6.5, frameon=False, loc="upper left")
    dn = D[D.cls == "DOWN"]
    fig.suptitle(f"R4 speed of the DOWN responses: {int(dn.too_fast.sum())}/{len(dn)} DOWN cell-sessions fall faster than calcium can "
                 f"(drop at 0.1-0.3 s / drop at 1 s > {TOO_FAST}, no earlier start; the GCaMP limit is 0.27-0.53).\n"
                 "Top: responses scaled to their change at 1 s; dashed = fastest possible calcium decrease. Bottom: the background "
                 "(pixels outside every footprint) at the same events.", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.55 / fig.get_figheight()))
    fig.savefig(os.path.join(a.out, f"R4_down_check{SUF}.png"), dpi=130); plt.close(fig)
    print(D.groupby(["cls"]).speed.describe().round(2).to_string())
    print("DOWN too fast:", int(dn.too_fast.sum()), "/", len(dn))
    print(dn.groupby(["mouse", "event"]).too_fast.agg(["sum", "count"]).to_string())
    print(BG.groupby("event").bg_change_in_sd.describe()[["mean", "min", "max"]].round(3).to_string())
    print("->", a.out)


if __name__ == "__main__":
    main()
