"""withdrawal_timing_check.py - are the hand-scored withdrawal marks late? (2026-10-23, user: "I checked but +- several sec
I did delay?")

The withdrawal marks (ScoringAB reflexEvents, type 1) are key presses, stored as pressed (lagMs = 250 in the files is applied
only to the machine-learning copy of the labels, not to reflexEvents). Three references that carry no human delay:

  motion onset   the first frame after the delivery where the paw / body motion of the video rises above the 1 s before the
                 delivery (behavior_auto <rid>_bfeatures.csv, native 30 fps: b_edge_me = motion at the mouse outline, s_me =
                 side-camera motion in the mouse box; robust z per session, mean of the two, onset = 2 frames > baseline + 3 SD)
  logger onset   Day 2 only: the pressure-logger onset of the delivery (contact)
  release        Day 2 only: the end of that press (|pressure - running median| back under the threshold): the paw left the probe
                 or the probe was lifted

Per withdrawal mark: the delivery before it (<= 3 s), the motion onset after that delivery, and mark - reference. Also the mean
motion around the marks (if the marks are late by d, the motion peak sits d before the mark).

Writes analysis\\withdrawal_timing\\: withdrawal_timing_events.csv, withdrawal_timing_sessions.csv, withdrawal_timing.png

  python withdrawal_timing_check.py
"""
import glob
import os

import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553"
OUT = os.path.join(B, "analysis", "withdrawal_timing")
FPS = 30.0
SC = {1: os.path.join(B, "scoring_mini1p", "20261019_final"), 2: os.path.join(B, "scoring_mini1p", "20261021_final")}
FEAT = {1: os.path.join(B, "behavior_auto", "20261019"), 2: os.path.join(B, "behavior_auto", "20261021")}
LOGD = os.path.join(B, "scoring_mini1p", "20261021_day2", "logger_deliveries")


def rz(x):
    x = np.asarray(x, float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) * 1.4826
    return (x - med) / (mad if mad > 0 else 1.0)


def motion(F):
    m = np.nanmean(np.vstack([rz(F.b_edge_me), rz(F.s_me)]), 0)
    m = pd.Series(m).rolling(3, center=True, min_periods=1).mean().to_numpy()      # 0.1 s
    return np.nan_to_num(m)


def onset_after(m, f, stop):
    """first frame >= f (0-based index of frame f) with 2 frames > mean + 3 SD of the 30 frames before f; None if none by stop"""
    b = m[max(0, f - 30):f]
    if len(b) < 10:
        return None
    thr = np.mean(b) + 3 * max(np.std(b), 0.25)
    for t in range(f, min(stop, len(m) - 1)):
        if m[t] > thr and m[t + 1] > thr:
            return t
    return None


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, trig = [], {1: [], 2: []}
    for d in (1, 2):
        for p in sorted(glob.glob(os.path.join(SC[d], "ScoringAB_*.mat"))):
            S = sio.loadmat(p, squeeze_me=True)
            mouse = str(S["mouseID"]).strip()
            ph = "pre" if str(S["phase"]).lower().startswith("base") else "post"
            rid = f"{mouse}_d{d}_{ph}"
            fp = os.path.join(FEAT[d], f"{rid}_bfeatures.csv")
            if not os.path.isfile(fp):
                print("no features:", rid); continue
            F = pd.read_csv(fp)
            m = motion(F)                                        # index i = camera frame i + 1
            re = np.atleast_2d(S["reflexEvents"]).astype(int).reshape(-1, 2)
            W = np.sort(re[re[:, 1] == 1, 0])
            dF = np.sort(np.atleast_1d(S["dFrames"]).astype(int))
            stim = [str(x) for x in np.atleast_1d(S["stimNames"])]
            dT = np.atleast_1d(S["dTypes"]).astype(int)[np.argsort(np.atleast_1d(S["dFrames"]).astype(int))]
            rel = {}
            if d == 2:
                L = pd.read_csv(os.path.join(LOGD, f"{rid}_logger_deliveries.csv"))
                rel = dict(zip(L.onset_cam_frame.astype(int), np.round(L.end_session_s * FPS).astype(int)))
            for w in W:
                k = np.flatnonzero(dF <= w)
                if not len(k):
                    continue
                f = dF[k[-1]]
                lat = (w - f) / FPS
                if lat > 3:
                    continue
                on = onset_after(m, f - 1, min(w - 1 + 15, f - 1 + 90))
                r = rel.get(int(f))
                st = stim[dT[k[-1]] - 1] if 1 <= dT[k[-1]] <= len(stim) else "?"
                rows.append(dict(day=d, mouse=mouse, phase=ph, rid=rid, stim=st, delivery_frame=int(f), mark_frame=int(w),
                                 mark_minus_delivery_s=lat,
                                 motion_onset_frame=(on + 1) if on is not None else np.nan,
                                 mark_minus_motion_s=((w - (on + 1)) / FPS) if on is not None else np.nan,
                                 release_frame=r if r is not None else np.nan,
                                 mark_minus_release_s=((w - r) / FPS) if r is not None else np.nan,
                                 syncOff=float(S.get("syncOff", 0) or 0)))
                seg = m[max(0, w - 1 - 90):w - 1 + 46]
                if len(seg) == 136:
                    trig[d].append(seg)
    E = pd.DataFrame(rows)
    E.to_csv(os.path.join(OUT, "withdrawal_timing_events.csv"), index=False)
    g = E.groupby(["day", "mouse", "phase"])
    Ssum = pd.DataFrame(dict(
        n_marks=g.size(),
        mark_minus_delivery_median_s=g.mark_minus_delivery_s.median(),
        motion_found_pct=g.mark_minus_motion_s.apply(lambda v: 100 * v.notna().mean()),
        mark_minus_motion_median_s=g.mark_minus_motion_s.median(),
        mark_minus_motion_p90_s=g.mark_minus_motion_s.quantile(0.9),
        late_over_1s_pct=g.mark_minus_motion_s.apply(lambda v: 100 * (v.dropna() > 1).mean() if v.notna().any() else np.nan),
        mark_minus_release_median_s=g.mark_minus_release_s.median())).reset_index()
    Ssum.to_csv(os.path.join(OUT, "withdrawal_timing_sessions.csv"), index=False)
    print(Ssum.round(2).to_string())
    for d in (1, 2):
        X = E[E.day == d]
        print(f"Day {d}: n = {len(X)}; mark - delivery median {X.mark_minus_delivery_s.median():.2f} s; mark - motion onset median "
              f"{X.mark_minus_motion_s.median():.2f} s (IQR {X.mark_minus_motion_s.quantile(.25):.2f}-{X.mark_minus_motion_s.quantile(.75):.2f}), "
              f"> 1 s: {100 * (X.mark_minus_motion_s.dropna() > 1).mean():.0f}%"
              + (f"; mark - release median {X.mark_minus_release_s.median():.2f} s" if d == 2 else ""))
    # figure
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    bins = np.arange(-1, 3.01, 0.1)
    for d, col in ((1, "#e67e22"), (2, "#2471a3")):
        X = E[E.day == d]
        ax[0].hist(X.mark_minus_delivery_s, bins=bins, color=col, alpha=0.5, label=f"Day {d} (n = {len(X)})")
        ax[1].hist(X.mark_minus_motion_s.dropna(), bins=bins, color=col, alpha=0.5, label=f"Day {d} (n = {X.mark_minus_motion_s.notna().sum()})")
        if trig[d]:
            T = np.vstack(trig[d]); tt = (np.arange(136) - 90) / FPS
            ax[2].plot(tt, T.mean(0), color=col, lw=2, label=f"Day {d}")
            ax[2].fill_between(tt, T.mean(0) - T.std(0) / np.sqrt(len(T)), T.mean(0) + T.std(0) / np.sqrt(len(T)), color=col, alpha=0.2, lw=0)
            pk = tt[np.argmax(T.mean(0)[:91])]
            ax[2].axvline(pk, color=col, ls=":", lw=1.2)
            ax[2].text(pk, ax[2].get_ylim()[1] if False else T.mean(0).max(), f" peak {pk:+.2f} s", color=col, fontsize=10, va="bottom")
    ax[0].set_xlabel("withdrawal mark - delivery (s)"); ax[0].set_title("A  mark after the delivery", loc="left")
    ax[1].set_xlabel("withdrawal mark - motion onset in the video (s)"); ax[1].set_title("B  mark after the movement starts", loc="left")
    ax[2].axvline(0, color="k", lw=1)
    ax[2].set_xlabel("time from the withdrawal mark (s)"); ax[2].set_ylabel("paw / body motion (robust z)")
    ax[2].set_title("C  mean motion around the marks", loc="left")
    for a in ax[:2]:
        a.axvline(0, color="k", lw=1); a.set_ylabel("withdrawal marks")
    for a in ax:
        a.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "withdrawal_timing.png"), dpi=140)
    plt.close(fig)
    print("->", OUT)


if __name__ == "__main__":
    main()
