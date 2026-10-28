"""withdrawal_kinematics.py  -  does SBI-553 blunt the sensation, or the muscle?

THE QUESTION THIS ANSWERS, AND WHY THE COUNTS CANNOT
    Every behaviour count fell on the drug day, and baseline locomotion was
    already halved before any stimulus. That pattern is what sedation looks
    like, so "fewer withdrawals" is ambiguous:

        analgesia          the animal feels less  -> it withdraws LESS OFTEN,
                           but when it does withdraw, the flick is normal
        motor suppression  the animal moves less  -> the flick itself is
                           smaller and slower, whether or not it felt anything

    A count cannot tell these apart. The KINEMATICS of the withdrawals that
    DID happen can, and that measurement has not been made. This is the same
    logic GrimACE used to catch opioid hyperactivity in its buprenorphine
    group: the pain score and the movement score are different axes, and you
    need both.

WHY NOT JUST USE THE EXISTING LOCOMOTION TRACE
    qc/locomotion/loco_raw_*.csv is sampled every 0.2 s. A paw withdrawal
    lasts 0.2-0.5 s, so that trace gives 1-2 samples per event and cannot
    resolve a rise time at all. This script decodes at the native
    30.00003 fps.

THE KEYPRESS PROBLEM, AND HOW IT IS AVOIDED
    Reflex events are human keypresses. ScoringAB stores lagMs = 250 but
    whether that correction was already applied to reflexEvents is not
    recorded unambiguously - reflex frame 9110 sits 2 frames (67 ms) after
    delivery frame 9108, which is too fast for a person to have watched a
    withdrawal and then pressed a key.

    So the keypress is used ONLY to place a search window, never as t = 0.
    Inside the window the motion ONSET is detected from the video itself, and
    every amplitude and slope is measured from that detected onset. Those
    numbers therefore carry no human delay at all.

    Latency is the one measure that still needs a reference, and it is taken
    from the DELIVERY frame - also a keypress. So absolute latency is not
    interpretable, but the Day 2 minus Day 1 DIFFERENCE is, because the same
    scorer pressed the same key on both days. The script prints it that way
    and never reports an absolute latency.

WHAT IT FOUND - THE ANSWER IS THAT THIS RIG CANNOT SUPPORT THE MEASUREMENT
    Run on M2 day 1 (99 scored withdrawals) the method returns numbers, and
    the numbers are not real. Three measurements say so:

    1  THE PERI-EVENT AVERAGE IS FLAT. Align every withdrawal to its
       keypress and the mean motion energy shows no peak at all
       (CHECK_M2_day1.png, left). 99 withdrawals should be visible in a
       signal that could measure them.

    2  DELIVERY WINDOWS ARE ONLY 1.05x QUIETER TIME. Median motion energy
       from 1.5 s before to 2.5 s after a delivery is 2.89, against 2.75
       inside the same stimulus blocks but more than 6 s from any delivery.
       A paw flick covers too little of a 720 x 480 frame to rise above the
       animal's own whole-body movement.

       This also refutes the obvious worry: the experimenter's hand does NOT
       dominate. It enters the frame at every delivery and moves it by 5 %.

    3  THE BACKLIGHT FLICKERS MORE THAN THE SIGNAL. Over one 6 s stretch the
       whole-frame mean brightness has sd 23.8 grey levels on a mean of
       61.1, peak-to-peak 66.5. Frame-to-frame the median change is 0.23 but
       the maximum is 13.84 - against a motion-energy median of 2.47. A
       uniform brightness step adds itself to |frame difference| everywhere,
       so single flicker steps are several times the whole signal. That is
       why the detected "onsets" pile up at 1-3 frames
       (CHECK_M2_day1.png, right): the detector is crossing flicker, not
       movement.

    SO THE NUMBERS BELOW ARE WRITTEN BUT MUST NOT BE READ AS KINEMATICS.
    They are kept so the negative result is reproducible.

WHAT WOULD BE NEEDED INSTEAD
    Paw-level tracking, which ignores both the hand and a global brightness
    step by construction. On this rig that is hard: the bottom camera looks
    UP THROUGH A HEXAGONAL MESH, the mouse occupies roughly 250 x 160 px of
    720 x 480, and a single paw is 10-20 px with mesh bars crossing it. The
    side camera is a backlit silhouette. Body-level pose is fine here and is
    already measured (measure_locomotion.py); paw keypoints are marginal.

    The mini2p rig is the opposite case - 1600 x 1200, lit from the side,
    face and paws both visible - so paw keypoints are feasible THERE. Which
    mirrors the step-1 result: the dataset with the question has the worse
    footage.

    For a next cohort: drop the backlight during delivery or add a
    front-lit camera, and the measurement becomes routine.

WHAT IS MEASURED, per withdrawal
    peak         maximum motion energy in the window, minus the local floor
    rise_s       detected onset -> peak
    slope        peak / rise_s, the speed of the flick
    dur_s        time above half of peak
    lat_s        delivery keypress -> detected onset  (differences only)

OUTPUT  ->  qc/kinematics/
    motion30_<day>_<mouse>.npy    the 30 fps motion trace, cached
    Withdrawal_events.csv         one row per withdrawal
    Withdrawal_summary.csv        per mouse per day
    fig_withdrawal_kinematics.png

USAGE
    python withdrawal_kinematics.py --mouse M2 --day 1 --check   # validate
    python withdrawal_kinematics.py                              # all 12

Hansol Lim - HEAL mini1p / SBI-553
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import cv2
import matplotlib
import numpy as np
import pandas as pd
import scipy.io as sio

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

ROOT = r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553"
VID = {1: os.path.join(ROOT, "videos", "cameraA"),
       2: os.path.join(ROOT, "videos", "day2", "bottom")}
SCO = {1: os.path.join(ROOT, "videos", "output_corrected"),
       2: os.path.join(ROOT, "videos", "output_day2_corrected")}
OUT = os.path.join(ROOT, "qc", "kinematics")

# The bottom camera looks up through the mesh, so a paw leaving the floor is
# the brightest change in the frame. Motion energy is taken over the whole
# frame minus a border: there is no backdrop to exclude from below, and the
# experimenter's hand enters from above only in the side view.
BORDER = 30
WIN = (-1.0, 2.0)       # search window around the keypress, seconds
FLOOR_W = 0.5           # local floor = median of this much time before onset
ONSET_K = 3.0           # onset = first crossing of floor + K x floor MAD
WD, FL = 1, 2           # reflexEvents type codes
UP, DOWN, INK = "#C1272D", "#1C6E8C", "#1A1A1A"
plt.rcParams.update({"font.size": 11, "axes.linewidth": 1.2,
                     "axes.titlesize": 12, "axes.labelsize": 11.5})


def mouse_id(name):
    m = re.match(r"(female|male)\s*(\d+)", os.path.basename(name).lower())
    return ("F" if m.group(1) == "female" else "M") + m.group(2)


def norm(s):
    """Alphanumerics only.

    The mat name is built from the video name by DELETING spaces and turning
    hyphens into underscores:
        "male2bottom0005 26-09-02 14-06-31-004"
        -> "ScoringAB_male2bottom000526_09_0214_06_31_004.mat"
    so mapping every non-alphanumeric to "_" does not reproduce it. Dropping
    the separators from both sides does.
    """
    return re.sub(r"[^0-9a-zA-Z]", "", s).lower()


def sessions(day):
    """(mouse, bottom video, ScoringAB mat) per animal for one day."""
    out = []
    mats = glob.glob(os.path.join(SCO[day], "ScoringAB_*.mat"))
    for v in sorted(glob.glob(os.path.join(VID[day], "*.avi"))):
        stem = os.path.splitext(os.path.basename(v))[0]
        hits = [m for m in mats
                if norm(os.path.basename(m)[10:-4]) == norm(stem)]
        if hits:
            out.append((mouse_id(stem), v, hits[0]))
        else:
            print(f"  no ScoringAB for {stem}")
    return out


def motion_trace(video, cache):
    """Mean |frame difference| per frame, at the native rate. Cached.

    Decoded sequentially rather than by seeking: DV is all-intra so seeking
    is exact, but 200+ seeks into a 6 GB file per session costs more than one
    pass, and one pass also gives the whole trace for later steps.
    """
    if os.path.exists(cache):
        return np.load(cache)
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    prev, vals = None, []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)[BORDER:-BORDER,
                                                 BORDER:-BORDER]
        g = cv2.resize(g, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
        vals.append(0.0 if prev is None
                    else float(np.abs(g.astype(np.int16)
                                      - prev.astype(np.int16)).mean()))
        prev = g
    cap.release()
    m = np.asarray(vals, np.float32)
    np.save(cache, m)
    return m


def kinematics(m, fps, kp_frame, del_frame):
    """One withdrawal, measured from a video-detected onset.

    Returns None when no onset is found in the window - that is a real
    outcome (the keypress does not correspond to a visible movement) and it
    is counted rather than silently dropped.
    """
    a = int(round(kp_frame + WIN[0] * fps))
    b = int(round(kp_frame + WIN[1] * fps))
    a, b = max(a, 1), min(b, len(m) - 1)
    if b - a < int(fps * .5):
        return None
    w = m[a:b]
    nf = int(FLOOR_W * fps)
    base = w[:nf]
    if len(base) < 3:
        return None
    floor = float(np.median(base))
    mad = float(np.median(np.abs(base - floor))) + 1e-9
    thr = floor + ONSET_K * 1.4826 * mad
    over = np.nonzero(w > thr)[0]
    if not len(over):
        return None
    i0 = int(over[0])
    ip = int(i0 + np.argmax(w[i0:]))
    peak = float(w[ip] - floor)
    if peak <= 0:
        return None
    rise = max((ip - i0) / fps, 1.0 / fps)
    half = floor + peak / 2.0
    j = ip
    while j < len(w) - 1 and w[j] > half:
        j += 1
    return dict(peak=peak, rise_s=rise, slope=peak / rise,
                dur_s=(j - i0) / fps,
                lat_s=(a + i0 - del_frame) / fps,
                floor=floor)


def one_session(mouse, video, mat, day, check=False):
    s = sio.loadmat(mat, squeeze_me=True, struct_as_record=False)
    fps = float(s["frameRate"])
    re_ev = np.asarray(s["reflexEvents"], float)
    dfr = np.asarray(s["dFrames"], float)
    cache = os.path.join(OUT, f"motion30_{day}_{mouse}.npy")
    m = motion_trace(video, cache)
    print(f"  {mouse} day{day}: {len(m)} frames at {fps:.3f} fps, "
          f"{int((re_ev[:, 1] == WD).sum())} withdrawals")

    rows, miss = [], 0
    for fr, kind in re_ev:
        if int(kind) != WD:
            continue
        prev = dfr[dfr <= fr]
        if not len(prev):
            continue
        k = kinematics(m, fps, fr, prev[-1])
        if k is None:
            miss += 1
            continue
        k.update(mouse=mouse, day=day, kp_frame=int(fr),
                 del_frame=int(prev[-1]))
        rows.append(k)
    print(f"      {len(rows)} with a detected onset, {miss} without")

    if check and rows:
        lags = np.arange(WIN[0], WIN[1], 1 / fps)
        W = []
        for r in rows:
            a = int(round(r["kp_frame"] + WIN[0] * fps))
            seg = m[a:a + len(lags)]
            if len(seg) == len(lags):
                W.append(seg - r["floor"])
        W = np.stack(W)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(lags, W.T, color="#BBBBBB", lw=.4)
        ax[0].plot(lags, W.mean(0), color=UP, lw=2.5)
        ax[0].axvline(0, color=INK, lw=1.4)
        ax[0].set_xlabel("time from the withdrawal KEYPRESS (s)")
        ax[0].set_ylabel("motion energy - local floor")
        ax[0].set_title(f"{mouse} day{day}: {len(W)} withdrawals\n"
                        f"grey = single events, red = mean")
        ax[1].hist([r["rise_s"] for r in rows], bins=20, color=UP)
        ax[1].set_xlabel("rise time, detected onset -> peak (s)")
        ax[1].set_ylabel("withdrawals")
        ax[1].set_title("is the flick resolved at 30 fps?\n"
                        "a single-sample rise means it is not")
        for A in ax:
            A.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        p = os.path.join(OUT, f"CHECK_{mouse}_day{day}.png")
        fig.savefig(p, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"      wrote {p}")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mouse", default=None)
    ap.add_argument("--day", type=int, default=None, choices=[1, 2])
    ap.add_argument("--check", action="store_true",
                    help="write the per-session validation figure")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    rows = []
    for day in ([a.day] if a.day else [1, 2]):
        for mouse, video, mat in sessions(day):
            if a.mouse and mouse != a.mouse:
                continue
            rows += one_session(mouse, video, mat, day, a.check)
    if not rows:
        raise SystemExit("no withdrawals measured")
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(OUT, "Withdrawal_events.csv"), index=False)
    print(f"\nwrote {len(D)} events -> Withdrawal_events.csv")

    g = D.groupby(["mouse", "day"]).agg(
        n=("peak", "size"), peak=("peak", "median"),
        rise_s=("rise_s", "median"), slope=("slope", "median"),
        dur_s=("dur_s", "median"), lat_s=("lat_s", "median")).reset_index()
    g.to_csv(os.path.join(OUT, "Withdrawal_summary.csv"), index=False)
    print(g.to_string(index=False))
    if a.day is None and a.mouse is None:
        report(g)
    return g


def report(g):
    """Paired within animal, n = 6, exact Wilcoxon - floor p = 0.031."""
    L = ["", "===== withdrawal kinematics, Day 2 (SBI-553) vs Day 1 "
             "(vehicle) =====",
         "paired Wilcoxon across the 6 mice; with n = 6 the exact test "
         "cannot go below p = 0.031", ""]
    piv = {}
    for col, what in (("n", "number of withdrawals"),
                      ("peak", "peak motion energy"),
                      ("slope", "flick speed (peak / rise)"),
                      ("rise_s", "rise time"),
                      ("dur_s", "movement duration")):
        w = g.pivot(index="mouse", columns="day", values=col).dropna()
        if len(w) < 3:
            continue
        piv[col] = w
        ratio = (w[2] / w[1]).median()
        try:
            _, p = wilcoxon(w[1], w[2])
        except ValueError:
            p = np.nan
        L.append(f"  {what:<28s} median Day2/Day1 = {ratio:5.2f}   "
                 f"p = {p:.3f}   ({int(len(w))} mice)")
    if "lat_s" in g.columns:
        w = g.pivot(index="mouse", columns="day", values="lat_s").dropna()
        if len(w) >= 3:
            _, p = wilcoxon(w[1], w[2])
            L += ["", f"  latency CHANGE (Day2 - Day1) = "
                      f"{(w[2] - w[1]).median():+.3f} s, p = {p:.3f}",
                  "    absolute latency is not reported: it is measured from "
                  "a keypress and carries the",
                  "    scorer's reaction time. The between-day difference "
                  "does not."]
    L += ["", "HOW TO READ THIS",
          "  count falls, kinematics HOLD   -> consistent with analgesia: "
          "fewer withdrawals, normal flicks",
          "  count falls, kinematics FALL   -> motor suppression is doing at "
          "least part of the work",
          "  Neither is proof. Day 1 was always vehicle and Day 2 always "
          "drug, so order and day",
          "  are confounded with treatment in this design."]
    txt = "\n".join(L)
    with open(os.path.join(OUT, "Withdrawal_report.txt"), "w",
              encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)
    figure(piv)


def figure(piv):
    ks = [k for k in ("n", "peak", "slope", "rise_s", "dur_s") if k in piv]
    if not ks:
        return
    ttl = {"n": "number of withdrawals", "peak": "peak motion energy",
           "slope": "flick speed  (peak / rise)", "rise_s": "rise time (s)",
           "dur_s": "movement duration (s)"}
    fig, ax = plt.subplots(1, len(ks), figsize=(3.3 * len(ks), 4.4))
    ax = np.atleast_1d(ax)
    for A, k in zip(ax, ks):
        w = piv[k]
        for mo in w.index:
            A.plot([0, 1], [w.loc[mo, 1], w.loc[mo, 2]], color="#BBBBBB",
                   lw=1.3, marker="o", ms=5, mfc="white", mec=INK, zorder=2)
        A.plot([0, 1], [w[1].median(), w[2].median()], color=UP, lw=3,
               zorder=3)
        try:
            _, p = wilcoxon(w[1], w[2])
        except ValueError:
            p = np.nan
        A.set_xticks([0, 1])
        A.set_xticklabels(["Day 1\nvehicle", "Day 2\nSBI-553"])
        A.set_xlim(-.35, 1.35)
        A.set_title(f"{ttl[k]}\nDay2/Day1 = {(w[2] / w[1]).median():.2f},  "
                    f"p = {p:.3f}")
        A.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Does the drug reduce how OFTEN the paw is withdrawn, or "
                 "how the withdrawal MOVES?\n"
                 "one line per mouse, red = median; amplitude and speed are "
                 "measured from a video-detected onset, so they carry no "
                 "keypress delay", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, .88])
    p = os.path.join(OUT, "fig_withdrawal_kinematics.png")
    fig.savefig(p, dpi=135, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
