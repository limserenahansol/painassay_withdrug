"""camera_drop_locator.py - where did a camera drop frames? (side vs bottom of one session)

Both cameras are triggered by the same SYNC pulses, so normally camera frame n of the side view and
of the bottom view are the same moment. When one file is shorter (Day 2, 5941 post: side 29763,
bottom 29781 = logger pulses) it lost frames somewhere. The mouse's motion is seen by both cameras:
the per-frame motion energy (mean |frame - previous frame|, 4x downsampled grey) of the two views is
cross-correlated in sliding 60-s windows; the lag that maximises the correlation (side behind bottom
in frames) steps up where frames were dropped.

  python camera_drop_locator.py --bottom <avi> --side <avi> --out <png> [--win-s 60 --step-s 15]
"""
import argparse

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def energy(path):
    c = cv2.VideoCapture(path)
    prev, e = None, []
    while True:
        ok, fr = c.read()
        if not ok:
            break
        g = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA).astype(np.float32)
        e.append(0.0 if prev is None else float(np.mean(np.abs(g - prev))))
        prev = g
    return np.asarray(e)


def lag_curve(eb, es, fps, win_s, step_s, max_lag):
    w, st = int(win_s * fps), int(step_s * fps)
    out = []
    for s0 in range(0, min(len(eb), len(es)) - w - max_lag, st):
        b = eb[s0:s0 + w]; b = (b - b.mean()) / (b.std() + 1e-9)
        best, bl = -np.inf, 0
        for L in range(-max_lag, max_lag + 1):
            if s0 + L < 0:
                continue
            s = es[s0 + L:s0 + L + w]
            if len(s) < w:
                continue
            s = (s - s.mean()) / (s.std() + 1e-9)
            r = float(np.mean(b * s))
            if r > best:
                best, bl = r, L
        out.append((s0 / fps, -bl, best))           # side frame k shows bottom frame k + dropped
    return np.asarray(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bottom", required=True)
    ap.add_argument("--side", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--win-s", type=float, default=60.0)
    ap.add_argument("--step-s", type=float, default=15.0)
    ap.add_argument("--max-lag", type=int, default=40)
    a = ap.parse_args(argv)
    eb, es = energy(a.bottom), energy(a.side)
    L = lag_curve(eb, es, a.fps, a.win_s, a.step_s, a.max_lag)
    np.savetxt(a.out.replace(".png", ".csv"), L, delimiter=",", header="window_start_s,side_frames_dropped_before,r", fmt="%.3f")
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax[0].plot(L[:, 0], L[:, 1], "o-", ms=3)
    ax[0].set_ylabel("frames dropped by the side camera\n(lag of side behind bottom)")
    ax[0].set_title(f"bottom {len(eb)} frames, side {len(es)} frames: difference {len(eb) - len(es)}")
    ax[1].plot(L[:, 0], L[:, 2], "o-", ms=3, color="k"); ax[1].set_ylabel("motion-energy r at that lag")
    ax[1].set_xlabel("window start (s, bottom-camera time)")
    fig.tight_layout(); fig.savefig(a.out, dpi=100)
    print(f"bottom {len(eb)}, side {len(es)}; lag (frames) from {L[0, 1]:.0f} to {L[-1, 1]:.0f}; "
          f"median r {np.median(L[:, 2]):.2f}")
    steps = np.flatnonzero(np.diff(L[:, 1]) != 0)
    for i in steps:
        print(f"  lag {L[i, 1]:.0f} -> {L[i + 1, 1]:.0f} between {L[i, 0]:.0f} and {L[i + 1, 0] + a.win_s:.0f} s (r {L[i + 1, 2]:.2f})")


if __name__ == "__main__":
    main()
