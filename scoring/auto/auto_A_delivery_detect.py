"""auto_A_delivery_detect.py  -  automatic stimulus-delivery detection (camera A).

Replaces the 28 minutes of watching in score_A_stimulus_delivery.m. The
experimenter's hand and the applicator are a large, high-contrast intruder in
the backlit bottom view, so intrusion events are detectable at robust z ~ 20.

WHAT THIS DOES AND DOES NOT DO
    does      finds every candidate delivery event and its contact frame
    does NOT  decide WHICH of the four stimuli it was

    The four applicators look the same from below at 720x480, so the type has
    to come from a human, and the type is what the normalisation denominator
    needs. So this script is NOT a replacement for manual scoring.

    Use it as a CROSS-CHECK on score_AB_dual_view.m: after you have scored a
    session by hand, compare the number of deliveries you marked against the
    number of candidates found here. A mismatch means you either missed a
    delivery or marked one twice, and the Candidates CSV tells you the
    timestamp to go back and look at.

    It is also the starting point for full automation in the next cohort,
    where the delivery ORDER can be read off the randomisation sheet and
    matched to these timestamps.

USAGE
    python auto_A_delivery_detect.py <bottom_video> [--out DIR] [--session N]
    python auto_A_delivery_detect.py <folder_of_bottom_videos> --out DIR

OUTPUT  (into <out>/, default = <video folder>/output)
    Candidates_<vid>.csv    event, onset/peak/offset frame + time, intrusion z
    Candidates_<vid>.png    the intrusion trace with detected events marked

Hansol Lim - HEAL mini1p / SBI-553
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import cv2
import numpy as np

# ── detection settings ──────────────────────────────────────────────────────
DARK_THR = 45        # a pixel below this is "dark"; from the WT QC (mouse ~0-40)
STEP = 3             # analyse every Nth frame (30 fps -> 10 Hz is plenty)
Z_ON = 6.0           # robust z to open an event
Z_OFF = 3.0          # robust z to close it (hysteresis, avoids flicker)
MIN_EVENT_S = 0.30   # ignore blips shorter than this
MERGE_GAP_S = 1.00   # events closer than this are one delivery
MAX_EVENT_S = 25.0   # longer than this is not a delivery (hand left in frame)


def intrusion_trace(path: str, step: int = STEP, dark_thr: int = DARK_THR):
    """Dark-pixel area per sampled frame. The mouse contributes a near-constant
    area (QC: CV 0.22), so a hand entering the frame shows up as a large step."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    kern = np.ones((7, 7), np.uint8)
    area, frames = [], []
    i = 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        if i % step == 0:
            g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            m = (g < dark_thr).astype(np.uint8)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kern)
            area.append(int(m.sum()))
            frames.append(i)
        i += 1
        if i % 9000 == 0:
            print(f"    {i}/{n} frames", end="\r", flush=True)
    cap.release()
    print(f"    {i}/{n} frames read      ")
    return np.asarray(frames), np.asarray(area, float), fps, i


def robust_z(v: np.ndarray) -> np.ndarray:
    base = np.median(v)
    mad = np.median(np.abs(v - base))
    scale = 1.4826 * mad
    if scale < 1e-9:                     # degenerate: fall back to SD
        scale = v.std() + 1e-9
    return (v - base) / scale


def find_events(frames, z, fps, step):
    """Hysteresis threshold -> merge -> length filter. Contact frame = peak z."""
    on = z >= Z_ON
    off = z < Z_OFF
    segs, i, N = [], 0, len(z)
    while i < N:
        if not on[i]:
            i += 1
            continue
        s = i
        while i < N and not off[i]:
            i += 1
            continue
        segs.append((s, min(i, N - 1)))
    if not segs:
        return []

    # merge segments separated by a short gap
    merge_gap = MERGE_GAP_S * fps / step
    merged = [list(segs[0])]
    for s, e in segs[1:]:
        if s - merged[-1][1] <= merge_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    ev = []
    min_len = max(1, int(MIN_EVENT_S * fps / step))
    max_len = MAX_EVENT_S * fps / step
    for s, e in merged:
        if (e - s + 1) < min_len:
            continue
        pk = s + int(np.argmax(z[s:e + 1]))
        ev.append(dict(onset=int(frames[s]), peak=int(frames[pk]),
                       offset=int(frames[e]), z=float(z[pk]),
                       long=bool((e - s + 1) > max_len)))
    return ev


def plot(frames, z, ev, fps, out_png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = frames / fps
    fig, ax = plt.subplots(figsize=(15, 4.2))
    ax.plot(t, z, lw=0.7, color="#1F2937")
    ax.axhline(Z_ON, color="#C0483B", ls="--", lw=1, label=f"on  z={Z_ON}")
    ax.axhline(Z_OFF, color="#B45309", ls=":", lw=1, label=f"off z={Z_OFF}")
    for k, e in enumerate(ev, 1):
        ax.axvline(e["peak"] / fps, color="#2E8B6F", lw=1.1, alpha=.85)
        ax.text(e["peak"] / fps, z.max() * 1.02, str(k), fontsize=7,
                ha="center", color="#2E8B6F")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("intrusion (robust z)")
    ax.set_title(f"{title}   -   {len(ev)} candidate deliveries")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def run_one(path: str, outdir: str, session: str, dark_thr: int = DARK_THR):
    vid = os.path.splitext(os.path.basename(path))[0]
    print(f"\n=== {os.path.basename(path)} ===")
    frames, area, fps, ntot = intrusion_trace(path, STEP, dark_thr)
    z = robust_z(area)
    ev = find_events(frames, z, fps, STEP)

    print(f"  {fps:.2f} fps, {ntot} frames ({ntot/fps/60:.1f} min)")
    print(f"  baseline dark area {np.median(area):.0f} px, max z {z.max():.1f}")
    print(f"  {len(ev)} candidate deliveries")
    nlong = sum(e["long"] for e in ev)
    if nlong:
        print(f"  {nlong} event(s) longer than {MAX_EVENT_S:.0f} s - flagged, "
              f"check these first")
    if len(ev) < 4:
        print("  WARNING: fewer than 4 events. Check DARK_THR / Z_ON for this rig.")

    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, f"Candidates_{vid}.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Session", "Candidate", "Onset frame", "Onset s",
                    "Peak frame", "Peak s", "Offset frame", "Offset s",
                    "Peak z", "Long flag", "Stim code", "Stimulus", "Confirmed"])
        for k, e in enumerate(ev, 1):
            w.writerow([session, k, e["onset"], f'{e["onset"]/fps:.4f}',
                        e["peak"], f'{e["peak"]/fps:.4f}',
                        e["offset"], f'{e["offset"]/fps:.4f}',
                        f'{e["z"]:.2f}', int(e["long"]), "", "", ""])
    png_path = os.path.join(outdir, f"Candidates_{vid}.png")
    plot(frames, z, ev, fps, png_path, vid)
    print(f"  wrote {csv_path}")
    print(f"  wrote {png_path}")
    print("  NEXT: score this session in score_AB_dual_view.m, then check that "
          f"your delivery count matches the {len(ev)} candidates above.")
    return csv_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", help="bottom-view video file, or a folder of them")
    ap.add_argument("--out", default=None, help="output folder")
    ap.add_argument("--session", default="", help="session number for the CSV")
    ap.add_argument("--dark-thr", type=int, default=DARK_THR)
    a = ap.parse_args()

    if os.path.isdir(a.target):
        vids = sorted(f for f in os.listdir(a.target)
                      if f.lower().endswith((".avi", ".mp4", ".mov")))
        if not vids:
            raise SystemExit(f"no videos in {a.target}")
        paths = [os.path.join(a.target, v) for v in vids]
        outdir = a.out or os.path.join(a.target, "output")
    else:
        paths = [a.target]
        outdir = a.out or os.path.join(os.path.dirname(a.target) or ".", "output")

    for p in paths:
        run_one(p, outdir, a.session, a.dark_thr)


if __name__ == "__main__":
    sys.exit(main())
