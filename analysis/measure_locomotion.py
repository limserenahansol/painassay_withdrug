"""measure_locomotion.py  -  is the drug day CALMER? Measured, not judged.

WHY THIS MATTERS MORE THAN IT LOOKS
    The primary readout of this experiment is a count of pain behaviours, and
    it went down on the drug day. There are two explanations and they are not
    the same finding:

        analgesia   the animal feels less, so it responds less
        sedation    the animal does everything less, including responding

    A sedated mouse produces exactly the same drop in every behaviour count.
    Neurotensin receptor agonists are known to cause hypolocomotion and
    hypothermia, so this is a live possibility for SBI-553, not a technicality.

    General movement is the discriminator. A biased analgesic should leave
    locomotion roughly intact; sedation cannot. And unlike facial expression,
    movement IS recoverable from a backlit silhouette - it needs position over
    time, not internal detail.

WHAT IT COMPUTES, per session
    motion_index     mean absolute frame-to-frame pixel change inside the
                     arena, normalised by the animal's own silhouette area so
                     a bigger mouse is not automatically a busier one
    speed_px_s       displacement of the silhouette centroid
    frac_moving      fraction of samples above a movement threshold
    area_px          median silhouette area - a crude posture proxy; a
                     hunched, still animal presents a smaller, rounder blob
    elongation       silhouette width / height, again posture

    Reported for the baseline and for each stimulus block, because sedation
    should be present throughout while an analgesic effect is specific to the
    stimulus periods. That contrast is the actual test.

LIMITS, stated up front
    This is a silhouette. It cannot separate grooming from walking, and it
    cannot see the face at all (measured: ~4 grey levels of internal contrast
    in a 720x480 backlit frame - there is no facial information recorded).
    It is a screen for gross sedation, nothing finer.

    THE ILLUMINATION CAVEAT. The Day 2 recordings are dimmer than Day 1
    (backdrop 175 vs 200 grey levels in the frames checked). motion_index is a
    pixel-difference measure, so lower contrast lowers it whether or not the
    animal moved less - it is reported but must NOT be read as sedation on its
    own. speed_px_s and frac_moving come from the silhouette CENTROID and are
    geometric, so they survive an illumination change. Read those first.

ALWAYS RUN --check FIRST
    It writes overlay frames with the detected silhouette outlined, so you can
    see the tracker is on the animal. This matters: a first version used a
    fixed bottom edge that included the mesh floor, and since the floor (~40)
    is darker than the animal (~27-68 against a ~175-200 backdrop), the
    tracker locked onto the floor. The floor does not move, so it would have
    reported near-zero locomotion on both days and looked plausible.

USAGE
    python measure_locomotion.py --day1 <side folder> --day2 <side> --check
    python measure_locomotion.py --day1 <side folder> --day2 <side folder>
    python measure_locomotion.py --day1 ... --day2 ... --step 6 --out <folder>

Hansol Lim - HEAL mini1p / SBI-553
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import cv2
import numpy as np
import pandas as pd

# Horizontal extent of the arena in the 720x480 side view. The BOTTOM edge is
# found per frame, not fixed: the mesh floor is much darker than the animal
# (measured: floor ~40, animal ~27-68, backdrop ~175-200 grey levels), so a
# band that includes the floor makes the floor the darkest blob and the
# tracker locks onto it instead of the mouse. That is exactly what happened
# with a fixed y1 = 430.
X0, X1 = 60, 680
Y_TOP = 150
DARK_PCT = 8.0          # the animal is the darkest few per cent ABOVE the floor
MIN_AREA = 800          # px, below this the blob is noise


def floor_row(gray):
    """Row where the bright backdrop gives way to the dark mesh floor."""
    rm = gray.mean(axis=1).astype(np.float32)
    d = np.diff(rm[Y_TOP:])
    return Y_TOP + int(np.argmin(d))
BLOCKS = [("baseline", 0, 300), ("block1", 300, 600), ("block2", 660, 960),
          ("block3", 1020, 1320), ("block4", 1380, 1680)]
MOVE_THR = 12.0         # px/s of centroid speed counted as "moving"


def mouse_id(name):
    """female1side0007 ... -> F1 ; male3side0010 ... -> M3"""
    m = re.match(r"(female|male)\s*(\d+)", os.path.basename(name).lower())
    if not m:
        return os.path.basename(name)[:12]
    return ("F" if m.group(1) == "female" else "M") + m.group(2)


def silhouette(gray, yfloor=None):
    if yfloor is None:
        yfloor = floor_row(gray)
    # a few rows of margin: the animal's feet sit right on the floor line and
    # including it would merge animal and floor into one blob
    y1 = max(yfloor - 4, Y_TOP + 20)
    band = gray[Y_TOP:y1, X0:X1]
    if band.size == 0:
        return None
    # Otsu, not a fixed percentile. A percentile assumes the animal occupies a
    # known fraction of the band, and it does not - it changes with posture and
    # with how close the mouse is to the camera. The 8th-percentile version
    # captured only the darkest core and undercut the area by roughly half.
    # Otsu splits dark animal from bright backdrop wherever that split lies.
    thr, _ = cv2.threshold(band, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx((band < thr).astype(np.uint8), cv2.MORPH_OPEN,
                         np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, st, cent = cv2.connectedComponentsWithStats(m, 8)
    if n < 2:
        return None
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    if st[i, cv2.CC_STAT_AREA] < MIN_AREA:
        return None
    return dict(area=float(st[i, cv2.CC_STAT_AREA]),
                w=float(st[i, cv2.CC_STAT_WIDTH]),
                h=float(st[i, cv2.CC_STAT_HEIGHT]),
                x=int(st[i, cv2.CC_STAT_LEFT] + X0),
                y=int(st[i, cv2.CC_STAT_TOP] + Y_TOP),
                cx=float(cent[i][0] + X0), cy=float(cent[i][1] + Y_TOP),
                yfloor=yfloor,
                mask=(lab == i), band=(Y_TOP, y1, X0, X1))


def one_video(path, step):
    """Sequential decode; seeking 10k times would cost far more than decoding."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"    cannot open {os.path.basename(path)}")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    rows = []
    prev_band = None
    prev_c = None
    k = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if k % step:
            k += 1
            continue
        ok, fr = cap.retrieve()
        if not ok:
            break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        s = silhouette(g)
        y0b, y1b, x0b, x1b = s["band"] if s else (Y_TOP, Y_TOP + 200, X0, X1)
        band = g[y0b:y1b, x0b:x1b].astype(np.int16)
        t = k / fps
        dt = step / fps
        md = np.nan
        if (prev_band is not None and s is not None
                and prev_band.shape == band.shape):
            d = np.abs(band - prev_band)
            # only inside the animal, and per pixel of animal, so size is out
            md = float(d[s["mask"]].mean())
        sp = np.nan
        if prev_c is not None and s is not None:
            sp = float(np.hypot(s["cx"] - prev_c[0], s["cy"] - prev_c[1]) / dt)
        rows.append(dict(t=t, motion=md, speed=sp,
                         area=s["area"] if s else np.nan,
                         elong=(s["w"] / s["h"]) if s and s["h"] else np.nan))
        prev_band = band
        prev_c = (s["cx"], s["cy"]) if s else None
        k += 1
    cap.release()
    return pd.DataFrame(rows)


def summarise(D, day, mouse):
    out = []
    for name, lo, hi in BLOCKS:
        g = D[(D.t >= lo) & (D.t < hi)]
        if not len(g):
            continue
        out.append(dict(day=day, mouse=mouse, period=name,
                        n_samples=len(g),
                        motion_index=float(np.nanmean(g.motion)),
                        speed_px_s=float(np.nanmedian(g.speed)),
                        frac_moving=float(np.nanmean(
                            (g.speed > MOVE_THR).astype(float))),
                        area_px=float(np.nanmedian(g.area)),
                        elongation=float(np.nanmedian(g.elong))))
    return out


def check(jobs, out, mouse=None):
    """Draw the detected silhouette on sampled frames so it can be eyeballed.

    Checks EVERY animal by default, not just the first video in the folder.
    Verifying one mouse and assuming the rest is how the earlier version's
    floor-locking bug survived as long as it did.
    """
    TIMES = [120, 420, 780, 1100, 1450, 1600]
    tiles = []
    for day, folder in jobs:
        vids = sorted(glob.glob(os.path.join(folder, "*.avi")))
        if mouse:
            vids = [v for v in vids
                    if os.path.basename(v).lower().startswith(mouse.lower())]
        if not vids:
            print(f"  {day}: no videos in {folder}")
            continue
      # fall through to the per-video loop below
        for v in vids:
            _check_one(day, v, TIMES, tiles)
    _check_write(tiles, out)


def _check_one(day, path, TIMES, tiles):
        vids = [path]
        cap = cv2.VideoCapture(vids[0])
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        row, hits = [], 0
        for t in TIMES:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
            ok, fr = cap.read()
            if not ok:
                continue
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            s = silhouette(g)
            vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
            yf = s["yfloor"] if s else floor_row(g)
            cv2.line(vis, (0, yf), (vis.shape[1], yf), (0, 200, 255), 1)
            if s:
                hits += 1
                cv2.rectangle(vis, (s["x"], s["y"]),
                              (s["x"] + int(s["w"]), s["y"] + int(s["h"])),
                              (0, 0, 255), 2)
                cv2.circle(vis, (int(s["cx"]), int(s["cy"])), 4,
                           (0, 255, 0), -1)
                cv2.putText(vis, f"area {int(s['area'])}", (6, 34),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1)
            cv2.putText(vis, f"{day}  {t}s", (6, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1)
            row.append(cv2.resize(vis, (360, 240)))
        cap.release()
        print(f"  {day}: silhouette found in {hits}/{len(TIMES)} frames "
              f"({os.path.basename(vids[0])})")
        if row:
            tiles.append(np.hstack(row))


def _check_write(tiles, out):
    if tiles:
        w = max(t.shape[1] for t in tiles)
        tiles = [np.hstack([t, np.zeros((t.shape[0], w - t.shape[1], 3),
                                        np.uint8)])
                 if t.shape[1] < w else t for t in tiles]
        p = os.path.join(out, "CHECK_silhouette.png")
        cv2.imwrite(p, np.vstack(tiles))
        print(f"\nwrote {p}")
        print("red box = detected animal, green dot = centroid, "
              "orange line = floor edge.")
        print("If the box is on the floor or the mesh, do NOT trust the "
              "numbers.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--day1", required=True, help="Day 1 SIDE video folder")
    ap.add_argument("--day2", default=None, help="Day 2 SIDE video folder")
    ap.add_argument("--step", type=int, default=6,
                    help="sample every Nth frame; 6 gives 5 Hz, plenty for "
                         "locomotion (default 6)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label1", default="Day 1 no drug")
    ap.add_argument("--label2", default="Day 2 drug")
    ap.add_argument("--check", action="store_true",
                    help="write overlay frames showing the detected "
                         "silhouette and exit; verify before measuring")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    jobs = [(a.label1, a.day1)]
    if a.day2:
        jobs.append((a.label2, a.day2))

    if a.check:
        check(jobs, a.out)
        return

    rows = []
    for day, folder in jobs:
        vids = sorted(glob.glob(os.path.join(folder, "*.avi"))
                      + glob.glob(os.path.join(folder, "*.mp4")))
        print(f"{day}: {len(vids)} video(s) in {folder}")
        for v in vids:
            mid = mouse_id(v)
            print(f"  {mid}  {os.path.basename(v)}")
            D = one_video(v, a.step)
            if D is None or not len(D):
                continue
            rows += summarise(D, day, mid)
            D.to_csv(os.path.join(a.out, f"loco_raw_{day.split()[1]}_{mid}.csv"),
                     index=False)

    S = pd.DataFrame(rows)
    if S.empty:
        raise SystemExit("nothing measured")
    S.to_csv(os.path.join(a.out, "Locomotion_summary.csv"), index=False)
    print(f"\nwrote {os.path.join(a.out, 'Locomotion_summary.csv')}")

    print("\n=== whole session (mean over periods) ===")
    print(f"  {'day':16s} {'mouse':6s} {'motion':>8s} {'speed':>8s} "
          f"{'moving%':>8s} {'area':>7s} {'elong':>6s}")
    for (day, mouse), g in S.groupby(["day", "mouse"]):
        print(f"  {day:16s} {mouse:6s} {g.motion_index.mean():8.2f} "
              f"{g.speed_px_s.mean():8.2f} "
              f"{100 * g.frac_moving.mean():8.1f} "
              f"{g.area_px.mean():7.0f} {g.elongation.mean():6.2f}")

    if not a.day2:
        print("\n  Day 1 only. Re-run with --day2 for the comparison.")
        return

    from scipy import stats
    print("\n=== Day 1 vs Day 2, paired on the mice present in both ===")
    print(f"  {'measure':14s} {'n':>2s} {'Day1':>8s} {'Day2':>8s} "
          f"{'ratio':>6s} {'p':>7s} {'floor':>6s}")
    res = []
    for meas in ("motion_index", "speed_px_s", "frac_moving", "area_px",
                 "elongation"):
        w = (S.groupby(["day", "mouse"])[meas].mean().reset_index()
             .pivot_table(index="mouse", columns="day", values=meas))
        if a.label1 not in w.columns or a.label2 not in w.columns:
            continue
        w = w.dropna()
        x, y = w[a.label1].to_numpy(), w[a.label2].to_numpy()
        n = len(x)
        p = np.nan
        if n >= 2 and not np.allclose(y - x, 0):
            try:
                p = float(stats.wilcoxon(x, y, zero_method="wilcox",
                                         method="exact").pvalue)
            except ValueError:
                pass
        floor = 2.0 / (2 ** n) if n else np.nan
        rat = y.mean() / x.mean() if x.mean() else np.nan
        res.append(dict(measure=meas, n_mice=n, day1=x.mean(), day2=y.mean(),
                        ratio=rat, p_wilcoxon=p, p_floor=floor,
                        n_down=int((y < x).sum())))
        print(f"  {meas:14s} {n:2d} {x.mean():8.2f} {y.mean():8.2f} "
              f"{rat:6.2f} {p:7.3f} {floor:6.3f}")
    pd.DataFrame(res).to_csv(
        os.path.join(a.out, "Locomotion_day_stats.csv"), index=False)

    print("\n  HOW TO READ THIS")
    print("  Movement clearly reduced across the WHOLE session, baseline "
          "included\n    -> sedation is in play; the behaviour-count drop "
          "cannot be called analgesia\n       on its own.")
    print("  Movement preserved at baseline, behaviour counts down only in "
          "the\n  stimulus blocks -> that pattern is what analgesia looks "
          "like.")
    print("  Escape / rearing from the manual scoring is the second, "
          "independent\n  check: on Day 1 it behaved as exploration, not "
          "pain.")


if __name__ == "__main__":
    main()
