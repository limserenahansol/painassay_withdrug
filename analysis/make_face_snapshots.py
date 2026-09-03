"""make_face_snapshots.py  -  high-contrast face snapshots, one mouse, both days.

WHAT THIS CAN AND CANNOT SETTLE
    It can show you the face as clearly as the footage allows.
    It cannot tell you whether that face means pain or sedation.

    Orbital tightening and ear flattening are Mouse Grimace Scale action units
    (Langford et al., Nat Methods 2010) AND signs of sedation. A drowsy mouse
    has half-closed eyes and flattened ears; so does a mouse in pain. No
    still image separates them, however good the contrast.

    The discriminator is ACTIVITY, not the face:

        pain      grimace present, locomotion preserved or increased
        sedation  grimace-like face, locomotion reduced

    Neurotensin receptor agonists cause hypolocomotion and hypothermia, so for
    SBI-553 sedation is a live hypothesis, not a technicality. Use
    measure_locomotion.py for the discriminating measurement, and the
    escape/rearing counts from the manual scoring as an independent check.

WHY A CONTACT SHEET
    Which frame shows the face you saw is something you know and the script
    does not. So it prints a grid of many frames with the timestamp on each,
    at several contrast settings. Pick the frame numbers that show it and pass
    them back with --times for a large single-frame version.

USAGE
    python make_face_snapshots.py --mouse male2 --out <folder>
    python make_face_snapshots.py --mouse male2 --out <folder> \\
                                  --times 640 812 1105 --zoom

Hansol Lim - HEAL mini1p / SBI-553
"""
from __future__ import annotations

import argparse
import glob
import os

import cv2
import numpy as np

BASE = r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\videos"
SIDE = {"Day1": os.path.join(BASE, "cameraB"),
        "Day2": os.path.join(BASE, "day2", "side")}
X0, X1, Y_TOP = 60, 680, 150
# A plausible animal, measured on both days. Boxes outside this are a failed
# detection, and they must be dropped rather than cropped and magnified - an
# 8 px box blown up to fill a tile looks like a photograph of something.
MIN_W, MAX_W = 55, 320
MIN_H, MAX_H = 45, 230
MIN_AREA = 2500
# FIXED crop and FIXED magnification, so every tile has the same scale.
# Fitting the crop to the animal made each tile a different magnification,
# which would show up as an ear-size difference between days that is purely
# an artefact of the zoom.
CROP_W, CROP_H = 190, 155
MAG = 2.4


def floor_row(g):
    """Row where the bright backdrop gives way to the dark mesh floor."""
    rm = g.mean(axis=1).astype(np.float32)
    return Y_TOP + int(np.argmin(np.diff(rm[Y_TOP:])))


def find_video(folder, mouse):
    """Match the mouse token at the START of the filename.

    A substring test is wrong here: "male2" is inside "female2", so asking for
    male2 silently returned the female2 recording. Anchoring at the start of
    the basename is unambiguous because every file is named
    <mouse><view><session> ....
    """
    m = mouse.lower()
    hits = [v for v in sorted(glob.glob(os.path.join(folder, "*.avi")))
            if os.path.basename(v).lower().startswith(m)]
    return hits[0] if hits else None


def animal_box(g, plate):
    """Locate the animal by how much it darkened the static scene.

    Thresholding on darkness alone does not work here: the cylinder's shadow
    band is as dark as the mouse and touches it, so every threshold returns a
    blob spanning the whole arena. Differencing against a median plate keeps
    only what changed, which is the animal.
    """
    yf = floor_row(g)
    y1 = max(yf - 4, Y_TOP + 20)
    d = cv2.subtract(plate[Y_TOP:y1, X0:X1], g[Y_TOP:y1, X0:X1])
    d = cv2.GaussianBlur(d, (5, 5), 0)
    _, m = cv2.threshold(d, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
    if n < 2:
        return None
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    w, h = int(st[i, cv2.CC_STAT_WIDTH]), int(st[i, cv2.CC_STAT_HEIGHT])
    area = int(st[i, cv2.CC_STAT_AREA])
    # reject implausible detections instead of magnifying them
    if not (MIN_W <= w <= MAX_W and MIN_H <= h <= MAX_H
            and area >= MIN_AREA):
        return None
    return dict(x=int(st[i, cv2.CC_STAT_LEFT] + X0),
                y=int(st[i, cv2.CC_STAT_TOP] + Y_TOP),
                w=w, h=h, area=area, yfloor=yf)


def fixed_crop(g, bb):
    """A CROP_W x CROP_H window on the animal's upper body, same size always.

    Centred horizontally on the animal and placed at the top of it, because
    that is where the head and ears sit whichever way the mouse is facing.
    """
    cx = bb["x"] + bb["w"] // 2
    cy = bb["y"] + int(.42 * bb["h"])
    x0 = int(np.clip(cx - CROP_W // 2, 0, g.shape[1] - CROP_W))
    y0 = int(np.clip(cy - CROP_H // 2, 0, g.shape[0] - CROP_H))
    return g[y0:y0 + CROP_H, x0:x0 + CROP_W]


def median_plate(path, n=100):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    acc = []
    for i in np.linspace(0, max(total - 2, 1), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            acc.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
    cap.release()
    if not acc:
        return None
    return np.median(np.stack(acc), axis=0).astype(np.uint8)


def boost(img, mode):
    """Contrast treatments. 'hi' is the one to look at for ear outline."""
    if mode == "raw":
        return img
    lo, hi = np.percentile(img, 1), np.percentile(img, 99)
    s = np.clip((img.astype(np.float32) - lo) * 255.0 / max(hi - lo, 1),
                0, 255).astype(np.uint8)
    if mode == "stretch":
        return s
    c = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(s)
    if mode == "hi":
        return cv2.fastNlMeansDenoising(c, None, 8, 7, 21)
    # 'edge' - outline only, which is where ear angle actually lives
    e = cv2.Laplacian(cv2.GaussianBlur(c, (3, 3), 0), cv2.CV_16S, ksize=3)
    e = cv2.convertScaleAbs(e)
    return cv2.addWeighted(c, .6, 255 - e, .4, 0)


def sheet(path, plate, times, out, tag, mouse):
    """Contact sheet with NO animal detection - deliberately.

    An earlier version located the animal and dropped frames where that
    failed. On Day 1 it kept 23 of 27 frames spread over the session; on Day 2
    it kept 12 of 27, ALL from the second half, because a stationary mouse
    contaminates its own median background plate and the difference vanishes.
    Comparing a full session against its second half is not a comparison.

    So: the same fixed arena window every frame, every sampled timepoint
    present, nothing rejected. Lower magnification, but an honest sample.
    """
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    tiles = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, fr = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        yf = floor_row(g)
        crop = g[Y_TOP:max(yf + 6, Y_TOP + 40), X0:X1]
        big = cv2.resize(crop, (330, 190), interpolation=cv2.INTER_LANCZOS4)
        v = cv2.cvtColor(boost(big, "hi"), cv2.COLOR_GRAY2BGR)
        cv2.putText(v, f"{int(t)}s", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, .55,
                    (0, 255, 255), 1)
        tiles.append(v)
    cap.release()
    if not tiles:
        print(f"    {tag}: could not read any frame")
        return None
    nrej = 0
    ncol = 6
    rows = []
    for i in range(0, len(tiles), ncol):
        r = tiles[i:i + ncol]
        while len(r) < ncol:
            r.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(r))
    grid = np.vstack(rows)
    band = np.zeros((32, grid.shape[1], 3), np.uint8)
    cv2.putText(band, f"{mouse}  {tag}   {len(tiles)} frames, every 60 s, "
                      f"NONE dropped   |   whole arena window, identical "
                      f"crop and scale in both days   |   CLAHE contrast",
                (6, 22), cv2.FONT_HERSHEY_SIMPLEX, .52, (255, 255, 255), 1)
    p = os.path.join(out, f"SHEET_{mouse}_{tag}.png")
    cv2.imwrite(p, np.vstack([band, grid]))
    print(f"  {p}  ({len(tiles)} frames)")
    return p


def big_frames(path, plate, times, out, tag, mouse):
    """One large panel per requested time, four contrast treatments."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    for t in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, fr = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        bb = animal_box(g, plate)
        if bb is None:
            print(f"    {t}s: no plausible animal found - skipped")
            continue
        crop = fixed_crop(g, bb)
        if crop.shape[:2] != (CROP_H, CROP_W):
            print(f"    {t}s: crop ran off the frame - skipped")
            continue
        # same fixed magnification as the contact sheet, just bigger on screen
        W = int(CROP_W * 2.6)
        H = int(CROP_H * 2.6)
        big = cv2.resize(crop, (W, H), interpolation=cv2.INTER_LANCZOS4)
        panels = []
        for mode, nm in (("raw", "as recorded"), ("stretch", "stretched"),
                         ("hi", "CLAHE + denoise"), ("edge", "outline")):
            v = cv2.cvtColor(boost(big, mode), cv2.COLOR_GRAY2BGR)
            cv2.putText(v, nm, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, .6,
                        (0, 255, 255), 1)
            panels.append(v)
        row = np.hstack(panels)
        band = np.zeros((32, row.shape[1], 3), np.uint8)
        cv2.putText(band, f"{mouse}  {tag}  t = {int(t)} s   "
                          f"animal {bb['w']}x{bb['h']} px", (6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
        p = os.path.join(out, f"FACE_{mouse}_{tag}_{int(t)}s.png")
        cv2.imwrite(p, np.vstack([band, row]))
        print(f"  {p}")
    cap.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mouse", default="male2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--times", type=float, nargs="*", default=None,
                    help="specific seconds to render large; omit for the "
                         "contact sheet")
    ap.add_argument("--days", nargs="*", default=["Day1", "Day2"])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    print(__doc__.split("USAGE")[0].strip()[:0] or "", end="")
    print("NOTE: a still face cannot separate pain from sedation. Both give "
          "orbital\n      tightening and flattened ears. Activity is the "
          "discriminator.\n")

    for tag in a.days:
        folder = SIDE.get(tag)
        if not folder:
            continue
        v = find_video(folder, a.mouse)
        if not v:
            print(f"{tag}: no {a.mouse} video in {folder}")
            continue
        print(f"{tag}: {os.path.basename(v)}")
        P = median_plate(v)
        if P is None:
            print("    could not build a background plate")
            continue
        if a.times:
            big_frames(v, P, a.times, a.out, tag, a.mouse)
        else:
            # spread across baseline and all four stimulus blocks
            times = list(np.arange(60, 1680, 60))
            sheet(v, P, times, a.out, tag, a.mouse)

    if not a.times:
        print("\nPick the frames that show the face you saw, then:")
        print(f"  python make_face_snapshots.py --mouse {a.mouse} "
              f"--out <folder> --times 640 812 1105")


if __name__ == "__main__":
    main()
