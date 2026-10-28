"""grimace_feasibility.py  -  can PainFace / GrimACE run on footage we already have?

STEP 1 of the automated-pain-scoring plan, measured rather than guessed.

The question is not "is the mouse in the frame" but "is the FACE resolvable".
The mouse grimace scale needs four action units - orbital tightening, nose
bulge, ear position, whisker change (Langford et al., Nat Methods 2010).
PainFace (Zylka lab, PAIN 2024) crops a box around each and runs a CNN on
the pixels inside it. So the decisive numbers are, per rig:

    1  how many pixels across is the EYE
    2  is the face LIT, or is the animal a backlit silhouette
    3  how often is the head in a scoreable pose
    4  how much does the head implant occlude

This script pulls frames from both rigs, crops the head at native
resolution, and prints a scale bar in ORIGINAL pixels so the answer is
checkable rather than asserted.

WHAT IT FOUND  (see grimace_feasibility.png)

    mini1p / SBI-553   720 x 480, DV, 29.97 fps.  Side camera is BACKLIT:
        the light sits behind the animal so the paw reads against the mesh.
        The mouse is a black silhouette. The only face structure visible is
        the pinna, glowing because it is translucent and lit from behind.
        NO EYE IS VISIBLE IN ANY FRAME SAMPLED. This is a lighting-geometry
        problem, not a resolution problem - a bigger sensor would record the
        same silhouette. MGS is not recoverable. 1 of 4 action units at best.

    mini2p / CEA-Ntsr1  1600 x 1200, MJPEG, 25 fps.  MiceVideo1 is a true
        lateral view, diffusely lit from the side, and the face IS visible:
        eye with corneal reflection, pinna, snout, whisker pad. But the eye
        measures ~9 px across. Orbital tightening is a graded change in
        eyelid aperture; at 9 px the full open-to-closed range is 4-5 px, so
        a 0/1/2 score is not supportable. A continuous eyelid-aperture
        measurement is - and for a regressor against calcium that is the
        better variable anyway.

    Head implant: the miniscope and its cable sit above and behind the ear.
        In a lateral frame they crowd the ear but do NOT occlude eye, nose or
        whisker pad. So a 3-unit version (orbital, nose, whisker) is the
        realistic target, not PainFace's 4-unit 0-8 score. GrimACE (Lab
        Animal 2026) is the one published pipeline validated on
        head-implanted mice, but it is a fixed-geometry chamber and cannot
        be applied retrospectively to either rig.

USAGE
    python grimace_feasibility.py                    # both rigs, default frames
    python grimace_feasibility.py --times 160 420    # your own timestamps

Hansol Lim - HEAL mini1p / SBI-553 and CEA-Ntsr1 mini2p
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess

import cv2
import numpy as np

FFMPEG = (r"C:\Users\hsollim\Downloads\ffmpeg-master-latest-win64-gpl-shared"
          r"\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg")
SBI = (r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\videos\cameraB")
M2P = (r"I:\20260911_CEANTSR1_65_15_pain_ 10min_pin_heat_2026-09-11_"
       r"15-56-48\MiceVideo1\MiceVideo")
OUT = (r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\qc")


def grab(video, t, dst):
    """One frame at t seconds, no re-encode loss beyond the codec's own."""
    subprocess.run([FFMPEG, "-v", "error", "-ss", str(t), "-i", video,
                    "-frames:v", "1", "-y", dst], check=True)
    return cv2.imread(dst, cv2.IMREAD_GRAYSCALE)


def stretch(img, lo_pct=1, hi_pct=99):
    lo, hi = np.percentile(img, [lo_pct, hi_pct])
    return np.clip((img.astype(np.float32) - lo) / max(hi - lo, 1) * 255,
                   0, 255).astype(np.uint8)


def panel(img, box, zoom, label, note, px_bar=10):
    """Crop at native resolution, magnify with NEAREST, print a px scale bar.

    NEAREST on purpose: any smooth interpolation invents detail the sensor
    never recorded, which is exactly the thing being measured here.
    """
    y0, y1, x0, x1 = box
    crop = stretch(img[y0:y1, x0:x1])
    big = cv2.cvtColor(cv2.resize(crop, None, fx=zoom, fy=zoom,
                                  interpolation=cv2.INTER_NEAREST),
                       cv2.COLOR_GRAY2BGR)
    h = big.shape[0]
    cv2.rectangle(big, (18, h - 34), (18 + px_bar * zoom, h - 26),
                  (60, 60, 240), -1)
    cv2.putText(big, f"{px_bar} px at native resolution", (18, h - 42),
                cv2.FONT_HERSHEY_SIMPLEX, .55, (60, 60, 240), 2)
    # dark stroke first: white-on-white was unreadable over the diffuser
    for txt, y, sz in ((label, 30, .8), (note, 56, .55)):
        cv2.putText(big, txt, (18, y), cv2.FONT_HERSHEY_SIMPLEX, sz,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(big, txt, (18, y), cv2.FONT_HERSHEY_SIMPLEX, sz,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return big


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--times", nargs=2, type=float, default=[160, 1260],
                    help="seconds: mini2p frame, then SBI frame")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    scratch = os.path.join(OUT, "_frames")
    os.makedirs(scratch, exist_ok=True)

    v_m = sorted(glob.glob(os.path.join(M2P, "*.avi")))[0]
    # "*male2side*" also matches female2side - "male2side" is a substring of
    # "female2side". Anchor it to the start of the filename.
    v_s = sorted(glob.glob(os.path.join(SBI, "male2side*.avi")))[0]
    im_m = grab(v_m, a.times[0], os.path.join(scratch, "m2p.png"))
    im_s = grab(v_s, a.times[1], os.path.join(scratch, "sbi.png"))
    print(f"mini2p  {im_m.shape[1]} x {im_m.shape[0]}   {os.path.basename(v_m)}")
    print(f"mini1p  {im_s.shape[1]} x {im_s.shape[0]}   {os.path.basename(v_s)}")

    p_m = panel(im_m, (700, 860, 1140, 1360), 6,
                "mini2p side (1600x1200), head implant",
                "eye + corneal reflection, pinna, snout - eye ~9 px across")
    p_s = panel(im_s, (200, 360, 250, 470), 6,
                "mini1p / SBI-553 side (720x480), no implant",
                "backlit silhouette - pinna glows, NO eye anywhere")
    H = max(p_m.shape[0], p_s.shape[0])

    def pad(p):
        return cv2.copyMakeBorder(p, 0, H - p.shape[0], 0, 8,
                                  cv2.BORDER_CONSTANT, value=(0, 0, 0))

    sheet = np.hstack([pad(p_m), pad(p_s)])
    bar = np.zeros((78, sheet.shape[1], 3), np.uint8)
    cv2.putText(bar, "Can the mouse grimace scale be scored on footage we "
                     "already have?", (18, 28), cv2.FONT_HERSHEY_SIMPLEX,
                .72, (255, 255, 255), 2)
    cv2.putText(bar, "NEAREST-neighbour magnification only - smooth "
                     "interpolation would invent detail the sensor never "
                     "recorded", (18, 56), cv2.FONT_HERSHEY_SIMPLEX, .5,
                (170, 170, 170), 1)
    sheet = np.vstack([bar, sheet])
    p = os.path.join(OUT, "grimace_feasibility.png")
    cv2.imwrite(p, sheet)
    print(f"\nwrote {p}")
    return p


if __name__ == "__main__":
    main()
