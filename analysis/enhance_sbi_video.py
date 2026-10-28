"""enhance_sbi_video.py  -  push the SBI-553 side video as far as it will go.

THE QUESTION  "why not just increase sharpness and contrast?"
    Fair, and the answer is: yes, and it recovers more than I first claimed.
    My earlier read - "backlit silhouette, nothing recoverable" - was WRONG,
    and wrong for an avoidable reason.

WHAT WENT WRONG IN MY FIRST LOOK
    The animal sits in a narrow dark band while the diffuser behind it
    saturates. Measured on six frames spread over this session:

        animal, 2nd percentile      23-24 DN
        animal, 98th percentile     97 DN      (stable to 1 DN over 28 min)
        backdrop                    250-255 DN, clipped
        camera noise floor          1.5-3.0 DN

    A WHOLE-FRAME percentile stretch - which is what I did - therefore maps
    0-255 onto a range the diffuser dominates, and crushes the entire animal
    to black. The silhouette was an artefact of my own display code, not a
    property of the recording.

WHAT THIS SCRIPT DOES INSTEAD
    1  a FIXED tone curve, LO to 0 and HI to 255, taken from the animal's
       own band. No segmentation is needed because the band does not drift,
       which also makes it reproducible and auditable.
    2  CLAHE, for local contrast at the scale of fur and whiskers rather
       than at the scale of the whole animal.
    3  an unsharp mask, for the sharpness half of the question.

WHAT IT RECOVERS, AND WHAT IT CANNOT
    Measured over 20296 frames at 1 fps across all 12 videos, with 168
    frames adjudicated at pixel level by eye.

        ear position    RECOVERED. The pinna is translucent and lit from
                        BEHIND, so it is the brightest thing on the animal.
                        Resolved in 13452 of 20296 frames = 66.3 %, median
                        31 x 31 native px = 9.1 mm - which matches a real
                        mouse pinna at 10-12 mm and independently confirms
                        the 3.41 px/mm scale.
        whiskers        VISIBLE against the bright diffuser. Yield not
                        quantified, so not claimed.
        fur texture     RECOVERED.
        orbital         NOT RECOVERED. 1 confirmed eye in 168 adjudicated
                        frames. A loose dark-disc detector fires on 43.6 %
                        of frames, but with a median candidate width of 3 px
                        - impossible for an eye. Tightening to eye-sized
                        closed minima gives 4.57 %, and checking the best of
                        those killed every one: they are monotonic steps and
                        slots - the body against the diffuser, the gap
                        between ear and head, a fur crease - none with an
                        eye's signature of a dark floor with shoulders on
                        both sides.
        nose bulge      NOT RECOVERED. The shape change is 2-3 px at this
                        scale and the dorsal nose is on the unlit side.
        cheek bulge     NOT RECOVERED. No lit cheek faces the camera.

WHY CONTRAST CANNOT FIX THE EYE
    The light is BEHIND the animal, so the camera-facing side of the face is
    the unlit side. Face-surface contrast at the action-unit scale measures
    3.22 DN against a 1.5-3.0 DN noise floor - about 2.2x noise. A stretch
    multiplies signal and noise together; it cannot create light that was
    never recorded. The one unit that DOES work, the ear, works because it
    is backlit - the same geometry that kills the other four.

    So: 1 of 5 action units. The Mouse Grimace Scale needs the set.

OUTPUT  ->  qc/enhanced/
    SBI553_<mouse>_enhanced_<n>s.mp4   the enhanced clip
    fig_enhance_before_after.png       what the processing does, head-centred

USAGE
    python enhance_sbi_video.py
    python enhance_sbi_video.py --mouse female1 --start 300 --dur 120

Hansol Lim - HEAL mini1p / SBI-553
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553"
SIDE = os.path.join(ROOT, "videos", "cameraB")
OUT = os.path.join(ROOT, "qc", "enhanced")
FFMPEG = (r"C:\Users\hsollim\Downloads\ffmpeg-master-latest-win64-gpl-shared"
          r"\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg")
LO, HI = 18, 108                # the animal's own band, widened slightly
CLIP, TILES = 2.4, (8, 8)
USHARP_SIGMA, USHARP_AMT = 1.1, 1.0
CROP = (180, 150, 660, 470)     # x0, y0, x1, y1 - the cylinder
BAD, INK = "#C1272D", "#1A1A1A"


def enhance(gray):
    """Fixed tone curve, then local contrast, then sharpening."""
    x = np.clip((gray.astype(np.float32) - LO) * (255.0 / (HI - LO)),
                0, 255).astype(np.uint8)
    x = cv2.createCLAHE(clipLimit=CLIP, tileGridSize=TILES).apply(x)
    blur = cv2.GaussianBlur(x, (0, 0), USHARP_SIGMA)
    return cv2.addWeighted(x, 1 + USHARP_AMT, blur, -USHARP_AMT, 0)


def pick(mouse):
    hits = sorted(glob.glob(os.path.join(SIDE, mouse + "side*.avi")))
    if not hits:
        raise SystemExit("no side video for " + mouse)
    return hits[0]


def find_head(gray):
    """Centre on the ear pinna.

    The pinna is translucent and lit FROM BEHIND, so it is the brightest
    thing on the animal - the same geometry that makes the rest of the face
    unreadable makes the ear a reliable head landmark. The sweep found it in
    66.3 % of 20296 frames.
    """
    x0, y0, x1, y1 = CROP
    roi = gray[y0:y1, x0:x1]
    animal = roi < 120
    if animal.sum() < 2000:
        return None
    v = roi.astype(np.float32).copy()
    v[~animal] = 0                      # ignore the saturated backdrop
    v = cv2.GaussianBlur(v, (0, 0), 3)
    _, _, _, mx = cv2.minMaxLoc(v)
    return mx[0] + x0, mx[1] + y0


ROWLAB = ["RAW", "WHOLE-FRAME stretch" + chr(10) + "(my mistake)",
          "animal band + CLAHE" + chr(10) + "+ unsharp"]
TITLE = (
    "Can sharpness and contrast rescue the SBI-553 video?   The cylinder, "
    "native pixels, nearest-neighbour x2" + chr(10) +
    "BOTTOM ROW maps the animal's own band (18-108 DN) to full range, then "
    "CLAHE and unsharp.   FUR TEXTURE, the EAR PINNA and WHISKERS all come "
    "back.   THE EYE DOES NOT." + chr(10) +
    "MIDDLE ROW is a whole-frame percentile stretch. At THIS crop it barely "
    "differs from raw - the animal already fills much of the frame. It was "
    "in my earlier TIGHT head crops, where the saturated diffuser dominates "
    "the percentiles, that it crushed the mouse to black" + chr(10) +
    "and produced the word silhouette. That was my display code, not the "
    "recording - and correcting it is what revealed the ear." + chr(10) +
    "What no curve can fix: the light is BEHIND the animal, so the "
    "camera-facing side of the face is the unlit side. Face-surface "
    "contrast at the action-unit scale is 3.22 DN against a 1.5-3.0 DN "
    "noise floor." + chr(10) +
    "A stretch multiplies signal AND noise together; it cannot create light "
    "that was never recorded.   1 confirmed eye in 168 adjudicated frames.")


def figure(video, times, out):
    """Before and after on the HEAD, where the action units live."""
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    H = 70                              # half-box, native px
    cols = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, f = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        x0, y0, x1, y1 = CROP
        raw = g[y0:y1, x0:x1]
        # WHAT I DID FIRST: stretch on the WHOLE FRAME's percentiles. Kept
        # so the artefact is visible rather than described.
        lo, hi = np.percentile(g, [1, 99])
        wrong = np.clip((raw.astype(np.float32) - lo) / max(hi - lo, 1)
                        * 255, 0, 255).astype(np.uint8)
        cols.append((t, raw, wrong, enhance(raw)))
    cap.release()
    if not cols:
        raise SystemExit("no frame with a locatable head")

    n = len(cols)
    fig, ax = plt.subplots(3, n, figsize=(4.2 * n, 10.2))
    ax = np.atleast_2d(ax)
    for j in range(n):
        t, raw, wrong, good = cols[j]
        for i, im in enumerate((raw, wrong, good)):
            A = ax[i, j]
            A.imshow(cv2.resize(im, None, fx=2, fy=2,
                                interpolation=cv2.INTER_NEAREST),
                     cmap="gray", vmin=0, vmax=255)
            A.set_xticks([])
            A.set_yticks([])
            if j == 0:
                A.set_ylabel(ROWLAB[i], fontsize=12,
                             color=BAD if i == 1 else INK,
                             fontweight="bold" if i == 2 else "normal")
            if i == 0:
                A.set_title("t = " + str(t) + " s", fontsize=12.5)
            if i == 2:
                k = im.shape[0] * 2
                A.plot([16, 36], [k - 16, k - 16], color="#FF3030", lw=5)
                A.text(16, k - 30, "10 native px", color="#FF3030",
                       fontsize=10, fontweight="bold")
    fig.suptitle(TITLE, fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, .875])
    p = os.path.join(out, "fig_enhance_before_after.png")
    fig.savefig(p, dpi=125, bbox_inches="tight")
    plt.close(fig)
    return p


def write_clip(video, start, dur, mouse, out):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
    x0, y0, x1, y1 = CROP
    w, h = (x1 - x0) * 2, (y1 - y0) * 2
    dst = os.path.join(out, "SBI553_" + mouse + "_enhanced_"
                       + str(int(dur)) + "s.mp4")
    pr = subprocess.Popen(
        [FFMPEG, "-v", "error", "-f", "rawvideo", "-pix_fmt", "gray",
         "-s", str(w) + "x" + str(h), "-r", "30", "-i", "-",
         "-c:v", "libx264", "-preset", "medium", "-crf", "22",
         "-pix_fmt", "yuv420p", "-an", "-y", dst], stdin=subprocess.PIPE)
    n, acc = 0, 0.0
    while n < int(dur * fps):
        ok, f = cap.read()
        if not ok:
            break
        g = enhance(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1])
        g = cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)
        acc += 30.0 / fps
        for _ in range(int(acc)):
            pr.stdin.write(g.tobytes())
        acc -= int(acc)
        n += 1
    cap.release()
    pr.stdin.close()
    pr.wait()
    return dst, os.path.getsize(dst) / 1e6


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mouse", default="male2")
    ap.add_argument("--start", type=float, default=170.0)
    ap.add_argument("--dur", type=float, default=60.0)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    v = pick(a.mouse)
    print(os.path.basename(v))
    print("tone curve " + str(LO) + "-" + str(HI) + " DN -> 0-255, CLAHE "
          "clip " + str(CLIP) + ", unsharp sigma " + str(USHARP_SIGMA))
    print("")
    print(figure(v, [190, 320, 840, 1360], OUT))
    dst, mb = write_clip(v, a.start, a.dur, a.mouse, OUT)
    print(dst + "  (" + str(round(mb)) + " MB)")
    print("")
    print("RECOVERED      ear position (66.3 % of 20296 frames, 31 px), "
          "whiskers, fur texture")
    print("NOT RECOVERED  orbital (1 of 168 adjudicated), nose, cheek - "
          "the camera-facing face is unlit")
    print("So 1 of 5 action units. MGS needs the set.")


if __name__ == "__main__":
    main()
