"""make_ppt_clip.py  -  a 30 s representative clip with the labels drawn on,
for a lab-meeting slide.

FOR PRESENTATION ONLY. This script never writes into videos\\output and never
touches your scoring. It only reads.

WHAT IT DOES
  1. ranks the six mice on video quality inside their Pin prick block
  2. among the mice that have been scored, picks the best one
  3. finds the busiest 30 s window inside that Pin prick block, so the clip
     actually shows behaviour rather than a resting mouse
  4. renders bottom + side side by side, with the label track underneath and
     a moving cursor - the same layout you score in
  5. encodes H.264 MP4, which is what PowerPoint plays reliably, plus a few
     still PNGs for a static slide

USAGE
  python make_ppt_clip.py                      # pick everything automatically
  python make_ppt_clip.py --mouse female3      # force the mouse
  python make_ppt_clip.py --start 1100 --dur 30
  python make_ppt_clip.py --stimulus "Pin prick"

OUTPUT  ->  ppt_clips\\
  <mouse>_<stimulus>_<start>s_30s.mp4
  <mouse>_<stimulus>_frame*.png
  Clip_quality_report.csv

Hansol Lim - HEAL mini1p / SBI-553
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import subprocess

import cv2
import numpy as np
import pandas as pd
from scipy.io import loadmat

ROOT = r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\videos"
FFMPEG_GUESS = [
    r"C:\Users\hsollim\Downloads\ffmpeg-master-latest-win64-gpl-shared"
    r"\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe",
    "ffmpeg",
]

# side-view segmentation settings validated in qc/WT_RECORDING_QC.md
SIDE_ROI = (90, 370, 110, 600)
SIDE_THR = 75

AFF = {1: "paw attending", 2: "licking / biting",
       3: "guarding", 4: "escape / rearing"}
AFF_BGR = {1: (191, 115, 51), 2: (38, 89, 217),
           3: (179, 77, 140), 4: (89, 166, 51)}
REF = {1: "withdrawal", 2: "flinch"}
REF_BGR = {1: (26, 191, 230), 2: (140, 140, 140)}
TRACK_ROWS = ["paw attending", "licking / biting", "guarding",
              "escape / rearing", "withdrawal", "flinch"]


def ffmpeg():
    for c in FFMPEG_GUESS:
        if os.path.exists(c) or shutil.which(c):
            return c
    return None


def mouse_key(name):
    m = re.match(r"([a-z]+\d)", os.path.basename(name).lower())
    return m.group(1) if m else os.path.basename(name)


def pair_videos():
    out = {}
    for b in glob.glob(os.path.join(ROOT, "cameraA", "*.avi")):
        k = mouse_key(b)
        cands = [s for s in glob.glob(os.path.join(ROOT, "cameraB", "*.avi"))
                 if mouse_key(s) == k]
        if cands:
            out[k] = (b, cands[0])
    return dict(sorted(out.items()))


def load_scoring(k):
    for f in glob.glob(os.path.join(ROOT, "output", "ScoringAB_*.mat")):
        if mouse_key(f[len("ScoringAB_"):] if "ScoringAB_" in f
                     else f) == k or mouse_key(os.path.basename(f)[10:]) == k:
            return loadmat(f), f
    return None, None


def stim_blocks(M):
    fps = float(np.asarray(M["frameRate"]).ravel()[0])
    dF = np.asarray(M["dFrames"]).ravel().astype(int)
    dT = np.asarray(M["dTypes"]).ravel().astype(int)
    names = [str(np.asarray(x).ravel()[0]).strip()
             for x in np.asarray(M["stimNames"]).ravel()][:4]
    out = {}
    for i, n in enumerate(names, start=1):
        m = dT == i
        if m.sum():
            out[n] = dict(code=i, n=int(m.sum()),
                          t0=(dF[m].min() - 1) / fps, t1=(dF[m].max() - 1) / fps,
                          frames=dF[m])
    return out, fps, dF, dT, names


def quality(bpath, spath, t0, t1, nsample=25):
    """Cheap image-quality metrics inside [t0, t1]."""
    res = {}
    for tag, path in (("bottom", bpath), ("side", spath)):
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        sharp, clip, bright, con, found = [], [], [], [], 0
        n = 0
        for t in np.linspace(t0, min(t1, t0 + 300), nsample):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
            ok, im = cap.read()
            if not ok:
                continue
            n += 1
            g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            sharp.append(cv2.Laplacian(g, cv2.CV_64F).var())
            clip.append(float(((g >= 254) | (g <= 1)).mean()))
            bright.append(float(g.mean()))
            if tag == "side":
                y0, y1, x0, x1 = SIDE_ROI
                sub = g[y0:y1, x0:x1]
                m = (sub < SIDE_THR).astype(np.uint8) * 255
                m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
                cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
                cnts = [c for c in cnts if 6000 <= cv2.contourArea(c) <= 70000]
                if cnts:
                    found += 1
                    mm = np.zeros_like(sub)
                    cv2.drawContours(mm, [max(cnts, key=cv2.contourArea)],
                                     -1, 255, -1)
                    con.append(abs(float(sub[mm > 0].mean())
                                   - float(sub[mm == 0].mean()))
                               / (float(sub[mm == 0].std()) + 1e-6))
        cap.release()
        res[f"{tag}_sharpness"] = float(np.median(sharp)) if sharp else np.nan
        res[f"{tag}_clipped_pct"] = 100 * float(np.median(clip)) if clip else np.nan
        res[f"{tag}_brightness"] = float(np.median(bright)) if bright else np.nan
        if tag == "side":
            res["side_detect_pct"] = 100 * found / max(n, 1)
            res["side_contrast_sd"] = float(np.median(con)) if con else np.nan
    return res


def busiest_window(M, t0, t1, dur, fps):
    """The `dur`-second window in [t0, t1] with the most labelled events."""
    sc = np.asarray(M["score"]).ravel().astype(int)
    rx = np.asarray(M.get("reflexEvents", np.empty((0, 2)))).reshape(-1, 2)
    d = np.diff(np.concatenate(([0], (sc > 0).astype(np.int8), [0])))
    onsets = np.flatnonzero(d == 1)
    ref = rx[:, 0].astype(int) if rx.size else np.empty(0, int)
    best, bt = -1, t0
    for s in np.arange(t0, max(t0, t1 - dur) + 0.01, 1.0):
        a, b = int(s * fps) + 1, int((s + dur) * fps)
        # held behaviours make a better demo than 1-frame reflex marks,
        # so they count for more when choosing the window
        k = 3 * int(((onsets >= a) & (onsets <= b)).sum()) \
            + int(((ref >= a) & (ref <= b)).sum())
        if k > best:
            best, bt = k, s
    return float(bt), int(best)


def draw(fb, fs, sc, rx, dF, dT, names, i0, i1, cur, fps, hdr, zoom=False):
    """One composited output frame."""
    if zoom:
        # crop each view to the part that carries information, so the mouse
        # is big enough to read from the back of the room. The side crop is
        # the QC-validated arena ROI widened a little for context.
        y0, y1, x0, x1 = SIDE_ROI
        # reach further DOWN than the segmentation ROI: that ROI deliberately
        # stops above the mesh floor, but for a slide the paws and the floor
        # line are exactly what the audience needs to see.
        fs = fs[max(0, y0 - 20):min(fs.shape[0], y1 + 105),
                max(0, x0 - 40):min(fs.shape[1], x1 + 40)]
        fb = fb[int(fb.shape[0] * 0.02):int(fb.shape[0] * 0.98),
                int(fb.shape[1] * 0.20):int(fb.shape[1] * 0.90)]
    H = 480
    if fb.shape[0] != H:
        fb = cv2.resize(fb, (int(fb.shape[1] * H / fb.shape[0]), H))
    if fs.shape[0] != H:
        fs = cv2.resize(fs, (int(fs.shape[1] * H / fs.shape[0]), H))
    bar = np.zeros((H, 6, 3), np.uint8)
    bar[:, :, 1] = 255
    vid = np.hstack([fb, bar, fs])
    W = vid.shape[1]

    cv2.rectangle(vid, (0, 0), (fb.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(vid, "BOTTOM (stimulus)", (8, 19), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 255), 2)
    cv2.rectangle(vid, (fb.shape[1] + 6, 0), (W, 26), (0, 0, 0), -1)
    cv2.putText(vid, "SIDE (behaviour)", (fb.shape[1] + 14, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # ---- label track for the whole clip, with a cursor ----
    RH, PAD, LBL = 26, 8, 190
    TH = RH * len(TRACK_ROWS) + 2 * PAD + 44   # +18 for the caption line
    tr = np.full((TH, W, 3), 245, np.uint8)
    x0, x1 = LBL, W - 20
    span = max(i1 - i0, 1)

    def cx(f):
        return int(x0 + (f - i0) / span * (x1 - x0))

    for r, nm in enumerate(TRACK_ROWS):
        y = PAD + r * RH
        cv2.rectangle(tr, (x0, y + 3), (x1, y + RH - 3), (225, 225, 225), -1)
        cv2.putText(tr, nm, (8, y + RH - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (40, 40, 40), 1)
    # A 2-frame mark is ~3 px on this track: invisible on a projector. Widening
    # it into a bar would be worse, because the audience reads bar width as
    # duration. So anything shorter than BRIEF_S is drawn as a DOT instead -
    # visible, and obviously a point event rather than an interval.
    BRIEF_S = 0.25
    brief_any = False
    for code in AFF:
        seg = (sc[i0:i1] == code).astype(np.int8)
        dd = np.diff(np.concatenate(([0], seg, [0])))
        for a, b in zip(np.flatnonzero(dd == 1), np.flatnonzero(dd == -1)):
            y = PAD + (code - 1) * RH
            if (b - a) / fps < BRIEF_S:
                brief_any = True
                cv2.circle(tr, (cx(i0 + a), y + RH // 2), 6,
                           AFF_BGR[code], -1)
            else:
                cv2.rectangle(tr, (cx(i0 + a), y + 3),
                              (max(cx(i0 + b), cx(i0 + a) + 3), y + RH - 3),
                              AFF_BGR[code], -1)
    if rx.size:
        for f, ty in rx.astype(int):
            if i0 <= f <= i1 and int(ty) in REF_BGR:
                y = PAD + (3 + int(ty)) * RH
                cv2.rectangle(tr, (cx(f) - 3, y + 3), (cx(f) + 3, y + RH - 3),
                              REF_BGR[int(ty)], -1)
    # deliveries: black line through every row
    for f, ty in zip(dF, dT):
        if i0 <= f <= i1:
            cv2.line(tr, (cx(f), PAD), (cx(f), PAD + len(TRACK_ROWS) * RH),
                     (0, 0, 0), 2)
    # cursor
    cv2.line(tr, (cx(cur), PAD - 4),
             (cx(cur), PAD + len(TRACK_ROWS) * RH + 4), (0, 0, 220), 3)
    # time axis
    yb = PAD + len(TRACK_ROWS) * RH + 18
    for s in range(0, int(span / fps) + 1, 5):
        f = i0 + int(s * fps)
        cv2.line(tr, (cx(f), yb - 12), (cx(f), yb - 6), (90, 90, 90), 1)
        cv2.putText(tr, f"{s}s", (cx(f) - 10, yb + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1)
    cap = "black line = stimulus delivered   |   red line = current frame"
    if brief_any:
        cap += f"   |   dot = brief mark (< {BRIEF_S:g} s)"
    cv2.putText(tr, cap, (LBL, yb + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                (70, 70, 70), 1)

    head = np.full((34, W, 3), 20, np.uint8)
    cv2.putText(head, hdr, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.66,
                (255, 255, 255), 2)
    return np.vstack([head, vid, tr])


def cut_clip(bpath, spath, start, dur, dest, fps):
    """Extract the same time window from both cameras into a small folder you
    can score on its own. Re-encoded as MJPEG/AVI, which MATLAB's VideoReader
    opens reliably and which seeks fast because every frame is a keyframe."""
    ff = ffmpeg()
    if not ff:
        raise SystemExit("ffmpeg not found - cannot cut the clip")
    a = os.path.join(dest, "cameraA")
    b = os.path.join(dest, "cameraB")
    os.makedirs(a, exist_ok=True)
    os.makedirs(b, exist_ok=True)
    outs = []
    for src, folder, tag in ((bpath, a, "bottom"), (spath, b, "side")):
        stem = os.path.splitext(os.path.basename(src))[0]
        dst = os.path.join(folder, f"{stem}__ppt_{int(start)}s_{int(dur)}s.avi")
        cmd = [ff, "-y", "-loglevel", "error",
               "-ss", f"{start:.3f}", "-i", src, "-t", f"{dur:.3f}",
               "-c:v", "mjpeg", "-q:v", "3", "-an", dst]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(dst) or not os.path.getsize(dst):
            print("  ffmpeg FAILED cutting " + tag + ":")
            print("   " + (r.stderr or "no stderr").strip()[:500])
            raise SystemExit(1)
        cap = cv2.VideoCapture(dst)
        nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        print(f"  {tag:6s} -> {dst}   ({nf} frames)")
        outs.append(dst)
    return outs


def render_from(folder, outdir, zoom, dur_hint):
    """Render the PPT video from a scoring done on a cut clip."""
    mats = sorted(glob.glob(os.path.join(folder, "output", "ScoringAB_*.mat")))
    if not mats:
        raise SystemExit(
            "no ScoringAB_*.mat in " + os.path.join(folder, "output")
            + "\nScore the cut clip first (see RUN_ppt_scoring.m).")
    M = loadmat(mats[0])
    fps = float(np.asarray(M["frameRate"]).ravel()[0])
    sc = np.asarray(M["score"]).ravel().astype(int)
    nUsed = int(float(np.asarray(M["nUsed"]).ravel()[0]))
    rx = np.asarray(M.get("reflexEvents", np.empty((0, 2)))).reshape(-1, 2)
    dF = np.asarray(M["dFrames"]).ravel().astype(int)
    dT = np.asarray(M["dTypes"]).ravel().astype(int)
    names = [str(np.asarray(x).ravel()[0]).strip()
             for x in np.asarray(M["stimNames"]).ravel()][:4]

    bs = sorted(glob.glob(os.path.join(folder, "cameraA", "*.avi")))
    ss = sorted(glob.glob(os.path.join(folder, "cameraB", "*.avi")))
    if not (bs and ss):
        raise SystemExit("cut videos not found in " + folder)
    cb, cs = cv2.VideoCapture(bs[0]), cv2.VideoCapture(ss[0])
    i0, i1 = 1, nUsed
    tmp = os.path.join(outdir, "_frames")
    os.makedirs(tmp, exist_ok=True)
    for f in glob.glob(os.path.join(tmp, "*.png")):
        os.remove(f)

    stem = os.path.splitext(os.path.basename(bs[0]))[0]
    # Build a readable slide title instead of a truncated filename. The mouse
    # comes from the filename, the stimulus from what was actually marked, and
    # the original time window from the __ppt_<start>s_<dur>s tag that --cut
    # appended.
    who = mouse_key(stem)
    used = sorted({int(x) for x in dT if 1 <= int(x) <= 4})
    stim_txt = " + ".join(names[i - 1] for i in used) if used else "no stimulus"
    m = re.search(r"__ppt_(\d+)s_(\d+)s", stem)
    if m:
        t0, dd = int(m.group(1)), int(m.group(2))
        when = f"{t0}-{t0 + dd} s of the session"
    else:
        when = f"{nUsed / fps:.0f} s clip"
    base = "PPT_" + re.sub(r"[^A-Za-z0-9_]+", "_", stem)[:60]
    hdr = (f"{who}   |   {stim_txt}   |   {when}   |   "
           f"{len(dF)} deliveries   |   manual scoring")
    stills, n = [], 0
    for i in range(i0, i1 + 1):
        ok1, fb = cb.read()
        ok2, fs = cs.read()
        if not (ok1 and ok2):
            break
        img = draw(fb, fs, sc, rx, dF, dT, names, i0, i1, i, fps, hdr, zoom=zoom)
        cv2.imwrite(os.path.join(tmp, f"f{n:05d}.png"), img)
        if n in (0, (i1 - i0) // 3, 2 * (i1 - i0) // 3, max(i1 - i0 - 1, 0)):
            p2 = os.path.join(outdir, f"{base}_frame{n:05d}.png")
            cv2.imwrite(p2, img)
            stills.append(p2)
        n += 1
    cb.release()
    cs.release()
    print(f"rendered {n} frame(s) from {os.path.basename(mats[0])}")
    encode(tmp, outdir, base, fps, n, stills)


def encode(tmp, outdir, base, fps, n, stills):
    ff = ffmpeg()
    mp4 = os.path.join(outdir, base + ".mp4")
    if ff and n:
        cmd = [ff, "-y", "-loglevel", "error", "-framerate", f"{fps:g}",
               "-i", os.path.join(tmp, "f%05d.png"),
               "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2:0:0:black",
               "-c:v", "libx264", "-preset", "slow", "-crf", "18",
               "-pix_fmt", "yuv420p", mp4]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not os.path.getsize(mp4):
            print("  ffmpeg FAILED:")
            print("   " + (r.stderr or "no stderr").strip()[:600])
            print(f"  the PNG frames are still in {tmp}")
        else:
            print(f"wrote {mp4}  ({os.path.getsize(mp4)/1e6:.1f} MB)")
            print("  H.264 / yuv420p, which is what PowerPoint plays reliably.")
            shutil.rmtree(tmp, ignore_errors=True)
    else:
        print(f"ffmpeg not found - the PNG frames are in {tmp}")
    for p2 in stills:
        print(f"wrote {p2}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mouse", default=None)
    ap.add_argument("--stimulus", default="pin")
    ap.add_argument("--start", type=float, default=None)
    ap.add_argument("--dur", type=float, default=30.0)
    ap.add_argument("--zoom", action="store_true",
                    help="crop both views to the arena so the mouse is "
                         "bigger on a projected slide")
    ap.add_argument("--out", default=os.path.join(ROOT, "..", "ppt_clips"))
    ap.add_argument("--cut", action="store_true",
                    help="STEP 1: cut the 30 s window out of both cameras into "
                         "ppt_clips\\score_me so you can score just that")
    ap.add_argument("--render-scored", action="store_true",
                    help="STEP 3: render the PPT video from the scoring you "
                         "did on the cut clip")
    a = ap.parse_args()
    outdir = os.path.abspath(a.out)
    os.makedirs(outdir, exist_ok=True)
    scoredir = os.path.join(outdir, "score_me")

    # ---- STEP 3: render from a scoring done on the cut clip ----
    if a.render_scored:
        render_from(scoredir, outdir, a.zoom, a.dur)
        return

    pairs = pair_videos()
    print(f"{len(pairs)} mouse/mice with both cameras: {', '.join(pairs)}\n")

    rows = []
    for k, (b, s) in pairs.items():
        M, mf = load_scoring(k)
        scored = M is not None
        t0, t1, stim, ndel = 300.0, 600.0, "", 0
        if scored:
            blocks, fps, *_ = stim_blocks(M)
            hit = [n for n in blocks if a.stimulus.lower() in n.lower()]
            if hit:
                stim = hit[0]
                t0, t1 = blocks[stim]["t0"], blocks[stim]["t1"]
                ndel = blocks[stim]["n"]
        q = quality(b, s, t0, t1)
        rows.append(dict(mouse=k, scored=int(scored), stimulus=stim,
                         n_deliveries=ndel, t0=round(t0, 1), t1=round(t1, 1), **q))
    Q = pd.DataFrame(rows)
    Q.to_csv(os.path.join(outdir, "Clip_quality_report.csv"), index=False)

    print("=== video quality inside the target block ===")
    cols = ["mouse", "scored", "stimulus", "n_deliveries", "side_detect_pct",
            "side_contrast_sd", "side_sharpness", "side_clipped_pct",
            "bottom_sharpness", "bottom_clipped_pct"]
    print(Q[cols].to_string(index=False))

    cand = Q[Q.scored == 1].copy()
    if cand.empty:
        raise SystemExit("no scored sessions - nothing to label")
    # rank: side detection first, then contrast, then sharpness
    # A composite z-score, not a sum of ranks. Summing three ranks over
    # three mice gave every mouse exactly 6.0 - a tie that hid a real
    # difference: female1 had 9 % of its side view CLIPPED against
    # female2's 0.005 %. Blown-out highlights are the thing you notice on a
    # projector, so clipping carries the most weight here.
    def z(col, sign=1):
        v = cand[col].astype(float)
        s = v.std(ddof=0)
        return sign * (v - v.mean()) / s if s > 1e-9 else v * 0.0
    cand["score_quality"] = (2.0 * z("side_clipped_pct", -1)
                             + 1.0 * z("side_contrast_sd", +1)
                             + 1.0 * z("side_sharpness", +1)
                             + 0.5 * z("bottom_clipped_pct", -1)
                             + 0.5 * z("n_deliveries", +1))
    cand = cand.sort_values("score_quality", ascending=False)
    print("\n=== ranked (scored mice only) ===")
    print(cand[["mouse", "score_quality", "side_clipped_pct",
                "side_contrast_sd", "side_sharpness",
                "n_deliveries"]].to_string(index=False))
    print("  (side_detect_pct was 100 % for all six, so it carries no "
          "information here)")

    pick = a.mouse or cand.iloc[0]["mouse"]
    if pick not in pairs:
        raise SystemExit(f"{pick} not found")
    bpath, spath = pairs[pick]
    M, mf = load_scoring(pick)
    blocks, fps, dF, dT, names = stim_blocks(M)
    hit = [n for n in blocks if a.stimulus.lower() in n.lower()]
    if not hit:
        raise SystemExit(f"{pick} has no block matching '{a.stimulus}'")
    stim = hit[0]
    bt0, bt1 = blocks[stim]["t0"], blocks[stim]["t1"]

    if a.start is None:
        start, nev = busiest_window(M, bt0, bt1, a.dur, fps)
        print(f"\nchose {pick}, '{stim}' block {bt0:.0f}-{bt1:.0f} s")
        print(f"busiest {a.dur:.0f} s window starts at {start:.1f} s "
              f"({nev} labelled events in it)")
    else:
        start, nev = a.start, -1
        print(f"\nusing {pick}, '{stim}', start {start:.1f} s (you set it)")

    # ---- STEP 1: cut the window out so it can be re-scored on its own ----
    if a.cut:
        print(f"\ncutting {a.dur:.0f} s from {start:.1f} s into\n  {scoredir}")
        cut_clip(bpath, spath, start, a.dur, scoredir, fps)
        print("\nNEXT")
        print("  1. in MATLAB open  scoring\\RUN_ppt_scoring.m  and press F5")
        print("     (it points straight at the folder above)")
        print("  2. score the clip - mode 3, and HOLD the a/s/d/f keys")
        print("  3. then run:")
        print("       python make_ppt_clip.py --render-scored --zoom")
        return

    sc = np.asarray(M["score"]).ravel().astype(int)
    rx = np.asarray(M.get("reflexEvents", np.empty((0, 2)))).reshape(-1, 2)
    i0 = int(start * fps) + 1
    i1 = int((start + a.dur) * fps)

    cb, cs = cv2.VideoCapture(bpath), cv2.VideoCapture(spath)
    cb.set(cv2.CAP_PROP_POS_FRAMES, i0 - 1)
    cs.set(cv2.CAP_PROP_POS_FRAMES, i0 - 1)
    tmp = os.path.join(outdir, "_frames")
    os.makedirs(tmp, exist_ok=True)
    for f in glob.glob(os.path.join(tmp, "*.png")):
        os.remove(f)

    base = f"{pick}_{stim.replace(' ', '')}_{int(start)}s_{int(a.dur)}s"
    hdr = (f"{pick}   |   {stim}   |   {start:.0f}-{start + a.dur:.0f} s   "
           f"|   manual scoring")
    stills, n = [], 0
    for i in range(i0, i1 + 1):
        ok1, fb = cb.read()
        ok2, fs = cs.read()
        if not (ok1 and ok2):
            break
        img = draw(fb, fs, sc, rx, dF, dT, names, i0, i1, i, fps, hdr,
                   zoom=a.zoom)
        cv2.imwrite(os.path.join(tmp, f"f{n:05d}.png"), img)
        if n in (0, (i1 - i0) // 3, 2 * (i1 - i0) // 3, i1 - i0 - 1):
            p = os.path.join(outdir, f"{base}_frame{n:05d}.png")
            cv2.imwrite(p, img)
            stills.append(p)
        n += 1
    cb.release(); cs.release()
    print(f"rendered {n} frame(s)")

    ff = ffmpeg()
    mp4 = os.path.join(outdir, base + ".mp4")
    if ff and n:
        # libx264 with yuv420p needs EVEN width and height. The composite is
        # whatever the crops add up to, which can easily be odd - that made
        # ffmpeg fail silently and leave a 0-byte file. Pad up to even.
        cmd = [ff, "-y", "-loglevel", "error", "-framerate", f"{fps:g}",
               "-i", os.path.join(tmp, "f%05d.png"),
               "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2:0:0:black",
               "-c:v", "libx264", "-preset", "slow", "-crf", "18",
               "-pix_fmt", "yuv420p", mp4]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not os.path.getsize(mp4):
            print("  ffmpeg FAILED:")
            print("   " + (r.stderr or "no stderr").strip()[:600])
            print(f"  the PNG frames are still in {tmp}")
        else:
            print(f"wrote {mp4}  ({os.path.getsize(mp4)/1e6:.1f} MB)")
            print("  H.264 / yuv420p, which is what PowerPoint plays reliably.")
            shutil.rmtree(tmp, ignore_errors=True)
    else:
        print(f"ffmpeg not found - the PNG frames are in {tmp}")
    for p in stills:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
