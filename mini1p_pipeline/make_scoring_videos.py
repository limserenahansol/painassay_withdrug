"""Scoring copies of the BOTTOM camera videos: same frames, paws made visible.

The bottom view is backlit: fur ~5-20 grey, paws ~20-40, mesh/background bright. Each frame
goes through the tone curve tested on 5941 (2026-10-20): 0-50 -> 0-190, 50-255 -> 190-255
(no CLAHE: it washed the paws out again; no temporal smoothing: it would blur a fast
withdrawal). Written as H.264 at 30 fps (the real, SYNC-triggered rate; the AVI header says
60), one output frame per input frame. Every output is checked frame-for-frame in count
against its input; a mismatch is an error and the file is renamed *_BAD.mp4.

The side camera is not copied: its white background is intended and the scorer opens the
original side AVI.

  python make_scoring_videos.py --in-dir K:\\mini1p_pain_behavior\\bottom \\
      --out-dir C:\\...\\videos_scoring\\20261019\\bottom_enhanced [--ffmpeg PATH] [--only 5941]
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np

FF_DEFAULT = r"C:\Users\hsollim\Downloads\ffmpeg-master-latest-win64-gpl-shared\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe"


def tone_lut():
    x = np.arange(256, dtype=np.float32)
    return np.where(x <= 50, x * 190 / 50, 190 + (x - 50) * 65 / 205).clip(0, 255).astype(np.uint8)


def count_frames(ffprobe, path):
    r = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-count_packets",
                        "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", path],
                       capture_output=True, text=True)
    return int(r.stdout.strip().split(",")[0])


def convert(src, dst, ffmpeg, fps):
    cap = cv2.VideoCapture(src)
    n_in = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    lut = tone_lut()
    tmp = dst + ".partial.mp4"
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{W}x{H}",
           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "fast", "-crf", "16",
           "-pix_fmt", "yuv420p", "-r", str(fps), tmp]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = 0
    t0 = time.time()
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        p.stdin.write(cv2.LUT(g, lut).tobytes())
        n += 1
    p.stdin.close()
    p.wait()
    return n_in, n, tmp, time.time() - t0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ffmpeg", default=FF_DEFAULT)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--only", default="", help="substring of the file names to convert")
    a = ap.parse_args(argv)
    ffprobe = os.path.join(os.path.dirname(a.ffmpeg), "ffprobe.exe")
    os.makedirs(a.out_dir, exist_ok=True)
    srcs = sorted(glob.glob(os.path.join(a.in_dir, "*.avi")))
    srcs = [s for s in srcs if a.only in os.path.basename(s)]
    log = []
    bad = 0
    for s in srcs:
        dst = os.path.join(a.out_dir, os.path.splitext(os.path.basename(s))[0] + ".mp4")
        if os.path.isfile(dst):
            print(f"exists, skipped: {dst}")
            continue
        n_in, n_read, tmp, dt = convert(s, dst, a.ffmpeg, a.fps)
        n_out = count_frames(ffprobe, tmp)
        ok = n_out == n_read == n_in
        final = dst if ok else dst.replace(".mp4", "_BAD.mp4")
        os.replace(tmp, final)
        bad += not ok
        rec = dict(source=s, output=final, frames_in_header=n_in, frames_read=n_read, frames_out=n_out,
                   equal=ok, seconds=round(dt, 1), bytes=os.path.getsize(final))
        log.append(rec)
        print(f"{'OK ' if ok else 'BAD'} {os.path.basename(s)}: in {n_in}, read {n_read}, out {n_out} "
              f"({dt:.0f} s, {rec['bytes'] / 1e9:.2f} GB)")
    with open(os.path.join(a.out_dir, "scoring_videos_log.json"), "a", encoding="utf-8") as f:
        for r in log:
            f.write(json.dumps(r) + "\n")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
