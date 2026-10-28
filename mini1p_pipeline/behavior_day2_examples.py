"""behavior_day2_examples.py - show how the Day-2 automatic movement and grooming were annotated.

Every grooming bout of the 8 sessions and 2 movement bouts per session (the longest and a random one):
  clip  (-1 s .. bout end + 1 s, max 8 s): bottom (scoring tone curve) | side, the label on top in red while the
         bout is on, and a track underneath (movement cyan, grooming pink, grooming probability black, cursor)
  sheet one row per bout: side and bottom frame at the bout middle + bottom frames at the start / middle / end
Output: behavior_auto\\20261021\\examples\\day2_move_groom_examples.mp4 + _sheet.png

  python behavior_day2_examples.py
"""
import glob
import os
import re
import subprocess

import cv2
import numpy as np
import pandas as pd

BASE = r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553"
VID = BASE + r"\mini1p_SBI_day2"
AUTO = BASE + r"\behavior_auto\20261021"
OUT = AUTO + r"\examples"
FF = r"C:\Users\hsollim\Downloads\ffmpeg-master-latest-win64-gpl-shared\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe"
FPS = 30.0
SIDE_DROP = {"5941_d2_post": (5100, 18)}


def lut():
    x = np.arange(256, dtype=np.float32)
    return np.where(x <= 50, x * 190 / 50, 190 + (x - 50) * 65 / 205).clip(0, 255).astype(np.uint8)


def video(folder, mouse, phase):
    for p in sorted(glob.glob(os.path.join(folder, "*.avi"))):
        nm = os.path.basename(p).lower()
        if re.match(r"\s*(\d+)", nm).group(1) == mouse and (("base" in nm) == (phase == "pre")):
            return p


def spans(m):
    d = np.diff(np.r_[0, np.asarray(m, int), 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def side_index(tag, f):
    if tag in SIDE_DROP and f >= SIDE_DROP[tag][0]:
        return f - SIDE_DROP[tag][1]
    return f


def main():
    os.makedirs(OUT, exist_ok=True)
    L = lut()
    rng = np.random.default_rng(0)
    items = []
    for z in sorted(glob.glob(os.path.join(AUTO, "*_move_groom.npz"))):
        tag = os.path.basename(z).replace("_move_groom.npz", "")
        Z = np.load(z)
        for s, e in spans(Z["groom"]):
            items.append((tag, "GROOMING", int(s), int(e)))
        mv = [x for x in spans(Z["move"]) if x[1] - x[0] >= 15]
        if mv:
            longest = max(mv, key=lambda x: x[1] - x[0])
            pick = [longest] + [mv[i] for i in rng.choice(len(mv), min(1, len(mv)), replace=False) if mv[i] != longest]
            items += [(tag, "MOVEMENT", int(s), int(e)) for s, e in pick]
    tmp = os.path.join(OUT, "_tmp.avi")
    W = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"MJPG"), 30, (1080, 520))
    sheet = []
    for tag, kind, s, e in items:
        m, ph = tag.split("_")[0], tag.split("_")[2]
        Z = np.load(os.path.join(AUTO, f"{tag}_move_groom.npz"))
        cb, cs = cv2.VideoCapture(video(VID, m, ph)), cv2.VideoCapture(video(os.path.join(VID, "side"), m, ph))
        f0 = max(0, s - 30); f1 = min(len(Z["move"]) - 1, min(e, s + 210) + 30)
        cb.set(cv2.CAP_PROP_POS_FRAMES, f0); cs.set(cv2.CAP_PROP_POS_FRAMES, side_index(tag, f0))
        mid = (s + min(e, s + 210)) // 2
        keep = {}
        for f in range(f0, f1):
            okb, b = cb.read(); oks, sd = cs.read()
            if not (okb and oks):
                break
            fr = np.full((520, 1080, 3), 255, np.uint8)
            fr[30:390, :540] = cv2.resize(cv2.LUT(b, L), (540, 360)); fr[30:390, 540:] = cv2.resize(sd, (540, 360))
            on = s <= f < e
            cv2.putText(fr, f"{kind} (automatic)  {tag}  {f / FPS:.1f} s   bout {s / FPS:.1f}-{e / FPS:.1f} s", (8, 21),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 220) if on else (90, 90, 90), 2)
            # track: movement / grooming / probability over the clip
            n = f1 - f0
            xs = lambda k: int(10 + (k - f0) / max(n - 1, 1) * 1060)
            for k in range(f0, f1):
                if Z["move"][k]:
                    cv2.line(fr, (xs(k), 410), (xs(k), 430), (200, 180, 25), 1)
                if Z["groom"][k]:
                    cv2.line(fr, (xs(k), 440), (xs(k), 460), (165, 80, 220), 1)
            pts = np.array([[xs(k), int(505 - 40 * float(Z["ps"][k]))] for k in range(f0, f1)], np.int32)
            cv2.polylines(fr, [pts], False, (0, 0, 0), 1)
            cv2.line(fr, (xs(f), 400), (xs(f), 515), (0, 0, 255), 2)
            for txt, y in (("movement", 425), ("grooming", 455), ("p(groom)", 500)):
                cv2.putText(fr, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1)
            W.write(fr)
            for key, ff in (("start", s), ("mid", mid), ("end", min(e, s + 210) - 1)):
                if f == ff:
                    keep[key] = (cv2.resize(cv2.LUT(b, L), (270, 180)), cv2.resize(sd, (270, 180)))
        for _ in range(10):
            W.write(np.full((520, 1080, 3), 255, np.uint8))
        if len(keep) == 3:
            row = np.hstack([keep["mid"][1], keep["start"][0], keep["mid"][0], keep["end"][0]])
            lab = np.full((24, row.shape[1], 3), 255, np.uint8)
            cv2.putText(lab, f"{kind}  {tag}  {s / FPS:.1f}-{e / FPS:.1f} s  |  side (middle) | bottom: start, middle, end", (4, 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200) if kind == "GROOMING" else (120, 100, 0), 1)
            sheet.append(np.vstack([lab, row]))
        print(f"{kind:9s} {tag} {s / FPS:6.1f}-{e / FPS:6.1f} s", flush=True)
    W.release()
    out = os.path.join(OUT, "day2_move_groom_examples.mp4")
    subprocess.run([FF, "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-crf", "22", "-pix_fmt", "yuv420p", out], check=True)
    os.remove(tmp)
    cv2.imwrite(os.path.join(OUT, "day2_move_groom_examples_sheet.png"), np.vstack(sheet))
    pd.DataFrame(items, columns=["session", "kind", "start_frame", "end_frame"]).to_csv(os.path.join(OUT, "examples.csv"), index=False)
    print("->", out)


if __name__ == "__main__":
    main()
