"""Close-up of chosen grooming bouts (numbers G1.. as in behavior_groom_figures.py):
  grooming_closeup_G<k>.png   12 side-view frames over the bout (+1 s before / after), larger
  grooming_closeup.mp4        the same bouts as video at real speed (side | bottom view)

  python behavior_groom_closeup.py 3 10 12 15 20
"""
import os
import subprocess
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2
import numpy as np
import scipy.io as sio
from behavior_final_move_groom import S, valid_name, spans, AUTO_DIR, FPS, BASE

OUT = BASE + r"\behavior_auto\20261019\grooming"
FF = r"C:\Users\hsollim\Downloads\ffmpeg-master-latest-win64-gpl-shared\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe"
want = [int(x) for x in sys.argv[1:]]
bouts = []
for tag, stem, side, *_ in S:
    A = sio.loadmat(rf"{AUTO_DIR}\ScoringAB_{valid_name(stem)}.mat", squeeze_me=True)
    for s, e in spans(np.asarray(A["groomVec"]).astype(bool)):
        bouts.append((tag, stem, side, s, e))
tmp = OUT + r"\closeup_tmp.avi"
W = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"MJPG"), 30, (1080, 400))
for k in want:
    tag, stem, side, s, e = bouts[k - 1]
    cs = cv2.VideoCapture(rf"K:\mini1p_pain_behavior\{side}")
    cb = cv2.VideoCapture(rf"{BASE}\videos_scoring\20261019\bottom_enhanced\{stem}.mp4")
    a, b = max(0, s - 30), e + 30
    tl = []
    for f in np.linspace(a, b - 1, 12).astype(int):
        cs.set(cv2.CAP_PROP_POS_FRAMES, int(f)); ok, im = cs.read()
        c = cv2.resize(im[40:455, 150:630], (320, 277))
        inb = s <= f < e
        cv2.putText(c, "%.1f s%s" % (f / FPS, "" if inb else " (outside)"), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 255) if inb else (120, 120, 120), 2)
        tl.append(c)
    img = np.vstack([np.hstack(tl[:6]), np.hstack(tl[6:])])
    head = np.full((40, img.shape[1], 3), 255, np.uint8)
    cv2.putText(head, "G%d  %s  grooming %.1f-%.1f s (frames from 1 s before to 1 s after)" % (k, tag, s / FPS, e / FPS),
                (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (140, 30, 200), 2)
    cv2.imwrite(OUT + r"\grooming_closeup_G%d.png" % k, np.vstack([head, img]))
    cs.set(cv2.CAP_PROP_POS_FRAMES, a); cb.set(cv2.CAP_PROP_POS_FRAMES, a)
    for f in range(a, b):
        ok1, x = cs.read(); ok2, y = cb.read()
        if not (ok1 and ok2):
            break
        fr = np.zeros((400, 1080, 3), np.uint8)
        fr[40:, :540] = cv2.resize(x, (540, 360)); fr[40:, 540:] = cv2.resize(y, (540, 360))
        lab = "GROOMING" if s <= f < e else "before / after"
        cv2.putText(fr, "G%d  %s  %.1f s   %s" % (k, tag, f / FPS, lab), (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 80, 220) if s <= f < e else (180, 180, 180), 2)
        W.write(fr)
    for _ in range(15):
        W.write(np.zeros((400, 1080, 3), np.uint8))
W.release()
out = OUT + r"\grooming_closeup.mp4"
subprocess.run([FF, "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", out], check=True)
os.remove(tmp)
print(out)
