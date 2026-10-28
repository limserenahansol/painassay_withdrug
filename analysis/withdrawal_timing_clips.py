"""withdrawal_timing_clips.py - a few sessions, the hand-scored withdrawal / flinch marks against the side video (2026-10-23, user:
"check 4x slower ... see when the foot is moving ... if you find the rule such as always +0.3 s delayed, apply it to the others").

Per session 8 marks (5 withdrawals within 1.5 s of a delivery, 2 flinches, 1 withdrawal > 1.5 s after the delivery; seeded):
  strips\\<rid>_<kind>_<frame>.png   side camera around the mouse, every 2nd frame from 1.0 s before to 0.27 s after the mark
                                     (20 tiles, each labelled with its time from the mark; red = the mark, blue = the delivery mark)
  <rid>_marks_4x_slower.mp4          side (large) + bottom camera, 1.5 s before to 0.7 s after each mark, 4x slower, the time
                                     from the mark written on every frame and the mark / delivery flashed
  events.csv                         the marks used

  python withdrawal_timing_clips.py
"""
import glob
import os
import subprocess

import cv2
import numpy as np
import pandas as pd
import scipy.io as sio

B = r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553"
OUT = os.path.join(B, "analysis", "withdrawal_timing", "clips")
FF = r"C:\Users\hsollim\Downloads\ffmpeg-master-latest-win64-gpl-shared\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe"
FPS = 30.0
SC = {1: os.path.join(B, "scoring_mini1p", "20261019_final"), 2: os.path.join(B, "scoring_mini1p", "20261021_final")}
FEAT = {1: os.path.join(B, "behavior_auto", "20261019"), 2: os.path.join(B, "behavior_auto", "20261021")}
PICK = [(1, "5941", "pre"), (1, "1191", "post"), (2, "5939", "post"), (2, "1191", "pre")]


def load(d, mouse, ph):
    for p in glob.glob(os.path.join(SC[d], "ScoringAB_*.mat")):
        S = sio.loadmat(p, squeeze_me=True)
        if str(S["mouseID"]).strip() == mouse and (("pre" if str(S["phase"]).lower().startswith("base") else "post") == ph):
            return S
    raise FileNotFoundError((d, mouse, ph))


def frames(path, f0, f1):
    """0-based frames f0..f1 (inclusive) of a video"""
    c = cv2.VideoCapture(path)
    c.set(cv2.CAP_PROP_POS_FRAMES, f0)
    out = []
    for _ in range(f0, f1 + 1):
        ok, im = c.read()
        out.append(im if ok else None)
    c.release()
    return out


def main(videos=True):
    os.makedirs(os.path.join(OUT, "strips"), exist_ok=True)
    rows = []
    for d, mouse, ph in PICK:
        rid = f"{mouse}_d{d}_{ph}"
        S = load(d, mouse, ph)
        off = int(round(float(S.get("syncOff", 0) or 0) * FPS))
        Fe = pd.read_csv(os.path.join(FEAT[d], f"{rid}_bfeatures.csv"))
        re = np.atleast_2d(S["reflexEvents"]).astype(int).reshape(-1, 2)
        dF = np.sort(np.atleast_1d(S["dFrames"]).astype(int))
        cand = []
        for w, typ in re:
            k = np.flatnonzero(dF <= w)
            if not len(k):
                continue
            lat = (w - dF[k[-1]]) / FPS
            cand.append((int(w), "withdrawal" if typ == 1 else "flinch", lat, int(dF[k[-1]])))
        C = pd.DataFrame(cand, columns=["mark", "kind", "lat", "deliv"])
        C = C[(C.mark > 60) & (C.mark < len(Fe) - 60)]
        rng = np.random.default_rng(int(mouse) + d)
        sel = []
        for kind, cond, n in (("withdrawal", C.lat <= 1.5, 5), ("flinch", C.lat <= 1.5, 2), ("withdrawal", C.lat > 1.5, 1)):
            Z = C[(C.kind == kind) & cond]
            if len(Z):
                sel += list(Z.iloc[rng.choice(len(Z), min(n, len(Z)), replace=False)].itertuples(index=False))
        sel = sorted(sel, key=lambda r: r.mark)
        vw = cv2.VideoWriter(os.path.join(OUT, "_tmp.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 30, (1280, 560))
        for r in sel:
            w = r.mark
            cx = int(np.nanmedian(Fe.s_cx.iloc[max(0, w - 31):w + 9])) if np.isfinite(Fe.s_cx.iloc[w - 1]) else 400
            top = int(np.nanmedian(Fe.s_top.iloc[max(0, w - 31):w + 9])) if np.isfinite(Fe.s_top.iloc[w - 1]) else 240
            x0 = int(np.clip(cx - 170, 0, 720 - 340)); y0 = int(np.clip(top - 10, 0, 480 - 230))
            bx = int(np.nanmedian(Fe.b_cx.iloc[max(0, w - 31):w + 9])) if np.isfinite(Fe.b_cx.iloc[w - 1]) else 360
            by = int(np.nanmedian(Fe.b_cy.iloc[max(0, w - 31):w + 9])) if np.isfinite(Fe.b_cy.iloc[w - 1]) else 240
            bx0 = int(np.clip(bx - 170, 0, 720 - 340)); by0 = int(np.clip(by - 115, 0, 480 - 230))
            # strip: every 2nd frame, -30 .. +8 frames from the mark (frame indices are 1-based in the scoring file);
            # each tile = side camera (feet included) over the bottom camera (soles on the mesh)
            side = frames(str(S["sidFile"]), w - 1 - 30 + off, w - 1 + 8 + off)
            bot = frames(str(S["botFile"]), w - 1 - 30, w - 1 + 8)
            tiles = []
            for j, k in enumerate(range(-30, 9, 2)):
                im, ib = side[k + 30], bot[k + 30]
                ts = np.zeros((230, 340, 3), np.uint8) if im is None else im[y0:y0 + 230, x0:x0 + 340].copy()
                tb = np.zeros((230, 340, 3), np.uint8) if ib is None else cv2.resize(ib, (340, 230))   # the whole bottom view
                t = np.vstack([ts, np.full((4, 340, 3), 255, np.uint8), tb])
                col = (0, 0, 255) if k == 0 else (255, 120, 0) if (w + k) == r.deliv else None
                if col:
                    cv2.rectangle(t, (0, 0), (339, t.shape[0] - 1), col, 5)
                cv2.putText(t, f"{k / FPS:+.2f} s", (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                cv2.putText(t, f"{k / FPS:+.2f} s", (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(t, f"#{j + 1}", (280, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                tiles.append(t)
            grid = np.vstack([np.hstack(tiles[i:i + 5]) for i in range(0, 20, 5)])
            head = np.full((40, grid.shape[1], 3), 255, np.uint8)
            cv2.putText(head, f"{rid}  {r.kind} mark at frame {w} ({w / FPS:.2f} s); delivery mark {(r.deliv - w) / FPS:+.2f} s; "
                        f"tiles = side (top) + bottom camera every 2 frames; red = mark, blue = delivery", (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 0), 1)
            fn = os.path.join(OUT, "strips", f"{rid}_{r.kind}_{w}.png")
            cv2.imwrite(fn, np.vstack([head, grid]))
            rows.append(dict(rid=rid, day=d, mouse=mouse, phase=ph, kind=r.kind, mark_frame=w, delivery_frame=r.deliv,
                             mark_minus_delivery_s=round(r.lat, 3), strip=fn, syncOff=float(S.get("syncOff", 0) or 0)))
            if not videos:
                continue
            # 4x slower clip: -1.5 .. +0.7 s, each frame written 4 times
            a, b = w - 1 - 45, w - 1 + 21
            sv = frames(str(S["sidFile"]), a + off, b + off)
            bv = frames(str(S["botFile"]), a, b)
            for i, (si, bi) in enumerate(zip(sv, bv)):
                k = a + i - (w - 1)
                fr = np.zeros((560, 1280, 3), np.uint8)
                if si is not None:
                    fr[40:520, 0:720] = si
                if bi is not None:
                    fr[40 + 120:40 + 120 + 240, 720:1280] = cv2.resize(bi, (560, 373))[66:306]
                cv2.putText(fr, f"{rid}  {r.kind}   time from YOUR MARK: {k / FPS:+.2f} s   (4x slower)", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                if k == 0 or k == 1:
                    cv2.rectangle(fr, (2, 40), (717, 519), (0, 0, 255), 8)
                    cv2.putText(fr, "YOUR MARK", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 4)
                if (a + i + 1) == r.deliv:
                    cv2.putText(fr, "DELIVERY MARK", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 160, 0), 4)
                cv2.putText(fr, "side camera", (10, 545), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                cv2.putText(fr, "bottom camera", (730, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                for _ in range(4):
                    vw.write(fr)
            for _ in range(20):
                vw.write(np.zeros((560, 1280, 3), np.uint8))
        vw.release()
        if not videos:
            os.remove(os.path.join(OUT, "_tmp.avi"))
            continue
        out = os.path.join(OUT, f"{rid}_marks_4x_slower.mp4")
        subprocess.run([FF, "-y", "-loglevel", "error", "-i", os.path.join(OUT, "_tmp.avi"), "-c:v", "libx264", "-crf", "23",
                        "-pix_fmt", "yuv420p", out], check=True)
        os.remove(os.path.join(OUT, "_tmp.avi"))
        print(rid, len(sel), "marks ->", out)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "events.csv"), index=False)
    print("->", OUT)


if __name__ == "__main__":
    import sys
    main(videos="--strips-only" not in sys.argv)
