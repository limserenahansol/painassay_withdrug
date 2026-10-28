"""logger_delivery_check.py - check the logger delivery onsets on the video, by eye.

For every session, deliveries in three groups:
  BOTH        found by logger_deliveries.py AND by the logger's own Schmitt indicator (|dt| <= 0.5 s)
  MINE ONLY   found offline only (e.g. a downward press, which the indicator cannot see)
  LOGGER ONLY the indicator fired but logger_deliveries.py found no press (e.g. tool handling)
For a random 3 of BOTH and every MINE ONLY / LOGGER ONLY (up to --max-each), a clip of -1.0 .. +1.5 s:
bottom view (tone curve of the scoring copies) | side view, the logger signal underneath with a cursor,
the onset frame marked. Played at 1/3 speed. Also a sheet PNG (bottom view at -0.3, 0, +0.2, +0.5 s).

  python logger_delivery_check.py --table <day2_session_table.csv> --deliveries <logger_deliveries dir>
      --root <Rogger> --bottom-dir <dir of bottom AVIs> --side-dir <dir of side AVIs> --out <dir>
"""
import argparse
import glob
import os
import re
import subprocess

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FF = r"C:\Users\hsollim\Downloads\ffmpeg-master-latest-win64-gpl-shared\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe"
FPS = 30.0
CAT = {"BOTH": (40, 160, 40), "MINE ONLY": (30, 120, 230), "LOGGER ONLY": (200, 60, 200)}


def lut():
    x = np.arange(256, dtype=np.float32)
    return np.where(x <= 50, x * 190 / 50, 190 + (x - 50) * 65 / 205).clip(0, 255).astype(np.uint8)


def video_for(folder, mouse, phase):
    for p in sorted(glob.glob(os.path.join(folder, "*.avi"))):
        nm = os.path.basename(p).lower()
        if re.match(r"\s*(\d+)", nm).group(1) == mouse and (("base" in nm) == (phase == "pre")):
            return p
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--deliveries", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--bottom-dir", required=True)
    ap.add_argument("--side-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-each", type=int, default=8)
    ap.add_argument("--n-both", type=int, default=3)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    T = pd.read_csv(a.table)
    L = lut()
    summary = []
    for _, r in T.iterrows():
        rid = f"{r.mouse}_d{int(r.day)}_{r.phase}"
        p = [q for q in glob.glob(os.path.join(a.root, "**", f"{r.logger_session}_*_pressure.csv"), recursive=True) if "__MACOSX" not in q][0]
        D = pd.read_csv(p)
        c, x, ind = D.ttl_count.to_numpy(), D.corrected.to_numpy(float), D.indicator.to_numpy()
        base = pd.Series(x).rolling(799, center=True, min_periods=79).median().to_numpy()
        d = x - base
        E = pd.read_csv(os.path.join(a.deliveries, f"{rid}_logger_deliveries.csv"))
        mine = E.onset_cam_frame.to_numpy()
        on = np.flatnonzero(np.diff(np.r_[0, ind]) == 1)
        logger = c[on]
        logger = logger[logger >= 1]
        both = [f for f in mine if np.any(np.abs(logger - f) <= 15)]
        mine_only = [f for f in mine if not np.any(np.abs(logger - f) <= 15)]
        log_only = [f for f in logger if not np.any(np.abs(mine - f) <= 15)]
        rng = np.random.default_rng(int(r.mouse) + (r.phase == "post"))
        pick = [("BOTH", f) for f in rng.choice(both, min(a.n_both, len(both)), replace=False)] if both else []
        pick += [("MINE ONLY", f) for f in mine_only[:a.max_each]] + [("LOGGER ONLY", f) for f in log_only[:a.max_each]]
        summary.append(dict(recording_id=rid, mine=len(mine), logger_indicator=len(logger), both=len(both),
                            mine_only=len(mine_only), logger_only=len(log_only), clips=len(pick)))
        vb, vs = video_for(a.bottom_dir, str(r.mouse), r.phase), video_for(a.side_dir, str(r.mouse), r.phase)
        if not (vb and vs) or not pick:
            print(f"{rid}: no videos or nothing to check"); continue
        cb, cs = cv2.VideoCapture(vb), cv2.VideoCapture(vs)
        tmp = os.path.join(a.out, "_tmp.avi")
        W = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"MJPG"), 10, (960, 480))
        sheet = []
        cf = c.astype(float) / FPS
        for cat, f in sorted(pick, key=lambda z: z[1]):
            f = int(f)
            f0, f1 = f - 30, f + 45
            # logger panel: |d| symlog-like over -2 .. +2 s
            tsel = (cf >= (f - 60) / FPS) & (cf <= (f + 60) / FPS)
            fig = plt.figure(figsize=(9.6, 1.6), dpi=100)
            ax = fig.add_axes([0.06, 0.2, 0.92, 0.72])
            ax.plot(cf[tsel] - f / FPS, d[tsel], lw=0.8, color="k")
            ax.set_yscale("symlog", linthresh=2000); ax.set_ylim(-1e7, 1e7)
            ax.axhline(3e5, ls="--", color="0.6", lw=0.6); ax.axhline(-3e5, ls="--", color="0.6", lw=0.6)
            ax.axvline(0, color="r", lw=0.8); ax.set_xlim(-2, 2); ax.tick_params(labelsize=7)
            ax.set_xlabel("s from the onset", fontsize=7, labelpad=0)
            fig.canvas.draw()
            panel = np.asarray(fig.canvas.buffer_rgba())[:, :, :3][:, :, ::-1].copy()
            xof = lambda tsec: int(round(ax.transData.transform((tsec, 0))[0]))
            plt.close(fig)
            cb.set(cv2.CAP_PROP_POS_FRAMES, f0 - 1); cs.set(cv2.CAP_PROP_POS_FRAMES, f0 - 1)
            for k in range(f0, f1):
                ok1, b = cb.read(); ok2, s = cs.read()
                if not (ok1 and ok2):
                    break
                fr = np.zeros((480, 960, 3), np.uint8)
                fr[0:320, 0:480] = cv2.resize(cv2.LUT(b, L), (480, 320))
                fr[0:320, 480:960] = cv2.resize(s, (480, 320))
                pn = panel.copy()
                xx = xof((k - f) / FPS)
                cv2.line(pn, (xx, 0), (xx, pn.shape[0] - 1), (0, 0, 255), 1)
                fr[320:480] = cv2.resize(pn, (960, 160))
                col = CAT[cat]
                cv2.putText(fr, f"{rid}  {cat}  onset {f / FPS:.2f} s (frame {f})   t = {(k - f) / FPS:+.2f} s", (8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
                if k == f:
                    cv2.rectangle(fr, (2, 2), (957, 317), (0, 0, 255), 3)
                for _ in range(2 if k == f else 1):
                    W.write(fr)
                if k - f in (-9, 0, 6, 15):
                    sheet.append((cat, f, k - f, cv2.resize(cv2.LUT(b, L), (240, 160))))
            for _ in range(5):
                W.write(np.zeros((480, 960, 3), np.uint8))
        W.release()
        out = os.path.join(a.out, f"{rid}_logger_check.mp4")
        subprocess.run([FF, "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-crf", "22", "-pix_fmt", "yuv420p", out], check=True)
        os.remove(tmp)
        # sheet: one row per checked delivery
        rows = {}
        for cat, f, dk, im in sheet:
            rows.setdefault((cat, f), []).append(im)
        R = []
        for (cat, f), ims in rows.items():
            row = np.hstack(ims + [np.zeros((160, 240, 3), np.uint8)] * (4 - len(ims)))
            lab = np.full((24, row.shape[1], 3), 255, np.uint8)
            cv2.putText(lab, f"{cat}  {f / FPS:.2f} s   (bottom view at -0.3, 0, +0.2, +0.5 s)", (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CAT[cat], 1)
            R.append(np.vstack([lab, row]))
        cv2.imwrite(os.path.join(a.out, f"{rid}_logger_check_sheet.png"), np.vstack(R))
        print(f"{rid}: mine {len(mine)}, indicator {len(logger)}, both {len(both)}, mine only {len(mine_only)}, "
              f"logger only {len(log_only)} -> {out}")
    pd.DataFrame(summary).to_csv(os.path.join(a.out, "logger_check_summary.csv"), index=False)


if __name__ == "__main__":
    main()
