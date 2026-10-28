"""behavior_day2_grooming_maybe.py - Day-2 "maybe grooming" clips for your judgement (like Day-1 rounds 2 and 3).

The Day-1 grooming model found about half of the grooming in a session it had not seen (leave-one-session-out), so
Day-2 misses are likely where the grooming probability rises but stays below the bout threshold (0.4). Candidates:
peaks of the smoothed probability 0.1-0.4, >= 5 s apart, not within 3 s of a bout, outside the first 15 s; the
highest per session (up to --per-session). Each clip: -2 s .. +4 s around the peak, bottom | side, numbered.
Your answer "grooming = #3, #7, ..." is applied with behavior_day2_grooming_apply.py (bouts added, model untouched).
Output: behavior_auto\\20261021\\grooming\\day2_grooming_maybe.mp4, _sheet.png, day2_grooming_maybe.csv

  python behavior_day2_grooming_maybe.py [--per-session 6]
"""
import argparse
import os
import subprocess
import sys

import cv2
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from behavior_day2_examples import AUTO, FF, FPS, VID, lut, side_index, spans, video  # noqa: E402

OUT = os.path.join(AUTO, "grooming")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-session", type=int, default=6)
    a = ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    L = lut()
    items = []
    for m in ("5941", "5937", "5939", "1191"):
        for ph in ("pre", "post"):
            tag = f"{m}_d2_{ph}"
            Z = np.load(os.path.join(AUTO, f"{tag}_move_groom.npz"))
            ps, gr = Z["ps"].astype(float), Z["groom"].astype(bool)
            near = np.zeros(len(gr), bool)
            for s, e in spans(gr):
                near[max(0, s - 90):e + 90] = True
            pk, _ = find_peaks(np.nan_to_num(ps), height=0.1, distance=int(5 * FPS))
            pk = [p for p in pk if ps[p] < 0.4 and not near[p] and p > 15 * FPS]
            pk = sorted(pk, key=lambda p: -ps[p])[:a.per_session]
            items += [(tag, int(p), float(ps[p])) for p in sorted(pk)]
    tmp = os.path.join(OUT, "_tmp_maybe.avi")
    W = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"MJPG"), 30, (1080, 420))
    sheet, rows = [], []
    for k, (tag, p, pv) in enumerate(items, 1):
        m, ph = tag.split("_")[0], tag.split("_")[2]
        cb, cs = cv2.VideoCapture(video(VID, m, ph)), cv2.VideoCapture(video(os.path.join(VID, "side"), m, ph))
        f0, f1 = max(0, p - 60), p + 120
        cb.set(cv2.CAP_PROP_POS_FRAMES, f0); cs.set(cv2.CAP_PROP_POS_FRAMES, side_index(tag, f0))
        keep = {}
        for f in range(f0, f1):
            okb, b = cb.read(); oks, sd = cs.read()
            if not (okb and oks):
                break
            fr = np.full((420, 1080, 3), 255, np.uint8)
            fr[40:400, :540] = cv2.resize(cv2.LUT(b, L), (540, 360)); fr[40:400, 540:] = cv2.resize(sd, (540, 360))
            cv2.putText(fr, f"MAYBE #{k}   {tag}   {f / FPS:.1f} s   (grooming probability peak {pv:.2f} at {p / FPS:.1f} s)",
                        (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)
            W.write(fr)
            for key, ff in (("a", p - 30), ("b", p), ("c", p + 45)):
                if f == ff:
                    keep[key] = (cv2.resize(cv2.LUT(b, L), (240, 160)), cv2.resize(sd, (240, 160)))
        for _ in range(15):
            W.write(np.full((420, 1080, 3), 255, np.uint8))
        if len(keep) == 3:
            row = np.hstack([keep["b"][1], keep["a"][0], keep["b"][0], keep["c"][0], keep["c"][1]])
            lab = np.full((24, row.shape[1], 3), 255, np.uint8)
            cv2.putText(lab, f"MAYBE #{k}  {tag}  peak {p / FPS:.1f} s (p {pv:.2f}) | side peak | bottom -1 s, peak, +1.5 s | side +1.5 s",
                        (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200), 1)
            sheet.append(np.vstack([lab, row]))
        rows.append(dict(clip=k, session=tag, peak_frame=p, peak_s=round(p / FPS, 2), prob=round(pv, 3), your_answer=""))
        print(f"MAYBE #{k:2d} {tag} {p / FPS:7.1f} s p {pv:.2f}", flush=True)
    W.release()
    out = os.path.join(OUT, "day2_grooming_maybe.mp4")
    subprocess.run([FF, "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", out], check=True)
    os.remove(tmp)
    if sheet:
        cv2.imwrite(os.path.join(OUT, "day2_grooming_maybe_sheet.png"), np.vstack(sheet))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "day2_grooming_maybe.csv"), index=False)
    print(f"{len(rows)} clips -> {out}")


if __name__ == "__main__":
    main()
