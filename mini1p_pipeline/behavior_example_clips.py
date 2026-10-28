"""Example clips of every annotated behaviour: WHY it is withdrawal, flinch, attending,
lick/bite, guarding, escape/rearing, general movement, grooming.

For each behaviour, up to --n typical examples are taken from all sessions (taps: the delivery
with a tap at the median latency; held behaviours: bouts of median duration that do not overlap
another behaviour; spread over mice). Each clip shows side | bottom view, the definition of the
behaviour, and an annotation track under the video (one row per behaviour, black lines =
deliveries, red line = the current frame, dots = taps), from 1.5 s before to 2 s after.

Outputs (--out-dir): behaviour_examples_<behaviour>.mp4, behaviour_examples_all.mp4,
behaviour_examples_sheet.png (the middle frame of every example)

  python behavior_example_clips.py --final-dir DIR [DIR2 ...] --out-dir OUT [--n 3]

10-23 (review): both days (16 sessions); a readable timeline (canvas 1080 x 640, rows 24 px, labels 0.62); withdrawal and
flinch examples from deliveries with only that reflex (they were the same deliveries); each tap example ends with a 4x slower
replay of +-0.5 s around the tap; guarding (0 bouts) has no row.
"""
import argparse
import glob
import os
import subprocess
import sys

import cv2
import numpy as np
import scipy.io as sio

FPS = 30.0
BASE = r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553"
FF = r"C:\Users\hsollim\Downloads\ffmpeg-master-latest-win64-gpl-shared\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe"
DEF = {
    "withdrawal": "paw lifted / pulled away from the stimulus right after contact (tap)",
    "flinch": "fast shake or flick of the stimulated paw (tap)",
    "attending": "mouse turns its head to look at / sniff the stimulated paw, no mouth contact (held)",
    "lick/bite": "mouth contact with the stimulated paw: licking or biting (held)",
    "guarding": "paw held up off the floor > 2 s, no weight on it (held)",
    "escape/rearing": "stands up on the hind legs / tries to climb or jump out (held)",
    "general movement": "AUTOMATIC: walks a few steps or turns the body (centroid / body-axis change)",
    "grooming": "AUTOMATIC (examples checked by eye): forepaws stroke the face or the head licks the flank, body still",
}
ROWS = ["withdrawal", "flinch", "attending", "lick/bite", "guarding", "escape/rearing", "general movement", "grooming"]
TRACK_ROWS = [r for r in ROWS if r != "guarding"]          # guarding: 0 bouts in all 16 sessions (no row, no clip)
CW, CH, TOP, CAMB = 1080, 660, 52, 404                    # canvas, header, camera bottom
COL = {"withdrawal": (30, 190, 230), "flinch": (120, 120, 120), "attending": (190, 115, 50), "lick/bite": (40, 90, 215),
       "guarding": (180, 75, 140), "escape/rearing": (90, 165, 50), "general movement": (200, 180, 25), "grooming": (165, 80, 220)}


def spans(m):
    d = np.diff(np.r_[0, np.asarray(m, int), 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def load(p):
    S = sio.loadmat(p, squeeze_me=True)
    n = int(S["nUsed"])
    sc = np.atleast_1d(S["score"]).astype(int)[:n]
    re = np.atleast_2d(S["reflexEvents"]).astype(int).reshape(-1, 2)
    held = {"attending": sc == 1, "lick/bite": sc == 2, "guarding": sc == 3, "escape/rearing": sc == 4,
            "general movement": np.asarray(S["moveVec"]).astype(bool).ravel()[:n],
            "grooming": np.asarray(S["groomVec"]).astype(bool).ravel()[:n]}
    day = "Day 1" if "20261019" in p else "Day 2" if "20261021" in p else ""
    # 10-23: the side camera of 5941 Day 2 post lost 18 frames at ~170 s; the scorer's syncOff (-0.6 s) maps side frame
    # k + off to bottom frame k - the clips must read the side video with the same offset
    off = int(round(float(S["syncOff"]) * FPS)) if "syncOff" in S and np.size(S["syncOff"]) else 0
    return dict(tag=f"{S['mouseID']} {day} {S['phase']}".replace("  ", " "), n=n, held=held, W=np.sort(re[re[:, 1] == 1, 0] - 1),
                F=np.sort(re[re[:, 1] == 2, 0] - 1), dF=np.sort(np.atleast_1d(S["dFrames"]).astype(int) - 1),
                bot=str(S["botFile"]), side=str(S["sidFile"]), off=off)


def candidates(sessions, beh):
    out = []
    for si, d in enumerate(sessions):
        if beh in ("withdrawal", "flinch"):
            ev, other = (d["W"], d["F"]) if beh == "withdrawal" else (d["F"], d["W"])
            for f in d["dF"]:
                e = ev[(ev >= f) & (ev <= f + 60)]
                if len(e):
                    pure = not ((other >= f) & (other <= f + 60)).any()     # only this reflex after the delivery
                    out.append(dict(s=si, centre=int(e[0]), start=int(e[0]), end=int(e[0]) + 1, key=(e[0] - f) / FPS, deliv=int(f),
                                    pure=pure))
        else:
            for a, b in spans(d["held"][beh]):
                other = any(d["held"][o][max(0, a - 15):b + 15].any() for o in d["held"] if o not in (beh, "general movement"))
                if b - a >= 6 and not other:
                    out.append(dict(s=si, centre=(a + b) // 2, start=int(a), end=int(b), key=(b - a) / FPS, deliv=-1))
    return out


def pick(c, n, sessions=None):
    if not c:
        return []
    if len([x for x in c if x.get("pure", True)]) >= n:      # taps: deliveries with only this reflex first
        c = [x for x in c if x.get("pure", True)]
    med = np.median([x["key"] for x in c])
    c = sorted(c, key=lambda x: abs(x["key"] - med))
    if sessions is not None:
        # 10-23 (user: new marks on this slide): both days in turn - Day 1, Day 2, Day 1 - so the reflexes are also shown
        # against the Day-2 clicker deliveries (an independent delivery time; on Day 1 a delivery its reflex came before was
        # moved to 0.07 s before that reflex by the timing fix)
        day = lambda x: "Day 2" if "Day 2" in sessions[x["s"]]["tag"] else "Day 1"
        order = ["Day 1", "Day 2"]
        chosen, used = [], set()
        for k in range(n):
            want = order[k % 2]
            x = next((x for x in c if day(x) == want and x["s"] not in used), None) or \
                next((x for x in c if x["s"] not in used and x not in chosen), None) or next((x for x in c if x not in chosen), None)
            if x is None:
                break
            chosen.append(x); used.add(x["s"])
        return chosen
    chosen, used = [], set()
    for x in c:                       # one per session first, then the rest
        if x["s"] not in used:
            chosen.append(x); used.add(x["s"])
        if len(chosen) >= n:
            return chosen
    for x in c:
        if x not in chosen:
            chosen.append(x)
        if len(chosen) >= n:
            break
    return chosen


def track(d, f, t0, t1, width, ref=None, ref_lab="the delivery"):
    """ref = the frame the time axis counts from (the delivery, or the start of a held bout); fixed ticks"""
    rows = len(TRACK_ROWS)
    rh = 26
    ref = f if ref is None else ref
    img = np.full((CH - CAMB, width, 3), 255, np.uint8)
    x = lambda fr: int(245 + (fr - t0) / max(t1 - t0, 1) * (width - 260))
    for i, nm in enumerate(TRACK_ROWS):
        y = 6 + i * rh
        cv2.putText(img, nm, (6, y + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        if nm in ("withdrawal", "flinch"):
            ev = d["W"] if nm == "withdrawal" else d["F"]
            for e in ev[(ev >= t0) & (ev <= t1)]:
                cv2.circle(img, (x(e), y + 12), 7, COL[nm], -1)
        else:
            v = d["held"][nm]
            for a, b in spans(v[t0:t1]):
                cv2.rectangle(img, (x(t0 + a), y + 3), (x(t0 + b), y + 21), COL[nm], -1)
    for e in d["dF"][(d["dF"] >= t0) & (d["dF"] <= t1)]:      # deliveries: black lines across all rows
        cv2.line(img, (x(e), 2), (x(e), rows * rh + 6), (0, 0, 0), 2)
    cv2.line(img, (x(f), 0), (x(f), rows * rh + 10), (0, 0, 255), 3)
    ya = rows * rh + 8
    cv2.line(img, (x(t0), ya), (x(t1), ya), (90, 90, 90), 1)
    for k in range(int(np.ceil((t0 - ref) / FPS)), int((t1 - ref) / FPS) + 1):     # time axis, 0 = the reference (fixed)
        xx = x(ref + k * FPS)
        cv2.line(img, (xx, ya), (xx, ya + 6), (90, 90, 90), 2)
        cv2.putText(img, f"{k:+d} s" if k else "0", (xx - 16, ya + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (60, 60, 60), 2)
    cv2.putText(img, f"time from {ref_lab}", (6, ya + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (60, 60, 60), 2)
    cv2.putText(img, "black line = delivery, red line = the frame shown, dot = tap, bar = held behaviour", (6, ya + 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (60, 60, 60), 2)
    return img


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--final-dir", required=True, nargs="+")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    sessions = [load(p) for fd in a.final_dir for p in sorted(glob.glob(os.path.join(fd, "ScoringAB_*.mat")))]
    print(len(sessions), "sessions")
    sheet, all_tmp = [], os.path.join(a.out_dir, "_all.avi")
    Wall = cv2.VideoWriter(all_tmp, cv2.VideoWriter_fourcc(*"MJPG"), 30, (CW, CH))
    for beh in TRACK_ROWS:
        ex = pick(candidates(sessions, beh), a.n, sessions)
        print(f"{beh}: {len(ex)} examples", [(sessions[x['s']]['tag'], round(x['centre'] / FPS, 1)) for x in ex])
        if not ex:
            continue
        tmp = os.path.join(a.out_dir, "_tmp.avi")
        W = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"MJPG"), 30, (CW, CH))
        for j, x in enumerate(ex, 1):
            d = sessions[x["s"]]
            ref = x["deliv"] if x["deliv"] >= 0 else x["start"]
            t0, t1 = max(0, ref - 45), min(d["n"] - 1, max(x["end"], ref) + 60)
            cs, cb = cv2.VideoCapture(d["side"]), cv2.VideoCapture(d["bot"])
            cs.set(cv2.CAP_PROP_POS_FRAMES, max(0, t0 + d["off"])); cb.set(cv2.CAP_PROP_POS_FRAMES, t0)
            mid = None
            keep = {}
            for f in range(t0, t1):
                ok1, s_ = cs.read(); ok2, b_ = cb.read()
                if not (ok1 and ok2):
                    break
                keep[f] = (s_, b_)
            tap = x["deliv"] >= 0
            seq = [(f, False) for f in range(t0, t1) if f in keep]
            if tap:                                              # 4x slower replay of +-0.5 s around the tap
                seq += [(f, True) for f in range(x["centre"] - 15, x["centre"] + 16) if f in keep for _ in range(4)]
            for f, slow in seq:
                s_, b_ = keep[f]
                fr = np.full((CH, CW, 3), 255, np.uint8)
                fr[TOP:CAMB, :540] = cv2.resize(s_, (540, CAMB - TOP)); fr[TOP:CAMB, 540:] = cv2.resize(b_, (540, CAMB - TOP))
                on = (x["start"] <= f < x["end"]) if not tap else (abs(f - x["centre"]) <= 4)
                cv2.putText(fr, f"{beh.upper()}  example {j}/{len(ex)}   {d['tag']}   {f / FPS:.1f} s" + ("   4x SLOWER" if slow else ""),
                            (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 200) if on else (60, 60, 60), 2)
                cv2.putText(fr, DEF[beh], (8, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                tr = track(d, f, t0, t1, CW, ref=ref, ref_lab="the delivery" if tap else "the bout start")
                fr[CAMB:CH] = tr[:CH - CAMB]
                W.write(fr); Wall.write(fr)
                if f == x["centre"] and not slow:
                    mid = fr.copy()
            for _ in range(12):
                W.write(np.full((CH, CW, 3), 255, np.uint8)); Wall.write(np.full((CH, CW, 3), 255, np.uint8))
            if mid is not None:
                sheet.append(cv2.resize(mid, (540, 330)))
        W.release()
        out = os.path.join(a.out_dir, f"behaviour_examples_{beh.replace('/', '_').replace(' ', '_')}.mp4")
        subprocess.run([FF, "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", out], check=True)
        os.remove(tmp)
    Wall.release()
    subprocess.run([FF, "-y", "-loglevel", "error", "-i", all_tmp, "-c:v", "libx264", "-crf", "22", "-pix_fmt", "yuv420p",
                    os.path.join(a.out_dir, "behaviour_examples_all.mp4")], check=True)
    os.remove(all_tmp)
    if sheet:
        rows = [np.hstack(sheet[i:i + 3] + [np.full((330, 540, 3), 255, np.uint8)] * (3 - len(sheet[i:i + 3])))
                for i in range(0, len(sheet), 3)]
        cv2.imwrite(os.path.join(a.out_dir, "behaviour_examples_sheet.png"), np.vstack(rows))
    print("done:", a.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
