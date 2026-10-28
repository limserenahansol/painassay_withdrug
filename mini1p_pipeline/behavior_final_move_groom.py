"""Automatic GENERAL MOVEMENT and GROOMING for every session of a day (the only two automatic
behaviours the user keeps, 2026-10-20), merged with the user's manual scoring when it exists.

  movement  behavior_auto_annotate.general_movement (silhouette artefacts bridged; nothing in
            the first 15 s, before the room light changes)
  grooming  behavior_grooming.py probability (1-s smoothed); a bout starts where it is > 0.4,
            continues while > 0.2, gaps < 2 s are closed, >= 1 s. Every clip the user judged
            overrides the model (judged grooming = grooming, judged not = not).

Writes per session
  <auto_dir>\\ScoringAB_<video>.mat   starting file for the scorer (mode 4): your manual marks if
                                     you scored the session, + moveVec + groomVec; j / k step
                                     through every grooming bout and every "maybe" (0.1-0.4)
  <final_dir>\\ScoringAB_<video>.mat  for sessions you scored: manual + movement + grooming
                                     (what the analysis reads)
  <final_dir>\\move_groom_summary.csv % time and bouts per schedule block
"""
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import behavior_auto_annotate as ba  # noqa: E402
from behavior_auto import bouts  # noqa: E402
from behavior_supervised import BLOCKS  # noqa: E402

FPS = 30.0
BASE = r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553"
FD = BASE + r"\behavior_auto\20261019"
GD = FD + r"\grooming"
SC = BASE + r"\scoring_mini1p"
AUTO_DIR, FINAL_DIR = SC + r"\20261019_auto", SC + r"\20261019_final"
# tag, bottom video stem, side video, sessionNo, sex, phase, stimulus order, manual scoring (or "")
S = [("5941_d1_pre", "5941 female0006 26-10-19 11-58-26-004", "5941 female0004 26-10-19 11-58-28-014.avi", "1", "F", "Baseline",
      ["Mild touch", "Pin prick", "Light touch", "Heat"], SC + r"\20261019\ScoringAB_x5941Female000626_10_1911_58_26_004.mat"),
     ("5941_d1_post", "5941 female sbi0007 26-10-19 12-27-38-009", "5941 femalesbi0005 26-10-19 12-27-39-016.avi", "7", "F", "Post-treatment",
      ["Pin prick", "Heat", "Mild touch", "Light touch"], SC + r"\20261019\ScoringAB_x5941FemaleSbi000726_10_1912_27_38_009.mat"),
     ("5937_d1_pre", "5937malebase0009 26-10-19 13-01-17-002", "5937malebase0006 26-10-19 13-01-19-001.avi", "2", "M", "Baseline",
      ["Heat", "Mild touch", "Pin prick", "Light touch"], SC + r"\20261019\ScoringAB_x5937malebase000926_10_1913_01_17_002.mat"),
     ("5937_d1_post", "5937malecontrol0010 26-10-19 13-20-59-011", "5937malecontrol0007 26-10-19 13-21-01-005.avi", "8", "M", "Post-treatment",
      ["Pin prick", "Heat", "Light touch", "Mild touch"], SC + r"\20261019\ScoringAB_x5937malecontrol001026_10_1913_20_59_011.mat"),
     ("5939_d1_pre", "5939male base0011 26-10-19 13-48-46-002", "5939male base0008 26-10-19 13-48-53-008.avi", "3", "M", "Baseline",
      ["Heat", "Light touch", "Pin prick", "Mild touch"], SC + r"\20261019\ScoringAB_x5939maleBase001126_10_1913_48_46_002.mat"),
     ("5939_d1_post", "5939sbi0012 26-10-19 14-17-29-001", "5939sbi0009 26-10-19 14-17-30-003.avi", "9", "M", "Post-treatment",
      ["Light touch", "Mild touch", "Pin prick", "Heat"], SC + r"\20261019\ScoringAB_x5939sbi001226_10_1914_17_29_001.mat"),
     ("1191_d1_pre", "1191femalebase0013 26-10-19 14-44-10-003", "1191femalebase0010 26-10-19 14-44-11-012.avi", "4", "F", "Baseline",
      ["Pin prick", "Mild touch", "Heat", "Light touch"], SC + r"\20261019\ScoringAB_x1191femalebase001326_10_1914_44_10_003.mat"),
     ("1191_d1_post", "1191femalecontrol0014 26-10-19 15-04-33-004", "1191femalecontrol0011 26-10-19 15-04-35-007.avi", "10", "F", "Post-treatment",
      ["Light touch", "Pin prick", "Heat", "Mild touch"], SC + r"\20261019\ScoringAB_x1191femalecontrol001426_10_1915_04_33_004.mat")]


def valid_name(s):
    # MATLAB matlab.lang.makeValidName for these file names
    s = re.sub(r"\s+(\S)", lambda m: m.group(1).upper(), s)
    s = re.sub(r"[^A-Za-z0-9_]", "_", s)
    return ("x" + s if not re.match(r"[A-Za-z]", s) else s)[:63]


def spans(m):
    d = np.diff(np.r_[0, np.asarray(m, int), 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def grooming_bouts(ps, labels, n):
    seed = ps > 0.4
    g, on, off = bouts(ps > 0.2, int(1.0 * FPS), int(2.0 * FPS))
    out = np.zeros(n, bool)
    for a, b in zip(on, off):
        if seed[a:b].any():
            out[a:b] = True
    for _, r in labels.iterrows():               # the user's judgement wins
        c = int(round(r.t_s * FPS))
        out[max(0, c - 45):c + 46] = bool(r.grooming)
    out[:int(15 * FPS)] = False
    return out


def main():
    os.makedirs(AUTO_DIR, exist_ok=True)
    os.makedirs(FINAL_DIR, exist_ok=True)
    L = pd.read_csv(GD + r"\grooming_labels_user.csv")
    rej = GD + r"\grooming_rejected_bouts.csv"
    global REJ
    REJ = pd.read_csv(rej) if os.path.isfile(rej) else pd.DataFrame(columns=["session", "start_s", "end_s"])
    old = os.path.join(AUTO_DIR, "review_deliveries_5941sbi_done")
    os.makedirs(old, exist_ok=True)
    summ = []
    for tag, stem, side, sno, sex, phase, stim, manual in S:
        F = pd.read_csv(rf"{FD}\{tag}_bfeatures.csv")
        Z = np.load(rf"{GD}\grooming_prob_{tag}.npz")
        n = min(len(F), len(Z["ps"]))
        move, _, _, _, _ = ba.general_movement(F.iloc[:n].reset_index(drop=True))
        move[:int(15 * FPS)] = False
        ps = Z["ps"][:n]
        groom = grooming_bouts(ps, L[L.session == tag], n)
        for _, r in REJ[REJ.session == tag].iterrows():        # bouts the user rejected on the figures
            groom[int(r.start_s * FPS) - 15:int(r.end_s * FPS) + 15] = False
        # "maybe" grooming: peaks 0.1-0.4 outside the bouts (where misses are likely)
        from scipy.signal import find_peaks
        pk, _ = find_peaks(np.where(groom, 0, ps), height=0.1, distance=int(10 * FPS))
        near_bout = np.convolve(groom.astype(float), np.ones(int(4 * FPS) + 1), "same") > 0   # within 2 s
        items = [(int(s), f"check AUTO grooming ({(e - s) / FPS:.1f} s)") for s, e in spans(groom)]
        items += [(int(f), f"maybe grooming? (p {ps[f]:.2f})") for f in pk if ps[f] < 0.4 and not near_bout[f]]
        items.sort()
        merged = []
        for f, w in items:                       # one stop per second
            if merged and f - merged[-1][0] < FPS:
                merged[-1] = (merged[-1][0], merged[-1][1] + " + " + w)
            else:
                merged.append((f, w))
        items = merged
        name = f"ScoringAB_{valid_name(stem)}.mat"
        out = dict(nUsed=float(n), frameRate=FPS, fpsTrue=FPS, hdrFps=FPS, mode=4.0, lagMs=250.0, obsWindow=30.0,
                   guardMin=2.0, syncOff=0.0, offFrames=0.0, sessionNo=sno, mouseID=tag.split("_")[0], sexID=sex,
                   dayNo="1", phase=phase, stimNames=np.array(stim, dtype=object),
                   affNames=np.array(["None", "Paw attending", "Licking or biting", "Sustained lifting / guarding",
                                      "Escape / rearing"], dtype=object),
                   affKeys=np.array(["a", "s", "d", "f"], dtype=object),
                   refNames=np.array(["Paw withdrawal", "Flinch / flick"], dtype=object),
                   score=np.zeros((n, 1)), uncert=np.zeros((n, 1), bool), reflexEvents=np.zeros((0, 2)),
                   dFrames=np.zeros((0, 1)), dTypes=np.zeros((0, 1)), dSec=np.zeros((0, 1)),
                   botFile=rf"{BASE}\videos_scoring\20261019\bottom_enhanced\{stem}.mp4",
                   sidFile=rf"K:\mini1p_pain_behavior\{side}")
        if manual and not os.path.isfile(manual):
            print(f"{tag}: manual scoring not found yet ({manual}) - movement + grooming only")
            manual = ""
        if manual:
            M = sio.loadmat(manual, squeeze_me=True)
            m_n = min(n, int(M["nUsed"]))
            sc = np.zeros(n); sc[:m_n] = np.atleast_1d(M["score"]).astype(float)[:m_n]
            un = np.zeros(n, bool); un[:m_n] = np.atleast_1d(M["uncert"]).astype(bool)[:m_n]
            dF = np.atleast_1d(M["dFrames"]).astype(float)
            dT = np.atleast_1d(M["dTypes"]).astype(float)
            # the stimulus of a delivery = the stimulus of its schedule block (user, 2026-10-20:
            # the block order is right, keys pressed in the wrong block were slips)
            blk = np.zeros(len(dF), int)
            for k, (_, s0, s1) in enumerate(BLOCKS[1:], 1):
                tt = (dF - 1) / FPS
                blk[(tt >= s0) & (tt < s1 + 10)] = k
            fix = (blk > 0) & (dT != blk)
            dT = np.where(blk > 0, blk, dT)
            out.update(score=sc.reshape(-1, 1), uncert=un.reshape(-1, 1),
                       reflexEvents=np.atleast_2d(M["reflexEvents"]).astype(float).reshape(-1, 2),
                       dFrames=dF.reshape(-1, 1), dTypes=dT.reshape(-1, 1),
                       dSec=((dF - 1) / FPS).reshape(-1, 1), typesFixedToBlock=float(fix.sum()))
            if fix.any():
                print(f"{tag}: {fix.sum()} delivery type(s) set to their block's stimulus")
        out.update(moveVec=move.reshape(-1, 1), moveName="General movement (turn / steps)",
                   groomVec=groom.reshape(-1, 1), groomName="Grooming",
                   groomProb=ps.astype(np.float32).reshape(-1, 1),
                   checkFrames=(np.array([f for f, _ in items], float) + 1).reshape(-1, 1),
                   checkWhy=np.array([w for _, w in items], dtype=object),
                   startFrom=("your manual scoring + " if manual else "") + "AUTO general movement + AUTO grooming",
                   manualSource=manual, created=time.strftime("%Y-%m-%d %H:%M"),
                   autoInfo=dict(movement="behavior_auto_annotate.general_movement (artefacts bridged, first 15 s skipped)",
                                 grooming="behavior_grooming.py model, hysteresis 0.4 / 0.2, gaps < 2 s closed, >= 1 s; "
                                          "user-judged clips override",
                                 grooming_check="leave-one-session-out: no false alarm in 50 judged non-grooming "
                                                "clips; about half of the judged grooming clips found"))
        p_auto = os.path.join(AUTO_DIR, name)
        if os.path.isfile(p_auto) and "refDFrames" in [w[0] for w in sio.whosmat(p_auto)]:
            os.replace(p_auto, os.path.join(old, name))      # the finished delivery review of 5941 SBI
        sio.savemat(p_auto, out, do_compression=True)
        if manual:
            sio.savemat(os.path.join(FINAL_DIR, name), out, do_compression=True)
        for b, s0, s1 in BLOCKS:
            sl = slice(int(s0 * FPS), min(n, int(s1 * FPS)))
            gs = [x for x in spans(groom) if s0 * FPS <= x[0] < s1 * FPS]
            ms = [x for x in spans(move) if s0 * FPS <= x[0] < s1 * FPS]
            summ.append(dict(session=tag, mouse=tag.split("_")[0], phase=phase, block=b,
                             block_stim="baseline" if b == "baseline" else stim[int(b[-1]) - 1],
                             movement_pct=round(100 * move[sl].mean(), 2), movement_bouts=len(ms),
                             grooming_pct=round(100 * groom[sl].mean(), 2), grooming_bouts=len(gs),
                             grooming_s=round(float(groom[sl].sum() / FPS), 1)))
        print(f"{tag}: movement {move.mean() * 100:.1f}% ({len(spans(move))} bouts), grooming {groom.sum() / FPS:.0f} s "
              f"({len(spans(groom))} bouts), {len(items)} stops for j  -> {p_auto}" + ("  + final" if manual else ""))
    pd.DataFrame(summ).to_csv(os.path.join(FINAL_DIR, "move_groom_summary.csv"), index=False)
    print("summary:", os.path.join(FINAL_DIR, "move_groom_summary.csv"))


if __name__ == "__main__":
    main()
