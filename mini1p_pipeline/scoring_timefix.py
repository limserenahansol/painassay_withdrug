"""scoring_timefix.py - corrected COPIES of the final scoring files (2026-10-23, agreed with the user; the originals are not
touched): scoring_mini1p\\20261019_final_timefix, scoring_mini1p\\20261021_final_timefix.

Why: the withdrawal / flinch marks are key presses made after seeing the movement. Against the movement in the video they are
late by a constant amount (analysis\\withdrawal_timing: automatic movement start over all 1,467 marks, Day 1 0.40 s, Day 2 0.37 s;
two independent visual judges on 32 marks, 0.47 / 0.43 s, agreement within 1 frame in 30/32).

Rules
  1  every withdrawal and flinch mark (reflexEvents) moves earlier: Day 1 by 12 frames (0.40 s), Day 2 by 11 frames (0.37 s)
  2  a delivery whose own reflex (the user's 2-s rule: from the delivery to 2 s later or the next delivery) would now come BEFORE
     it is moved to 2 frames (0.07 s, the median Day-2 click -> paw-movement interval) before that reflex; never closer than
     15 frames (0.5 s) to the previous delivery. Other deliveries stay. (User, 10-23: "only for those cases".)
  3  Day 2 clicker (Rogger) fixes, checked on the pressure trace 10-23: 5941 d2 pre 209.93 s = an indicator onset with no press,
     0.93 s after the real press at 209.0 s -> removed; presses inside a held indicator that had no delivery -> added at the
     press onset (|pressure - running median| leaves 0.1e6): 5941 d2 pre 214.30 s (user: "this is the real stimulus"),
     1191 d2 pre 334.08 s, 1191 d2 post 652.29 s; stimulus = that of the nearest delivery (same block). Light clicks
     (0.1-0.3e6, indicator on) are kept, and so are the double presses < 0.5 s apart (the first one is used, as before).

Writes the copies and timefix_changes.csv (every changed delivery) + timefix_summary.csv.

  python scoring_timefix.py
"""
import glob
import os
import shutil

import numpy as np
import pandas as pd
import scipy.io as sio

BASE = r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\scoring_mini1p"
FPS = 30.0
SH = {1: 12, 2: 11}
SRC = {1: os.path.join(BASE, "20261019_final"), 2: os.path.join(BASE, "20261021_final")}
DST = {d: p + "_timefix" for d, p in SRC.items()}
REMOVE = {("5941", "pre"): [209.93]}                                   # session seconds (onset_session_s of the logger list)
ADD = {("5941", "pre"): [214.30], ("1191", "pre"): [334.08], ("1191", "post"): [652.29]}


def phase_of(S):
    return "pre" if str(np.squeeze(S["phase"])).lower().startswith("base") else "post"


def main():
    rows, summ = [], []
    for d in (1, 2):
        os.makedirs(DST[d], exist_ok=True)
        for p in sorted(glob.glob(os.path.join(SRC[d], "ScoringAB_*.mat"))):
            S = sio.loadmat(p)                                         # raw (not squeezed): written back in the same form
            mouse = str(np.squeeze(S["mouseID"])).strip(); ph = phase_of(S)
            re = np.asarray(S["reflexEvents"], float).reshape(-1, 2).astype(int)
            dF0 = np.asarray(S["dFrames"], float).ravel().astype(int)
            dT0 = np.asarray(S["dTypes"], float).ravel().astype(int)
            o = np.argsort(dF0); dF, dT = dF0[o].copy(), dT0[o].copy()
            # rule 3 (Day 2 clicker)
            if d == 2:
                for t in REMOVE.get((mouse, ph), []):
                    k = int(np.argmin(np.abs((dF - 1) / FPS - t)))
                    assert abs((dF[k] - 1) / FPS - t) < 0.05, (mouse, ph, t)
                    rows.append(dict(day=d, mouse=mouse, phase=ph, change="removed (indicator onset, no press)", old_frame=int(dF[k]),
                                     new_frame=np.nan, stimulus_type=int(dT[k])))
                    dF, dT = np.delete(dF, k), np.delete(dT, k)
                for t in ADD.get((mouse, ph), []):
                    f = int(round(t * FPS)) + 1                           # dSec = (frame - 1) / 30
                    k = int(np.argmin(np.abs(dF - f)))
                    typ = int(dT[k])
                    rows.append(dict(day=d, mouse=mouse, phase=ph, change="added (press inside a held indicator)", old_frame=np.nan,
                                     new_frame=f, stimulus_type=typ))
                    dF, dT = np.r_[dF, f], np.r_[dT, typ]
                o = np.argsort(dF); dF, dT = dF[o], dT[o]
            # rule 1: reflexes earlier
            re2 = re.copy(); re2[:, 0] = np.maximum(1, re[:, 0] - SH[d])
            # rule 2: deliveries that their own reflex now precedes
            dF2 = dF.copy()
            for k in range(len(dF)):
                nxt = dF[k + 1] if k + 1 < len(dF) else 10 ** 9
                own = (re[:, 0] >= dF[k]) & (re[:, 0] <= min(dF[k] + int(2 * FPS), nxt - 1))
                if not own.any():
                    continue
                first = int(re2[own, 0].min())
                if first < dF2[k] + 0:                                    # the shifted reflex comes before (or at) its delivery
                    if first <= dF2[k]:
                        new = first - 2
                        if k > 0:
                            new = max(new, dF2[k - 1] + 15)
                        if new < dF2[k]:
                            rows.append(dict(day=d, mouse=mouse, phase=ph, change="moved earlier (reflex before it)", old_frame=int(dF2[k]),
                                             new_frame=int(new), stimulus_type=int(dT[k])))
                            dF2[k] = new
            # write back in the stored orientation
            shp = S["dFrames"].shape
            S["dFrames"] = dF2.astype(float).reshape((1, -1) if shp[0] == 1 else (-1, 1))
            S["dTypes"] = dT.astype(float).reshape(S["dFrames"].shape)
            S["dSec"] = ((dF2 - 1) / FPS).reshape(S["dFrames"].shape)
            S["reflexEvents"] = re2.astype(float)
            S["timefix"] = np.array([f"10-23: reflexes -{SH[d]} frames; deliveries moved only where their own reflex came first; "
                                     "Day-2 clicker fixes (see scoring_timefix.py)"], dtype=object)
            out = os.path.join(DST[d], os.path.basename(p))
            sio.savemat(out, {k: v for k, v in S.items() if not k.startswith("__")}, do_compression=True)
            n_mov = sum(1 for r in rows if r["day"] == d and r["mouse"] == mouse and r["phase"] == ph and r["change"].startswith("moved"))
            summ.append(dict(day=d, mouse=mouse, phase=ph, reflex_marks=len(re), deliveries_before=len(dF0), deliveries_after=len(dF2),
                             deliveries_moved=n_mov))
        for f in glob.glob(os.path.join(SRC[d], "*")):                 # keep the other files of the folder (tables, figures)
            if not os.path.basename(f).startswith("ScoringAB_") and os.path.isfile(f):
                shutil.copy(f, DST[d])
    C = pd.DataFrame(rows); Sm = pd.DataFrame(summ)
    C.to_csv(os.path.join(BASE, "timefix_changes.csv"), index=False)
    Sm.to_csv(os.path.join(BASE, "timefix_summary.csv"), index=False)
    print(Sm.to_string())
    mv = C[C.change.str.startswith("moved")]
    for d in (1, 2):
        X = mv[mv.day == d]
        print(f"Day {d}: deliveries moved {len(X)}; by median {np.median(X.old_frame - X.new_frame) / FPS if len(X) else 0:.2f} s "
              f"(max {(X.old_frame - X.new_frame).max() / FPS if len(X) else 0:.2f} s)")
    print(C[~C.change.str.startswith("moved")].to_string())


if __name__ == "__main__":
    main()
