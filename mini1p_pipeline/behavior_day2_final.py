"""behavior_day2_final.py - Day-2 final scoring files (what every Day-2 analysis reads), like Day 1's 20261019_final.

For every Day-2 session you have saved in scoring_mini1p\\20261021_day2 (ScoringAB_*.mat, matched by mouseID + phase):
  deliveries    = the pressure-logger deliveries (logger_deliveries.py, your rule: the logger indicator + the downward
                  presses), type = their block's stimulus; your own taps are kept as userDFrames / userDTypes;
                  a delivery < 0.5 s after the previous one is the same contact (you, 2026-10-22: the 19 such
                  deliveries "might be mistake" = indicator re-triggers) and is dropped (counted in the table)
  reflexes, held behaviours (attending, lick/bite, guarding, escape)  = your scoring
  moveVec, groomVec = the Day-2 automatic movement and grooming (behavior_day2_move_groom.py; same code and model
                  as Day 1; you checked the grooming)
Written to scoring_mini1p\\20261021_final\\ScoringAB_<the same name>.mat (your files are never changed) and a table
day2_final_sessions.csv (which sessions are in, the counts).

  python behavior_day2_final.py
"""
import glob
import os
import time

import numpy as np
import pandas as pd
import scipy.io as sio

BASE = r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553"
MAN = BASE + r"\scoring_mini1p\20261021_day2"
FIN = BASE + r"\scoring_mini1p\20261021_final"
AUTO = BASE + r"\behavior_auto\20261021"
TABLE = MAN + r"\day2_session_table.csv"


def main():
    os.makedirs(FIN, exist_ok=True)
    T = pd.read_csv(TABLE)
    rows = []
    for p in sorted(glob.glob(os.path.join(MAN, "ScoringAB_*.mat"))):
        S = sio.loadmat(p, squeeze_me=True)
        m = str(S["mouseID"]).strip()
        ph = "pre" if str(S["phase"]).lower().startswith("base") else "post"
        r = T[(T.mouse.astype(str) == m) & (T.phase == ph)]
        if not len(r):
            print(f"{os.path.basename(p)}: mouse {m} {ph} not in the Day-2 table - skipped"); continue
        r = r.iloc[0]
        rid = f"{m}_d2_{ph}"
        n = int(S["nUsed"])
        L = sio.loadmat(os.path.join(MAN, f"DeliveryTimes_{rid}_LOGGER.mat"), squeeze_me=True)
        lf, lt = np.atleast_1d(L["deliveryFrames"]).astype(float), np.atleast_1d(L["deliveryType"]).astype(float)
        ok = (lf >= 1) & (lf <= n)
        # indicator re-triggers: < 0.5 s (15 frames) after the previous kept delivery = the same contact
        order = np.argsort(lf, kind="stable"); lf, lt, ok = lf[order], lt[order], ok[order]
        last, retrig = -1e9, np.zeros(len(lf), bool)
        for k_ in range(len(lf)):
            if not ok[k_]:
                continue
            if lf[k_] - last < 0.5 * 30:
                retrig[k_] = True
            else:
                last = lf[k_]
        ok = ok & ~retrig
        Z = np.load(os.path.join(AUTO, f"{rid}_move_groom.npz"))
        mv = np.zeros(n, bool); gr = np.zeros(n, bool)
        k = min(n, len(Z["move"])); mv[:k] = Z["move"][:k]; gr[:k] = Z["groom"][:k]
        out = {k_: S[k_] for k_ in S if not k_.startswith("__") and k_ not in ("refDFrames", "refDTypes", "refReflex", "refScore",
                                                                                 "checkFrames", "checkWhy", "T", "rep")}
        out.update(userDFrames=np.atleast_1d(S["dFrames"]).astype(float).reshape(-1, 1),
                   userDTypes=np.atleast_1d(S["dTypes"]).astype(float).reshape(-1, 1),
                   dFrames=lf[ok].reshape(-1, 1), dTypes=lt[ok].reshape(-1, 1), dSec=((lf[ok] - 1) / 30.0).reshape(-1, 1),
                   stimNames=np.array([r[f"block{j}"] for j in range(1, 5)], dtype=object),
                   moveVec=mv.reshape(-1, 1), groomVec=gr.reshape(-1, 1),
                   moveName="General movement (turn / steps)", groomName="Grooming", dayNo="2",
                   deliverySource="pressure logger (logger indicator + downward presses; < 0.5 s re-triggers dropped)",
                   manualSource=p, created=time.strftime("%Y-%m-%d %H:%M"),
                   autoInfo=dict(movement="behavior_auto_annotate.general_movement, Day-2 features (behavior_day2_move_groom.py)",
                                 grooming="Day-1 grooming model on Day-2 features, hysteresis 0.4 / 0.2"))
        sio.savemat(os.path.join(FIN, os.path.basename(p)), out, do_compression=True)
        re = np.atleast_2d(S["reflexEvents"]).reshape(-1, 2)
        sc = np.atleast_1d(S["score"]).astype(int)
        rows.append(dict(recording_id=rid, treatment=r.treatment, file=os.path.basename(p), frames_scored=n,
                         logger_deliveries=int(ok.sum()), retriggers_dropped=int(retrig.sum()), your_tapped_deliveries=int(np.atleast_1d(S["dFrames"]).size),
                         withdrawal=int((re[:, 1] == 1).sum()), flinch=int((re[:, 1] == 2).sum()),
                         escape_s=round((sc == 4).sum() / 30, 1), movement_pct=round(100 * mv.mean(), 1), grooming_s=round(gr.sum() / 30, 1)))
        print(f"{rid}: {int(ok.sum())} logger deliveries, {rows[-1]['withdrawal']} withdrawal, {rows[-1]['flinch']} flinch -> {FIN}")
    pd.DataFrame(rows).to_csv(os.path.join(FIN, "day2_final_sessions.csv"), index=False)
    done = {x["recording_id"] for x in rows}
    todo = [f"{m}_d2_{ph}" for m, ph in zip(T.mouse.astype(str), T.phase) if f"{m}_d2_{ph}" not in done]
    print(f"{len(done)} of {len(T)} sessions in; still to score: {', '.join(todo) if todo else 'none'}")


if __name__ == "__main__":
    main()
