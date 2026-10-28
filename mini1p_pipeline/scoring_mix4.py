"""scoring_mix4.py - 10-25 (user): a scoring COPY where each event uses the marks (original key presses or the 10-23 timing-corrected
ones) that gave the most significant results for that event (exploratory; chosen on 75 single-neuron + population tests computed
with both marks x both windows, see the deck notes). Originals and _timefix copies are not touched.

  heat deliveries and withdrawal marks   <- the ORIGINAL scoring   (scoring_mini1p\\2026xxxx_final)
  every other delivery and the flinch marks <- the timing-corrected copy (scoring_mini1p\\2026xxxx_final_timefix)
  everything else (affective score, movement / grooming vectors, ...) <- the timing-corrected copy

Writes scoring_mini1p\\20261019_final_mix4 and 20261021_final_mix4 (+ mix4_summary.csv).
  python scoring_mix4.py
"""
import glob
import os
import shutil

import numpy as np
import pandas as pd
import scipy.io as sio

BASE = r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\scoring_mini1p"
FPS = 30.0
OLD_STIM = {"Heat"}                 # deliveries of these stimuli from the original marks
OLD_REF = {"Paw withdrawal"}        # reflex marks of these kinds from the original marks
DAYS = {1: "20261019_final", 2: "20261021_final"}


def names(S, key):
    return [str(np.squeeze(x)) for x in np.asarray(S[key]).ravel()]


def main():
    summ = []
    for d, stem in DAYS.items():
        old_dir, new_dir, out_dir = (os.path.join(BASE, stem + s) for s in ("", "_timefix", "_mix4"))
        os.makedirs(out_dir, exist_ok=True)
        for pn in sorted(glob.glob(os.path.join(new_dir, "ScoringAB_*.mat"))):
            po = os.path.join(old_dir, os.path.basename(pn))
            N, O = sio.loadmat(pn), sio.loadmat(po)
            stim, ref = names(N, "stimNames"), names(N, "refNames")
            assert stim == names(O, "stimNames") and ref == names(O, "refNames"), pn
            old_t = {i + 1 for i, s in enumerate(stim) if s in OLD_STIM}
            old_r = {i + 1 for i, s in enumerate(ref) if s in OLD_REF}
            dFn, dTn = np.asarray(N["dFrames"], float).ravel(), np.asarray(N["dTypes"], float).ravel()
            dFo, dTo = np.asarray(O["dFrames"], float).ravel(), np.asarray(O["dTypes"], float).ravel()
            keep_n = ~np.isin(dTn, list(old_t)); take_o = np.isin(dTo, list(old_t))
            dF = np.r_[dFn[keep_n], dFo[take_o]]; dT = np.r_[dTn[keep_n], dTo[take_o]]
            o = np.argsort(dF, kind="stable"); dF, dT = dF[o], dT[o]
            rn = np.asarray(N["reflexEvents"], float).reshape(-1, 2); ro = np.asarray(O["reflexEvents"], float).reshape(-1, 2)
            re = np.r_[rn[~np.isin(rn[:, 1], list(old_r))], ro[np.isin(ro[:, 1], list(old_r))]]
            re = re[np.argsort(re[:, 0], kind="stable")]
            S = dict(N)
            shp = N["dFrames"].shape
            S["dFrames"] = dF.reshape((1, -1) if shp[0] == 1 else (-1, 1))
            S["dTypes"] = dT.reshape(S["dFrames"].shape)
            S["dSec"] = ((dF - 1) / FPS).reshape(S["dFrames"].shape)
            S["reflexEvents"] = re
            S["mix4"] = np.array(["10-25: heat deliveries and withdrawal marks from the original scoring; the rest from the "
                                  "timing-corrected copy (scoring_mix4.py)"], dtype=object)
            sio.savemat(os.path.join(out_dir, os.path.basename(pn)), {k: v for k, v in S.items() if not k.startswith("__")},
                        do_compression=True)
            summ.append(dict(day=d, file=os.path.basename(pn), deliveries=len(dF), heat_from_old=int(take_o.sum()),
                             heat_in_new=int((~keep_n).sum()), reflex=len(re), withdrawal_from_old=int(np.isin(ro[:, 1], list(old_r)).sum()),
                             flinch_from_new=int((~np.isin(rn[:, 1], list(old_r))).sum())))
        for f in glob.glob(os.path.join(new_dir, "*")):
            if not os.path.basename(f).startswith("ScoringAB_") and os.path.isfile(f):
                shutil.copy(f, out_dir)
    S_ = pd.DataFrame(summ)
    S_.to_csv(os.path.join(BASE, "mix4_summary.csv"), index=False)
    print(S_.to_string())


if __name__ == "__main__":
    main()
