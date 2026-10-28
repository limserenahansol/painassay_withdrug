"""logger_reference_for_scorer.py - the logger deliveries as a READ-ONLY reference in the scorer (mode 4).

You score the deliveries (keys 1-4) and the behaviours yourself; the logger's deliveries are drawn
next to yours in the grey reference rows of the label track, so every place where the logger says
"delivery" can be compared with what you see. j / k jump to 1 s before the next / previous logger
delivery (the title says which stimulus and whether it came from the logger indicator or is an added
downward press). A left click on a grey mark takes it into your marks, a right click rejects it.

One starting file per session (scorer mode 4), empty editable layer:
  score / uncert / moveVec / groomVec all zero, no deliveries or taps of yours yet
  refDFrames / refDTypes = the logger deliveries (camera frame, 1-based; type = block = key 1-4)
  checkFrames / checkWhy  = the same deliveries (the j / k stops)

  python logger_reference_for_scorer.py --table <day2_session_table.csv> --deliveries <logger_deliveries dir> --out <dir>
Writes <out>/LoggerReference_<mouse>_d2_<pre|post>.mat; the launcher hands it to the scorer as autoFile.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import scipy.io as sio


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--deliveries", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    T = pd.read_csv(a.table)
    S = pd.read_csv(os.path.join(a.deliveries, "logger_sessions.csv")).set_index("recording_id")
    for _, r in T.iterrows():
        rid = f"{r.mouse}_d{int(r.day)}_{r.phase}"
        E = pd.read_csv(os.path.join(a.deliveries, f"{rid}_logger_deliveries.csv"))
        E = E[(E.block >= 1) & (E.onset_cam_frame >= 1)].sort_values("onset_cam_frame")
        n = int(S.at[rid, "max_ttl_count"])
        stim = [r[f"block{k}"] for k in range(1, 5)]
        fr = E.onset_cam_frame.to_numpy(float)
        why = np.array([f"logger: {s} ({'added downward press' if 'downward' in src else 'logger indicator'})"
                        for s, src in zip(E.stimulus, E.source)], dtype=object)
        out = dict(score=np.zeros((n, 1)), uncert=np.zeros((n, 1), bool), moveVec=np.zeros((n, 1), bool),
                   groomVec=np.zeros((n, 1), bool), reflexEvents=np.zeros((0, 2)), dFrames=np.zeros((0, 1)),
                   dTypes=np.zeros((0, 1)), stimNames=np.array(stim, dtype=object).reshape(1, -1), frameRate=30.0, nUsed=0.0,
                   refDFrames=fr.reshape(-1, 1), refDTypes=E.block.to_numpy(float).reshape(-1, 1),
                   refReflex=np.zeros((0, 2)), refScore=np.zeros((n, 1)),
                   checkFrames=fr.reshape(-1, 1), checkWhy=why,
                   sessionNo=str(int(r.session_no)), mouseID=str(r.mouse), sexID=str(r.sex), dayNo=str(int(r.day)),
                   phase="Baseline" if r.phase == "pre" else "Post-treatment",
                   startFrom="empty: you score deliveries and behaviours; grey rows = the logger deliveries",
                   loggerSource=os.path.join(a.deliveries, f"{rid}_logger_deliveries.csv"), created=time.strftime("%Y-%m-%d %H:%M"))
        p = os.path.join(a.out, f"LoggerReference_{rid}.mat")
        sio.savemat(p, out, do_compression=True)
        print(f"{os.path.basename(p)}: {len(fr)} logger deliveries as reference, {n} frames")


if __name__ == "__main__":
    main()
