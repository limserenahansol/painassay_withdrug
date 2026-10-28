"""logger_to_delivery_times.py - logger deliveries -> DeliveryTimes_*.mat for the scorer's MODE 2.

Mode 2 (BEHAVIOUR ONLY) of score_AB_dual_view_mini1p.m imports a DeliveryTimes_*.mat and shows
its deliveries as pre-loaded marks, so only the behaviour has to be scored. This writes one per
session from <rid>_logger_deliveries.csv (logger_deliveries.py), in the scorer's own format:
  deliveryFrames = camera frame of the onset (1-based) = ttl_count at the onset sample
  deliverySec    = (deliveryFrames - 1) / 30   (the scorer's dSec: mode 2 maps it back with
                   round(sec * 30) + 1, i.e. the same frame)
  deliveryType   = the block of the delivery = key 1-4 = this session's stimulus order
  stimNames      = the session's 4 stimuli in block order (session table)
  numFrames      = the logger's last ttl_count (the camera frames taken; the scorer checks it
                   against the video)
Deliveries outside every block (deliveryType 0 in the scorer = unknown) are left out and listed.

  python logger_to_delivery_times.py --table <day2_session_table.csv> --deliveries <dir> --out <dir>
"""
import argparse
import os

import numpy as np
import pandas as pd
import scipy.io as sio


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--deliveries", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    T = pd.read_csv(a.table)
    S = pd.read_csv(os.path.join(a.deliveries, "logger_sessions.csv")).set_index("recording_id")
    for _, r in T.iterrows():
        rid = f"{r.mouse}_d{int(r.day)}_{r.phase}"
        E = pd.read_csv(os.path.join(a.deliveries, f"{rid}_logger_deliveries.csv"))
        out_blk = E[E.block == 0]
        E = E[(E.block >= 1) & (E.onset_cam_frame >= 1)].sort_values("onset_cam_frame")
        stim = [r[f"block{k}"] for k in range(1, 5)]
        fr = E.onset_cam_frame.to_numpy(float)
        typ = E.block.to_numpy(float)
        n_per = np.array([int((typ == k).sum()) for k in range(1, 5)], float)
        name = f"DeliveryTimes_{rid}_LOGGER.mat"
        sio.savemat(os.path.join(a.out, name), {
            "deliveryFrames": fr.reshape(-1, 1), "deliverySec": ((fr - 1) / 30.0).reshape(-1, 1),
            "deliveryType": typ.reshape(-1, 1), "stimNames": np.array(stim, dtype=object).reshape(1, -1),
            "nPerStim": n_per.reshape(1, -1), "nUnknown": 0.0, "frameRate": 30.0,
            "numFrames": float(S.at[rid, "max_ttl_count"]), "sessionNo": str(int(r.session_no)),
            "vidName": rid, "source": "pressure logger (logger_deliveries.py)"})
        print(f"{name}: {len(E)} deliveries " + ", ".join(f"{s} {int(n)}" for s, n in zip(stim, n_per))
              + (f"; left out (outside blocks) at {', '.join(f'{v:.1f} s' for v in out_blk.onset_session_s)}" if len(out_blk) else ""))


if __name__ == "__main__":
    main()
