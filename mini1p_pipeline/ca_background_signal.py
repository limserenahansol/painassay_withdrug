"""Background (neuropil / focus) signal of each recording, for a background-corrected trace version.

In the 5941 baseline video several cells dipped TOGETHER in steps during movement bouts, and the
population mean correlated 0.43 with the frame brightness: a diffuse signal outside the cells
(neuropil, out-of-focus light, focus shifts with movement) is mixed into the traces. Here, per
recording, b(t) = mean of the motion-corrected movie over BACKGROUND pixels: inside the lens mask,
outside every candidate footprint (dilated 6 px). Written next to the traces as
<rid>_background.npz (frame, b, and b as fractional change from its 180-s running median);
ca_trace_outputs.py --bg-regress then removes it from every cell (least squares per cell).

  python ca_background_signal.py --day-root <mini1p_out/20261019/5941_d1>
"""
import argparse
import json
import os
import sys

import h5py
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.ndimage import binary_dilation
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import safe_io  # noqa: E402,F401  (a figure save retries on a transient Windows write error)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--day-root", required=True)
    ap.add_argument("--block", type=int, default=500)
    a = ap.parse_args(argv)
    tag = os.path.basename(os.path.normpath(a.day_root))
    M = sio.loadmat(os.path.join(a.day_root, "3_extract", f"{tag}_extract.mat"), squeeze_me=True,
                    struct_as_record=False, variable_names=["spatial_weights", "movie_mask", "frame_table"])
    Sp = np.asarray(M["spatial_weights"], np.float32)
    lens = np.asarray(M["movie_mask"]).astype(bool) if np.size(M["movie_mask"]) > 1 else np.ones(Sp.shape[:2], bool)
    cells = binary_dilation((Sp > 0.1 * Sp.max(axis=(0, 1), keepdims=True)).any(2), iterations=6)
    bg = lens & ~cells
    print(f"{tag}: background = {bg.mean() * 100:.1f}% of the frame ({bg.sum()} px)")
    ft = M["frame_table"]
    FT = pd.DataFrame({k: np.atleast_1d(getattr(ft, k)) for k in ("recording_id", "isx_frame", "valid", "h5_row")})
    od = os.path.join(a.day_root, "5_analysis", "traces")
    os.makedirs(od, exist_ok=True)
    for rid in FT.recording_id.astype(str).unique():
        h5 = os.path.join(a.day_root, "1_movie", f"{rid}_mc.h5")
        F = FT[(FT.recording_id.astype(str) == rid)]
        with h5py.File(h5, "r") as f:
            d = f["mov"]                                  # (frames, W, H) of MATLAB [H W n]
            n = d.shape[0]
            bgT = bg.T                                    # to the h5py (W, H) layout
            b = np.zeros(n)
            for i0 in range(0, n, a.block):
                Y = d[i0:i0 + a.block].astype(np.float32)
                b[i0:i0 + len(Y)] = Y[:, bgT].mean(1)
        # per-recording rows: h5 rows of this recording are 1..n in its own mc file
        fr = F.isx_frame.values.astype(int)
        valid = F.valid.values.astype(bool)
        bb = np.full(len(fr), np.nan)
        rows_in_rec = np.cumsum(valid) - 1
        bb[valid] = b[rows_in_rec[valid]]
        med = pd.Series(bb).rolling(int(180 * 30), center=True, min_periods=300).median().values
        np.savez_compressed(os.path.join(od, f"{rid}_background.npz"), frame=fr, b=bb.astype(np.float32),
                            b_frac=((bb - med) / med).astype(np.float32), n_bg_px=int(bg.sum()))
        print(f"  {rid}: background signal of {len(fr)} frames, fractional SD {np.nanstd((bb - med) / med) * 100:.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
