"""Synthetic Day-1 Ca data with KNOWN cell responses, on the REAL behaviour of each session, so that
ca_day1_analysis.py / ca_day1_extended.py / ca_longitudinal.py can be checked against the truth.

Per mouse (all four, each with its real scoring files for pre and post), 60 cells:
  15 UP cells   respond to deliveries, size grows with the stimulus (light 0.4, mild 0.8, pin 1.4,
                heat 2.0 z); 8 DOWN cells (heat and pin, -0.8 z); 8 withdrawal cells (+1.5 z at each
                withdrawal); 8 movement cells (+0.8 z while moving); the rest no response.
  every cell: AR(1) noise (z, 30 Hz, phi 0.95), Ca kernel tau 1 s, a slow drift.
  SBI-553 mice, injection session: every evoked response x 0.5 (the known 'drug effect');
  vehicle mice: x 1.0.
Writes <out-root>/<m>_d1/5_analysis/traces/<rid>_traces.npz + _events.csv (events = threshold
crossings of the noiseless calcium, as a stand-in for OASIS) and 3_extract/<m>_d1_extract.mat
(footprints, mean image, cell_qc centroids), plus truth.csv.

  python ca_synthetic_day1.py --scoring-dir <final> --out-root <dir>
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ca_day1_analysis as d1  # noqa: E402

FPS = 30.0
SIZE = {"Light touch": 0.4, "Mild touch": 0.8, "Pin prick": 1.4, "Heat": 2.0}


def kernel(tau=1.0, rise=0.15, dur=6.0):
    t = np.arange(0, dur, 1 / FPS)
    k = (1 - np.exp(-t / rise)) * np.exp(-t / tau)
    return k / k.max()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scoring-dir", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--n-cells", type=int, default=60)
    a = ap.parse_args(argv)
    rng = np.random.default_rng(7)
    K = kernel()
    truth = []
    for m, arm, sex in d1.MICE:
        nc = a.n_cells
        kind = np.array(["UP"] * 15 + ["DOWN"] * 8 + ["withdrawal"] * 8 + ["movement"] * 8 + ["none"] * (nc - 39))
        H, W = 300, 360
        yy, xx = np.mgrid[:H, :W]
        cy, cx = rng.integers(20, H - 20, nc), rng.integers(20, W - 20, nc)
        Sp = np.stack([np.exp(-((yy - cy[j]) ** 2 + (xx - cx[j]) ** 2) / (2 * 4.0 ** 2)) for j in range(nc)], 2).astype(np.float32)
        tag = f"{m}_d1"
        exd = os.path.join(a.out_root, tag, "3_extract"); os.makedirs(exd, exist_ok=True)
        sio.savemat(os.path.join(exd, f"{tag}_extract.mat"), dict(
            spatial_weights=Sp, mean_image=(1000 + 200 * Sp.sum(2)).astype(np.float32),
            cell_qc=dict(centroid_xy=np.c_[cx, cy].astype(float)), SYNTHETIC="ca_synthetic_day1.py"), do_compression=True)
        td = os.path.join(a.out_root, tag, "5_analysis", "traces"); os.makedirs(td, exist_ok=True)
        for ph in ("pre", "post"):
            S, _ = d1.scoring_for(a.scoring_dir, m, ph)
            B = d1.behaviour(S)
            n = B["st"]["n_cam"] + 1
            fr = np.arange(n)
            gain = 0.5 if (arm == "SBI-553" and ph == "post") else 1.0
            drive = np.zeros((n, nc))
            for st in d1.STIM:
                for f in B["ev"][st]:
                    if f < n:
                        drive[f, kind == "UP"] += SIZE[st] * gain
                        if st in ("Heat", "Pin prick"):
                            drive[f, kind == "DOWN"] -= 0.8 * gain
            for f in B["ev"]["withdrawal"]:
                if f < n:
                    drive[f, kind == "withdrawal"] += 1.5 * gain
            ca = np.stack([np.convolve(drive[:, j], K)[:n] for j in range(nc)], 1)
            mv = B["st"]["move"][:n].astype(float)
            mvk = np.convolve(mv, np.ones(15) / 15, "same")
            ca[:, kind == "movement"] += 0.8 * mvk[:, None]
            e = rng.normal(0, 1, (n, nc))
            ar = np.zeros((n, nc))
            for i in range(1, n):
                ar[i] = 0.95 * ar[i - 1] + np.sqrt(1 - 0.95 ** 2) * e[i]
            drift = 0.3 * np.sin(2 * np.pi * fr[:, None] / (FPS * 600) + rng.uniform(0, 6.28, nc))
            z = (ca + ar + drift).astype(np.float32)
            z[: int(d1.SKIP_S * FPS)] = np.nan
            dff = 0.01 * z
            ev = []
            for j in range(nc):
                c = ca[:, j]
                up = np.where((c[1:] > 0.5) & (c[:-1] <= 0.5))[0] + 1
                for o in up:
                    if o / FPS >= d1.SKIP_S:
                        ev.append(dict(cell=j + 1, onset_frame=int(o), peak_frame=int(o + 6), onset_s=o / FPS,
                                       peak_s=(o + 6) / FPS, amplitude_dff=float(0.01 * c[min(o + 6, n - 1)]),
                                       peak_z=float(c[min(o + 6, n - 1)])))
                # spontaneous transients (gain-independent, so the rate change comes from evoked only)
                for o in rng.choice(np.arange(int(d1.SKIP_S * FPS), n - 10), rng.poisson(n / FPS / 60 * 2), replace=False):
                    ev.append(dict(cell=j + 1, onset_frame=int(o), peak_frame=int(o + 6), onset_s=o / FPS, peak_s=(o + 6) / FPS,
                                   amplitude_dff=0.01, peak_z=1.0))
            rid = f"{tag}_{ph}"
            np.savez_compressed(os.path.join(td, f"{rid}_traces.npz"), frame=fr, t_s=fr / FPS, valid=np.ones(n, bool),
                                x=dff, baseline=np.zeros_like(dff), dff=dff, z=z, c=ca.astype(np.float32),
                                s=np.zeros_like(dff), sn=np.full(nc, 0.01), g=np.full(nc, 0.967), tau_s=np.full(nc, 1.0),
                                cell_ids=np.arange(1, nc + 1), skip_s=d1.SKIP_S)
            pd.DataFrame(ev).sort_values(["cell", "onset_frame"]).to_csv(os.path.join(td, f"{rid}_events.csv"), index=False)
        for j in range(nc):
            truth.append(dict(mouse=m, arm=arm, cell=j + 1, kind=kind[j]))
        print(f"{m}: synthetic pre + post, {nc} cells")
    pd.DataFrame(truth).to_csv(os.path.join(a.out_root, "truth.csv"), index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
