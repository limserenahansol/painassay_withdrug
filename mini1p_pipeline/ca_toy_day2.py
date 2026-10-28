"""TOY Day 2 for the longitudinal pipeline (Day 2 has not been recorded yet).

Design assumed (crossover): every mouse gets the OTHER treatment on Day 2, so each mouse ends
with two baselines, one SBI-553 and one vehicle session. The toy Day 2 of a mouse is made from
its REAL Day-1 baseline session, with KNOWN changes written to toy_truth.json, so the
longitudinal analysis can be checked against them:
  field of view   footprints and mean image shifted by (dy, dx) = (6, -4) px and blurred a little;
                  15 % of the cells are not found again, 10 % new cells appear, new numbering
                  (so the cross-day matching has to work, and its accuracy is measured)
  Day 2 baseline  Day-1 baseline traces + independent noise (z SD 0.3), 10 % of transients lost
  Day 2 injection SBI-553 (mice that had vehicle on Day 1): z x 0.7 and 30 % of transients removed
                  (toy effect); vehicle (mice that had SBI-553 on Day 1): like the Day 2 baseline
  behaviour       the Day-1 baseline scoring file for both Day 2 sessions (same event times),
                  saved with phase Baseline / Post-treatment in <toy>/scoring
EVERY output is under <out-root> (default mini1p_out/TOY_day2) and every figure made from it must
say TOY.

  python ca_toy_day2.py --day1-root <mini1p_out/20261019> --scoring-dir <final> --out-root <TOY_day2>
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.ndimage import gaussian_filter, shift as ndshift

MICE = [("5941", "SBI-553"), ("5939", "SBI-553"), ("5937", "vehicle"), ("1191", "vehicle")]
SHIFT = (6, -4)
P_LOST, P_NEW = 0.15, 0.10
NOISE_Z = 0.3
SBI_SCALE, SBI_DROP, BASE_DROP = 0.7, 0.30, 0.10
FPS = 30.0


def scoring_for(scoring_dir, mouse, phase):
    for p in glob.glob(os.path.join(scoring_dir, "ScoringAB_*.mat")):
        S = sio.loadmat(p, squeeze_me=True, variable_names=["mouseID", "phase"])
        ph = "pre" if str(S["phase"]).lower().startswith("base") else "post"
        if str(S["mouseID"]).strip() == mouse and ph == phase:
            return p
    return None


def make_mouse(m, arm1, a, rng):
    tag1, tag2 = f"{m}_d1", f"{m}_d2"
    td1 = os.path.join(a.day1_root, tag1, "5_analysis", "traces")
    f_pre = os.path.join(td1, f"{tag1}_pre_traces.npz")
    if not os.path.isfile(f_pre):
        print(f"{m}: no Day-1 baseline traces yet"); return None
    T = dict(np.load(f_pre))
    E = pd.read_csv(os.path.join(td1, f"{tag1}_pre_events.csv"))
    M = sio.loadmat(os.path.join(a.day1_root, tag1, "3_extract", f"{tag1}_extract.mat"), squeeze_me=True,
                    variable_names=["spatial_weights", "mean_image"])
    Sp = np.asarray(M["spatial_weights"], np.float32)
    mean1 = np.asarray(M["mean_image"], np.float32)
    ids1 = T["cell_ids"].astype(int)                         # kept cells of Day 1
    K = len(ids1)
    found = rng.random(K) >= P_LOST
    n_new = int(round(P_NEW * K))
    # Day-2 numbering: shuffled
    order = rng.permutation(found.sum() + n_new)
    src = [("d1", int(c)) for c in ids1[found]] + [("new", i) for i in range(n_new)]
    src = [src[i] for i in order]
    H, W = mean1.shape
    S2 = np.zeros((H, W, len(src)), np.float32)
    for j, (kind, c) in enumerate(src):
        if kind == "d1":
            fp = Sp[:, :, c - 1]
        else:                                                 # a new cell: a round blob somewhere bright
            yy, xx = np.mgrid[:H, :W]
            cy, cx = rng.integers(60, H - 60), rng.integers(60, W - 60)
            fp = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 4.0 ** 2)).astype(np.float32)
        g = ndshift(gaussian_filter(fp, 0.7), SHIFT, order=1)
        S2[:, :, j] = g / max(g.max(), 1e-9)
    mean2 = ndshift(mean1, SHIFT, order=1, mode="nearest") + rng.normal(0, 0.01 * mean1.std(), mean1.shape)
    exd = os.path.join(a.out_root, tag2, "3_extract")
    os.makedirs(exd, exist_ok=True)
    sio.savemat(os.path.join(exd, f"{tag2}_extract.mat"), dict(spatial_weights=S2, mean_image=mean2.astype(np.float32),
                TOY="toy Day 2 made by ca_toy_day2.py from the Day-1 baseline"), do_compression=True)
    # traces
    td2 = os.path.join(a.out_root, tag2, "5_analysis", "traces")
    os.makedirs(td2, exist_ok=True)
    arm2 = "vehicle" if arm1 == "SBI-553" else "SBI-553"
    col1 = {int(c): j for j, c in enumerate(ids1)}
    truth_sessions = {}
    for ph in ("pre", "post"):
        sbi = ph == "post" and arm2 == "SBI-553"
        scale, drop = (SBI_SCALE, SBI_DROP) if sbi else (1.0, BASE_DROP)
        n = len(T["frame"])
        out = {k: T[k] for k in ("frame", "t_s", "valid", "skip_s") if k in T}
        arrs = {k: np.zeros((n, len(src)), np.float32) for k in ("x", "baseline", "dff", "z", "c", "s")}
        sn, g = np.zeros(len(src)), np.zeros(len(src))
        ev = []
        for j, (kind, c) in enumerate(src):
            jj = col1[c] if kind == "d1" else int(rng.integers(0, K))
            sh = 0 if kind == "d1" else int(rng.integers(30 * 30, n - 30 * 30))
            for k in arrs:
                v = np.roll(T[k][:, jj], sh)
                if k in ("z", "dff", "c", "s", "x"):
                    v = v * scale
                arrs[k][:, j] = v
            arrs["z"][:, j] += rng.normal(0, NOISE_Z, n).astype(np.float32)
            sn[j], g[j] = T["sn"][jj], T["g"][jj]
            e = E[E.cell == ids1[jj]].copy()
            if sh:
                fr = T["frame"]
                idx = (np.searchsorted(fr, e.onset_frame.values) + sh) % n
                e["onset_frame"] = fr[idx]; e["onset_s"] = fr[idx] / FPS
                idx = (np.searchsorted(fr, e.peak_frame.values) + sh) % n
                e["peak_frame"] = fr[idx]; e["peak_s"] = fr[idx] / FPS
            e = e[rng.random(len(e)) >= drop]
            e["cell"] = j + 1
            e["amplitude_dff"] = e.amplitude_dff * scale
            e["peak_z"] = e.peak_z * scale
            ev.append(e)
        out.update(arrs)
        out.update(sn=sn, g=g, tau_s=-1 / (FPS * np.log(np.clip(g, 1e-6, 0.999999))), cell_ids=np.arange(1, len(src) + 1))
        rid = f"{tag2}_{ph}"
        np.savez_compressed(os.path.join(td2, f"{rid}_traces.npz"), **out)
        pd.concat(ev).sort_values(["cell", "onset_frame"]).to_csv(os.path.join(td2, f"{rid}_events.csv"), index=False)
        truth_sessions[rid] = dict(treatment=("SBI-553" if sbi else ("vehicle" if ph == "post" else "baseline")),
                                   z_scale=scale, transients_removed=drop)
    # behaviour: the Day-1 baseline scoring for both sessions
    sd = os.path.join(a.out_root, "scoring")
    os.makedirs(sd, exist_ok=True)
    sp = scoring_for(a.scoring_dir, m, "pre")
    if sp:
        S = sio.loadmat(sp, squeeze_me=False)
        S = {k: v for k, v in S.items() if not k.startswith("__")}
        for ph, phase in (("pre", "Baseline"), ("post", "Post-treatment")):
            S["phase"] = np.array([phase]); S["TOY"] = np.array(["toy Day 2: the Day-1 baseline scoring"])
            S["day"] = np.array([[2]])
            sio.savemat(os.path.join(sd, f"ScoringAB_TOY_{m}_d2_{ph}.mat"), S)
    pairs = [dict(d2_cell=j + 1, d1_cell=(c if kind == "d1" else None)) for j, (kind, c) in enumerate(src)]
    return dict(mouse=m, day1_treatment=arm1, day2_treatment=arm2, shift_px=SHIFT, n_day1=K,
                n_day2=len(src), n_lost=int((~found).sum()), n_new=n_new, sessions=truth_sessions, pairs=pairs)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--day1-root", required=True)
    ap.add_argument("--scoring-dir", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--seed", type=int, default=20261020)
    a = ap.parse_args(argv)
    os.makedirs(a.out_root, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    truth = dict(note="TOY Day 2 - not data. Made from each mouse's real Day-1 baseline with known changes.",
                 shift_px=SHIFT, p_lost=P_LOST, p_new=P_NEW, noise_z=NOISE_Z,
                 sbi_toy_effect=dict(z_scale=SBI_SCALE, transients_removed=SBI_DROP), mice={})
    for m, arm in MICE:
        r = make_mouse(m, arm, a, rng)
        if r:
            truth["mice"][m] = r
            print(f"{m}: toy Day 2 ({r['day2_treatment']}): {r['n_day2']} cells ({r['n_lost']} of {r['n_day1']} lost, "
                  f"{r['n_new']} new)")
    with open(os.path.join(a.out_root, "toy_truth.json"), "w") as fh:
        json.dump(truth, fh, indent=1, default=lambda o: None if o is None else int(o) if isinstance(o, np.integer) else str(o))
    return 0


if __name__ == "__main__":
    sys.exit(main())
