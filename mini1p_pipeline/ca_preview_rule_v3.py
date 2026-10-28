"""Decision v3 for the loose previews (2026-10-21): v2 kept far too many non-neuron shapes.

keep = 1 only for
  a) a candidate that matches a cell kept in the ORIGINAL preview (footprint cosine >= 0.5;
     you said those greens are fine)
  b) a candidate at one of your screenshot marks
  c) a NEW candidate that clearly looks like a neuron:
       one compact body   mass in the biggest piece >= main_min, solidity >= sol_min
       normal size/shape  area_lo <= area ratio <= area_hi, axis ratio <= axis_max
       movie proof        event_image_corr >= ev_min (the movie at its own transients lights up
                          the footprint), and Ca-like transients
     thresholds default to the 25th percentile of the original greens (so a new cell must look at
     least as good as most of the cells you accepted)
everything else keep = 0. Writes the v2_loose review csv (backup *_before_v3.csv), column
decision_v3 with the reason, and prints the counts.

  python ca_preview_rule_v3.py --run <preview3min/<rid>> [--marks <csv>]
"""
import argparse
import os
import shutil

import numpy as np
import pandas as pd
import scipy.io as sio
from skimage.measure import label, regionprops


def shape(S):
    rows = []
    for j in range(S.shape[2]):
        f = S[:, :, j]
        m = f.max()
        if m <= 0:
            rows.append(dict(main_frac=0, solidity=0)); continue
        bw = f >= 0.3 * m
        L = label(bw, connectivity=2)
        props = regionprops(L, intensity_image=f)
        mass = np.array([p.intensity_mean * p.area for p in props])
        k = int(np.argmax(mass))
        rows.append(dict(main_frac=float(mass[k] / mass.sum()), solidity=float(props[k].solidity)))
    return pd.DataFrame(rows)


def load(xd, rd, tag):
    S = np.asarray(sio.loadmat(os.path.join(xd, f"{tag}_extract.mat"), variable_names=["spatial_weights"])["spatial_weights"], np.float32)
    return S, pd.read_csv(os.path.join(rd, f"{tag}_cell_review.csv"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--marks", nargs="*", default=[], help="one or more marks csv (all screenshots)")
    ap.add_argument("--variant", default="v2_loose", help="subfolder with the re-detected candidates")
    ap.add_argument("--q", type=float, default=25, help="percentile of the original greens used as the bar")
    ap.add_argument("--dry", action="store_true", help="count only, do not write the csv")
    a = ap.parse_args(argv)
    rid = os.path.basename(os.path.normpath(a.run))
    tag = rid.rsplit("_", 1)[0]
    S0, R0 = load(os.path.join(a.run, "3_extract"), os.path.join(a.run, "4_review"), tag)
    vd = os.path.join(a.run, a.variant)
    S1, R1 = load(os.path.join(vd, "3_extract"), os.path.join(vd, "4_review"), tag)
    csv = os.path.join(vd, "4_review", f"{tag}_cell_review.csv")
    if not os.path.isfile(csv.replace(".csv", "_before_v3.csv")):
        shutil.copy(csv, csv.replace(".csv", "_before_v3.csv"))
    sh0, sh1 = shape(S0), shape(S1)
    R0 = pd.concat([R0, sh0], axis=1); R1 = pd.concat([R1.drop(columns=[c for c in ("main_frac", "solidity") if c in R1]), sh1], axis=1)
    G = R0[R0.keep == 1]
    q = a.q
    thr = dict(main_min=float(np.percentile(G.main_frac, q)), sol_min=float(np.percentile(G.solidity, q)),
               area_lo=float(np.percentile(G.area_ratio, q / 2)), area_hi=float(np.percentile(G.area_ratio, 100 - q / 2)),
               axis_max=float(np.percentile(G.axis_ratio, 100 - q)), ev_min=float(np.nanpercentile(G.event_image_corr, q)))
    # a) matches an original green
    A0 = S0.reshape(-1, S0.shape[2]); A1 = S1.reshape(-1, S1.shape[2])
    A0 = A0 / np.maximum(np.linalg.norm(A0, axis=0), 1e-12); A1 = A1 / np.maximum(np.linalg.norm(A1, axis=0), 1e-12)
    cos = A1.T @ A0[:, (R0.keep == 1).values]
    match = cos.max(1) >= 0.5 if cos.size else np.zeros(len(R1), bool)
    # b) your marks
    marked = np.zeros(len(R1), bool)
    ms = [pd.read_csv(f) for f in a.marks if os.path.isfile(f) and os.path.getsize(f) > 10]
    if ms:
        M = pd.concat(ms, ignore_index=True)
        mx = S1.reshape(-1, S1.shape[2]).max(0)
        for _, m in M.iterrows():
            r, c = int(round(m.fov_row)), int(round(m.fov_col))
            hit = list(np.where(S1[r, c, :] >= 0.3 * np.maximum(mx, 1e-12))[0]) if 0 <= r < S1.shape[0] and 0 <= c < S1.shape[1] else []
            if hit:
                marked[max(hit, key=lambda i: S1[r, c, i] / max(mx[i], 1e-12))] = True
            else:
                d = np.hypot(R1.row - m.fov_row, R1.col - m.fov_col)
                if d.min() <= 10:
                    marked[int(np.argmin(d))] = True
    # c) clear neuron shape
    ok_shape = ((R1.main_frac >= thr["main_min"]) & (R1.solidity >= thr["sol_min"]) & R1.area_ratio.between(thr["area_lo"], thr["area_hi"])
                & (R1.axis_ratio <= thr["axis_max"]) & (R1.event_image_corr >= thr["ev_min"]) & (R1.ca_like == 1)
                & (R1.verdict != "not_cell"))
    keep = match | marked | ok_shape.values
    why = np.where(marked, "your mark", np.where(match, "kept in the original preview", np.where(ok_shape, "new: clear neuron shape", "")))
    R1["keep"] = keep.astype(int)
    R1["decision_v3"] = why
    if not a.dry:
        R1.drop(columns=["main_frac", "solidity"]).to_csv(csv, index=False)
    print(f"{rid}: bar from {len(G)} original greens: " + ", ".join(f"{k} {v:.2f}" for k, v in thr.items()))
    print(f"   {len(R1)} candidates -> keep {int(keep.sum())}: original greens matched {int(match.sum())}, your marks {int(marked.sum())}, "
          f"new clear neurons {int((ok_shape.values & ~match & ~marked).sum())}  (original preview kept {len(G)})")


if __name__ == "__main__":
    main()
