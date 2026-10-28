"""ca_pair_flag.py - kept neighbours whose traces move in OPPOSITE directions (a least-squares split of one soma?).

Two kept footprints that overlap (cosine >= 0.5) share pixels; when EXTRACT's least squares divides the signal between
them their dF/F traces become anti-correlated, whereas two real neighbouring neurons are positively correlated (shared
neuropil / background). Pairs with r < -0.2 in either session get an attention mark ("check: maybe one neuron split")
- both stay kept (you decided on Day 1 that they are two neurons); you decide on the max projection.
Writes the attention text into <tag>_cell_review.csv (backup *_before_pairflag.csv) and decisions/<tag>_pair_flags.csv.

  python ca_pair_flag.py --day-root <mini1p_out\\20261021\\5941_d2>
"""
import argparse
import os
import shutil

import numpy as np
import pandas as pd
import scipy.io as sio
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import safe_io  # noqa: E402,F401  (a figure save retries on a transient Windows write error)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--day-root", required=True)
    ap.add_argument("--min-cos", type=float, default=0.5)
    ap.add_argument("--max-r", type=float, default=-0.2)
    a = ap.parse_args(argv)
    tag = os.path.basename(os.path.normpath(a.day_root))
    csv = os.path.join(a.day_root, "4_cell_review", f"{tag}_cell_review.csv")
    R = pd.read_csv(csv)
    S = np.asarray(sio.loadmat(os.path.join(a.day_root, "3_extract", f"{tag}_extract.mat"),
                               variable_names=["spatial_weights"])["spatial_weights"], np.float32)
    keep = np.flatnonzero(R.keep.values == 1)
    F = S.reshape(-1, S.shape[2])[:, keep]
    F = F / np.maximum(np.linalg.norm(F, axis=0), 1e-12)
    C = F.T @ F
    tr = {}
    for ph in ("pre", "post"):
        p = os.path.join(a.day_root, "5_analysis", "traces", f"{tag}_{ph}_traces.npz")
        if os.path.isfile(p):
            T = np.load(p); tr[ph] = (list(T["cell_ids"].astype(int)), T["dff"])
    rows = []
    for x in range(len(keep)):
        for y in range(x + 1, len(keep)):
            if C[x, y] < a.min_cos:
                continue
            ia, ib = int(R.id.iloc[keep[x]]), int(R.id.iloc[keep[y]])
            rr = {}
            for ph, (ids, D) in tr.items():
                if ia in ids and ib in ids:
                    u, v = D[:, ids.index(ia)], D[:, ids.index(ib)]
                    ok = np.isfinite(u) & np.isfinite(v)
                    rr[ph] = float(np.corrcoef(u[ok], v[ok])[0, 1])
            rows.append(dict(a=ia, b=ib, cos=round(float(C[x, y]), 3), **{f"r_{k}": round(v, 3) for k, v in rr.items()}))
    P = pd.DataFrame(rows)
    if os.path.isdir(os.path.join(a.day_root, "5_analysis", "decisions")):
        P.to_csv(os.path.join(a.day_root, "5_analysis", "decisions", f"{tag}_pair_flags.csv"), index=False)
    bk = csv.replace(".csv", "_before_pairflag.csv")
    if not os.path.isfile(bk):
        shutil.copy(csv, bk)
    R["attention"] = R.get("attention", pd.Series([""] * len(R))).fillna("").astype(str)
    n = 0
    for r in P.itertuples():
        rmin = min(getattr(r, "r_pre", 0.0), getattr(r, "r_post", 0.0))
        if not np.isfinite(rmin) or rmin >= a.max_r:
            continue
        for me, other in ((r.a, r.b), (r.b, r.a)):
            k = R.index[R.id == me][0]
            txt = f"overlaps #{other} (cos {r.cos:.2f}) and their traces are anti-correlated (r {rmin:.2f}) - one neuron split? check"
            if txt not in R.at[k, "attention"]:
                R.at[k, "attention"] = (R.at[k, "attention"] + "; " if R.at[k, "attention"] else "") + txt
        n += 1
    R.to_csv(csv, index=False)
    print(f"{tag}: {len(P)} overlapping kept pairs, {n} with anti-correlated traces flagged")


if __name__ == "__main__":
    main()
