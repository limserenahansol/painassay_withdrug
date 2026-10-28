"""Match the EXTRACT candidates of two runs on the same movie (e.g. 2 x 2 partitions vs 1) by
footprint overlap and report what each run has that the other does not, with the distance to
the partition seam, plus a map of both.

  python ca_partition_compare.py --a RUN_A_extract.mat --a-review A.csv --b RUN_B_extract.mat
      --b-review B.csv --out PNG [--seam-x 0.5] [--seam-y 0.5]
"""
import argparse
import sys

import numpy as np
import pandas as pd
import scipy.io as sio


def load(p, csv):
    M = sio.loadmat(p, squeeze_me=True, variable_names=["spatial_weights"])
    S = np.asarray(M["spatial_weights"], float)
    S = S[:, :, None] if S.ndim == 2 else S
    R = pd.read_csv(csv)
    return S, R


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--a-review", required=True)
    ap.add_argument("--b", required=True); ap.add_argument("--b-review", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels", default="2x2 partitions,1 partition")
    a = ap.parse_args(argv)
    la, lb = a.labels.split(",")
    SA, RA = load(a.a, a.a_review)
    SB, RB = load(a.b, a.b_review)
    H, W = SA.shape[:2]
    fa = SA.reshape(H * W, -1); fb = SB.reshape(H * W, -1)
    fa = fa / np.maximum(np.linalg.norm(fa, axis=0), 1e-12); fb = fb / np.maximum(np.linalg.norm(fb, axis=0), 1e-12)
    C = fa.T @ fb                                     # cosine overlap
    keepA, keepB = RA.keep.values == 1, RB.keep.values == 1
    bestB, bestA = C.max(1), C.max(0)
    matchA = bestB >= 0.5
    matchB = bestA >= 0.5
    print(f"{la}: {SA.shape[2]} candidates ({keepA.sum()} keep); {lb}: {SB.shape[2]} candidates ({keepB.sum()} keep)")
    print(f"kept in {la} with no match (overlap >= 0.5) in {lb}: {np.sum(keepA & ~matchA)}; kept in {lb} with no match in {la}: {np.sum(keepB & ~matchB)}")
    both = [(i, int(np.argmax(C[i]))) for i in range(len(RA)) if matchA[i]]
    agree = sum(1 for i, j in both if keepA[i] == keepB[j])
    print(f"matched pairs {len(both)}, same keep decision in {agree}")
    cen = lambda S, k: (np.average(np.arange(H), weights=S[:, :, k].sum(1) + 1e-12), np.average(np.arange(W), weights=S[:, :, k].sum(0) + 1e-12))
    rows = []
    for i in np.where(keepA & ~matchA)[0]:
        r, c = cen(SA, i); rows.append(dict(run=la, id=int(RA.id[i]), row=round(r), col=round(c), dist_to_vertical_mid=round(abs(c - W / 2)), verdict=RA.verdict[i]))
    for j in np.where(keepB & ~matchB)[0]:
        r, c = cen(SB, j); rows.append(dict(run=lb, id=int(RB.id[j]), row=round(r), col=round(c), dist_to_vertical_mid=round(abs(c - W / 2)), verdict=RB.verdict[j]))
    T = pd.DataFrame(rows)
    print(T.to_string(index=False) if len(T) else "no unmatched kept candidates")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage import measure
    fig, ax = plt.subplots(1, 2, figsize=(18, 8.5))
    for k, (S, R, keep, match, lab) in enumerate(((SA, RA, keepA, matchA, la), (SB, RB, keepB, matchB, lb))):
        ax[k].imshow(S.max(2), cmap="gray")
        for i in range(S.shape[2]):
            cs = measure.find_contours(S[:, :, i] / max(S[:, :, i].max(), 1e-12), 0.5)
            if not cs:
                continue
            xy = max(cs, key=len)
            col = "#2ecc71" if keep[i] and match[i] else ("#ff00ff" if keep[i] else "#ff5050")
            ax[k].plot(xy[:, 1], xy[:, 0], color=col, lw=1.2 if keep[i] else 0.5)
        ax[k].axvline(W / 2, color="c", ls=":", lw=0.8)
        ax[k].set_title(f"{lab}: {S.shape[2]} candidates, {keep.sum()} keep (magenta = kept but not found in the other run)", fontsize=10)
        ax[k].axis("off")
    fig.suptitle("Same movie, same recipe: does the EXTRACT partition seam (dotted) change which cells are found?", fontsize=12)
    fig.tight_layout(); fig.savefig(a.out, dpi=100); plt.close(fig)
    T.to_csv(a.out.replace(".png", ".csv"), index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
