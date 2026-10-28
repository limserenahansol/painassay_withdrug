"""Duplicate / split-cell check of the kept cells of one mouse-day.

Two kept components can be ONE neuron: EXTRACT split a soma, or a rescued footprint landed next to
the day run's own fragment of it. Such a pair would count one neuron twice in every statistic.
A pair is flagged when the footprints overlap strongly (cosine >= --max-overlap, 0.5: the joint
least squares then splits one signal between them and the traces ANTI-correlate - seen in the
3-min test, cos 0.87 -> r -0.79), or overlap (cos >= --min-cos, 0.2) with |r| >= --min-r (0.7)
(local dF/F, both sessions pooled, 0.2-s bins). Nothing is changed: the table and a page of pictures say which pairs to look at (keep the
better one: delete the other in the review table and re-run ca_post_mouse.sh).

Outputs (--out): <tag>_duplicate_pairs.csv, <tag>_duplicate_pairs.png
  python ca_duplicate_check.py --day-root <mini1p_out/20261019/5941_d1> --out <...>/5_analysis/decisions
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import scipy.io as sio

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import safe_io  # noqa: E402,F401  (a figure save retries on a transient Windows write error)


def resolve(day_root, tag, cos_thr):
    """near-identical kept footprints (cos >= cos_thr) are one neuron: keep ONE per group - the one
    carrying your decision, else the higher SNR - and set the others keep = 0 (note says which).
    Two genuinely different overlapping neurons do not share this many pixels in one plane, and
    a joint least-squares fit of near-identical footprints splits the signal (anti-correlated
    traces), so leaving both would count one neuron twice with damaged traces."""
    rv = os.path.join(day_root, "4_cell_review", f"{tag}_cell_review.csv")
    R = pd.read_csv(rv)
    M = sio.loadmat(os.path.join(day_root, "3_extract", f"{tag}_extract.mat"), squeeze_me=True,
                    struct_as_record=False, variable_names=["spatial_weights", "cell_qc"])
    Sp = np.asarray(M["spatial_weights"], np.float32)
    kept = R.id[R.keep == 1].values.astype(int)
    F = Sp[:, :, kept - 1].reshape(-1, len(kept))
    nrm = np.linalg.norm(F, axis=0) + 1e-12
    C = (F.T @ F) / np.outer(nrm, nrm)
    np.fill_diagonal(C, 0)
    # same soma: near-identical footprints (cos >= cos_thr), or overlapping (cos >= 0.5) with the
    # centroids <= 6 px apart (about half a cell radius: two different neurons sit >= ~10 px apart)
    cxy = np.asarray(M["cell_qc"].centroid_xy, float)[kept - 1]
    D = np.hypot(cxy[:, None, 0] - cxy[None, :, 0], cxy[:, None, 1] - cxy[None, :, 1])
    same = (C >= cos_thr) | ((C >= 0.5) & (D <= 6.0))
    # greedy, best first (your decision before the rules, then SNR): a cell is kept unless it
    # duplicates one already kept. Duplicate = same soma (above), or - when at least one of the two
    # is a rule-only candidate - a footprint overlap of cos >= 0.5: you decided that spot on the
    # reference movie with ONE component, so a new candidate sitting on it is that neuron again.
    # Two of YOUR cells are only merged when they are the same soma.
    src = R.set_index("id").get("decision_source")
    snr = pd.to_numeric(R.set_index("id").get("snr"), errors="coerce") if "snr" in R else None
    is_user = np.array([src is not None and str(src.get(c, "")) == "user" for c in kept])
    sn = np.array([float(snr.get(c)) if snr is not None and np.isfinite(snr.get(c, np.nan)) else 0.0 for c in kept])
    order = sorted(range(len(kept)), key=lambda i: (-int(is_user[i]), -sn[i]))
    acc, changed = [], []
    for i in order:
        hit = None
        for j in acc:
            dup = same[i, j] or (C[i, j] >= 0.5 and not (is_user[i] and is_user[j]))
            if dup:
                hit = j
                break
        if hit is None:
            acc.append(i)
            continue
        c, best = int(kept[i]), int(kept[hit])
        k = R.index[R.id == c][0]
        R.loc[k, "keep"] = 0
        why = "same soma" if same[i, hit] else ("overlaps your cell" if is_user[hit] else "overlaps a better candidate")
        R.loc[k, "note"] = (f"duplicate of #{best} ({why}: footprint cos {C[i, hit]:.2f}, centroids {D[i, hit]:.1f} px "
                            f"apart) - one neuron, the better one kept")
        if "decision_source" in R:
            R.loc[k, "decision_source"] = "auto-duplicate"
        changed.append((c, best, round(float(C[i, hit]), 3)))
    if changed:
        bk = rv.replace("_cell_review.csv", "_cell_review_before_dedup.csv")
        if not os.path.isfile(bk):
            import shutil
            shutil.copyfile(rv, bk)
        R.to_csv(rv, index=False)
    print(f"{tag}: {len(changed)} kept duplicates set to keep 0 (same soma, or a rule candidate overlapping a kept cell at cos >= 0.5): " +
          ", ".join(f"#{c} -> #{b} ({cs})" for c, b, cs in changed))
    return changed


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--day-root", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--resolve", type=float, default=0.0,
                    help="> 0: keep one of each group of kept footprints with cos >= this (edits the review table) and stop")
    ap.add_argument("--min-cos", type=float, default=0.2)
    ap.add_argument("--min-r", type=float, default=0.7)
    ap.add_argument("--max-overlap", type=float, default=0.5)
    a = ap.parse_args(argv)
    tag = os.path.basename(os.path.normpath(a.day_root))
    if a.resolve > 0:
        resolve(a.day_root, tag, a.resolve)
        return 0
    os.makedirs(a.out, exist_ok=True)
    M = sio.loadmat(os.path.join(a.day_root, "3_extract", f"{tag}_extract.mat"), squeeze_me=True,
                    variable_names=["spatial_weights", "mean_image"])
    Sp = np.asarray(M["spatial_weights"], np.float32)
    mi = np.asarray(M["mean_image"], float)
    files = sorted(glob.glob(os.path.join(a.day_root, "5_analysis", "traces", f"{tag}_*_traces.npz")))
    if not files:
        sys.exit("no trace files")
    Z, ids = [], None
    for f in files:
        T = np.load(f)
        ids = T["cell_ids"].astype(int)
        d = T["dff"]
        n = len(d) // 6
        Z.append(np.nanmean(d[:n * 6].reshape(n, 6, -1), 1))
    Z = np.vstack(Z)
    ok = np.all(np.isfinite(Z), 1)
    C = np.corrcoef(Z[ok].T)
    F = Sp[:, :, ids - 1].reshape(-1, len(ids))
    nrm = np.linalg.norm(F, axis=0) + 1e-12
    cosm = (F.T @ F) / np.outer(nrm, nrm)
    rows = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            # same cell twice: strongly overlapping footprints (least squares then splits the
            # signal and the traces anti-correlate), or overlapping AND |r| high
            if cosm[i, j] >= a.max_overlap or (cosm[i, j] >= a.min_cos and abs(C[i, j]) >= a.min_r):
                rows.append(dict(cell_a=int(ids[i]), cell_b=int(ids[j]), footprint_cos=round(float(cosm[i, j]), 3),
                                 trace_r=round(float(C[i, j]), 3)))
    D = pd.DataFrame(rows, columns=["cell_a", "cell_b", "footprint_cos", "trace_r"])
    D.to_csv(os.path.join(a.out, f"{tag}_duplicate_pairs.csv"), index=False)
    # context: the distribution for all overlapping pairs
    iu = np.triu_indices(len(ids), 1)
    fig = plt.figure(figsize=(14, 3.2 + 2.2 * max(min(len(D), 8), 1)))
    ax0 = fig.add_axes([0.06, 1 - 2.6 / fig.get_figheight(), 0.4, 2.0 / fig.get_figheight()])
    ax0.scatter(cosm[iu], C[iu], s=4, color="0.6")
    ax0.scatter(D.footprint_cos, D.trace_r, s=14, color="red")
    ax0.axvline(a.min_cos, color="k", ls=":", lw=0.8); ax0.axhline(a.min_r, color="k", ls=":", lw=0.8)
    ax0.set_xlabel("footprint overlap (cosine)"); ax0.set_ylabel("trace correlation (r)")
    ax0.set_title(f"{tag}: {len(ids)} kept cells, {len(D)} possible duplicate pairs (red)", fontsize=10)
    for k, r in enumerate(D.head(8).itertuples()):
        y0 = 1 - (2.9 + 2.2 * (k + 1)) / fig.get_figheight()
        h = 2.0 / fig.get_figheight()
        i, j = list(ids).index(r.cell_a), list(ids).index(r.cell_b)
        fa, fb = Sp[:, :, r.cell_a - 1], Sp[:, :, r.cell_b - 1]
        yy, xx = np.unravel_index(np.argmax(fa + fb), fa.shape)
        r0, c0 = max(yy - 35, 0), max(xx - 35, 0)
        axi = fig.add_axes([0.03, y0, 0.12, h])
        crop = mi[r0:yy + 35, c0:xx + 35]
        axi.imshow(crop, cmap="gray")
        axi.contour(fa[r0:yy + 35, c0:xx + 35], levels=[0.4 * fa.max()], colors=["#2ecc71"])
        axi.contour(fb[r0:yy + 35, c0:xx + 35], levels=[0.4 * fb.max()], colors=["#e67e22"])
        axi.set_axis_off()
        axi.set_title(f"#{r.cell_a} (green) / #{r.cell_b} (orange)", fontsize=7)
        axt = fig.add_axes([0.18, y0, 0.8, h])
        seg = slice(0, min(3000, ok.sum()))
        za, zb = Z[ok][seg, i], Z[ok][seg, j]
        axt.plot(za / (np.nanstd(za) + 1e-9), lw=0.5, color="#2ecc71")
        axt.plot(zb / (np.nanstd(zb) + 1e-9) - 6, lw=0.5, color="#e67e22")
        axt.set_yticks([]); axt.set_xlabel("0.2-s bins (first 10 min)", fontsize=7)
        axt.set_title(f"overlap cos {r.footprint_cos}, r {r.trace_r}", fontsize=7)
    fig.savefig(os.path.join(a.out, f"{tag}_duplicate_pairs.png"), dpi=100); plt.close(fig)
    print(f"{tag}: {len(D)} possible duplicate pairs among {len(ids)} kept cells")
    return 0


if __name__ == "__main__":
    sys.exit(main())
