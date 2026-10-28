"""Which cells to check first in the GUI: the kept cells that you have NOT decided (rule / rescued)
ranked by how much they move the main result, with the shape flags of the rule.

Main result per cell = log2((post + 0.1) / (pre + 0.1)) of the transient rate in the stimulus
blocks, still frames (the four blocks pooled, weighted by their still minutes; ca_day1 table).
Influence = how far the mouse median moves when the cell is left out, + |its own log2 ratio|
weighted by its activity share (a very active cell with a large change counts most).
Flags: not Ca-like, > 1 piece, area < 0.5x or > 2x the typical cell, < 60 px from the lens edge,
negative skew (mostly decreases), a near-duplicate partner, and - once traces_roi exists - a
EXTRACT trace that agrees poorly (r < 0.5) with the raw footprint pixels minus the local neuropil.

  python ca_review_priority.py --base C:/Users/hsollim/Documents/HEAL_mini1p_SBI553 [--top 20]
  -> <mouse>_d1/5_analysis/decisions/<mouse>_d1_review_priority.csv
"""
import argparse
import os

import numpy as np
import pandas as pd

MICE = ["5941", "5939", "5937", "1191"]
STIM = ["Light touch", "Mild touch", "Pin prick", "Heat"]


def trace_agreement(root, tag):
    """per cell: correlation of the EXTRACT dF/F with the demixing-free ROI dF/F (traces_roi,
    ca_roi_traces.py), 0.2-s smoothed, both sessions; NaN if the ROI traces are not there"""
    from scipy.ndimage import uniform_filter1d
    out = {}
    for ph in ("pre", "post"):
        fa = os.path.join(root, "5_analysis", "traces", f"{tag}_{ph}_traces.npz")
        fb = os.path.join(root, "5_analysis", "traces_roi", f"{tag}_{ph}_traces.npz")
        if not (os.path.isfile(fa) and os.path.isfile(fb)):
            return {}
        A, Bt = np.load(fa), np.load(fb)
        if not np.array_equal(A["cell_ids"], Bt["cell_ids"]):
            return {}
        X, Y = np.asarray(A["dff"], float), np.asarray(Bt["dff"], float)
        ok = np.isfinite(X).all(1) & np.isfinite(Y).all(1)
        for j, c in enumerate(A["cell_ids"].astype(int)):
            x, y = uniform_filter1d(X[ok, j], 6), uniform_filter1d(Y[ok, j], 6)
            r = np.corrcoef(x, y)[0, 1] if np.std(x) > 0 and np.std(y) > 0 else np.nan
            out.setdefault(c, []).append(r)
    return {c: float(np.nanmean(v)) for c, v in out.items()}


def pooled_still(W):
    num = sum(W[f"still rate | {s}"].fillna(0) * W[f"still min | {s}"].fillna(0) for s in STIM)
    den = sum(W[f"still min | {s}"].fillna(0) for s in STIM)
    return num / den.replace(0, np.nan)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--top", type=int, default=20)
    a = ap.parse_args(argv)
    W = pd.read_csv(os.path.join(a.base, "analysis", "ca_day1", "ca_d1_cells_whole_session.csv"))
    W["mouse"] = W.mouse.astype(str)
    W["stim_still"] = pooled_still(W)
    for m in MICE:
        root = os.path.join(a.base, "mini1p_out", "20261019", f"{m}_d1")
        R = pd.read_csv(os.path.join(root, "4_cell_review", f"{m}_d1_cell_review.csv"))
        R = R[R.keep == 1].copy()
        P = W[W.mouse == m].pivot(index="cell", columns="phase", values="stim_still")
        if not {"pre", "post"} <= set(P.columns):
            continue
        P["l2"] = np.log2((P.post + 0.1) / (P.pre + 0.1))
        med = P.l2.median()
        loo = {c: abs(P.l2.drop(c).median() - med) for c in P.index}
        share = (P.pre + P.post) / (P.pre + P.post).sum()
        R = R.set_index("id").join(P[["pre", "post", "l2"]], how="inner")
        R["loo_shift"] = [loo.get(c, np.nan) for c in R.index]
        R["influence"] = R.loo_shift * len(R) + (R.l2.abs() * share.reindex(R.index)) * len(R)
        agr = trace_agreement(root, f"{m}_d1")
        R["r_extract_vs_roi"] = [agr.get(c, np.nan) for c in R.index]
        typ = R.area_px.median()
        fl = []
        for c, r in R.iterrows():
            f = []
            if r.get("ca_like", 1) == 0:
                f.append("not Ca-like")
            if r.get("n_pieces", 1) > 1:
                f.append(f"{int(r.n_pieces)} pieces")
            if r.area_px < 0.5 * typ or r.area_px > 2 * typ:
                f.append(f"area {r.area_px / typ:.1f}x")
            if r.get("edge_px", 999) < 60:
                f.append("lens edge")
            if r.get("skew", 1) < 0:
                f.append("negative skew")
            if isinstance(r.get("reasons"), str) and "same soma" in r.reasons:
                f.append("near-duplicate?")
            if np.isfinite(r.get("r_extract_vs_roi", np.nan)) and r.r_extract_vs_roi < 0.5:
                f.append(f"EXTRACT vs raw pixels r {r.r_extract_vs_roi:.2f}")
            fl.append("; ".join(f))
        R["flags"] = fl
        X = R[R.decision_source.astype(str) != "user"].copy()
        X["rank_score"] = X.influence.rank(ascending=False) + 0.5 * (X["flags"] == "").astype(int) * len(X) / 4
        X = X.sort_values("rank_score")
        cols = ["decision_source", "verdict", "flags", "pre", "post", "l2", "influence", "r_extract_vs_roi", "row", "col", "snr", "reasons"]
        out = os.path.join(root, "5_analysis", "decisions", f"{m}_d1_review_priority.csv")
        Y = X[[c for c in cols if c in X]].head(a.top).rename(columns={"pre": "stim_still_rate_pre", "post": "stim_still_rate_post",
                                                                      "l2": "log2_post_over_pre"})
        Y.index.name = "cell_id"
        Y.round(3).to_csv(out)
        print(f"{m}: {len(X)} cells without your decision; top {len(Y)} -> {out}")
        print(Y[["decision_source", "flags", "stim_still_rate_pre", "stim_still_rate_post", "log2_post_over_pre"]].head(8).round(2).to_string())


if __name__ == "__main__":
    main()
