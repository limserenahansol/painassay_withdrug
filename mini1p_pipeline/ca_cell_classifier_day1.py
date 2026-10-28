"""ca_cell_classifier_day1.py - learn YOUR Day-1 keep / delete decisions, to rank new Day-2 candidates (not to decide).

Labels: your final Day-1 selection (green + orange kept = 1, every other EXTRACT candidate = 0; rescued footprints left
out). Features of every candidate, all available on any day:
  review table   area_ratio, axis_ratio, n_pieces, edge_px, snr, skew, neg_ratio, n_events, rise_s, decay_half_s,
                 ca_like, max_S_corr, event_image_corr
  footprint      mass in the biggest piece, solidity of that piece, compactness (area / convex-hull area)
  max dF/F, Cn   soma contrast: footprint core (>= 50 % of peak) vs the ring 1.5-2.5 radii around it, in noise units
                 (what you look at on the max projection: a compact bright body), and the same on the Cn image
Model: gradient-boosted trees (sklearn HistGradientBoosting, depth 3), scored LEAVE ONE MOUSE OUT (trained on three
mice, tested on the fourth) - the honest estimate for a new field of view / a new day. Compared with my fixed rules
(ca_rule_vs_user_day1.py). Writes analysis\\ca_selfcheck\\classifier_day1_lomo.csv, classifier_day1_roc.png and the
model trained on all four mice (cell_classifier_day1.joblib) for ca_day2_decisions.py ranking.

  python ca_cell_classifier_day1.py
"""
import os
import sys

import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ca_day2_decisions import shape_numbers  # noqa: E402

BASE = r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553"
MICE = ["5941", "5939", "5937", "1191"]
REV = ["area_ratio", "axis_ratio", "n_pieces", "edge_px", "snr", "skew", "neg_ratio", "n_events", "rise_s", "decay_half_s",
       "ca_like", "max_S_corr", "event_image_corr"]
FEATS = REV + ["main_frac", "solidity", "contrast_max", "contrast_cn", "core_px"]


def contrast(img, f, r=10):
    """footprint core (>= 50 % of its peak) vs a ring 1.5-2.5 radii from the centroid, in robust SD of the ring"""
    from scipy.ndimage import binary_dilation
    core = f >= 0.5 * f.max()
    if not core.any():
        return np.nan
    ys, xs = np.nonzero(core)
    cy, cx = ys.mean(), xs.mean()
    y0, y1 = int(max(cy - 3 * r, 0)), int(min(cy + 3 * r + 1, img.shape[0]))
    x0, x1 = int(max(cx - 3 * r, 0)), int(min(cx + 3 * r + 1, img.shape[1]))
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.hypot(yy - cy, xx - cx)
    ring = (d >= 1.5 * r) & (d <= 2.5 * r) & ~binary_dilation(f[y0:y1, x0:x1] > 0.1 * f.max(), iterations=2)
    w = img[y0:y1, x0:x1]
    if ring.sum() < 10:
        return np.nan
    med = np.median(w[ring]); sd = 1.4826 * np.median(np.abs(w[ring] - med)) + 1e-9
    return float((img[core].mean() - med) / sd)


def features(root, tag):
    R = pd.read_csv(os.path.join(root, tag, "4_cell_review", f"{tag}_cell_review.csv"))
    xd = os.path.join(root, tag, "3_extract")
    S = np.asarray(sio.loadmat(os.path.join(xd, f"{tag}_extract.mat"), variable_names=["spatial_weights"])["spatial_weights"], np.float32)
    P = sio.loadmat(os.path.join(xd, f"{tag}_projections.mat"), variable_names=["max_dff", "Cn"])
    mx, cn = np.asarray(P["max_dff"], float), np.asarray(P["Cn"], float)
    mf, so = shape_numbers(S)
    F = pd.DataFrame({c: pd.to_numeric(R[c], errors="coerce").values if c in R else np.nan for c in REV})
    F["main_frac"], F["solidity"] = mf, so
    F["contrast_max"] = [contrast(mx, S[:, :, j]) if S[:, :, j].max() > 0 else np.nan for j in range(S.shape[2])]
    F["contrast_cn"] = [contrast(cn, S[:, :, j]) if S[:, :, j].max() > 0 else np.nan for j in range(S.shape[2])]
    F["core_px"] = [(S[:, :, j] >= 0.5 * S[:, :, j].max()).sum() if S[:, :, j].max() > 0 else 0 for j in range(S.shape[2])]
    F["id"] = R.id.values
    F["verdict"] = R.verdict.astype(str).values
    return F, R


def model():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.08, max_iter=200, l2_regularization=1.0,
                                          random_state=0)


def main():
    out = os.path.join(BASE, "analysis", "ca_selfcheck"); os.makedirs(out, exist_ok=True)
    root = os.path.join(BASE, "mini1p_out", "20261019")
    D = []
    for m in MICE:
        F, R = features(root, f"{m}_d1")
        F["mouse"] = m
        F["user"] = (R.keep.values == 1).astype(int)
        D.append(F[F.verdict != "rescued"])
    D = pd.concat(D, ignore_index=True)
    D.to_csv(os.path.join(out, "classifier_day1_features.csv"), index=False)
    rows, fig = [], plt.figure(figsize=(6, 5))
    pred_all = np.full(len(D), np.nan)
    for m in MICE:
        tr, te = D.mouse != m, D.mouse == m
        clf = model().fit(D.loc[tr, FEATS], D.loc[tr, "user"])
        p = clf.predict_proba(D.loc[te, FEATS])[:, 1]
        pred_all[te.values] = p
        y = D.loc[te, "user"].values
        auc = roc_auc_score(y, p)
        fpr, tpr, _ = roc_curve(y, p)
        plt.plot(fpr, tpr, label=f"{m} held out: AUC {auc:.2f}")
        for thr in (0.3, 0.5, 0.7):
            k = p >= thr
            tp = int((k & (y == 1)).sum())
            rows.append(dict(mouse=m, threshold=thr, auc=round(auc, 3), you_kept=int(y.sum()), predicted_keep=int(k.sum()),
                             agree=tp, precision=round(tp / max(k.sum(), 1), 3), recall=round(tp / max(y.sum(), 1), 3)))
    D["p_keep_lomo"] = pred_all
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(out, "classifier_day1_lomo.csv"), index=False)
    D[["mouse", "id", "user", "p_keep_lomo"] + FEATS].to_csv(os.path.join(out, "classifier_day1_predictions.csv"), index=False)
    plt.plot([0, 1], [0, 1], "k:", lw=0.8)
    plt.xlabel("false keeps (fraction of your deletions)"); plt.ylabel("your keeps found (fraction)")
    plt.title("Learning your Day-1 decisions, one mouse held out", fontsize=10); plt.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, "classifier_day1_roc.png"), dpi=130); plt.close(fig)
    import joblib
    full = model().fit(D[FEATS], D.user)
    joblib.dump(dict(model=full, features=FEATS), os.path.join(out, "cell_classifier_day1.joblib"))
    pd.set_option("display.width", 200)
    print(T.to_string(index=False))
    for thr in (0.3, 0.5, 0.7):
        g = T[T.threshold == thr]
        print(f"pooled thr {thr}: precision {g.agree.sum() / g.predicted_keep.sum():.3f}, recall {g.agree.sum() / g.you_kept.sum():.3f}")


if __name__ == "__main__":
    main()
