"""A second opinion on every candidate: a classifier trained on YOUR 91 decisions of the 5941
5-min test, applied to the candidates of another run, next to the rules' verdict.

The rules (extract_cell_review) reproduce your 5941 decisions 91/91, but they were written on that
one field of view. This model learns the same decisions from the shape / trace features alone
(logistic regression, standardised; leave-one-out accuracy on 5941 is printed), so where it and
the rules disagree on another mouse, that candidate is worth looking at first in the GUI.
Features (length-independent): area_ratio, axis_ratio, n_pieces, snr, skew, neg_ratio, ca_like,
event_image_corr, max_S_corr, rise_s, decay_half_s.

Outputs (--out): <tag>_second_opinion.csv (rule keep, model probability, disagree flag) and
<tag>_second_opinion.png (disagreements on the mean image).
  python ca_candidate_second_opinion.py --train <5941 5c_cell_review_loose csv> --review <day review csv>
      --extract <day extract mat> --out <dir> --tag <tag>
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import scipy.io as sio

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FEAT = ["area_ratio", "axis_ratio", "n_pieces", "snr", "skew", "neg_ratio", "ca_like", "event_image_corr",
        "max_S_corr", "rise_s", "decay_half_s"]


def prep(D, med=None):
    X = D.reindex(columns=FEAT).apply(pd.to_numeric, errors="coerce").astype(float)
    X["snr"] = np.log(np.clip(X["snr"], 0.1, None))
    X["axis_ratio"] = np.log(np.clip(X["axis_ratio"], 1, None))
    X["area_ratio"] = np.log(np.clip(X["area_ratio"], 0.01, None))
    if med is None:
        med = X.median()
    return X.fillna(med), med


def main(argv=None):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneOut, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--review", required=True)
    ap.add_argument("--extract", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    T = pd.read_csv(a.train)
    T = T[T.get("reviewed", 1) == 1] if "reviewed" in T else T
    Xt, med = prep(T)
    yt = T.keep.astype(int).values
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=5000))
    pcv = cross_val_predict(clf, Xt, yt, cv=LeaveOneOut(), method="predict_proba")[:, 1]
    acc = np.mean((pcv >= 0.5) == yt)
    clf.fit(Xt, yt)
    coef = dict(zip(FEAT, np.round(clf[-1].coef_[0], 2)))
    print(f"second opinion: trained on {len(T)} of your decisions ({yt.sum()} kept); leave-one-out accuracy {acc:.3f}")
    R = pd.read_csv(a.review)
    Xr, _ = prep(R, med)
    p = clf.predict_proba(Xr)[:, 1]
    rule = (R.verdict.astype(str) != "not_cell").astype(int).values
    keep = R.keep.astype(int).values if "keep" in R else rule
    out = pd.DataFrame(dict(id=R.id, verdict=R.verdict, rule_keep=rule, current_keep=keep, model_p_keep=np.round(p, 3)))
    out["disagree"] = ((p >= 0.5).astype(int) != out.current_keep) & ((p >= 0.75) | (p <= 0.25))
    out.to_csv(os.path.join(a.out, f"{a.tag}_second_opinion.csv"), index=False)
    D = out[out.disagree]
    print(f"{a.tag}: {len(R)} candidates; the model disagrees confidently with the current keep on {len(D)}: "
          + ", ".join(f"#{r.id} ({'keep' if r.current_keep else 'delete'} now, model p {r.model_p_keep:.2f})"
                      for r in D.itertuples()))
    if a.extract and os.path.isfile(a.extract):
        from skimage import measure
        M = sio.loadmat(a.extract, squeeze_me=True, variable_names=["spatial_weights", "mean_image"])
        Sp = np.asarray(M["spatial_weights"], np.float32)
        mi = np.asarray(M["mean_image"], float)
        fig, ax = plt.subplots(figsize=(10, 10 * mi.shape[0] / mi.shape[1] + 0.6))
        ax.imshow(mi, cmap="gray", vmin=np.percentile(mi, 1), vmax=np.percentile(mi, 99.7))
        for r in out.itertuples():
            if int(r.id) > Sp.shape[2]:
                continue
            fp = Sp[:, :, int(r.id) - 1]
            cs = measure.find_contours(fp / max(fp.max(), 1e-9), 0.5)
            if not cs:
                continue
            xy = max(cs, key=len)
            if r.disagree:
                col = "#00e5ff" if r.current_keep else "#ff00ff"
                ax.plot(xy[:, 1], xy[:, 0], color=col, lw=1.6)
                ax.text(xy[:, 1].mean() + 8, xy[:, 0].mean(), f"{r.id} ({r.model_p_keep:.2f})", color=col, fontsize=7)
            else:
                ax.plot(xy[:, 1], xy[:, 0], color="#2ecc71" if r.current_keep else "#e74c3c", lw=0.5, alpha=0.6)
        ax.plot([], [], color="#00e5ff", label="kept now, model says delete")
        ax.plot([], [], color="#ff00ff", label="deleted now, model says keep")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_axis_off()
        ax.set_title(f"{a.tag}: second opinion from a model of your 5941 decisions (LOO accuracy {acc:.2f}); "
                     f"{len(D)} confident disagreements to look at first", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, f"{a.tag}_second_opinion.png"), dpi=110); plt.close(fig)
    with open(os.path.join(a.out, f"{a.tag}_second_opinion_model.txt"), "w") as fh:
        fh.write(f"trained on {len(T)} decisions, LOO accuracy {acc:.3f}\ncoefficients (standardised): {coef}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
