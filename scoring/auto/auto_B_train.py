"""auto_B_train.py  -  train and apply an automatic behaviour classifier from
your own manual scoring.

INPUTS
    Features_<vid>.csv       from auto_B_features.py   (side view, per frame)
    TrainingLabels_<vid>.csv from score_AB_dual_view.m (your manual scoring)

    The two are joined on the `frame` column, so the SIDE video passed to
    auto_B_features.py must be the same session you scored.

WHICH BEHAVIOURS
    All four affective codes written by score_AB_dual_view.m are modelled:

        code 1  Paw attending                  hold a
        code 2  Licking or biting              hold s
        code 3  Sustained lifting / guarding   hold d   (> 2 s rule applied)
        code 4  Escape / rearing               hold f

    Licking-or-biting is included now that the two are pooled. What the video
    could not support was the licking-vs-biting distinction - the tongue is
    ~5 px. "Is the mouth on the paw or not" is a head-to-paw posture question,
    which the side-view silhouette can carry.

    The two REFLEXIVE measures are still not modelled: at 30 fps a paw
    withdrawal is 1.5-4.5 frames. That is a rig limit, not a model limit, and
    they stay manual until the frame rate goes up. Pass --behaviours to
    override any of this.

WHY affective_code_ml
    Training uses `affective_code_ml`, the lag-corrected label column, and drops
    every frame with exclude_from_training = 1. Using the raw `affective_code`
    would teach the model that behaviours start ~250 ms later than they do.

HONEST EVALUATION
    Frame-level F1 is optimistic: neighbouring frames are highly correlated, so
    a random split leaks. This script therefore
      * groups cross-validation folds BY SESSION (never splits a session), and
      * reports BOUT-level recall and false-positive rate as well as frame F1,
        because a bout you miss entirely matters more than a few ragged edges.
    With only one session it falls back to a chronological split and says so.

USAGE
    train:
      python auto_B_train.py train --features F1.csv F2.csv --labels L1.csv L2.csv
                                   [--out model.joblib]
    apply:
      python auto_B_train.py apply --model model.joblib --features Fnew.csv
                                   [--out-csv Predicted.csv]

Hansol Lim - HEAL mini1p / SBI-553
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

BEHAV = {1: "Paw attending",
         2: "Licking or biting",
         3: "Sustained lifting / guarding",
         4: "Escape / rearing"}
MIN_DUR_S = {3: 2.0}          # guarding must last > 2 s, as in the manual rules
FPS_DEFAULT = 30.0
DROP = {"frame", "time_s", "found"}


def row_rate(df, fallback=FPS_DEFAULT, label=""):
    """Rows per second in a features file.

    This is NOT always the video frame rate: auto_B_features.py --step N
    analyses every Nth frame, so a 30 fps video with --step 3 gives 10 rows/s.
    Every duration rule here is in seconds, so it has to be converted with the
    ROW rate. Getting this wrong silently scales the > 2 s guarding threshold
    by N and deletes every bout.
    """
    if "time_s" not in df.columns or len(df) < 3:
        return fallback
    dt = np.median(np.diff(df["time_s"].to_numpy()))
    if not np.isfinite(dt) or dt <= 0:
        return fallback
    hz = 1.0 / dt
    if abs(hz - fallback) > 0.5:
        print(f"  NOTE{label}: features are {hz:.2f} rows/s, not {fallback:.2f}. "
              f"Using {hz:.2f} for all duration rules.")
    return hz


def load_pair(fpath, lpath):
    F = pd.read_csv(fpath)
    L = pd.read_csv(lpath)
    need = {"frame", "affective_code_ml", "exclude_from_training"}
    missing = need - set(L.columns)
    if missing:
        raise SystemExit(
            f"{os.path.basename(lpath)} lacks {sorted(missing)}.\n"
            "It was written by an older score_B. Re-score with "
            "score_AB_dual_view.m, which writes the lag-corrected columns.")
    D = F.merge(L[["frame", "affective_code_ml", "exclude_from_training"]],
                on="frame", how="inner")
    if D.empty:
        raise SystemExit(f"no overlapping frames between\n  {fpath}\n  {lpath}")
    D["session"] = os.path.basename(lpath)
    print(f"  {os.path.basename(fpath)[:44]:46s} {len(D):6d} frames "
          f"({len(F)} feat / {len(L)} label rows)")
    return D


def bout_metrics(y_true, y_pred, fps, min_gap_s=0.2):
    """Bout-level recall and false-positive count. A true bout counts as
    detected if the prediction overlaps it at all."""
    def bouts(v):
        d = np.diff(np.concatenate(([0], v.astype(np.int8), [0])))
        s = np.flatnonzero(d == 1)
        e = np.flatnonzero(d == -1)
        # merge bouts separated by a very short gap
        out = []
        for a, b in zip(s, e):
            if out and a - out[-1][1] < min_gap_s * fps:
                out[-1] = (out[-1][0], b)
            else:
                out.append((a, b))
        return out

    T, P = bouts(y_true), bouts(y_pred)
    hit = sum(any(p0 < b1 and b0 < p1 for p0, p1 in P) for b0, b1 in T)
    fp = sum(not any(b0 < p1 and p0 < b1 for b0, b1 in T) for p0, p1 in P)
    return len(T), hit, len(P), fp


def smooth_pred(pred, fps, med_s=0.30, close_s=0.40):
    """Clean up frame-wise predictions before any duration rule is applied.

    A classifier's raw output is fragmented: a single real bout comes back as a
    run of ONs broken by a few OFF frames. Applying a minimum-duration rule to
    that directly deletes the whole bout, because no individual fragment is
    long enough. So: median-filter, then bridge short gaps, THEN enforce the
    duration. Without this step the > 2 s guarding rule removes everything.
    """
    p = pred.astype(np.int8).copy()
    k = max(3, int(round(med_s * fps)) | 1)
    if k > 1 and len(p) >= k:                       # median filter
        pad = k // 2
        q = np.pad(p, pad, mode="edge")
        p = (np.lib.stride_tricks.sliding_window_view(q, k).mean(axis=1)
             >= 0.5).astype(np.int8)
    gap = int(round(close_s * fps))
    if gap > 0:                                     # bridge short OFF gaps
        d = np.diff(np.concatenate(([0], p, [0])))
        starts = np.flatnonzero(d == 1)
        ends = np.flatnonzero(d == -1)
        for i in range(len(starts) - 1):
            if starts[i + 1] - ends[i] <= gap:
                p[ends[i]:starts[i + 1]] = 1
    return p.astype(int)


def enforce_min_duration(pred, min_frames):
    """Drop predicted bouts shorter than min_frames (the > 2 s guarding rule).
    Run smooth_pred() first, or this deletes fragmented real bouts."""
    if min_frames <= 1:
        return pred
    out = pred.copy()
    d = np.diff(np.concatenate(([0], pred.astype(np.int8), [0])))
    for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if b - a < min_frames:
            out[a:b] = 0
    return out


def postprocess(pred, code, fps, min_dur_s):
    """Smooth, then apply the behaviour's minimum-duration rule."""
    p = smooth_pred(pred, fps)
    mf = int(round(min_dur_s.get(code, 0) * fps))
    if mf > 1:
        p = enforce_min_duration(p, mf)
    return p, mf


def do_train(args):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import precision_recall_fscore_support
    import joblib

    if len(args.features) != len(args.labels):
        raise SystemExit("--features and --labels must have the same length")

    print("=== loading ===")
    D = pd.concat([load_pair(f, l) for f, l in zip(args.features, args.labels)],
                  ignore_index=True)
    fps = row_rate(D, args.fps)
    n_sess = D["session"].nunique()
    print(f"  {len(D)} frames from {n_sess} session(s)")

    keep = D["exclude_from_training"] == 0
    print(f"  dropping {int((~keep).sum())} excluded frame(s) "
          f"({100 * (~keep).mean():.1f}%)")
    D = D[keep].reset_index(drop=True)

    feat_cols = [c for c in D.columns
                 if c not in DROP | {"affective_code_ml", "exclude_from_training",
                                     "session"}
                 and pd.api.types.is_numeric_dtype(D[c])]
    X = D[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    groups = D["session"].to_numpy()
    print(f"  {len(feat_cols)} features")

    codes = args.behaviours or sorted(BEHAV)
    models, report = {}, []

    for code in codes:
        name = BEHAV.get(code, f"code {code}")
        y = (D["affective_code_ml"] == code).to_numpy().astype(int)
        pos = int(y.sum())
        print(f"\n=== {name}  (code {code}) ===")
        print(f"  positive frames {pos}/{len(y)}  ({100 * y.mean():.2f}%)")
        if pos < 60:
            print("  SKIPPED: fewer than 60 positive frames. Score more "
                  "sessions before modelling this behaviour.")
            report.append((name, pos, None, None, None, None, None, "too few labels"))
            continue

        # ---- cross-validated, grouped by session ----
        if n_sess >= 2:
            n_splits = min(n_sess, 5)
            splitter = GroupKFold(n_splits=n_splits).split(X, y, groups)
            scheme = f"GroupKFold by session, {n_splits} folds"
        else:
            cut = int(0.7 * len(X))
            splitter = [(np.arange(cut), np.arange(cut, len(X)))]
            scheme = "chronological 70/30 split (ONLY 1 SESSION - optimistic)"
        print(f"  validation: {scheme}")

        oof = np.full(len(y), -1)
        for tr, te in splitter:
            if y[tr].sum() < 20 or y[te].sum() < 5:
                continue
            m = HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.08, max_depth=6,
                l2_regularization=1.0, early_stopping=True,
                validation_fraction=0.15, random_state=0,
                class_weight="balanced")
            m.fit(X[tr], y[tr])
            oof[te] = m.predict(X[te])

        ev = oof >= 0
        if ev.sum() == 0 or y[ev].sum() == 0:
            print("  could not evaluate: no fold had enough positives.")
            report.append((name, pos, None, None, None, None, None,
                           "no evaluable fold"))
            continue

        yp, mf = postprocess(oof[ev].astype(int), code, fps, MIN_DUR_S)
        print("  post-processing: median filter + gap closing" +
              (f", then the > {MIN_DUR_S[code]:g} s minimum-duration rule"
               if mf > 1 else ""))

        p, r, f1, _ = precision_recall_fscore_support(
            y[ev], yp, average="binary", zero_division=0)
        nb, hit, npd, fp = bout_metrics(y[ev], yp, fps)
        print(f"  FRAME  precision {p:.2f}  recall {r:.2f}  F1 {f1:.2f}")
        print(f"  BOUT   {hit}/{nb} true bouts detected "
              f"({100 * hit / max(nb, 1):.0f}%),  "
              f"{fp}/{npd} predicted bouts are false positives")
        report.append((name, pos, p, r, f1, hit / max(nb, 1),
                       fp / max(npd, 1), scheme))

        # ---- refit on everything for deployment ----
        full = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, max_depth=6,
            l2_regularization=1.0, early_stopping=True,
            validation_fraction=0.15, random_state=0, class_weight="balanced")
        full.fit(X, y)
        models[code] = full

    if not models:
        print("\nNothing trained. Score more sessions and re-run.")
        return 1

    joblib.dump(dict(models=models, feat_cols=feat_cols, fps=fps,
                     min_dur_s=MIN_DUR_S, behav=BEHAV), args.out)
    print(f"  duration rules were applied at {fps:.2f} rows/s")
    print(f"\nmodel saved: {args.out}")

    print("\n=== summary ===")
    print(f"  {'behaviour':32s} {'pos':>7s} {'prec':>6s} {'rec':>6s} "
          f"{'F1':>6s} {'bout rec':>9s} {'bout FP':>8s}")
    for nm, pos, p, r, f1, br, bfp, note in report:
        if p is None:
            print(f"  {nm:32s} {pos:7d}   -- {note}")
        else:
            print(f"  {nm:32s} {pos:7d} {p:6.2f} {r:6.2f} {f1:6.2f} "
                  f"{br:8.0%} {bfp:7.0%}")
    print("\n  Trust the BOUT columns over frame F1. A model is worth using when "
          "bout recall is high\n  and bout FP is low; ragged frame edges matter "
          "much less for the per-stimulus\n  durations you actually report.")
    return 0


def do_apply(args):
    import joblib
    B = joblib.load(args.model)
    models, feat_cols = B["models"], B["feat_cols"]
    F = pd.read_csv(args.features)
    fps = row_rate(F, B["fps"], label=" (new file)")
    if abs(fps - B["fps"]) > 0.5:
        print(f"  WARNING: the model was trained on {B['fps']:.2f} rows/s but "
              f"this file is {fps:.2f} rows/s.")
        print("  Re-extract features with the SAME --step used for training, "
              "or the rolling-window features mean different things.")
    miss = [c for c in feat_cols if c not in F.columns]
    if miss:
        raise SystemExit(f"features file lacks {len(miss)} column(s), e.g. "
                         f"{miss[:5]}. Regenerate with the same "
                         f"auto_B_features.py settings.")
    X = F[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()

    out = F[["frame", "time_s"]].copy()
    pred_code = np.zeros(len(F), int)
    for code, m in models.items():
        p, _ = postprocess(m.predict(X).astype(int), code, fps, B["min_dur_s"])
        out[f"pred_{code}"] = p
        pred_code[(p == 1) & (pred_code == 0)] = code
        print(f"  {B['behav'].get(code, code):32s} {p.sum():7d} frames "
              f"({100 * p.mean():.2f}%)")
    out["pred_affective_code"] = pred_code

    dest = args.out_csv or os.path.join(
        os.path.dirname(args.features) or ".",
        "Predicted_" + os.path.basename(args.features).replace("Features_", ""))
    out.to_csv(dest, index=False)
    print(f"\nwrote {dest}")
    print("  NOTE: these are predictions, not scores. Spot-check them against "
          "the video\n  before using them in an analysis, and never mix "
          "predicted and manual\n  sessions inside one comparison.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train from manual scoring")
    t.add_argument("--features", nargs="+", required=True)
    t.add_argument("--labels", nargs="+", required=True)
    t.add_argument("--out", default="behaviour_model.joblib")
    t.add_argument("--fps", type=float, default=FPS_DEFAULT)
    t.add_argument("--behaviours", nargs="*", type=int, default=None,
                   help="affective codes to model (default 1 4 5)")
    t.set_defaults(func=do_train)

    a = sub.add_parser("apply", help="apply a trained model")
    a.add_argument("--model", required=True)
    a.add_argument("--features", required=True)
    a.add_argument("--out-csv", default=None)
    a.set_defaults(func=do_apply)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
