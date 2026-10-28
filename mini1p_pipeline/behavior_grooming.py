"""Grooming detector trained on the user's confirmed examples (active learning, 2026-10-20).

Round 1: the user judged 29 candidate clips (3 s each) of 6 sessions: grooming = #1, 13, 14,
15, 20, 22, 23 (face washing in front of the camera, and licking the flank with the head turned
into the body); the rest are not (sitting still with the paws together is NOT grooming: the
strokes are what matters).

Features per frame (side camera = behavior_side_grid.py, bottom = behavior_features.py), each
normalised within its session so different lighting does not matter:
  side  motion in the 3 x 3 cells over the mouse: total, share of the most active cell and of
        each row / column, rhythm (power in 1-3, 3-6, 6-10, 10-15 Hz of the total and of the
        most active cell, 2-s window), posture (height / width, area, top), body sway
  bottom  centroid speed (silhouette artefacts bridged), area, body / edge motion
  each in 1-s and 2-s windows centred on the frame
Model: random forest; positives = the frames of the clips judged grooming, negatives = the
other clips + random frames of all 8 sessions (> 5 s from a grooming clip). Checked by leaving
each judged clip out of training in turn and scoring it.

Outputs (--out-dir): grooming_prob_<tag>.npz (per-frame probability, valid mask), grooming_cv.csv,
grooming_model.joblib, grooming_candidates_r<k>.csv (next clips for the user to judge)

  python behavior_grooming.py --labels LABELS.csv --out-dir DIR [--round 2]
  LABELS.csv: session,t_s,grooming (0/1) - every clip the user has judged so far
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import behavior_auto_annotate as ba  # noqa: E402

FPS = 30.0
FD = r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\behavior_auto\20261019"
GD = FD + r"\grooming"
TAGS = ["5941_d1_pre", "5941_d1_post", "5937_d1_pre", "5937_d1_post", "5939_d1_pre", "5939_d1_post",
        "1191_d1_pre", "1191_d1_post"]
HALF = 45                      # a judged clip = its centre +- 1.5 s


def sm(x, w):
    return pd.Series(x).rolling(w, center=True, min_periods=1).mean().values


def band_power(x, step=5, win=60):
    """fraction of power in 1-3, 3-6, 6-10, 10-15 Hz, 2-s window centred on each frame"""
    n = len(x)
    xp = np.r_[np.full(win // 2, x[0]), x, np.full(win // 2, x[-1])]
    centres = np.arange(0, n, step)
    idx = centres[:, None] + np.arange(win)[None, :]
    seg = xp[idx]
    seg = seg - seg.mean(1, keepdims=True)
    P = np.abs(np.fft.rfft(seg * np.hanning(win), axis=1)) ** 2
    f = np.fft.rfftfreq(win, 1 / FPS)
    tot = P[:, f > 0.5].sum(1) + 1e-9
    out = []
    for lo, hi in ((1, 3), (3, 6), (6, 10), (10, 15)):
        out.append(np.interp(np.arange(n), centres, P[:, (f >= lo) & (f < hi)].sum(1) / tot))
    return np.stack(out, 1)


def features(tag):
    Z = np.load(rf"{GD}\{tag}_bsgrid.npz")
    S = pd.DataFrame(Z["X"], columns=[str(c) for c in Z["names"]])
    F = pd.read_csv(rf"{FD}\{tag}_bfeatures.csv")
    n = min(len(S), len(F))
    S, F = S.iloc[:n].reset_index(drop=True), F.iloc[:n].reset_index(drop=True)
    bright = S.sg_bright.values
    found = np.isfinite(S.sg_me0.values)
    valid = found & (bright > 0.6 * np.nanmedian(bright)) & (bright < 1.4 * np.nanmedian(bright))
    S = S.interpolate(limit_direction="both")
    V = S[[f"sg_me{k}" for k in range(9)]].values
    tot = V.mean(1)
    norm = np.median(tot[valid]) + 1e-6
    vs = np.stack([sm(V[:, k], 9) for k in range(9)], 1)
    share = vs / (vs.sum(1, keepdims=True) + 1e-6)
    srt = np.sort(share, 1)
    kmax = vs.argmax(1)
    vmax = V[np.arange(n), kmax]
    rows = np.stack([share[:, 3 * r:3 * r + 3].sum(1) for r in range(3)], 1)
    cols = np.stack([share[:, [c, c + 3, c + 6]].sum(1) for c in range(3)], 1)
    h, w = S.sg_h.values, S.sg_w.values
    hm = np.median(h[valid])
    sspd = np.r_[0, np.hypot(np.diff(sm(S.sg_cx.values, 9)), np.diff(sm(S.sg_cy.values, 9)))] / hm
    bad = ba.silhouette_artefacts(F)
    G = F[["b_cx", "b_cy"]].astype(float).copy()
    G[bad] = np.nan
    bspd = np.r_[0, np.hypot(np.diff(sm(ba.clean(G.b_cx), 9)), np.diff(sm(ba.clean(G.b_cy), 9)))]
    base = {"tot": tot / norm, "vmax": vmax / norm, "share1": srt[:, -1], "share2": srt[:, -2],
            "row0": rows[:, 0], "row1": rows[:, 1], "row2": rows[:, 2], "col0": cols[:, 0], "col1": cols[:, 1],
            "col2": cols[:, 2], "aspect": h / np.maximum(w, 1), "area": S.sg_area.values / np.median(S.sg_area.values[valid]),
            "top": (S.sg_top.values - np.median(S.sg_top.values[valid])) / hm, "sway": sspd, "bspeed": bspd,
            "barea": ba.clean(F.b_area) / np.nanmedian(F.b_area), "bbody": ba.rz(ba.clean(F.b_body_me)),
            "bedge": ba.rz(ba.clean(F.b_edge_me)), "bshape": ba.rz(ba.clean(F.b_shape_chg))}
    cols_ = {}
    for k, x in base.items():
        x = np.nan_to_num(np.asarray(x, float))
        cols_[f"{k}_m1"] = sm(x, 31)
        cols_[f"{k}_m2"] = sm(x, 61)
    cols_["tot_sd2"] = pd.Series(base["tot"]).rolling(61, center=True, min_periods=1).std().values
    for nm, x in (("tot", base["tot"]), ("vmax", base["vmax"])):
        B = band_power(np.nan_to_num(x))
        for j, b in enumerate(("1_3", "3_6", "6_10", "10_15")):
            cols_[f"{nm}_bp{b}"] = sm(B[:, j], 15)
    X = pd.DataFrame(cols_).fillna(0)
    return X, valid


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out-dir", default=GD)
    ap.add_argument("--round", type=int, default=2, help="number of the NEXT judging round")
    ap.add_argument("--per-session", type=int, default=5)
    ap.add_argument("--min-prob", type=float, default=0.0)
    a = ap.parse_args(argv)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    import joblib
    L = pd.read_csv(a.labels)
    rng = np.random.default_rng(0)
    FX, VAL = {}, {}
    for t in TAGS:
        FX[t], VAL[t] = features(t)
        print(f"features {t}: {FX[t].shape}, valid {100 * VAL[t].mean():.1f}%", flush=True)
    names = list(FX[TAGS[0]].columns)
    # labelled clip frames + background frames
    rows, y, grp, sess = [], [], [], []
    for i, r in L.iterrows():
        c = int(round(r.t_s * FPS))
        fr = np.arange(max(0, c - HALF), min(len(FX[r.session]), c + HALF + 1))
        fr = fr[VAL[r.session][fr]]
        rows += [(r.session, f) for f in fr]; y += [int(r.grooming)] * len(fr); grp += [i] * len(fr); sess += [r.session] * len(fr)
    for t in TAGS:
        near = np.zeros(len(FX[t]), bool)
        for _, r in L[(L.session == t) & (L.grooming == 1)].iterrows():
            c = int(round(r.t_s * FPS)); near[max(0, c - 150):c + 150] = True
        pool = np.where(VAL[t] & ~near)[0]
        pool = pool[pool >= 450]
        for f in rng.choice(pool, min(1500, len(pool)), replace=False):
            rows.append((t, int(f))); y.append(0); grp.append(-1); sess.append(t)
    Xall = np.stack([FX[t].values[f] for t, f in rows]).astype(np.float32)
    y, grp = np.array(y), np.array(grp)
    frames = np.array([f for _, f in rows]); sess = np.array(sess)

    def rf():
        return RandomForestClassifier(500, min_samples_leaf=5, max_features="sqrt", class_weight="balanced_subsample",
                                      n_jobs=-1, random_state=0)
    # leave one judged clip out
    cv = []
    for i, r in L.iterrows():
        c = int(round(r.t_s * FPS))
        te = grp == i
        excl = (sess == r.session) & (np.abs(frames - c) < 150)
        tr = ~te & ~excl
        m = rf().fit(Xall[tr], y[tr])
        fr = np.arange(max(0, c - HALF), min(len(FX[r.session]), c + HALF + 1))
        p = m.predict_proba(FX[r.session].values[fr].astype(np.float32))[:, 1]
        cv.append(dict(session=r.session, t_s=r.t_s, grooming=int(r.grooming), cv_score=float(np.mean(p)),
                       cv_max=float(np.max(sm(p, 15)))))
    CV = pd.DataFrame(cv)
    auc = roc_auc_score(CV.grooming, CV.cv_score) if CV.grooming.nunique() > 1 else np.nan
    CV.to_csv(os.path.join(a.out_dir, "grooming_cv.csv"), index=False)
    print(f"leave-one-clip-out: AUC {auc:.3f} over {len(CV)} judged clips ({CV.grooming.sum()} grooming)")
    print(CV.sort_values("cv_score", ascending=False).round(3).to_string(index=False))
    # final model, every frame of every session
    m = rf().fit(Xall, y)
    joblib.dump(dict(model=m, features=names, labels=L), os.path.join(a.out_dir, "grooming_model.joblib"), compress=3)
    cands = []
    for t in TAGS:
        p = m.predict_proba(FX[t].values.astype(np.float32))[:, 1]
        p[~VAL[t]] = 0
        p[:450] = 0
        ps = sm(p, 31)
        np.savez_compressed(os.path.join(a.out_dir, f"grooming_prob_{t}.npz"), p=p.astype(np.float32),
                            ps=ps.astype(np.float32), valid=VAL[t])
        judged = [int(round(x * FPS)) for x in L[L.session == t].t_s]
        picks = []
        for i in np.argsort(-ps):
            if len(picks) >= a.per_session or ps[i] < a.min_prob:
                break
            if all(abs(i - j) > 300 for j in picks + judged):
                picks.append(int(i))
        cands += [dict(session=t, t_s=round(i / FPS, 1), prob=round(float(ps[i]), 3)) for i in picks]
        print(f"{t}: frames with smoothed p > 0.5: {100 * np.mean(ps > 0.5):.1f}%; next candidates "
              f"{[round(i / FPS, 1) for i in picks]}", flush=True)
    C = pd.DataFrame(cands)
    C.insert(0, "cand", np.arange(1, len(C) + 1))
    C.to_csv(os.path.join(a.out_dir, f"grooming_candidates_r{a.round}.csv"), index=False)
    print(f"{len(C)} candidates for round {a.round}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
