"""Learn the user's manual behaviour annotation from the camera features and check how well an
automatic detector mimics it (5941 SBI session, 2026-10-20); define post-stimulus movement
(turning / stepping) as an extra behaviour; compare sessions.

Inputs
  --features  <tag>_bfeatures.csv of behavior_features.py (camera frame n = row n)
  --scoring   ScoringAB_*.mat of the dual-view scorer (the user's annotation = gold standard)
  --all-features-dir  folder with every session's <tag>_bfeatures.csv (session comparison)
Outputs (--out-dir)
  annotation_pattern.png / .csv    per stimulus: deliveries, % with withdrawal / flinch within
                                   2 s, latency; motion aligned to the annotated taps (the
                                   key-press lag); motion aligned to deliveries
  reflex_model.png / .json         cross-validated (leave one stimulus block out) detection of
                                   "withdrawal within 2 s" and "flinch within 2 s" from the
                                   post-delivery motion; per-fold and pooled AUC / accuracy
  delivery_detect.png / .json      can the deliveries themselves be found automatically?
  movement_after_stim.csv / .png   per delivery: centroid displacement and body rotation in
                                   0-3 s vs the 1 s before; "moved" = displacement > 25 px or
                                   rotation > 30 deg; % per stimulus
  sessions_movement.csv / .png     all sessions: per schedule block (baseline, 4 stimulus
                                   blocks): % time moving, turns per min, mean speed
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import scipy.io as sio

FPS = 30.0
# 16.5-min schedule: baseline 0-180 s, stimulus blocks of 180 s with 30-s rests
BLOCKS = [("baseline", 0, 180), ("stim1", 180, 360), ("stim2", 390, 570), ("stim3", 600, 780),
          ("stim4", 810, 990)]


def load_scoring(path):
    m = sio.loadmat(path, squeeze_me=True)
    names = [str(x) for x in np.atleast_1d(m["stimNames"])]
    dF = np.atleast_1d(m["dFrames"]).astype(int)
    dT = np.atleast_1d(m["dTypes"]).astype(int)
    re = np.atleast_2d(m["reflexEvents"]).astype(int)
    score = np.atleast_1d(m["score"]).astype(int)
    lag = float(m.get("lagMs", 250)) / 1000
    return dict(names=names, dF=dF, dT=dT, re=re, score=score, nUsed=int(m["nUsed"]), lag=lag)


def zs(x):
    x = np.asarray(x, float)
    mu, sd = np.nanmedian(x), 1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x))) + 1e-9
    return (x - mu) / sd


def window_feats(F, f0, specs):
    # windows in frames relative to the key press; baseline = 1.0-0.5 s before it (the
    # motion that the press marks peaks ~0.23 s BEFORE the press: the key-press lag)
    out = {}
    n = len(F)
    for col in ("b_edge_me", "b_body_me", "b_shape_chg", "b_speed", "b_turn", "s_me", "b_out_me"):
        x = F[col].values
        pre = x[max(0, f0 - 30):max(1, f0 - 15)]
        base = np.nanmean(pre) if len(pre) else np.nan
        for nm, a, b in specs:
            seg = x[min(n - 1, max(0, f0 + a)):min(n, max(1, f0 + b))]
            out[f"{col}_{nm}_max"] = (np.nanmax(seg) if len(seg) else np.nan) - base
            out[f"{col}_{nm}_mean"] = (np.nanmean(seg) if len(seg) else np.nan) - base
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--scoring", required=True)
    ap.add_argument("--all-features-dir", default="")
    ap.add_argument("--arms", default="", help="tag=arm,... e.g. 5941=sbi553,5937=vehicle")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    F = pd.read_csv(a.features)
    S = load_scoring(a.scoring)
    n = min(len(F), S["nUsed"])
    F = F.iloc[:n].reset_index(drop=True)
    names = S["names"]
    dF, dT, re = S["dF"], S["dT"], S["re"]
    wd = re[re[:, 1] == 1, 0] if re.size else np.array([], int)
    fl = re[re[:, 1] == 2, 0] if re.size else np.array([], int)
    t_del = dF / FPS
    block = np.array([next((b for b, s0, s1 in BLOCKS if s0 <= t < s1 + 15), "other") for t in t_del])

    # ---- 1. annotation pattern ---------------------------------------------------------------
    rows = []
    for i, (f, k) in enumerate(zip(dF, dT)):
        nxt = dF[i + 1] if i + 1 < len(dF) else n
        win = min(f + int(2 * FPS), nxt)
        w_hit = wd[(wd >= f) & (wd < win)]
        f_hit = fl[(fl >= f) & (fl < win)]
        rows.append(dict(delivery=i + 1, frame=int(f), t_s=f / FPS, stim=names[k - 1] if 1 <= k <= 4 else "unknown",
                         block=block[i], withdrawal=int(len(w_hit) > 0), flinch=int(len(f_hit) > 0),
                         wd_latency_s=(w_hit[0] - f) / FPS if len(w_hit) else np.nan,
                         fl_latency_s=(f_hit[0] - f) / FPS if len(f_hit) else np.nan,
                         gap_to_next_s=(nxt - f) / FPS))
    D = pd.DataFrame(rows)
    pat = D.groupby("stim").agg(n=("delivery", "size"), withdrawal_pct=("withdrawal", lambda x: 100 * x.mean()),
                                flinch_pct=("flinch", lambda x: 100 * x.mean()),
                                wd_latency_median_s=("wd_latency_s", "median"),
                                fl_latency_median_s=("fl_latency_s", "median"),
                                gap_median_s=("gap_to_next_s", "median")).round(2)
    pat.to_csv(os.path.join(a.out_dir, "annotation_pattern.csv"))
    D.to_csv(os.path.join(a.out_dir, "annotation_per_delivery.csv"), index=False)

    # motion aligned to the annotated taps and to the deliveries
    def aligned(col, ev, pre=2.0, post=3.0):
        x = zs(F[col].values)
        L = np.arange(-int(pre * FPS), int(post * FPS) + 1)
        A = [x[e + L] for e in ev if e + L[0] >= 0 and e + L[-1] < n]
        return L / FPS, (np.nanmean(A, 0) if A else np.full(len(L), np.nan)), len(A)

    fig, ax = plt.subplots(2, 3, figsize=(15, 7.5))
    lag_est = {}
    for j, (ev, lab) in enumerate(((wd, "withdrawal taps"), (fl, "flinch taps"), (dF, "deliveries (contact taps)"))):
        for col, c in (("b_edge_me", "tab:red"), ("b_body_me", "tab:orange"), ("s_me", "tab:blue"), ("b_out_me", "tab:green")):
            tl, m, k = aligned(col, ev)
            ax[0, j].plot(tl, m, color=c, label=col)
            if col == "b_edge_me" and np.isfinite(m).any():
                win = (tl > -1.5) & (tl < 0.5)
                lag_est[lab] = float(-tl[win][np.nanargmax(m[win])])
        ax[0, j].axvline(0, color="k", ls=":")
        ax[0, j].set_title(f"motion (robust z) aligned to {lab}, n={len(ev)}", fontsize=9)
        ax[0, j].set_xlabel("s from the key press")
    ax[0, 0].legend(fontsize=7, frameon=False)
    x = np.arange(len(pat))
    ax[1, 0].bar(x - 0.2, pat.withdrawal_pct, 0.4, label="withdrawal")
    ax[1, 0].bar(x + 0.2, pat.flinch_pct, 0.4, label="flinch")
    ax[1, 0].set_xticks(x, [f"{s}\n(n={k})" for s, k in zip(pat.index, pat.n)], fontsize=8)
    ax[1, 0].set_ylabel("% deliveries with the response within 2 s"); ax[1, 0].legend(frameon=False)
    for s_, g in D.groupby("stim"):
        ax[1, 1].hist(g.wd_latency_s.dropna(), bins=np.arange(0, 2.01, 0.1), alpha=0.5, label=s_)
    ax[1, 1].set_xlabel("withdrawal latency after delivery tap (s)"); ax[1, 1].legend(fontsize=7, frameon=False)
    ax[1, 2].scatter(D.t_s, D.stim.map({s: i for i, s in enumerate(names)}), c=D.withdrawal, cmap="coolwarm", s=14)
    ax[1, 2].set_yticks(range(4), names); ax[1, 2].set_xlabel("time in session (s)")
    ax[1, 2].set_title("deliveries (red = withdrawal annotated)", fontsize=9)
    fig.suptitle("Your annotation (5941 SBI session): what it contains and how the camera motion lines up with it")
    fig.tight_layout(); fig.savefig(os.path.join(a.out_dir, "annotation_pattern.png"), dpi=110); plt.close(fig)

    # ---- 2. reflex model: is there a withdrawal / flinch within 2 s of this delivery? -----------
    specs = [("resp", -14, 2), ("after", 2, 18), ("late", 18, 60)]   # -0.45..+0.05, ..0.6, ..2 s
    X = pd.DataFrame([window_feats(F, int(f), specs) for f in dF]).fillna(0)
    res = {}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for j, target in enumerate(("withdrawal", "flinch")):
        y = D[target].values
        prob = np.full(len(y), np.nan)
        folds = sorted(set(block))
        for b in folds:
            te = block == b
            tr = ~te
            if len(set(y[tr])) < 2:
                continue
            mdl = RandomForestClassifier(300, min_samples_leaf=3, class_weight="balanced", random_state=0)
            mdl.fit(X[tr], y[tr])
            prob[te] = mdl.predict_proba(X[te])[:, 1]
        ok = np.isfinite(prob)
        auc = roc_auc_score(y[ok], prob[ok]) if len(set(y[ok])) > 1 else np.nan
        pred = (prob >= 0.5).astype(int)
        bacc = balanced_accuracy_score(y[ok], pred[ok]) if len(set(y[ok])) > 1 else np.nan
        agree = float((pred[ok] == y[ok]).mean())
        mdl = RandomForestClassifier(300, min_samples_leaf=3, class_weight="balanced", random_state=0).fit(X, y)
        imp = pd.Series(mdl.feature_importances_, X.columns).sort_values(ascending=False).head(6)
        res[target] = dict(n=int(len(y)), positives=int(y.sum()), cv="leave one stimulus block out",
                           auc=round(float(auc), 3), balanced_accuracy=round(float(bacc), 3),
                           agreement=round(agree, 3), top_features=imp.round(3).to_dict())
        D[f"auto_{target}_prob"] = prob
        D[f"auto_{target}"] = pred
        from sklearn.metrics import roc_curve
        if np.isfinite(auc):
            fpr, tpr, _ = roc_curve(y[ok], prob[ok])
            ax[j].plot(fpr, tpr, "k-"); ax[j].plot([0, 1], [0, 1], "k:")
        ax[j].set_title(f"{target} within 2 s: AUC {auc:.2f}, agreement {100 * agree:.0f}% (n={len(y)})", fontsize=9)
        ax[j].set_xlabel("false positive rate"); ax[j].set_ylabel("true positive rate")
    fig.suptitle("Automatic vs your annotation, cross-validated (trained on 3 stimulus blocks, tested on the 4th)")
    fig.tight_layout(); fig.savefig(os.path.join(a.out_dir, "reflex_model.png"), dpi=110); plt.close(fig)
    D.to_csv(os.path.join(a.out_dir, "annotation_per_delivery.csv"), index=False)

    # ---- 3. can the deliveries be found automatically? -------------------------------------------
    from sklearn.metrics import roc_auc_score as AUC
    rng = np.random.default_rng(0)
    stim_frames = np.concatenate([np.arange(int(s0 * FPS), int(s1 * FPS)) for _, s0, s1 in BLOCKS[1:]])
    stim_frames = stim_frames[stim_frames < n - 60]
    far = np.array([f for f in stim_frames if np.min(np.abs(dF - f)) > 2 * FPS])
    neg = rng.choice(far, min(len(far), 4 * len(dF)), replace=False)
    dres = {}
    for col in ("b_out_me", "b_edge_me", "b_body_me", "s_me", "b_shape_chg"):
        x = zs(F[col].values)
        sc = lambda f: np.nanmax(x[max(0, f - 6):f + 7])  # +-0.2 s around the tap
        p_ = np.array([sc(f) for f in dF]); n_ = np.array([sc(f) for f in neg])
        dres[col] = round(float(AUC(np.r_[np.ones(len(p_)), np.zeros(len(n_))], np.r_[p_, n_])), 3)
    with open(os.path.join(a.out_dir, "delivery_detect.json"), "w") as f:
        json.dump(dict(question="does the camera motion at a delivery tap differ from other moments of "
                                "the stimulus blocks? (AUC 0.5 = no, 1 = perfectly)", auc=dres,
                       n_deliveries=int(len(dF)), n_other=int(len(neg))), f, indent=1)

    # ---- 4. post-stimulus movement (turning / stepping) ------------------------------------------
    mv = []
    for f in dF:
        f = int(f)
        pre = F.iloc[max(0, f - 30):f]
        post = F.iloc[f:min(n, f + 90)]
        if len(pre) < 5 or len(post) < 10:
            mv.append((np.nan, np.nan, np.nan)); continue
        c0 = np.nanmedian(pre[["b_cx", "b_cy"]].values, 0)
        disp = np.nanmax(np.hypot(post.b_cx - c0[0], post.b_cy - c0[1]))
        th0 = np.nanmedian(pre.b_theta)
        d = np.abs(post.b_theta - th0) % 180
        rot = np.nanmax(np.minimum(d, 180 - d))
        mv.append((disp, rot, float(np.nanmean(post.b_speed))))
    mv = np.array(mv)
    D["move_disp_px"], D["move_rot_deg"], D["move_speed"] = mv[:, 0], mv[:, 1], mv[:, 2]
    D["moved_after"] = ((D.move_disp_px > 25) | (D.move_rot_deg > 30)).astype(int)
    D.to_csv(os.path.join(a.out_dir, "movement_after_stim.csv"), index=False)
    mpat = D.groupby("stim").agg(n=("delivery", "size"), moved_pct=("moved_after", lambda x: 100 * x.mean()),
                                 moved_and_withdrawal_pct=("moved_after", lambda x: 100 * (x & D.loc[x.index, "withdrawal"]).mean())).round(1)
    mpat.to_csv(os.path.join(a.out_dir, "movement_after_stim_summary.csv"))

    # ---- 5. all sessions: movement per schedule block ---------------------------------------------
    srows = []
    if a.all_features_dir:
        arms = dict(kv.split("=") for kv in a.arms.split(",") if "=" in kv)
        for fn in sorted(os.listdir(a.all_features_dir)):
            if not fn.endswith("_bfeatures.csv"):
                continue
            tag = fn.replace("_bfeatures.csv", "")
            G = pd.read_csv(os.path.join(a.all_features_dir, fn))
            sp = G.b_speed.values
            moving = sp > 2.0                      # px/frame = 60 px/s
            for b, s0, s1 in BLOCKS:
                sel = (G.t_s >= s0) & (G.t_s < s1)
                if sel.sum() < 100:
                    continue
                turn = G.b_turn[sel].values
                srows.append(dict(session=tag, mouse=tag.split("_")[0], phase=tag.split("_")[-1],
                                  arm=arms.get(tag.split("_")[0], "") if tag.endswith("post") else "none",
                                  block=b, pct_time_moving=100 * np.nanmean(moving[sel.values]),
                                  turn_deg_per_s=np.nansum(turn) / (sel.sum() / FPS),
                                  mean_speed_px_s=np.nanmean(sp[sel.values]) * FPS,
                                  side_motion=np.nanmean(G.s_me[sel])))
        SM = pd.DataFrame(srows).round(3)
        SM.to_csv(os.path.join(a.out_dir, "sessions_movement.csv"), index=False)
        if len(SM):
            fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
            for j, (col, lab) in enumerate((("pct_time_moving", "% time moving (> 60 px/s)"),
                                            ("turn_deg_per_s", "body rotation (deg / s)"),
                                            ("side_motion", "side-camera motion energy"))):
                for mouse, g in SM.groupby("mouse"):
                    for ph, mk in (("pre", "o-"), ("post", "s--")):
                        h = g[g.phase == ph]
                        if len(h):
                            arm = h.arm.iloc[0] if ph == "post" else "baseline"
                            ax[j].plot(range(len(h)), h[col], mk, label=f"{mouse} {ph} ({arm})")
                ax[j].set_xticks(range(5), [b for b, _, _ in BLOCKS], fontsize=8)
                ax[j].set_title(lab, fontsize=10)
            ax[0].legend(fontsize=6, frameon=False, ncol=2)
            fig.suptitle("Movement per schedule block, every session (automatic; deliveries not needed)")
            fig.tight_layout(); fig.savefig(os.path.join(a.out_dir, "sessions_movement.png"), dpi=110); plt.close(fig)

    summ = dict(annotation=pat.reset_index().to_dict("records"), keypress_lag_s=lag_est, reflex_model=res,
                delivery_detect_auc=dres, movement_after_stim=mpat.reset_index().to_dict("records"))
    with open(os.path.join(a.out_dir, "behavior_supervised.json"), "w") as f:
        json.dump(summ, f, indent=1, default=float)
    print(json.dumps(summ, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
