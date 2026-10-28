"""Find stimulus deliveries automatically and score every session the same way (validated
against the user's annotation of 5941 SBI, 2026-10-20).

Detector: during the 4 stimulus blocks, the sum of the robust z-scores of the motion outside
the mouse (tool / hand, b_out_me) and of the whole-body motion (b_body_me) is smoothed
(0.1 s); its peaks at least min_gap apart and above a threshold are deliveries. The user's
taps come ~0.23 s after the motion peak (key-press lag), so a detection matches a tap when
tap - peak is within [-0.15, +0.6] s. The threshold is chosen on 3 blocks and tested on the
4th (leave one block out), and the pooled threshold is used for the other sessions.

Per detected delivery (every session): block, stimulus (from the session's order),
"withdrawal" (reflex model probability from behavior_supervised features, trained on the
whole annotated session), and "moved after" (centroid > 25 px or body axis > 30 deg in 0-3 s).

  python behavior_delivery_detect.py --train-features 5941_d1_post_bfeatures.csv \
      --train-scoring ScoringAB_....mat --features-dir DIR --out-dir OUT
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import behavior_supervised as bs  # noqa: E402

FPS = 30.0
BLOCKS = bs.BLOCKS
ORDER = {  # randomisation sheet, 2026-10-19
    "5941_d1_pre": ["Mild touch", "Pin prick", "Light touch", "Heat"],
    "5941_d1_post": ["Pin prick", "Heat", "Mild touch", "Light touch"],
    "5937_d1_pre": ["Heat", "Mild touch", "Pin prick", "Light touch"],
    "5937_d1_post": ["Pin prick", "Heat", "Light touch", "Mild touch"],
    "5939_d1_pre": ["Heat", "Light touch", "Pin prick", "Mild touch"],
    "5939_d1_post": ["Light touch", "Mild touch", "Pin prick", "Heat"],
    "1191_d1_pre": ["Pin prick", "Mild touch", "Heat", "Light touch"],
    "1191_d1_post": ["Light touch", "Pin prick", "Heat", "Mild touch"],
}
ARM = {"5941": "sbi553", "5939": "sbi553", "5937": "vehicle", "1191": "vehicle"}


def signal(F):
    s = bs.zs(F.b_out_me.values) + bs.zs(F.b_body_me.values)
    return np.convolve(np.nan_to_num(s), np.ones(3) / 3, "same")


def peaks(sig, frames_ok, thr, gap):
    from scipy.signal import find_peaks
    p, _ = find_peaks(np.where(frames_ok, sig, -np.inf), height=thr, distance=gap)
    return p


def match(det, taps, lo=-0.15, hi=0.6):
    # greedy one-to-one: each tap to the nearest unused detection with tap - det in [lo, hi] s
    used = set()
    hit = 0
    for t in taps:
        d = (t - det) / FPS
        c = [i for i in np.argsort(np.abs(d - 0.23)) if lo <= d[i] <= hi and i not in used]
        if c:
            used.add(c[0]); hit += 1
    return hit, len(det) - len(used), len(taps) - hit     # TP, FP, FN


def in_blocks(n):
    ok = np.zeros(n, bool)
    blk = np.full(n, "", object)
    for b, s0, s1 in BLOCKS[1:]:
        a, z = int(s0 * FPS), min(n, int((s1 + 10) * FPS))
        ok[a:z] = True
        blk[a:z] = b
    return ok, blk


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-features", required=True)
    ap.add_argument("--train-scoring", required=True)
    ap.add_argument("--features-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestClassifier

    F = pd.read_csv(a.train_features)
    S = bs.load_scoring(a.train_scoring)
    n = min(len(F), S["nUsed"])
    F = F.iloc[:n]
    taps = S["dF"].astype(int) - 1                     # 0-based rows
    sig = signal(F)
    ok, blk = in_blocks(n)
    gap = int(1.0 * FPS)
    thrs = np.arange(2, 40, 0.5)
    # leave one block out: choose the threshold on 3 blocks (best F1), test on the 4th
    cv = []
    for b, _, _ in BLOCKS[1:]:
        tr = ok & (blk != b)
        te = ok & (blk == b)
        best = max(thrs, key=lambda th: f1(*match(peaks(sig, tr, th, gap), taps[tr[taps]])))
        tp, fp, fn = match(peaks(sig, te, best, gap), taps[te[taps]])
        cv.append(dict(block=b, threshold=float(best), TP=tp, FP=fp, FN=fn,
                       precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1)))
    CV = pd.DataFrame(cv)
    thr = max(thrs, key=lambda th: f1(*match(peaks(sig, ok, th, gap), taps)))
    tp, fp, fn = match(peaks(sig, ok, thr, gap), taps)
    pooled = dict(threshold=float(thr), TP=tp, FP=fp, FN=fn, precision=tp / max(tp + fp, 1),
                  recall=tp / max(tp + fn, 1))
    CV.to_csv(os.path.join(a.out_dir, "delivery_detect_cv.csv"), index=False)

    # reflex (withdrawal) model trained on all annotated deliveries, applied to detections
    specs = [("resp", -14, 2), ("after", 2, 18), ("late", 18, 60)]
    D = pd.read_csv(os.path.join(os.path.dirname(a.out_dir), "supervised_5941_sbi", "annotation_per_delivery.csv")) \
        if os.path.isfile(os.path.join(os.path.dirname(a.out_dir), "supervised_5941_sbi", "annotation_per_delivery.csv")) else None
    X = pd.DataFrame([bs.window_feats(F, int(f) + 7, specs) for f in taps]).fillna(0)   # +7: peak -> tap
    y = D.withdrawal.values if D is not None else np.zeros(len(taps))
    rf = RandomForestClassifier(300, min_samples_leaf=3, class_weight="balanced", random_state=0).fit(X, y)

    rows = []
    for fn_ in sorted(os.listdir(a.features_dir)):
        if not fn_.endswith("_bfeatures.csv"):
            continue
        tag = fn_.replace("_bfeatures.csv", "")
        G = pd.read_csv(os.path.join(a.features_dir, fn_))
        m = len(G)
        okg, blkg = in_blocks(m)
        det = peaks(signal(G), okg, thr, gap)
        Xg = pd.DataFrame([bs.window_feats(G, int(f) + 7, specs) for f in det]).fillna(0)
        pw = rf.predict_proba(Xg)[:, 1] if len(det) else np.array([])
        for i, f in enumerate(det):
            pre = G.iloc[max(0, f - 30):f]
            post = G.iloc[f:min(m, f + 90)]
            c0 = np.nanmedian(pre[["b_cx", "b_cy"]].values, 0)
            disp = np.nanmax(np.hypot(post.b_cx - c0[0], post.b_cy - c0[1]))
            dth = np.abs(post.b_theta - np.nanmedian(pre.b_theta)) % 180
            rot = np.nanmax(np.minimum(dth, 180 - dth))
            b = blkg[f]
            k = int(b[-1]) - 1 if b else -1
            mouse, phase = tag.split("_")[0], tag.split("_")[-1]
            rows.append(dict(session=tag, mouse=mouse, phase=phase, arm=ARM.get(mouse, "") if phase == "post" else "none",
                             frame=int(f) + 1, t_s=(f + 1) / FPS, block=b,
                             stim=ORDER.get(tag, ["?"] * 4)[k] if k >= 0 else "?",
                             p_withdrawal=float(pw[i]), withdrawal=int(pw[i] >= 0.5),
                             moved_after=int((disp > 25) or (rot > 30))))
    A = pd.DataFrame(rows)
    A.to_csv(os.path.join(a.out_dir, "auto_deliveries_all_sessions.csv"), index=False)
    T = A.groupby(["mouse", "phase", "arm", "stim"]).agg(n=("frame", "size"),
                                                         withdrawal_pct=("withdrawal", lambda x: 100 * x.mean()),
                                                         moved_pct=("moved_after", lambda x: 100 * x.mean())).round(1).reset_index()
    T.to_csv(os.path.join(a.out_dir, "auto_per_session_stimulus.csv"), index=False)

    # figure: pre vs post per mouse, per stimulus, for withdrawal % and moved %
    stims = ["Light touch", "Mild touch", "Pin prick", "Heat"]
    fig, ax = plt.subplots(2, 4, figsize=(16, 7), sharey="row")
    for r, (col, lab) in enumerate((("withdrawal_pct", "withdrawal (% deliveries)"), ("moved_pct", "moved after (% deliveries)"))):
        for j, st in enumerate(stims):
            for mouse, g in T[T.stim == st].groupby("mouse"):
                pre = g[g.phase == "pre"][col].values
                post = g[g.phase == "post"][col].values
                if len(pre) and len(post):
                    c = "tab:red" if ARM.get(mouse) == "sbi553" else "tab:blue"
                    ax[r, j].plot([0, 1], [pre[0], post[0]], "o-", color=c, label=f"{mouse} ({ARM.get(mouse)})")
            ax[r, j].set_xticks([0, 1], ["baseline", "post"])
            ax[r, j].set_title(f"{st}", fontsize=10)
            if j == 0:
                ax[r, j].set_ylabel(lab)
    ax[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"AUTOMATIC per-delivery scoring, all sessions (red SBI-553, blue vehicle; detector "
                 f"precision {pooled['precision']:.2f} / recall {pooled['recall']:.2f} on the annotated session)")
    fig.tight_layout(); fig.savefig(os.path.join(a.out_dir, "auto_pre_post_by_arm.png"), dpi=110); plt.close(fig)
    out = dict(delivery_detector=dict(pooled=pooled, leave_one_block_out=CV.round(3).to_dict("records")),
               n_detected=A.groupby("session").size().to_dict())
    with open(os.path.join(a.out_dir, "delivery_detect.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))
    return 0


def f1(tp, fp, fn):
    return 2 * tp / max(2 * tp + fp + fn, 1)


if __name__ == "__main__":
    sys.exit(main())
