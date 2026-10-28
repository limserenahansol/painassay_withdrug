"""Starting file for a REVIEW (scorer mode 4) of a session that already has a manual scoring:
your marks are the editable layer, the automatic ones are shown next to them (read-only), and
the jump list (j / k in the scorer) holds every place where the two disagree.

Editable layer (what you correct, and what is saved):
  your deliveries + types, withdrawal / flinch taps, held behaviours (attending, escape ...)
  + the automatic general movement and grooming bouts (you have not scored those yet)
Reference layer (refDFrames, refDTypes, refReflex, refScore; drawn in 4 grey "AUTO" rows):
  ONLY the automatic marks that differ from yours - deliveries with none of yours within
  0.5 s, withdrawal / flinch taps with no tap of yours of the same kind within 0.3 s, escape /
  rearing bouts that do not overlap one of yours. In the scorer a left click takes one into
  yours, a right click rejects it.
Jump list (checkFrames + checkWhy), merged when < 1 s apart:
  - a delivery of yours with no automatic one within 0.5 s, and the reverse
  - a delivery of yours where you and the algorithm disagree on withdrawal or flinch
    (any tap within 2 s or until the next delivery)
  - an escape / rearing bout in one layer only
  - every automatic general movement and grooming bout (new categories, never checked)

  python behavior_review_merge.py --manual CORRECTED.mat --auto-dir DIR --auto-mat AUTO.mat --out MERGED.mat
"""
import argparse
import os
import sys
import time

import numpy as np
import scipy.io as sio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from behavior_auto_annotate import match, response_table  # noqa: E402

FPS = 30.0


def spans(m):
    d = np.diff(np.r_[0, np.asarray(m, int), 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--manual", required=True, help="your ScoringAB (types corrected)")
    ap.add_argument("--auto-dir", required=True, help="folder with auto_annotation_arrays.npz")
    ap.add_argument("--auto-mat", required=True, help="the automatic ScoringAB of behavior_auto_annotate.py")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    M = sio.loadmat(a.manual, squeeze_me=True)
    A = sio.loadmat(a.auto_mat, squeeze_me=True)
    Z = np.load(os.path.join(a.auto_dir, "auto_annotation_arrays.npz"))
    n = int(M["nUsed"])
    u_del = np.atleast_1d(M["dFrames"]).astype(int) - 1                  # 0-based rows
    o = np.argsort(u_del)
    u_del, u_typ = u_del[o], np.atleast_1d(M["dTypes"]).astype(int)[o]
    re = np.atleast_2d(M["reflexEvents"]).astype(int)
    u_w, u_f = np.sort(re[re[:, 1] == 1, 0] - 1), np.sort(re[re[:, 1] == 2, 0] - 1)
    u_score = np.atleast_1d(M["score"]).astype(int)[:n]
    a_del, a_typ = Z["a_del"].astype(int), Z["a_typ"].astype(int)
    # automatic reflex taps: those around your deliveries and those around its own deliveries
    w = np.unique(np.r_[Z["w_ev_u"], Z["w_ev"]]).astype(int)
    f = np.unique(np.r_[Z["f_ev_u"], Z["f_ev"]]).astype(int)
    w = w[np.r_[True, np.diff(w) > 3]] if len(w) else w
    f = f[np.r_[True, np.diff(f) > 3]] if len(f) else f
    a_rear = np.asarray(Z["rear"], bool)[:n]

    items = []
    _, _, _, Mt = match(a_del, u_del, 15)
    mu, ma = {j for _, j in Mt}, {i for i, _ in Mt}
    items += [(int(x), "your delivery, no AUTO one") for j, x in enumerate(u_del) if j not in mu]
    items += [(int(x), "AUTO delivery, not in yours") for i, x in enumerate(a_del) if i not in ma]
    for nm, ue, ae in (("withdrawal", u_w, Z["w_ev_u"].astype(int)), ("flinch", u_f, Z["f_ev_u"].astype(int))):
        uy, ay = response_table(u_del, n - 1, ue), response_table(u_del, n - 1, ae)
        for x, p, q in zip(u_del, uy, ay):
            if p != q:
                items.append((int(x), f"{nm}: yours {'yes' if p else 'no'}, AUTO {'yes' if q else 'no'}"))
    u_rear = u_score == 4
    for s, e in spans(a_rear):
        if not u_rear[max(0, s - 15):e + 15].any():
            items.append((int(s), "AUTO escape/rearing, not in yours"))
    for s, e in spans(u_rear):
        if not a_rear[max(0, s - 15):e + 15].any():
            items.append((int(s), "your escape/rearing, no AUTO one"))
    move = np.asarray(A["moveVec"]).astype(bool).ravel()[:n]
    groom = np.asarray(A["groomVec"]).astype(bool).ravel()[:n]
    items += [(int(s), f"check AUTO general movement ({(e - s) / FPS:.1f} s)") for s, e in spans(move)]
    items += [(int(s), f"check AUTO grooming candidate ({(e - s) / FPS:.1f} s)") for s, e in spans(groom)]
    items.sort()
    chk, why = [], []
    for x, r in items:                      # merge items < 1 s apart into one stop
        if chk and x - chk[-1] < FPS:
            if r not in why[-1]:
                why[-1] += " + " + r
        else:
            chk.append(x); why.append(r)
    kinds = {}
    for _, r in items:
        k = r.split(" (")[0].split(":")[0]
        kinds[k] = kinds.get(k, 0) + 1

    # reference layer = only what differs from yours
    ref_del = np.array([i for i in range(len(a_del)) if i not in ma], int)
    w_d = np.array([x for x in w if not np.any(np.abs(u_w - x) <= 9)], int)
    f_d = np.array([x for x in f if not np.any(np.abs(u_f - x) <= 9)], int)
    ref_score = np.zeros(n)
    for s0, e0 in spans(a_rear):
        if not u_rear[max(0, s0 - 15):e0 + 15].any():
            ref_score[s0:e0] = 4
    print(f"AUTO layer (differences only): {len(ref_del)} deliveries, {len(w_d)} withdrawal, "
          f"{len(f_d)} flinch taps, {len(spans(ref_score == 4))} escape bouts")
    out = {k: M[k] for k in ("score", "uncert", "affNames", "affKeys", "refNames", "stimNames", "frameRate",
                             "fpsTrue", "hdrFps", "lagMs", "obsWindow", "guardMin", "syncOff", "offFrames",
                             "botFile", "sidFile") if k in M}
    out.update(score=u_score.astype(float).reshape(-1, 1),
               uncert=np.asarray(M["uncert"]).astype(bool).reshape(-1, 1),
               reflexEvents=re.astype(float), dFrames=(u_del + 1).astype(float).reshape(-1, 1),
               dTypes=u_typ.astype(float).reshape(-1, 1), dSec=(u_del / FPS).reshape(-1, 1),
               nUsed=float(n), mode=4.0, moveVec=move.reshape(-1, 1), groomVec=groom.reshape(-1, 1),
               moveName="General movement (turn / steps)", groomName="Grooming",
               refDFrames=(a_del[ref_del] + 1).astype(float).reshape(-1, 1),
               refDTypes=a_typ[ref_del].astype(float).reshape(-1, 1),
               refReflex=np.r_[np.c_[w_d + 1, np.ones(len(w_d))], np.c_[f_d + 1, 2 * np.ones(len(f_d))]].astype(float),
               refScore=ref_score.reshape(-1, 1),
               checkFrames=(np.array(chk) + 1).astype(float).reshape(-1, 1),
               checkWhy=np.array(why, dtype=object),
               startFrom="your annotation (types corrected) + AUTO general movement and grooming",
               manualSource=a.manual, autoSource=a.auto_mat, created=time.strftime("%Y-%m-%d %H:%M"))
    for k in ("sessionNo", "mouseID", "sexID", "dayNo", "phase"):
        out[k] = str(A.get(k, M.get(k, "")))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    sio.savemat(a.out, out, do_compression=True)
    print(f"{len(chk)} stops for {len(items)} discrepancies -> {a.out}")
    for k, v in sorted(kinds.items(), key=lambda kv: -kv[1]):
        print(f"  {v:4d}  {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
