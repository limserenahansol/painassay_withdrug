"""ca_rule_vs_user_day1.py - how well do my automatic keep rules reproduce YOUR Day-1 decisions? (self-evaluation)

Day 1 is the only day with your final decision on every candidate (green + orange kept, everything else deleted;
2026-10-21, on the max dF/F projection). The same rules decide the NEW Day-2 candidates (ca_day2_decisions.py), so
their agreement with you on Day 1 is the honest estimate of how much to trust them on Day 2:
  rule    keep unless verdict not_cell or outside your envelope (area <= 3.5 x soma, <= 3 pieces, axis <= 3.5)
  strict  rule AND the clear-neuron-shape bar (ca_transfer_decisions --strict-new: one compact body, solidity, size,
          axis, movie proof, Ca-like), the bar from the cells YOU kept in the OTHER three mice (leave one mouse out,
          percentile q of each measure)
For each mouse: kept by you / by the rule, precision (rule keeps you kept), recall (your keeps the rule kept).
Rescued footprints (verdict 'rescued') are left out (they are not EXTRACT candidates).
Writes analysis\\ca_selfcheck\\rule_vs_user_day1.csv

  python ca_rule_vs_user_day1.py
"""
import os
import sys

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ca_transfer_decisions import rule_keep  # noqa: E402
from ca_day2_decisions import shape_numbers  # noqa: E402

BASE = r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553"
MICE = ["5941", "5939", "5937", "1191"]


def load(m):
    tag = f"{m}_d1"
    R = pd.read_csv(os.path.join(BASE, "mini1p_out", "20261019", tag, "4_cell_review", f"{tag}_cell_review.csv"))
    S = np.asarray(sio.loadmat(os.path.join(BASE, "mini1p_out", "20261019", tag, "3_extract", f"{tag}_extract.mat"),
                               variable_names=["spatial_weights"])["spatial_weights"], np.float32)
    mf, so = shape_numbers(S)
    num = lambda c: pd.to_numeric(R[c], errors="coerce").values
    D = pd.DataFrame(dict(mouse=m, id=R.id.values, user=(R.keep.values == 1).astype(int), verdict=R.verdict.astype(str).values,
                          ar=num("area_ratio"), pieces=num("n_pieces"), ax=num("axis_ratio"), ev=num("event_image_corr"),
                          cl=np.nan_to_num(num("ca_like")), mf=mf, so=so, src=R.get("decision_source", "").astype(str).values))
    D["rule"] = [rule_keep(v, a, p, x) for v, a, p, x in zip(D.verdict, D.ar, D.pieces, D.ax)]
    return D[D.verdict != "rescued"].reset_index(drop=True)


def bar_from(K, q):
    return dict(main_min=np.nanpercentile(K.mf, q), sol_min=np.nanpercentile(K.so, q), area_lo=np.nanpercentile(K.ar, q / 2),
                area_hi=np.nanpercentile(K.ar, 100 - q / 2), axis_max=np.nanpercentile(K.ax, 100 - q),
                ev_min=np.nanpercentile(K.ev, q))


def strict(D, bar):
    return ((D.rule == 1) & (D.mf >= bar["main_min"]) & (D.so >= bar["sol_min"]) & (D.ar >= bar["area_lo"])
            & (D.ar <= bar["area_hi"]) & (D.ax <= bar["axis_max"]) & (D.ev >= bar["ev_min"]) & (D.cl == 1)).astype(int)


def score(y, p):
    tp = int(((y == 1) & (p == 1)).sum()); fp = int(((y == 0) & (p == 1)).sum()); fn = int(((y == 1) & (p == 0)).sum())
    return tp, fp, fn, tp / max(tp + fp, 1), tp / max(tp + fn, 1)


def main():
    out = os.path.join(BASE, "analysis", "ca_selfcheck"); os.makedirs(out, exist_ok=True)
    A = {m: load(m) for m in MICE}
    rows = []
    for m in MICE:
        D = A[m]
        others = pd.concat([A[o] for o in MICE if o != m])
        K = others[others.user == 1]
        for name, pred in [("rule", D.rule)] + [(f"strict q{q}", strict(D, bar_from(K, q))) for q in (5, 10, 20, 30)]:
            tp, fp, fn, pr, rc = score(D.user.values, np.asarray(pred))
            rows.append(dict(mouse=m, rule=name, candidates=len(D), you_kept=int(D.user.sum()), rule_kept=int(np.sum(pred)),
                             agree=tp, rule_only=fp, you_only=fn, precision=round(pr, 3), recall=round(rc, 3),
                             accuracy=round(float(np.mean(np.asarray(pred) == D.user.values)), 3)))
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(out, "rule_vs_user_day1.csv"), index=False)
    pd.set_option("display.width", 200)
    print(T.to_string(index=False))
    print(T.groupby("rule")[["agree", "rule_only", "you_only"]].sum().assign(
        precision=lambda x: (x.agree / (x.agree + x.rule_only)).round(3), recall=lambda x: (x.agree / (x.agree + x.you_only)).round(3)))


if __name__ == "__main__":
    main()
