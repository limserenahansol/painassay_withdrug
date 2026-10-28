"""Attention marks on kept cells you have not decided yourself (2026-10-21: "keep the previous rule's
cells but mark them; if they do not look good on Day 2, delete them").

Writes a column 'attention' into the day review table (backup *_before_attention.csv):
  ''                                        your decision / your mark / rescued (your kept cell)
  'rule only - not reviewed by you'         kept by the rule
  '...; strict neuron-shape rule would drop' (optional --strict <decisions.csv of a --strict-new
                                            --no-write run>: the rule cells that would not pass)
and a list <tag>_attention_cells.csv in 5_analysis/decisions. Keep decisions are NOT changed.

  python ca_mark_attention.py --day-root <mini1p_out/20261019/<m>_d1> [--strict <csv>]
"""
import argparse
import os
import shutil

import numpy as np
import pandas as pd


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--day-root", required=True)
    ap.add_argument("--strict", default="")
    a = ap.parse_args(argv)
    tag = os.path.basename(os.path.normpath(a.day_root))
    csv = os.path.join(a.day_root, "4_cell_review", f"{tag}_cell_review.csv")
    R = pd.read_csv(csv)
    bk = csv.replace(".csv", "_before_attention.csv")
    if not os.path.isfile(bk):
        shutil.copy(csv, bk)
    src = R.get("decision_source", pd.Series([""] * len(R))).astype(str)
    rule = (R.keep == 1) & (src != "user") & (R.verdict.astype(str) != "rescued")
    att = np.where(rule, "rule only - not reviewed by you", "")
    if a.strict and os.path.isfile(a.strict):
        D = pd.read_csv(a.strict).set_index("id")
        drop = np.array([bool(rule.iloc[j]) and int(R.id.iloc[j]) in D.index and int(D.at[int(R.id.iloc[j]), "keep"]) == 0
                         for j in range(len(R))])
        att = np.where(drop, "rule only - not reviewed by you; strict neuron-shape rule would drop", att)
    R["attention"] = att
    R.to_csv(csv, index=False)
    L = R[R.attention != ""][["id", "row", "col", "verdict", "attention"]]
    out = os.path.join(a.day_root, "5_analysis", "decisions", f"{tag}_attention_cells.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    L.to_csv(out, index=False)
    print(f"{tag}: {len(L)} kept cells with an attention mark ({int(L.attention.str.contains('would drop').sum())} the strict rule "
          f"would drop); keep unchanged ({int((R.keep == 1).sum())} kept) -> {out}")


if __name__ == "__main__":
    main()
