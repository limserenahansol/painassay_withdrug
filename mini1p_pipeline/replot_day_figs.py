"""replot_day_figs.py - redraw the A2 / A6 / C2 figures of ca_day1_analysis.py from the saved per-cell table, without
recomputing anything (2026-10-22: every star now shows the pair it compares). Checks that the tests come out as the saved
test tables (A2_blocks_tests, A6_blocks_still_tests, C2_arm_blocks_still) before writing the figures.

  CA_DAY=1 python replot_day_figs.py      (Day 1 -> analysis\\ca_day1)
  CA_DAY=2 python replot_day_figs.py      (Day 2 -> analysis\\ca_day2)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ca_day1_analysis as d1  # noqa: E402

B = r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\analysis"
out = os.path.join(B, f"ca_day{d1.DAY}")
W = pd.read_csv(os.path.join(out, f"ca_d{d1.DAY}_cells_whole_session.csv"))
W["mouse"] = W.mouse.astype(str)
ok = True
for fn, csv, key in ((d1.fig_blocks, f"ca_d{d1.DAY}_A2_blocks_tests.csv", "condition"),
                     (d1.fig_blocks_still, f"ca_d{d1.DAY}_A6_blocks_still_tests.csv", "block"),
                     (d1.fig_arm_blocks, f"ca_d{d1.DAY}_C2_arm_blocks_still.csv", "condition")):
    T = fn(W, out)
    S = pd.read_csv(os.path.join(out, csv))
    T["mouse"] = T.mouse.astype(str); S["mouse"] = S.mouse.astype(str)
    M = T.merge(S, on=["mouse", key], suffixes=("_new", "_saved"))
    dp = np.nanmax(np.abs(M.p_new - M.p_saved)) if len(M) else np.nan
    same_sig = bool(((M.p_new < 0.05) == (M.p_saved < 0.05)).all())
    print(f"{fn.__name__}: {len(M)}/{len(S)} rows matched, max |p difference| {dp:.2g}, same p < 0.05 calls: {same_sig}")
    ok &= len(M) == len(S) and same_sig
print("all tests as saved" if ok else "*** CHECK: a test differs from the saved table")
