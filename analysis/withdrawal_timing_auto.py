"""withdrawal_timing_auto.py - per withdrawal / flinch mark, when did the motion in the video start? (2026-10-23)

For every hand-scored withdrawal and flinch mark of the 16 sessions: the motion trace of withdrawal_timing_check.motion()
(b_edge_me + s_me, robust z, 0.1 s smoothing); the motion peak in the 1.0 s before to 0.17 s after the mark; the movement start =
the last frame before that peak where the trace was still below half-way between the baseline (median of 3-1.5 s before the mark)
and the peak. Only marks with a clear burst (peak >= baseline + 3) are timed.

Writes analysis\\withdrawal_timing\\auto_marks.csv and prints mark - start and mark - peak per session, day and kind.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import withdrawal_timing_check as W  # noqa: E402


def main():
    rows = []
    for d in (1, 2):
        for p in sorted(glob.glob(os.path.join(W.SC[d], "ScoringAB_*.mat"))):
            S = sio.loadmat(p, squeeze_me=True)
            mouse = str(S["mouseID"]).strip()
            ph = "pre" if str(S["phase"]).lower().startswith("base") else "post"
            rid = f"{mouse}_d{d}_{ph}"
            F = pd.read_csv(os.path.join(W.FEAT[d], f"{rid}_bfeatures.csv"))
            m = W.motion(F)
            re = np.atleast_2d(S["reflexEvents"]).astype(int).reshape(-1, 2)
            for w, typ in re:
                i = w - 1
                if i - 90 < 0 or i + 6 >= len(m):
                    continue
                base = float(np.median(m[i - 90:i - 45]))
                seg = m[i - 30:i + 6]
                pk = i - 30 + int(np.argmax(seg))
                if m[pk] < base + 3:
                    rows.append(dict(rid=rid, day=d, mouse=mouse, phase=ph, kind="withdrawal" if typ == 1 else "flinch", mark=int(w),
                                     clear=False, mark_minus_start_s=np.nan, mark_minus_peak_s=np.nan)); continue
                half = base + 0.5 * (m[pk] - base)
                st = pk
                while st > i - 45 and m[st - 1] >= half:
                    st -= 1
                rows.append(dict(rid=rid, day=d, mouse=mouse, phase=ph, kind="withdrawal" if typ == 1 else "flinch", mark=int(w),
                                 clear=True, mark_minus_start_s=(i - st) / W.FPS, mark_minus_peak_s=(i - pk) / W.FPS))
    A = pd.DataFrame(rows)
    A.to_csv(os.path.join(W.OUT, "auto_marks.csv"), index=False)
    C = A[A.clear]
    print(f"marks {len(A)}, with a clear motion burst {len(C)} ({100 * len(C) / len(A):.0f}%)")
    q = lambda v: f"{v.median():.2f} (IQR {v.quantile(.25):.2f}-{v.quantile(.75):.2f})"
    for key, Z in [("ALL", C)] + [(f"Day {d}", C[C.day == d]) for d in (1, 2)] + [(k, C[C.kind == k]) for k in ("withdrawal", "flinch")]:
        print(f"{key:11s} n={len(Z):4d}  mark - start {q(Z.mark_minus_start_s)}   mark - peak {q(Z.mark_minus_peak_s)}")
    print(C.groupby(["rid", "kind"]).mark_minus_start_s.median().unstack().round(2).to_string())


if __name__ == "__main__":
    main()
