"""make_session_segments.py - 10-26 (user: whole-session slides had no test): mean z of every neuron in each part of each session -
the no-stimulus start (15-180 s) and the 180-s block of each stimulus (identified from that stimulus' delivery times).

  python make_session_segments.py  ->  analysis\\_variants\\mix4\\pi_extra\\session\\segments_per_neuron.csv
"""
import os
import sys

os.environ.setdefault("SCORING_SUFFIX", "_mix4")
sys.argv = sys.argv[:1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import ca_day1_analysis as CA  # noqa: E402
import ca_event_analysis as EA  # noqa: E402

OUT = os.path.join(EA.BASE, "analysis", "_variants", "mix4", "pi_extra", "session", "segments_per_neuron.csv")
rows = []
for m in EA.MO:
    for d in (1, 2):
        for ph in ("pre", "post"):
            s = CA.load_session(EA.ROOT[d], m, ph, EA.SCORE[d], day=d)
            if s is None or s["B"] is None:
                continue
            Z = np.asarray(s["T"]["z"], float)
            t = s["T"]["frame"].astype(int) / EA.FPS
            ids = s["T"]["cell_ids"].astype(int)
            segs = {"no stimulus": (CA.SKIP_S, 180.0)}
            for ev in EA.STIM:
                tt = np.asarray(s["B"]["ev"].get(ev, []), float) / EA.FPS
                if len(tt):
                    med = np.median(tt)
                    for a, b in EA.BLOCK_S:
                        if a <= med < b:
                            segs[ev] = (a, b)
            for seg, (a, b) in segs.items():
                k = (t >= a) & (t < b)
                mz = np.nanmean(Z[k], 0)
                rows += [dict(mouse=m, day=d, phase=ph, cell=int(c), seg=seg, t0=a, t1=b, mean_z=float(v)) for c, v in zip(ids, mz)]
            print("done", m, d, ph, sorted(segs.items(), key=lambda x: x[1]), flush=True)
pd.DataFrame(rows).to_csv(OUT, index=False)
print("->", OUT, len(rows))
