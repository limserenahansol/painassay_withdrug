"""make_plainz_cache.py - 10-26 (user: "normally trace ppl showed only z score and then bar graph with AUC"): trial-mean z per neuron
around every event, NOT baseline-subtracted (the older psth caches subtract each neuron's mean over -3..0 s for display).
Same sessions, marks (SCORING_SUFFIX=_mix4) and trial rules as ca_event_analysis (trials in the first 15 s dropped).

  python make_plainz_cache.py  ->  analysis\\new_decks\\psth_cache_plainz_mix4.npz   {(mouse, day, phase, event): {mean, tt, ids, n}}
"""
import os
import sys

os.environ.setdefault("SCORING_SUFFIX", "_mix4")
sys.argv = sys.argv[:1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import ca_day1_analysis as CA  # noqa: E402
import ca_event_analysis as EA  # noqa: E402

OUT = os.path.join(EA.BASE, "analysis", "new_decks", "psth_cache_plainz_mix4.npz")
EVS = EA.STIM + EA.REFL + ["movement onset"]
lag = np.arange(int(-5 * EA.FPS), int(8 * EA.FPS) + 1)
D = {}
for m in EA.MO:
    for d in (1, 2):
        for ph in ("pre", "post"):
            s = CA.load_session(EA.ROOT[d], m, ph, EA.SCORE[d], day=d)
            if s is None or s["B"] is None:
                continue
            Z = np.asarray(s["T"]["z"], float)
            fr = s["T"]["frame"].astype(int)
            s["fr"] = fr
            for ev in EVS:
                idx, f = EA.ev_rows(s, s["B"]["ev"].get(ev, []))
                keep = (f / EA.FPS >= CA.SKIP_S) & (idx + lag[0] >= 0) & (idx + lag[-1] < len(fr))
                idx = idx[keep]
                if len(idx) < CA.MIN_TRIALS:
                    continue
                X = Z[idx[:, None] + lag[None, :]]                 # trials x time x cells
                D[(m, d, ph, ev)] = dict(mean=np.nanmean(X, 0), tt=lag / EA.FPS, ids=s["T"]["cell_ids"].astype(int), n=int(len(idx)))
            print("done", m, d, ph, flush=True)
np.savez_compressed(OUT, D=np.array(D, dtype=object))
print("->", OUT, len(D))
