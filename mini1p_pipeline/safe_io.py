"""safe_io.py - a figure save that survives a transient Windows write error.

2026-10-22: four PNGs failed within 20 s with OSError [Errno 22] Invalid argument on open(..., "w+b") (the tracking maps
of 5939 / 1191 / 5937 and a 1191 heatmap) and were writable again a minute later - another program (a viewer, a
scanner) held the file for a moment. Importing this module makes matplotlib's Figure.savefig retry (6 tries, 2-12 s
apart), then write next to the file and replace it.
"""
import os
import time

import matplotlib.figure as _mf

_orig = _mf.Figure.savefig


def _savefig(self, fname, *a, **k):
    if not isinstance(fname, (str, os.PathLike)):
        return _orig(self, fname, *a, **k)
    last = None
    for t in range(6):
        try:
            return _orig(self, fname, *a, **k)
        except OSError as e:                        # noqa: PERF203
            last = e
            print(f"safe_io: {os.path.basename(str(fname))}: {e} - retry {t + 1}")
            time.sleep(2 + 2 * t)
    root, ext = os.path.splitext(str(fname))
    tmp = root + "_tmpwrite" + ext
    _orig(self, tmp, *a, **k)
    for _ in range(5):
        try:
            os.replace(tmp, fname)
            return None
        except OSError as e:
            last = e
            time.sleep(3)
    raise last


if getattr(_mf.Figure.savefig, "__name__", "") != "_savefig":
    _mf.Figure.savefig = _savefig
