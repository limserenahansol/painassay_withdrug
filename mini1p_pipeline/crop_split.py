"""crop_split.py - split a wide or tall figure PNG into parts at its whitest band (between panels), so each part fills a slide
with readable fonts (2026-10-22, user: 'if the font is too small, separate the figure over 2 slides'). Nothing is redrawn:
the parts are exact crops of the original; make_new_deck.SPLIT shows them as 1/2, 2/2 (or 1/4 ...) slides.

  python crop_split.py      (the event-analysis / validation figures listed in JOBS)
"""
import os

import numpy as np
from PIL import Image

EA = r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\analysis\event_analysis"
VA = os.path.join(EA, "validation")


def white_line(a, axis, lo, hi):
    """index (row if axis=0, column if axis=1) between lo and hi fractions: the middle of the widest run of the whitest
    rows / columns (10-22: the middle of ALL the whitest columns could fall between a panel's y label and its axis)"""
    g = a.mean(2) if a.ndim == 3 else a
    w = (g > 245).mean(axis=1 - axis)                       # fraction of white pixels per row (axis 0) / column (axis 1)
    n = len(w)
    i0, i1 = int(lo * n), int(hi * n)
    seg = w[i0:i1]
    ok = (seg >= seg.max() - 1e-9).astype(int)
    e = np.diff(np.r_[0, ok, 0])
    st, en = np.flatnonzero(e == 1), np.flatnonzero(e == -1)
    k = int(np.argmax(en - st))
    return i0 + int((st[k] + en[k]) // 2)


def top_band(a):
    """rows of the top band (a figure legend) down to the end of the first white gap under it, or None"""
    g = a.mean(2) if a.ndim == 3 else a
    dark = (g < 200).any(1)
    r = np.flatnonzero(dark)
    if not len(r):
        return None
    i = r[0]
    while i < len(dark) and dark[i]:
        i += 1
    j = i
    while j < len(dark) and not dark[j]:
        j += 1
    return (0, j - 2) if j - i >= 3 and j < 0.2 * len(dark) else None


def split(path, mode, lo=0.35, hi=0.65, lo2=None, hi2=None, head=False):
    """mode 'rows' (top / bottom), 'cols' (left / right) or 'quad' (rows x cols); writes <stem>_p1.. (_q1.. for quad);
    head (rows): the top legend band goes above the lower part as well"""
    im = Image.open(path).convert("RGB")
    a = np.asarray(im)
    H, W = a.shape[:2]
    stem, ext = os.path.splitext(path)
    out = []
    if mode == "rows":
        y = white_line(a, 0, lo, hi)
        boxes = [(0, 0, W, y), (0, y, W, H)]
        names = ["_p1", "_p2"]
    elif mode == "cols":
        x = white_line(a, 1, lo, hi)
        boxes = [(0, 0, x, H), (x, 0, W, H)]
        names = ["_p1", "_p2"]
    else:
        y = white_line(a, 0, lo, hi)
        x = white_line(a, 1, lo2 if lo2 is not None else lo, hi2 if hi2 is not None else hi)
        t0, b1 = int(0.075 * H), int(0.955 * H)               # leave out the figure title band and the legend band
        boxes = [(0, t0, x, y), (x, t0, W, y), (0, y, x, b1), (x, y, W, b1)]
        names = ["_q1", "_q2", "_q3", "_q4"]
    dpi = im.info.get("dpi", (100, 100))
    head_flag, head = head, None
    if mode == "rows" and head_flag:
        hb = top_band(a)
        if hb:
            head = hb
    if mode == "quad":                                         # the column titles (above the first panel row) go on the lower parts too
        g = a.mean(2)
        dark = (g < 235).mean(1)
        ys = [r for r in range(t0 + int(0.01 * H), y) if dark[r] > 0.3]     # first row of panel frames
        if ys:
            head = (t0, ys[0] - 2)
    for k, (b, nm) in enumerate(zip(boxes, names)):
        part = im.crop(b)
        if head and (k >= 2 or (mode == "rows" and k == 1)):
            hd = im.crop((b[0], head[0], b[2], head[1]))
            new = Image.new("RGB", (part.width, part.height + hd.height), "white")
            new.paste(hd, (0, 0)); new.paste(part, (0, hd.height))
            part = new
        part.save(stem + nm + ext, dpi=dpi)
        out.append(stem + nm + ext)
    return out


JOBS = [(os.path.join(EA, "W1_window_search.png"), "rows"), (os.path.join(EA, "W1b_window_search_slow.png"), "rows"),
        (os.path.join(EA, "W2_named_windows.png"), "cols", False, 0.28, 0.34), (os.path.join(EA, "W3_timing.png"), "cols"),
        (os.path.join(EA, "Q1_motion_check.png"), "cols"), # P6b: drawn as two native parts by ca_event_analysis (10-23: the automatic cut split the legend)
        (os.path.join(EA, "P7_brain_behaviour.png"), "rows", True), (os.path.join(VA, "A1_rules_compared.png"), "rows"),
        (os.path.join(VA, "A2_functional_types.png"), "rows"), (os.path.join(VA, "A2b_types_responsive.png"), "rows"),
        (os.path.join(VA, "V4_key_findings_per_mouse.png"), "cols")]


if __name__ == "__main__":
    for job in JOBS:
        p, mode, hd, lo, hi = (job + (False, 0.35, 0.65)[len(job) - 2:])[:5]     # W2: cut left of panel B's y label
        if os.path.isfile(p):
            print(os.path.basename(p), mode, [os.path.basename(x) for x in split(p, mode, lo, hi, head=hd)])
    for m in ("5941", "5939", "5937", "1191"):
        p = os.path.join(EA, f"P4b_maps_{m}.png")
        if os.path.isfile(p):                                   # rows: Day 1 | Day 2; columns: 4 stimuli | 6 behaviours
            print(os.path.basename(p), [os.path.basename(x) for x in split(p, "quad", 0.42, 0.58, 0.36, 0.5)])
