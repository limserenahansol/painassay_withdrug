"""split_parts.py - cut one slide figure into 2-4 parts at its white bands (between panels / between the per-mouse figures of a
grid), so that every part is shown at a readable size (2026-10-22, user: 'if the font is too small, separate the figure over 2
slides'). Nothing is redrawn: the parts are exact crops. Used by assemble_master.py for the earlier (old-style) slides.

  parts(img, box_w, box_h) -> (list of PIL crops, scale of the smallest part)
"""
import numpy as np

# (rows, cols) layouts tried, fewest parts first
LAYOUTS = [(1, 1), (1, 2), (2, 1), (1, 3), (3, 1), (2, 2), (1, 4), (4, 1)]


def _cut(white, n, k):
    """position of the k-th of n-1 cuts: the middle of the widest fully-white run within +-12 % of k / n"""
    L = len(white)
    lo, hi = int((k / n - 0.12) * L), int((k / n + 0.12) * L)
    lo, hi = max(lo, 1), min(hi, L - 1)
    seg = white[lo:hi]
    ok = (seg >= seg.max() - 1e-9).astype(int)
    e = np.diff(np.r_[0, ok, 0])
    st, en = np.flatnonzero(e == 1), np.flatnonzero(e == -1)
    j = int(np.argmax(en - st))
    return lo + int((st[j] + en[j]) // 2), float(seg.max())


def parts(img, box_w, box_h, target=0.62, max_parts=4):
    """the crops of the layout with the fewest parts whose smallest part reaches `target` (shown width / native width);
    if none does, the layout with the largest smallest-part scale (up to max_parts parts)"""
    im = img.convert("RGB")
    a = np.asarray(im).mean(2)
    W, H = im.size
    dpi = (img.info.get("dpi", (100, 100))[0] or 100)
    wc = (a > 245).mean(0)                                     # white fraction per column
    wr = (a > 245).mean(1)                                     # ... per row
    best = None
    for r, c in LAYOUTS:
        if r * c > max_parts:
            continue
        xs = [0] + [_cut(wc, c, k)[0] for k in range(1, c)] + [W]
        ys = [0] + [_cut(wr, r, k)[0] for k in range(1, r)] + [H]
        # a cut must fall on a clean band (>= 97 % white), or the layout would cut through a panel
        clean = all(_cut(wc, c, k)[1] >= 0.97 for k in range(1, c)) and all(_cut(wr, r, k)[1] >= 0.97 for k in range(1, r))
        if not clean:
            continue
        boxes = [(xs[j], ys[i], xs[j + 1], ys[i + 1]) for i in range(r) for j in range(c)]
        sc = min(min(box_w / ((b[2] - b[0]) / dpi), box_h / ((b[3] - b[1]) / dpi)) for b in boxes)
        if best is None or (sc > best[0] + 1e-9 and (best[0] < target)):
            best = (sc, boxes)
        if sc >= target:
            best = (sc, boxes)
            break
    sc, boxes = best
    crops = [im.crop(b) for b in boxes]
    for c_ in crops:
        c_.info["dpi"] = (dpi, dpi)
    return crops, sc
