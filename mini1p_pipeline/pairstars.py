"""pairstars.py - significance marks that always show WHICH PAIR was tested (user rule, 2026-10-22: "if you write *** based
on P value you should well mark which pair they are"). Two forms only:

  bracket(ax, x1, x2, y, p)        a bracket from one compared element to the other, the star (or 'ns') above its middle
  vs_ref(ax, x, y, p, "vs 0")      a test against a reference (a change vs 0, a response vs chance): the star with the
                                   reference written under it

* p < 0.05, ** p < 0.01, *** p < 0.001, ns = p >= 0.05 (KEY below, for a figure footnote).
"""
import numpy as np

KEY = "* p < 0.05, ** p < 0.01, *** p < 0.001, ns = not significant"


def star(p):
    if p is None or not np.isfinite(p):
        return "n.a."
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def _tr(ax, axes_y):
    return ax.get_xaxis_transform() if axes_y else ax.transData


def bracket(ax, x1, x2, y, p, color="k", fs=11, show_ns=True, inset=0.03, label="", axes_y=False, below=False, h=None, lw=1.1):
    """bracket from x1 to x2 at height y (data units; axes fraction with axes_y=True) with the star of p over its middle;
    below=True hangs it under y (hooks up, text under). inset keeps two brackets that share an end apart. label = optional
    text before the star. Returns the star text, or None when nothing was drawn."""
    st = star(p)
    if st == "n.a." or (st == "ns" and not show_ns):
        return None
    if h is None:
        if axes_y:
            h = 0.018
        else:
            lo, hi = ax.get_ylim()
            h = 0.02 * (hi - lo)
    a, b = x1 + inset, x2 - inset
    s = -1 if below else 1
    tr = _tr(ax, axes_y)
    ax.plot([a, a, b, b], [y - s * h, y, y, y - s * h], color=color, lw=lw, clip_on=False, transform=tr, zorder=6,
            solid_capstyle="butt")
    txt = f"{label} {st}" if label else st
    ax.text((a + b) / 2, y + s * 0.3 * h, txt, ha="center", va="top" if below else "bottom", color=color,
            fontsize=fs if st != "ns" else fs - 1.5, fontweight="bold" if st != "ns" else "normal", clip_on=False,
            transform=tr, zorder=6)
    return st


def vs_ref(ax, x, y, p, ref="vs 0", color="k", fs=10, show_ns=True, axes_y=False, va="bottom"):
    """a test against a reference: the star (or 'ns') with the reference under it, e.g. '***' over 'vs 0'"""
    st = star(p)
    if st == "n.a." or (st == "ns" and not show_ns):
        return None
    txt = f"{st}\n{ref}" if va == "bottom" else f"{ref}\n{st}"
    ax.text(x, y, txt, ha="center", va=va, color=color, fontsize=fs, fontweight="bold" if st != "ns" else "normal",
            clip_on=False, transform=_tr(ax, axes_y), linespacing=0.95, zorder=6)
    return st
