"""figfoot.py - 10-24 (user: "all details in the notes, not on the slide; not wordy; x / y labels and colour scales clear"):
the long grey methods text that the figure scripts printed under their figures now goes into the slide notes.

note(fig, txt, lang)  remembers txt for this figure and draws in the figure only the star key (one line, readable size) when
                      txt has one; returns the text artist (or None)
when that figure is saved as <name>.png the remembered text is written next to it as <name>.png.notes.txt
notes_for(png)        the text for a PNG (also for the split parts <stem>_p1.png / _q2.png of crop_split.py), "" if none;
                      make_new_deck.Deck._slide puts it at the top of the slide notes
"""
import os
import re

import matplotlib.figure as _mf

STAR = re.compile(r"\*\s*p\s*<\s*0?\.05")
KEY_EN = "* p < 0.05, ** p < 0.01, *** p < 0.001, ns = not significant"
KEY_KR = "* p < 0.05, ** p < 0.01, *** p < 0.001, ns = 유의하지 않음"
SIDE = ".notes.txt"


def _kr(txt):
    return bool(re.search(r"[가-힣]", txt))


def note(fig, txt, lang=None, fs=10.5, y=0.008, color="0.25", star=True):
    txt = " ".join(str(txt).split())
    if not txt:
        return None
    old = getattr(fig, "_figfoot", "")
    fig._figfoot = (old + " " + txt).strip()
    if not (star and STAR.search(txt)) or getattr(fig, "_figfoot_star", False):
        return None
    fig._figfoot_star = True
    kr = (lang == "kr") if lang else _kr(txt)
    return fig.text(0.01, y, KEY_KR if kr else KEY_EN, fontsize=fs, color=color, ha="left", va="bottom",
                    fontname="Malgun Gothic" if kr else None)


_orig = _mf.Figure.savefig


def _savefig(self, fname, *a, **k):
    r = _orig(self, fname, *a, **k)
    t = getattr(self, "_figfoot", "")
    if t and isinstance(fname, (str, os.PathLike)):
        try:
            with open(os.fspath(fname) + SIDE, "w", encoding="utf-8") as fh:
                fh.write(t)
        except OSError:
            pass
    return r


_mf.Figure.savefig = _savefig


def notes_for(png):
    """the remembered text of a PNG or of the figure it was split from; '' if none"""
    if not png or not isinstance(png, (str, os.PathLike)):
        return ""
    png = os.fspath(png)
    cands = [png + SIDE]
    m = re.match(r"^(.*?)_(?:p|q)\d+(_kr)?\.png$", png)
    if m:
        cands.append(m.group(1) + (m.group(2) or "") + ".png" + SIDE)
    for c in cands:
        if os.path.isfile(c):
            try:
                return open(c, encoding="utf-8").read().strip()
            except OSError:
                return ""
    return ""
