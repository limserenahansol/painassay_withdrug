"""deck_diff.py - which slides changed between two versions of a deck (10-23: before vs after the timing fix).

Slides are matched by title in order (difflib alignment of the title lists, so inserted / dropped slides do not shift the
rest). Per matched pair: the text of every shape (title, key sentence, bottom, cards, tables), the notes and the pictures
(SHA-1 of each image blob) are compared. Numbers are pulled out of the texts so a change reads 'old -> new'.

  python deck_diff.py OLD.pptx NEW.pptx --out diff.csv
Writes diff.csv (one row per slide of either deck) and prints a summary.
"""
import argparse
import difflib
import hashlib
import re

import pandas as pd
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

NUM = re.compile(r"[-+]?\d+(?:\.\d+)?%?|\*{1,3}|\bns\b")


def shape_texts(shapes, out):
    for sh in shapes:
        if sh.shape_type == MSO_SHAPE_TYPE.GROUP:
            shape_texts(sh.shapes, out)
        elif sh.has_text_frame and sh.text_frame.text.strip():
            out.append(sh.text_frame.text.strip())
        elif getattr(sh, "has_table", False) and sh.has_table:
            for r in sh.table.rows:
                for c in r.cells:
                    if c.text.strip():
                        out.append(c.text.strip())


def pics(shapes, out):
    for sh in shapes:
        if sh.shape_type == MSO_SHAPE_TYPE.GROUP:
            pics(sh.shapes, out)
        elif sh.shape_type == MSO_SHAPE_TYPE.PICTURE:
            out.append(hashlib.sha1(sh.image.blob).hexdigest()[:12])
        elif sh.shape_type == MSO_SHAPE_TYPE.MEDIA:
            out.append("media")


def read(path):
    rows = []
    for i, s in enumerate(Presentation(path).slides, 1):
        t = []
        shape_texts(s.shapes, t)
        p = []
        pics(s.shapes, p)
        # 10-24: the decks have no title placeholder; the title is the text box at 0.25-0.6 in from the top (the first text box is
        # the section label, which made every slide of a section look alike)
        tb = [sh for sh in s.shapes if sh.has_text_frame and sh.text_frame.text.strip() and 0.25 <= (sh.top or 0) / 914400 < 0.6]
        title = tb[0].text_frame.text.strip() if tb else (t[0] if t else "")
        notes = s.notes_slide.notes_text_frame.text.strip() if s.has_notes_slide else ""
        rows.append(dict(no=i, title=title, texts=t, notes=notes, pics=p))
    return rows


def nums(s):
    return NUM.findall(s)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("old"); ap.add_argument("new"); ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    O, N = read(a.old), read(a.new)
    key = lambda r: re.sub(r"\s+", " ", re.sub(NUM, "#", r["title"])).strip().lower()
    # 10-24: pair by title (same title first, in order; then the most similar remaining title >= 0.6), so slides inserted
    # in between do not shift the pairs
    used, pairs = set(), []
    for i, r in enumerate(O):
        j = next((j for j, q in enumerate(N) if j not in used and key(q) == key(r)), None)
        if j is not None:
            used.add(j); pairs.append((i, j))
    paired_o = {i for i, _ in pairs}
    for i, r in enumerate(O):
        if i in paired_o:
            continue
        best = max(((difflib.SequenceMatcher(None, key(r), key(q)).ratio(), j) for j, q in enumerate(N) if j not in used), default=(0, None))
        if best[0] >= 0.6:
            used.add(best[1]); pairs.append((i, best[1]))
    res = []
    paired_o = {i for i, _ in pairs}
    for i, r in enumerate(O):
        if i not in paired_o:
            res.append(dict(old_no=r["no"], new_no=None, title_old=r["title"], title_new="", status="only in old"))
    for j, q in enumerate(N):
        if j not in used:
            res.append(dict(old_no=None, new_no=q["no"], title_old="", title_new=q["title"], status="only in new"))
    for _ in [0]:
        for i, j in sorted(pairs):
            o, n = O[i], N[j]
            body_o, body_n = "\n".join(o["texts"]), "\n".join(n["texts"])
            txt = body_o != body_n
            note = o["notes"] != n["notes"]
            pic = o["pics"] != n["pics"]
            no, nn = nums(body_o), nums(body_n)
            ch = []
            for x, y in zip(o["texts"], n["texts"]):
                if x != y:
                    d = [w for w in difflib.ndiff(x.split(), y.split()) if w[:1] in "+-"]
                    ch.append(" ".join(d)[:600])
            res.append(dict(old_no=o["no"], new_no=n["no"], title_old=o["title"], title_new=n["title"],
                            text_changed=txt, numbers_changed=no != nn, notes_changed=note, figure_changed=pic,
                            n_pics=len(n["pics"]), changes=" || ".join(ch)[:3000],
                            status="same" if not (txt or note or pic) else "changed"))
    R = pd.DataFrame(res)
    R.to_csv(a.out, index=False, encoding="utf-8-sig")
    print(f"old {len(O)} slides, new {len(N)} slides")
    print(R.status.value_counts().to_string())
    if "figure_changed" in R:
        C = R[R.status == "changed"]
        print(f"changed: figure {int(C.figure_changed.sum())}, numbers in text {int(C.numbers_changed.sum())}, "
              f"text {int(C.text_changed.sum())}, notes {int(C.notes_changed.sum())}")


if __name__ == "__main__":
    main()
