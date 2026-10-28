"""audit_rules.py - check every slide of a deck against the PI rules (2026-10-22):
  title on top | one or two sentences at the bottom (what we did / found) | a figure on content slides | readable fonts (how much
  the figure is shrunk on the slide) | n animals stated (slide, or notes) | the statistical test stated (slide, or notes; 'descriptive'
  counts). Axis labels, legends and colour bars cannot be read from the pptx; they are checked in the figure code and the previews.

  python audit_rules.py <deck.pptx> [<deck2.pptx> ...] --out <report.md>
"""
import argparse
import io
import os
import re

from PIL import Image
from pptx import Presentation
from pptx.util import Emu

N_RE = re.compile(r"\bn\s*=\s*\d|\b\d+\s*(mice|mouse|neurons|neuron-sessions)\b|\b(2|two|4|four)\s*\+\s*(2|two)\b|마리|마우스\s*\d|뉴런\s*\d|n\s*=|not applicable|해당 없음|simulated data|모의 데이터", re.I)
T_RE = re.compile(r"Wilcoxon|Mann-Whitney|McNemar|Fisher|Mantel|CMH|Cochran|binomial|time-shift|shift test|max-T|GLM|Poisson|"
                  r"Spearman|rank|bootstrap|permutation|descriptive|기술 통계|검정|t-test|exact test|BH q|q <=|chance|decod|mixed model|혼합 모형|Friedman|no test|no statistical test|slope|기울기|"
                  r"literature|문헌|hypergeometric|rank-sum", re.I)


def texts(slide):
    out = []
    for sh in slide.shapes:
        if sh.has_text_frame and sh.text_frame.text.strip():
            sz = [r.font.size.pt for p in sh.text_frame.paragraphs for r in p.runs if r.font.size] + \
                 [p.font.size.pt for p in sh.text_frame.paragraphs if p.font.size]
            out.append((Emu(sh.top or 0).inches, max(sz) if sz else 0, sh.text_frame.text.strip()))
        if sh.has_table:
            out.append((Emu(sh.top or 0).inches, 12, " ".join(c.text for r in sh.table.rows for c in r.cells)))
    return sorted(out)


def scale_of(sh):
    """displayed width / native width of the picture (1 = figure fonts as drawn; 0.7 = 30 % smaller)"""
    try:
        im = Image.open(io.BytesIO(sh.image.blob))
        dpi = im.info.get("dpi", (100, 100))[0] or 100
        return Emu(sh.width).inches / (im.size[0] / float(dpi))
    except Exception:
        return None


def audit(path):
    prs = Presentation(path)
    rows = []
    for i, s in enumerate(prs.slides, 1):
        T = texts(s)
        pics = [sh for sh in s.shapes if sh.shape_type == 13]
        title = next((t for top, sz, t in T if top < 1.0 and sz >= 18), "")
        sec = next((t for top, sz, t in T if top < 0.3 and sz <= 12), "")
        bottom = " ".join(t for top, sz, t in T if top >= 5.6 and len(t) >= 25)
        onslide = " ".join(t for _, _, t in T)
        notes = s.notes_slide.notes_text_frame.text if s.has_notes_slide else ""
        sc = [scale_of(p) for p in pics]
        sc = [x for x in sc if x]
        cards = not pics and any(sh.shape_type == 1 for sh in s.shapes)        # summary-card slides (rectangles, no figure)
        r = dict(i=i, sec=sec[:40], title=title[:80], pics=len(pics), cards=cards,
                 has_title=bool(title), has_bottom=bool(bottom) or cards,
                 scale=min(sc) if sc else None,
                 n_where="slide" if N_RE.search(onslide) else "notes" if N_RE.search(notes) else "-",
                 t_where="slide" if T_RE.search(onslide) else "notes" if T_RE.search(notes) else "-")
        r["small"] = r["scale"] is not None and r["scale"] < 0.70
        rows.append(r)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("decks", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lang", default="en", choices=["en", "kr"])
    a = ap.parse_args(argv)
    L = []
    for d in a.decks:
        R = audit(d)
        n = len(R)
        fig = [r for r in R if r["pics"]]
        c = lambda f: sum(1 for r in R if f(r))
        L.append(f"## {os.path.basename(d)}: {n} slides\n")
        L.append("| check | pass | fail |")
        L.append("|---|---|---|")
        L.append(f"| title on top | {c(lambda r: r['has_title'])} | {c(lambda r: not r['has_title'])} |")
        L.append(f"| sentence(s) at the bottom | {c(lambda r: r['has_bottom'])} | {c(lambda r: not r['has_bottom'])} |")
        L.append(f"| readable figure (shrunk to >= 70 %) | {sum(1 for r in fig if not r['small'])} | {sum(1 for r in fig if r['small'])} |")
        L.append(f"| n animals on the slide / only in notes / missing | {c(lambda r: r['n_where'] == 'slide')} / "
                 f"{c(lambda r: r['n_where'] == 'notes')} | {c(lambda r: r['n_where'] == '-')} |")
        L.append(f"| test (or 'descriptive') on the slide / only in notes / missing | {c(lambda r: r['t_where'] == 'slide')} / "
                 f"{c(lambda r: r['t_where'] == 'notes')} | {c(lambda r: r['t_where'] == '-')} |\n")
        bad = [r for r in R if not r["has_title"] or not r["has_bottom"] or r["small"] or r["n_where"] == "-" or r["t_where"] == "-"]
        if bad:
            L.append("| # | section | title | problem |")
            L.append("|---|---|---|---|")
            for r in bad:
                pr = []
                if not r["has_title"]:
                    pr.append("no title")
                if not r["has_bottom"]:
                    pr.append("no bottom sentence")
                if r["small"]:
                    pr.append(f"figure shrunk to {100 * r['scale']:.0f} %")
                if r["n_where"] == "-":
                    pr.append("n not stated")
                if r["t_where"] == "-":
                    pr.append("test not stated")
                L.append(f"| {r['i']} | {r['sec']} | {r['title']} | {', '.join(pr)} |")
            L.append("")
    if a.lang == "kr":
        KR = {"| check | pass | fail |": "| 기준 | 통과 | 미달 |", "title on top": "위에 제목", "sentence(s) at the bottom": "아래 설명 문장",
              "readable figure (shrunk to >= 70 %)": "그림 글자 크기 (슬라이드에서 70 % 이상 유지)",
              "n animals on the slide / only in notes / missing": "동물 수 n: 슬라이드 / 노트에만 | 없음",
              "test (or 'descriptive') on the slide / only in notes / missing": "통계 검정 (또는 '기술 통계'): 슬라이드 / 노트에만 | 없음",
              "| # | section | title | problem |": "| # | 섹션 | 제목 | 문제 |", "no title": "제목 없음", "no bottom sentence": "아래 문장 없음",
              "figure shrunk to": "그림 축소", "n not stated": "n 없음", "test not stated": "검정 없음", " slides": "장"}
        txt = "\n".join(L)
        for k, v in KR.items():
            txt = txt.replace(k, v)
        open(a.out, "w", encoding="utf-8").write(txt)
    else:
        open(a.out, "w", encoding="utf-8").write("\n".join(L))
    print("->", a.out)


if __name__ == "__main__":
    main()
