"""deck_pi_extra.py - the slides of the 10-23 evening figure sets (analysis\\pi_extra\\<set>\\summary.json, one script per set) for
make_new_deck.py, Day 1 + Day 2 decks only. Added NEXT to the existing slides (nothing replaced).

Each summary.json lists its figures with title / key / bottom / notes (EN, KR), a section (Q1 behaviour, Q2 activity, Q3 population,
Q3 neurons, Q4 stimulus or movement, Q5 sex, methods, supplement) and a priority: 'main' (PI deck, in front), 'backup' (PI deck,
grey and hidden: the notes start with [BACKUP]) or 'full' (FULL deck only). The slide section label is 'X-<section>', which
assemble_master.STORY places inside that question (Q1-Q5 order kept).
"""
import json
import os

PX = r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\analysis\pi_extra"
SETS = ["session", "population", "responsive", "flow", "space", "nature_style"]       # order inside a section
SECTIONS = ["methods", "Q1 behaviour", "Q2 activity", "Q3 population", "Q3 neurons", "Q4 stimulus or movement", "Q5 sex", "supplement"]
BACKUP = "[BACKUP] "
# titles that would be caught by an assemble_master.TAKE regex (which rewrites title + key of the older slide): renamed here
TITLE_FIX = {"How many neurons respond to each stimulus, and what they look like":
             ("UP and DOWN neurons per stimulus, with their traces", "자극별 UP·DOWN 뉴런과 그 trace")}


def _norm_section(s):
    s0 = (s or "").strip().lower()
    for k in SECTIONS:
        if s0 == k.lower() or s0.startswith(k.lower().split()[0] + " " + (k.lower().split()[1] if len(k.split()) > 1 else "")):
            return k
    for k, keys in (("Q1 behaviour", ("q1", "behav")), ("Q2 activity", ("q2", "activity")), ("Q3 population", ("population",)),
                    ("Q3 neurons", ("q3", "neuron")), ("Q4 stimulus or movement", ("q4", "stimulus or")), ("Q5 sex", ("q5", "sex")),
                    ("methods", ("method",)), ("supplement", ("supp",))):
        if any(x in s0 for x in keys):
            return k
    return "supplement"


def _figures(J):
    """the figure entries of one summary.json, whatever its layout (list, {'figures': [...]}, or {file: {...}})"""
    if isinstance(J, list):
        items = J
    elif isinstance(J, dict) and isinstance(J.get("figures"), list):
        items = J["figures"]
    elif isinstance(J, dict) and isinstance(J.get("figures"), dict):
        items = [dict(v, file=k) if isinstance(v, dict) else {} for k, v in J["figures"].items()]
    elif isinstance(J, dict):
        items = [dict(v, file=v.get("file", k)) for k, v in J.items() if isinstance(v, dict) and ("title_en" in v or "key_en" in v)]
    else:
        items = []
    out = []
    for it in items:
        f = it.get("file") or it.get("figure") or it.get("png") or it.get("name")
        if not f or not it.get("title_en"):
            continue
        out.append(dict(it, file=os.path.basename(str(f))))
    return out


def collect():
    rows = []
    for si, st in enumerate(SETS):
        js = os.path.join(PX, st, "summary.json")
        if not os.path.isfile(js):
            continue
        try:
            J = json.load(open(js, encoding="utf-8"))
        except ValueError:
            continue
        for k, it in enumerate(_figures(J)):
            rows.append(dict(it, _set=st, _si=si, _k=k, _sec=_norm_section(it.get("section")),
                             _pri={"main": 0, "backup": 1, "full": 2}.get(str(it.get("priority", "backup")).lower(), 1)))
    return rows


def add_extra(dk, lg, short=False):
    """every figure set, by section, main before backup; returns the number of slides made"""
    L = lambda en, kr: kr if (lg == "kr" and kr) else en
    rows = collect()
    rows.sort(key=lambda r: (SECTIONS.index(r["_sec"]), r["_pri"], r["_si"], r["_k"]))
    n = 0
    for r in rows:
        if short and r["_pri"] == 2:
            continue
        fe = os.path.join(PX, r["_set"], r["file"])
        stem, ext = os.path.splitext(fe)
        fk = stem + "_kr" + ext
        fig = fk if (lg == "kr" and os.path.isfile(fk)) else fe
        if not os.path.isfile(fig):
            continue
        notes = L(r.get("notes_en", ""), r.get("notes_kr", ""))
        if r["_pri"] >= 1:
            notes = BACKUP + notes
        te, tk = TITLE_FIX.get(r["title_en"], (r["title_en"], r.get("title_kr")))
        dk.slide("X-" + r["_sec"], L(te, tk), L(r.get("key_en", ""), r.get("key_kr")), fig,
                 L(r.get("bottom_en", ""), r.get("bottom_kr")), notes)
        n += 1
    return n
