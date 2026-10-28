"""make_variant_assembler.py - 10-24: the assembler of one window-mix deck variant = assemble_master.py + that variant's checked
text edits (analysis\\_variants\\<variant>\\text_edits.json, from the re-verification of every hand-typed number against the
variant's outputs). Writes assemble_master_<short>.py next to this file and a log of what was applied / skipped.

  python make_variant_assembler.py mix_new      -> assemble_master_mixnew.py
  python make_variant_assembler.py mix_old      -> assemble_master_mixold.py
"""
import json
import os
import py_compile
import sys

P = os.path.dirname(os.path.abspath(__file__))
VA = r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\analysis\_variants"


def main(variant):
    short = variant.replace("_", "")
    src = open(os.path.join(P, "assemble_master.py"), encoding="utf-8").read()
    E = json.load(open(os.path.join(VA, variant, "text_edits.json"), encoding="utf-8"))
    done, skipped = [], []
    for e in E:
        o, n = e["old"], e["new"]
        k = src.count(o)
        if k == 1:
            src = src.replace(o, n)
            done.append(e)
        elif k == 0 and src.count(n) >= 1:
            done.append({**e, "note": "already applied"})
        else:
            skipped.append({**e, "count": k})
    out = os.path.join(P, f"assemble_master_{short}.py")
    open(out, "w", encoding="utf-8", newline="\n").write(src)
    py_compile.compile(out, doraise=True)
    json.dump({"applied": done, "skipped": skipped}, open(os.path.join(VA, variant, "text_edits_applied.json"), "w",
                                                          encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{out}: {len(done)} edits applied, {len(skipped)} skipped")
    for s in skipped:
        print("  SKIPPED (found", s["count"], "times):", s["old"][:120])


if __name__ == "__main__":
    main(sys.argv[1])
