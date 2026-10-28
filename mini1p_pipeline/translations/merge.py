"""merge.py - translations\\kr_part*.json -> translations_kr.json (checks that every placeholder is kept)

  python translations\\merge.py
"""
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
D, bad = {}, []
for p in sorted(glob.glob(os.path.join(HERE, "kr_part*.json"))):
    for k, v in json.load(open(p, encoding="utf-8")).items():
        if sorted(re.findall(r"\{\d+\}", k)) != sorted(re.findall(r"\{\d+\}", v)):
            bad.append(k)
            continue
        D[k] = v
json.dump(D, open(os.path.join(os.path.dirname(HERE), "translations_kr.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=0)
print(f"{len(D)} templates -> translations_kr.json; {len(bad)} with placeholder mismatches left out")
for k in bad:
    print("  mismatch:", k[:120])
