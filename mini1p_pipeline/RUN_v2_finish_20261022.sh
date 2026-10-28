#!/bin/bash
# RUN_v2_finish_20261022.sh - after both build streams: the old-style KR decks again with the new translations, the four
# ALL-IN-ONE decks (FULL / PI x EN / KR), the rule audit (EVALUATION_KR.md) and previews of the PI decks.
PIPE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_pipeline
V=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/lab_meeting/decks_Day1_Day2/v2_20261022
w() { cygpath -w "$1"; }
echo "start $(date '+%F %T')"
python -I $PIPE/translations/merge.py | head -2
for name in Day1_PI_summary Day2_PI_summary Day1_per_mouse Day2_per_mouse Day1+2_longitudinal Day1_FULL Day2_FULL; do
  [ -f $V/OLD_STYLE/EN/$name.pptx ] || { echo "*** missing $name"; continue; }
  python -I $PIPE/pptx_translate.py --in "$(w $V/OLD_STYLE/EN/$name.pptx)" --out "$(w $V/OLD_STYLE/KR/${name}_KR.pptx)" \
    --dump "$(w $V/OLD_STYLE/logs/missing_kr_$name.json)" 2>&1 | head -1
done
for lg in en kr; do
  echo "[$(date '+%T')] assemble $lg"
  python -I $PIPE/assemble_master.py --lang $lg 2>&1 | tail -4
done
python -I $PIPE/audit_rules.py $(for f in ALL_PI_EN ALL_PI_KR ALL_FULL_EN ALL_FULL_KR; do echo "$(w $V/ALL_IN_ONE/$f.pptx)"; done) \
  --out "$(w $V/ALL_IN_ONE/EVALUATION_KR.md)" --lang kr
mkdir -p $V/ALL_IN_ONE/previews
for f in ALL_PI_EN ALL_PI_KR; do
  python -I $PIPE/pptx_preview.py "$(w $V/ALL_IN_ONE/$f.pptx)" "$(w $V/ALL_IN_ONE/previews)" --dpi 60 > /dev/null 2>&1
done
ls -la $V/ALL_IN_ONE/*.pptx
echo "all done $(date '+%F %T')"
