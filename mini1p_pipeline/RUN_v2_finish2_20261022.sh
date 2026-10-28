#!/bin/bash
# RUN_v2_finish2_20261022.sh - the four ALL-IN-ONE decks again (FULL / PI x EN / KR) after the 10-22 evening fixes (legibility
# redraws, pipeline slides, n / test on every earlier slide, archive index, the best-window re-run decks when present), the rule
# audit (EVALUATION_KR.md) and previews of the PI decks (UTF-8 output: the Korean warnings stopped the KR previews before).
PIPE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_pipeline
V=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/lab_meeting/decks_Day1_Day2/v2_20261022
w() { cygpath -w "$1"; }
echo "start $(date '+%F %T')"
for lg in en kr; do
  echo "[$(date '+%T')] assemble $lg"
  python -I -X utf8 $PIPE/assemble_master.py --lang $lg 2>&1 | tail -4
done
python -I -X utf8 $PIPE/audit_rules.py $(for f in ALL_PI_EN ALL_PI_KR ALL_FULL_EN ALL_FULL_KR; do echo "$(w $V/ALL_IN_ONE/$f.pptx)"; done) \
  --out "$(w $V/ALL_IN_ONE/EVALUATION_KR.md)" --lang kr
mkdir -p $V/ALL_IN_ONE/previews
rm -f $V/ALL_IN_ONE/previews/ALL_PI_*_slide*.png
for f in ALL_PI_EN ALL_PI_KR; do
  python -I -X utf8 $PIPE/pptx_preview.py "$(w $V/ALL_IN_ONE/$f.pptx)" "$(w $V/ALL_IN_ONE/previews)" --dpi 60 > $V/ALL_IN_ONE/previews/$f.log 2>&1 \
    || echo "*** preview $f failed"
done
ls -la $V/ALL_IN_ONE/*.pptx
echo "all done $(date '+%F %T')"
