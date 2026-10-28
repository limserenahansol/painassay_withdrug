#!/bin/bash
# RUN_v2_finish3_20261022.sh - the four ALL-IN-ONE decks after the triple check (story order, per-part take-home lines, four-bar
# figures, pipeline fixes, the new video), the rule audit, and 100-dpi previews of both PI decks plus the review manifests.
PIPE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_pipeline
V=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/lab_meeting/decks_Day1_Day2/v2_20261022
SP="C:\Users\hsollim\AppData\Local\Temp\claude\C--Users-hsollim\05a4ec72-aaab-4bd6-bf3c-2d10c312ad78\scratchpad"
w() { cygpath -w "$1"; }
echo "start $(date '+%F %T')"
for lg in en kr; do
  echo "[$(date '+%T')] assemble $lg"
  python -I -X utf8 $PIPE/assemble_master.py --lang $lg 2>&1 | tail -4 | cut -c1-700
done
python -I -X utf8 $PIPE/audit_rules.py $(for f in ALL_PI_EN ALL_PI_KR ALL_FULL_EN ALL_FULL_KR; do echo "$(w $V/ALL_IN_ONE/$f.pptx)"; done) \
  --out "$(w $V/ALL_IN_ONE/EVALUATION_KR.md)" --lang kr
grep -A8 "^| 기준" $V/ALL_IN_ONE/EVALUATION_KR.md | head -40
rm -rf "$SP\review\prev_EN" "$SP\review\prev_KR"
python -I -X utf8 "$SP\make_review_set.py"
python -I -X utf8 $PIPE/pptx_preview.py "$(w $V/ALL_IN_ONE/ALL_PI_EN.pptx)" "$SP\review\prev_EN" --dpi 100 > "$SP\review\prev_EN.log" 2>&1 &
python -I -X utf8 $PIPE/pptx_preview.py "$(w $V/ALL_IN_ONE/ALL_PI_KR.pptx)" "$SP\review\prev_KR" --dpi 100 > "$SP\review\prev_KR.log" 2>&1
wait
grep -h WARNING "$SP\review\prev_EN.log" "$SP\review\prev_KR.log" | head -20
ls -la $V/ALL_IN_ONE/*.pptx
echo "all done $(date '+%F %T')"
