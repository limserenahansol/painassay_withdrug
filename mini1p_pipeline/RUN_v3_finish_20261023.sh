#!/bin/bash
# RUN_v3_finish_20261023.sh - the four ALL-IN-ONE decks after the round-2 triple check, from NEW_STYLE_v3, into FINAL_20261023
# (ALL_IN_ONE keeps the 10-22 versions); rule audit; 100-dpi previews of both PI decks; review manifests (scratchpad\review3).
PIPE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_pipeline
V=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/lab_meeting/decks_Day1_Day2/v2_20261022
F=$V/FINAL_20261023
SP="C:\Users\hsollim\AppData\Local\Temp\claude\C--Users-hsollim\05a4ec72-aaab-4bd6-bf3c-2d10c312ad78\scratchpad"
w() { cygpath -w "$1"; }
export NEW_STYLE_DIR=NEW_STYLE_v3 OUT_DIR=FINAL_20261023
echo "start $(date '+%F %T')"
for lg in en kr; do
  echo "[$(date '+%T')] assemble $lg"
  python -I -X utf8 $PIPE/assemble_master.py --lang $lg 2>&1 | tail -5 | cut -c1-900
done
python -I -X utf8 $PIPE/audit_rules.py $(for f in ALL_PI_EN ALL_PI_KR ALL_FULL_EN ALL_FULL_KR; do echo "$(w $F/$f.pptx)"; done) \
  --out "$(w $F/EVALUATION_KR.md)" --lang kr
grep -A8 "^| 기준" $F/EVALUATION_KR.md | head -40
rm -rf "$SP\review3\prev_EN" "$SP\review3\prev_KR"
mkdir -p "$SP\review3"
python -I -X utf8 $PIPE/pptx_preview.py "$(w $F/ALL_PI_EN.pptx)" "$SP\review3\prev_EN" --dpi 100 > "$SP\review3\prev_EN.log" 2>&1 &
python -I -X utf8 $PIPE/pptx_preview.py "$(w $F/ALL_PI_KR.pptx)" "$SP\review3\prev_KR" --dpi 100 > "$SP\review3\prev_KR.log" 2>&1
wait
python -I -X utf8 "$SP\make_review_set3.py"
grep -h WARNING "$SP\review3\prev_EN.log" "$SP\review3\prev_KR.log" | head -30
ls -la $F/*.pptx
echo "all done $(date '+%F %T')"
