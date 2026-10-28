#!/bin/bash
# RUN_final_assemble_only_20261024.sh - re-assemble FINAL_20261023_timefix from NEW_STYLE_v5 (story / supplement fixes),
# then audit, PI previews and the comparison with the morning decks.
# rule audit, 100-dpi previews of both PI decks and the slide-by-slide comparison with the 10-23 morning decks.
PIPE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_pipeline
B="C:/Users/hsollim/Documents/HEAL_mini1p_SBI553"
A=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/analysis
V=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/lab_meeting/decks_Day1_Day2/v2_20261022
NS=$V/NEW_STYLE_v5
F=$V/FINAL_20261023_timefix
PC="$(cygpath -w $A/new_decks/psth_cache_win3_tf.npz)"
w() { cygpath -w "$1"; }
export SCORING_SUFFIX=_timefix
mkdir -p $F/previews
echo "[$(date '+%T')] assemble"
export NEW_STYLE_DIR=NEW_STYLE_v5 OUT_DIR=FINAL_20261023_timefix
for lg in en kr; do
  python -I -X utf8 $PIPE/assemble_master.py --lang $lg 2>&1 | tail -4 | cut -c1-1200
done
python -I -X utf8 $PIPE/audit_rules.py $(for f in ALL_PI_EN ALL_PI_KR ALL_FULL_EN ALL_FULL_KR; do echo "$(w $F/$f.pptx)"; done) \
  --out "$(w $F/EVALUATION_KR.md)" --lang kr > $NS/logs/audit.log 2>&1 || echo "*** audit failed"
python -I -X utf8 $PIPE/pptx_preview.py "$(w $F/ALL_PI_EN.pptx)" "$(w $F/previews/PI_EN)" --dpi 100 > $NS/logs/prev_en.log 2>&1 &
python -I -X utf8 $PIPE/pptx_preview.py "$(w $F/ALL_PI_KR.pptx)" "$(w $F/previews/PI_KR)" --dpi 100 > $NS/logs/prev_kr.log 2>&1
wait
O=$V/FINAL_20261023
for k in PI_EN FULL_EN; do
  python -I -X utf8 $PIPE/deck_diff.py "$(w $O/ALL_$k.pptx)" "$(w $F/ALL_$k.pptx)" --out "$(w $F/DIFF_$k.csv)" > $NS/logs/diff_$k.log 2>&1
  cat $NS/logs/diff_$k.log
done
ls -la $F/*.pptx
echo "all done $(date '+%F %T')"
