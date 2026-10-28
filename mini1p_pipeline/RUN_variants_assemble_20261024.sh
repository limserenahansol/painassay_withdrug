#!/bin/bash
# RUN_variants_assemble_20261024.sh - the final ALL_PI / ALL_FULL of the two window-mix variants (10-24), each with its own checked
# slide texts (assemble_master_mixnew.py / assemble_master_mixold.py, copies of assemble_master.py whose hand-typed numbers were
# re-verified on that variant's outputs):
#   PPT 1  FINAL_20261024_1_newmarks_mixwin   new (timing-corrected) marks, window per event
#   PPT 2  FINAL_20261024_2_oldmarks_mixwin   old marks, window per event (banner on every slide)
# For V2 the canonical analysis folders hold the old-marks outputs during its assembly, then V1 again.
PIPE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_pipeline
A=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/analysis
V=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/lab_meeting/decks_Day1_Day2/v2_20261022
VA=$A/_variants
SETS="event_analysis tracked_crossover sex_analysis pi_extra pipeline_figs behavior_crossover behavior_day1 behavior_day2"
w() { cygpath -w "$1"; }
restore() { for d in $SETS; do rm -rf $A/$d; cp -r $VA/$1/$d $A/$d; done; echo "[$(date '+%T')] restored <- $1"; }
export WIN_MODE=mix
one() {   # $1 variant, $2 NEW_STYLE dir, $3 output dir, $4 assemble script, $5 banner
  F=$V/$3; mkdir -p $F/previews
  export NEW_STYLE_DIR=$2 OUT_DIR=$3 DECK_BANNER="$5" VARIANT_KEY=$1
  for lg in en kr; do python -I -X utf8 $PIPE/$4 --lang $lg 2>&1 | grep -v "^ *story" | tail -2 | cut -c1-400; done
  python -I -X utf8 $PIPE/audit_rules.py $(for f in ALL_PI_EN ALL_PI_KR ALL_FULL_EN ALL_FULL_KR; do echo "$(w $F/$f.pptx)"; done) \
    --out "$(w $F/EVALUATION_KR.md)" --lang kr > /dev/null 2>&1 || echo "*** audit failed"
  python -I -X utf8 $PIPE/pptx_preview.py "$(w $F/ALL_PI_EN.pptx)" "$(w $F/previews/PI_EN)" --dpi 100 > /dev/null 2>&1
  python -I -X utf8 $PIPE/pptx_preview.py "$(w $F/ALL_PI_KR.pptx)" "$(w $F/previews/PI_KR)" --dpi 100 > /dev/null 2>&1
  python -I -X utf8 $PIPE/deck_diff.py "$(w $V/FINAL_20261023_timefix/ALL_PI_EN.pptx)" "$(w $F/ALL_PI_EN.pptx)" --out "$(w $F/DIFF_PI_EN.csv)" | tail -2
  echo "[$(date '+%T')] $3 done"
}
echo "start $(date '+%F %T')"
WHICH=${1:-both}                                    # new | old | both | mix4
if [ $WHICH = new ] || [ $WHICH = both ]; then
  python -I -X utf8 $PIPE/make_variant_assembler.py mix_new || { echo "*** assembler mix_new failed"; exit 1; }
  restore mix_new
  one mix_new NEW_STYLE_mixnew FINAL_20261024_1_newmarks_mixwin assemble_master_mixnew.py ""
fi
if [ $WHICH = mix4 ]; then                          # 10-25: marks and window chosen per event
  python -I -X utf8 $PIPE/make_variant_assembler.py mix4 || { echo "*** assembler mix4 failed"; exit 1; }
  restore mix4
  export SCORING_SUFFIX=_mix4
  one mix4 NEW_STYLE_mix4 FINAL_20261025_3_mixmarks_mixwin assemble_master_mix4.py       "Marks + window chosen per event (heat, withdrawal: original marks) - exploratory|사건별 표시·창 선택 (heat, withdrawal: 옛 표시) - 탐색적"
  unset SCORING_SUFFIX
  restore mix_new
  echo "all done $(date '+%F %T')"; exit 0
fi
if [ $WHICH = old ] || [ $WHICH = both ]; then
  python -I -X utf8 $PIPE/make_variant_assembler.py mix_old || { echo "*** assembler mix_old failed"; exit 1; }
  restore mix_old
  export SCORING_SUFFIX=""
  one mix_old NEW_STYLE_mixold FINAL_20261024_2_oldmarks_mixwin assemble_master_mixold.py       "OLD MARKS (before the 10-23 timing fix) - window per event|옛 표시 (10-23 타이밍 보정 전) - 사건별 창"
  unset SCORING_SUFFIX
  restore mix_new
fi
ls -la $V/FINAL_20261024_1_newmarks_mixwin/*.pptx $V/FINAL_20261024_2_oldmarks_mixwin/*.pptx 2>/dev/null
echo "all done $(date '+%F %T')"
