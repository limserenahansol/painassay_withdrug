#!/bin/bash
# RUN_bestwin_old_analyses_20261022.sh <day> - the earlier (old-style) event-locked analyses of one day again, now with the
# BEST window (CA_WIN=best: stimuli 0-1 s after vs the 0.5 s before, behaviours 0-0.5 s after vs the 1 s before; user 10-22:
# 'use this time window for ALL neuronal analysis'), into analysis/<folder>_bw. The old-window folders are not touched.
# Then the earlier Day-<day> deck from these folders (CA_AN_SUFFIX=_bw), EN + KR.
PIPE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_pipeline
BASE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553
V=$BASE/lab_meeting/decks_Day1_Day2/v2_20261022
w() { cygpath -w "$1"; }
d=$1
if [ "$d" = 1 ]; then
  OUT=$BASE/mini1p_out/20261019; SCORE='C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\scoring_mini1p\20261019_final'; SFX=""
else
  OUT=$BASE/mini1p_out/20261021; SCORE='C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\scoring_mini1p\20261021_final'; SFX="_day2"
fi
export CA_DAY=$d CA_WIN=best
A=$BASE/analysis
D1=$A/ca_day${d}_bw; EX=$A/ca_day${d}_extended_bw; RS=$A/ca_response_shape${SFX}_bw; RC=$A/ca_responsive_change${SFX}_bw
ROI=$A/ca_day${d}_roi_bw; RB=$A/ca_robustness${SFX}_bw
mkdir -p $V/OLD_STYLE_BW/EN $V/OLD_STYLE_BW/KR $V/OLD_STYLE_BW/logs
echo "#### Day $d best window $(date '+%F %T') ####"
python -I $PIPE/ca_day1_analysis.py --out-root "$(w $OUT)" --scoring-dir "$SCORE" --out "$(w $D1)" || echo "*** analysis failed"
python -I $PIPE/ca_day1_extended.py --out-root "$(w $OUT)" --scoring-dir "$SCORE" --day1 "$(w $D1)" --out "$(w $EX)" || echo "*** extended failed"
python -I $PIPE/ca_response_shape.py --out-root "$(w $OUT)" --scoring-dir "$SCORE" --day1 "$(w $D1)" --out "$(w $RS)" || echo "*** R3 failed"
python -I $PIPE/ca_down_check.py --out-root "$(w $OUT)" --scoring-dir "$SCORE" --day1 "$(w $D1)" --out "$(w $RS)" || echo "*** R4 failed"
for scr in ca_responsive_change ca_trial_common_mode ca_population_response_test ca_responsive_calibrated; do
  python -I $PIPE/$scr.py --out-root "$(w $OUT)" --scoring-dir "$SCORE" --out "$(w $RC)" || echo "*** $scr failed"
done
echo "[$(date '+%T')] ROI traces"
python -I $PIPE/ca_day1_analysis.py --out-root "$(w $OUT)" --scoring-dir "$SCORE" --out "$(w $ROI)" --traces traces_roi --z-norm noise || echo "*** ROI analysis failed"
python -I $PIPE/ca_down_check.py --out-root "$(w $OUT)" --scoring-dir "$SCORE" --day1 "$(w $ROI)" --out "$(w $RS)" --traces traces_roi || echo "*** R4 ROI failed"
for scr in ca_population_response_test ca_responsive_calibrated; do
  python -I $PIPE/$scr.py --out-root "$(w $OUT)" --scoring-dir "$SCORE" --out "$(w $RC)" --traces traces_roi || echo "*** $scr ROI failed"
done
python -I $PIPE/ca_raw_roi_check.py --out-root "$(w $OUT)" --scoring-dir "$SCORE" --day1 "$(w $D1)" --out "$(w $RS)" || echo "*** R9 failed"
python -I $PIPE/ca_field_change.py --out-root "$(w $OUT)" --scoring-dir "$SCORE" --out "$(w $RS)" || echo "*** R10 failed"
# R1 / R2 (whole-session rates under every choice) do not depend on the window: copied, so the summary figure can be redrawn
mkdir -p $RB && cp $A/ca_robustness${SFX}/R1_* $A/ca_robustness${SFX}/R2_* $RB/ 2>/dev/null
CA_AN_SUFFIX=_bw python -I $PIPE/ca_main_findings.py --base "$(w $BASE)" || echo "*** summary figure failed"
echo "[$(date '+%T')] deck"
name=Day${d}_FULL_bestwindow
CA_AN_SUFFIX=_bw python -I $PIPE/make_day1_complete_deck.py --base "C:/Users/hsollim/Documents/HEAL_mini1p_SBI553" \
  --out "$(w $V/OLD_STYLE_BW/EN/$name.pptx)" > $V/OLD_STYLE_BW/logs/$name.log 2>&1 || echo "*** deck failed"
python -I $PIPE/pptx_translate.py --in "$(w $V/OLD_STYLE_BW/EN/$name.pptx)" --out "$(w $V/OLD_STYLE_BW/KR/${name}_KR.pptx)" \
  --dump "$(w $V/OLD_STYLE_BW/logs/missing_kr_$name.json)" 2>&1 | head -1
echo "#### Day $d done $(date '+%F %T') ####"
