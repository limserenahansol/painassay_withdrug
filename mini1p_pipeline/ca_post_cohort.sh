#!/bin/bash
# ca_post_cohort.sh - after every mouse of day 1 is through ca_post_mouse.sh
#   1. ca_day1_analysis.py   baseline vs injection, whole session + event-locked, per mouse / arm
#   1b. ca_day1_extended.py  quality control, mixed model, encoding GLM, same cells, adaptation,
#      reliability, correlations, trajectories, spatial clustering
#   2. ca_behaviour_video.py again, now showing the cells with the largest Heat / Pin response
#   3. ca_toy_day2.py        TOY Day 2 (crossover) from each mouse's Day-1 baseline
#   4. ca_longitudinal.py    baseline (mean of 2 days) vs vehicle vs SBI-553, overall + event-locked
#   also: noise-z versions (analysis/*_znoise), ca_robustness.py (R1/R2), ca_response_shape.py (R3),
#         ca_rescue_check.py (5941 rescued cells vs the 5-min test)
PIPE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_pipeline
BASE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553
OUT=$BASE/mini1p_out/20261019
SCORE='C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\scoring_mini1p\20261019_final'
w() { cygpath -w "$1"; }
echo "#### cohort $(date '+%F %T') ####"
# WAIT_<mouse>.flag: a mouse is being (re)processed - wait for it, whatever the other switches say
while ls $OUT/WAIT_*.flag > /dev/null 2>&1; do sleep 30; done
# COHORT_HOLD.flag: wait for all 4 mice (run with CA_COHORT_RELEASE=1 when they are done)
[ -f $OUT/COHORT_HOLD.flag ] && [ -z "$CA_COHORT_RELEASE" ] && { echo "COHORT_HOLD.flag present - cohort waits for all 4 mice"; exit 0; }
# SELECTION_ONLY.flag: skipped, unless CA_FORCE_COHORT=1 (run on the mice whose selection you confirmed)
[ -f $OUT/SELECTION_ONLY.flag ] && [ -z "$CA_FORCE_COHORT" ] && { echo "SELECTION_ONLY.flag present - cohort analyses skipped"; exit 0; }
python -I $PIPE/ca_day1_analysis.py --out-root "$(w $OUT)" --scoring-dir "$SCORE" --out "$(w $BASE/analysis/ca_day1)" \
    || echo "*** day-1 analysis failed"
python -I $PIPE/ca_day1_analysis.py --out-root "$(w $OUT)" --scoring-dir "$SCORE" --out "$(w $BASE/analysis/ca_day1_bgcorr)"     --traces traces_bgcorr || echo "*** day-1 analysis (background-corrected traces) failed"
python -I $PIPE/ca_day1_extended.py --out-root "$(w $OUT)" --scoring-dir "$SCORE" --day1 "$(w $BASE/analysis/ca_day1)"     --out "$(w $BASE/analysis/ca_day1_extended)" || echo "*** day-1 extended analysis failed"
# the same with z = dF/F / the cell's noise SD over both sessions (one scale per cell): pre-vs-post SIZES
python -I $PIPE/ca_day1_analysis.py --out-root "$(w $OUT)" --scoring-dir "$SCORE" --out "$(w $BASE/analysis/ca_day1_znoise)" --z-norm noise     || echo "*** day-1 analysis (noise z) failed"
python -I $PIPE/ca_day1_extended.py --out-root "$(w $OUT)" --scoring-dir "$SCORE" --day1 "$(w $BASE/analysis/ca_day1_znoise)"     --out "$(w $BASE/analysis/ca_day1_extended_znoise)" --z-norm noise || echo "*** day-1 extended analysis (noise z) failed"
# robustness of the main result to each analysis choice, and the shape of the UP / DOWN responses
python -I $PIPE/ca_robustness.py --out-root "$(w $OUT)" --scoring-dir "$SCORE" --out "$(w $BASE/analysis/ca_robustness)"     || echo "*** robustness failed"
python -I $PIPE/ca_response_shape.py --out-root "$(w $OUT)" --scoring-dir "$SCORE" --day1 "$(w $BASE/analysis/ca_day1)"     --out "$(w $BASE/analysis/ca_response_shape)" || echo "*** response shape failed"
python -I $PIPE/ca_down_check.py --out-root "$(w $OUT)" --scoring-dir "$SCORE" --day1 "$(w $BASE/analysis/ca_day1)"     --out "$(w $BASE/analysis/ca_response_shape)" || echo "*** DOWN speed check failed"
# held-out responsive-cell change (R5), trial common mode (R6), population test (R7), calibrated responsive cells (R8)
for scr in ca_responsive_change ca_trial_common_mode ca_population_response_test ca_responsive_calibrated; do
  python -I $PIPE/$scr.py --out-root "$(w $OUT)" --scoring-dir "$SCORE" --out "$(w $BASE/analysis/ca_responsive_change)"       || echo "*** $scr failed"
done
# the same event analyses on the demixing-free ROI traces (5_analysis/traces_roi)
python -I $PIPE/ca_day1_analysis.py --out-root "$(w $OUT)" --scoring-dir "$SCORE" --out "$(w $BASE/analysis/ca_day1_roi)" --traces traces_roi --z-norm noise || echo "*** day-1 analysis (ROI traces) failed"
python -I $PIPE/ca_down_check.py --out-root "$(w $OUT)" --scoring-dir "$SCORE" --day1 "$(w $BASE/analysis/ca_day1_roi)" --out "$(w $BASE/analysis/ca_response_shape)" --traces traces_roi || echo "*** R4 (ROI traces) failed"
for scr in ca_population_response_test ca_responsive_calibrated; do
  python -I $PIPE/$scr.py --out-root "$(w $OUT)" --scoring-dir "$SCORE" --out "$(w $BASE/analysis/ca_responsive_change)" --traces traces_roi || echo "*** $scr (ROI traces) failed"
done
python -I $PIPE/ca_raw_roi_check.py --out-root "$(w $OUT)" --scoring-dir "$SCORE" --day1 "$(w $BASE/analysis/ca_day1)" --out "$(w $BASE/analysis/ca_response_shape)" || echo "*** R9 failed"
python -I $PIPE/ca_field_change.py --out-root "$(w $OUT)" --scoring-dir "$SCORE" --out "$(w $BASE/analysis/ca_response_shape)" || echo "*** R10 failed"
# which undecided cells move the main result most (GUI review order)
python -I $PIPE/ca_review_priority.py --base "$(w $BASE)" || echo "*** review priority failed"
# 5941: do the rescued cells carry the signal of the cells kept in the 5-min test?
python -I $PIPE/ca_rescue_check.py --day-root "$(w $OUT/5941_d1)"     --ref-mat "$(w $BASE/mini1p_out/test_5941_5min/4c_extract_seeded_loose/5941_d1_extract.mat)" --ref-name 5min_test     || echo "*** rescue check failed"
for m in 5941 5937 5939 1191; do
  for s in pre post; do
    [ -f $OUT/${m}_d1/5_analysis/traces/${m}_d1_${s}_traces.npz ] || continue
    python -I $PIPE/ca_behaviour_video.py --day-root "$(w $OUT/${m}_d1)" --rid ${m}_d1_$s --scoring-dir "$SCORE" \
        --out-dir "$(w $BASE/analysis/ca_videos)" --auto --dur 40 \
        --event-tests "$(w $BASE/analysis/ca_day1/ca_d1_cells_event_tests.csv)" || echo "*** ${m}_d1_$s: video failed"
  done
done
python -I $PIPE/ca_toy_day2.py --day1-root "$(w $OUT)" --scoring-dir "$SCORE" --out-root "$(w $BASE/mini1p_out/TOY_day2)" \
    || echo "*** toy day 2 failed"
python -I $PIPE/ca_longitudinal.py --day1-root "$(w $OUT)" --day1-scoring "$SCORE" \
    --day2-root "$(w $BASE/mini1p_out/TOY_day2)" --day2-scoring "$(w $BASE/mini1p_out/TOY_day2/scoring)" \
    --out "$(w $BASE/analysis/ca_longitudinal_TOY)" || echo "*** longitudinal failed"
python -I $PIPE/ca_main_findings.py --base "$(w $BASE)" || echo "*** main findings figure failed"
python -I $PIPE/ca_morning_summary.py --base "$(w $BASE)" --out "$(w $BASE/SUMMARY_20261021_KR.md)" || echo "*** summary failed"
echo "#### cohort done $(date '+%F %T') ####"
