#!/bin/bash
# ca_post_cohort_day2.sh - DAY 2 (2026-10-21): the same cohort analyses as Day 1 (made from ca_post_cohort.sh)
# with CA_DAY=2 (crossover groups: 5937 / 1191 SBI-553, 5941 / 5939 vehicle), then the REAL longitudinal analysis
# (Day 1 + Day 2: baseline mean vs vehicle vs SBI-553, only neurons found on both days)
#   1. ca_day1_analysis.py   baseline vs injection, whole session + event-locked, per mouse / arm
#   1b. ca_day1_extended.py  quality control, mixed model, encoding GLM, same cells, adaptation,
#      reliability, correlations, trajectories, spatial clustering
#   2. ca_behaviour_video.py again, now showing the cells with the largest Heat / Pin response
#   also: noise-z versions (analysis/*_znoise), ca_robustness.py (R1/R2), ca_response_shape.py (R3),
PIPE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_pipeline
BASE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553
OUT=$BASE/mini1p_out/20261021
export CA_DAY=2
SCORE='C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\scoring_mini1p\20261021_final'
w() { cygpath -w "$1"; }
echo "#### cohort $(date '+%F %T') ####"
# WAIT_<mouse>.flag: a mouse is being (re)processed - wait for it, whatever the other switches say
while ls $OUT/WAIT_*.flag > /dev/null 2>&1; do sleep 30; done
# COHORT_HOLD.flag: wait for all 4 mice (run with CA_COHORT_RELEASE=1 when they are done)
[ -f $OUT/COHORT_HOLD.flag ] && [ -z "$CA_COHORT_RELEASE" ] && { echo "COHORT_HOLD.flag present - cohort waits for all 4 mice"; exit 0; }
# SELECTION_ONLY.flag: skipped, unless CA_FORCE_COHORT=1 (run on the mice whose selection you confirmed)
[ -f $OUT/SELECTION_ONLY.flag ] && [ -z "$CA_FORCE_COHORT" ] && { echo "SELECTION_ONLY.flag present - cohort analyses skipped"; exit 0; }
python -I $PIPE/ca_day1_analysis.py --out-root "$(w $OUT)" --scoring-dir "$SCORE" --out "$(w $BASE/analysis/ca_day2)" \
    || echo "*** day-1 analysis failed"
python -I $PIPE/ca_day1_analysis.py --out-root "$(w $OUT)" --scoring-dir "$SCORE" --out "$(w $BASE/analysis/ca_day2_bgcorr)"     --traces traces_bgcorr || echo "*** day-1 analysis (background-corrected traces) failed"
python -I $PIPE/ca_day1_extended.py --out-root "$(w $OUT)" --scoring-dir "$SCORE" --day1 "$(w $BASE/analysis/ca_day2)"     --out "$(w $BASE/analysis/ca_day2_extended)" || echo "*** day-1 extended analysis failed"
# the same with z = dF/F / the cell's noise SD over both sessions (one scale per cell): pre-vs-post SIZES
python -I $PIPE/ca_day1_analysis.py --out-root "$(w $OUT)" --scoring-dir "$SCORE" --out "$(w $BASE/analysis/ca_day2_znoise)" --z-norm noise     || echo "*** day-1 analysis (noise z) failed"
python -I $PIPE/ca_day1_extended.py --out-root "$(w $OUT)" --scoring-dir "$SCORE" --day1 "$(w $BASE/analysis/ca_day2_znoise)"     --out "$(w $BASE/analysis/ca_day2_extended_znoise)" --z-norm noise || echo "*** day-1 extended analysis (noise z) failed"
# robustness of the main result to each analysis choice, and the shape of the UP / DOWN responses
python -I $PIPE/ca_robustness.py --out-root "$(w $OUT)" --scoring-dir "$SCORE" --out "$(w $BASE/analysis/ca_robustness_day2)"     || echo "*** robustness failed"
python -I $PIPE/ca_response_shape.py --out-root "$(w $OUT)" --scoring-dir "$SCORE" --day1 "$(w $BASE/analysis/ca_day2)"     --out "$(w $BASE/analysis/ca_response_shape_day2)" || echo "*** response shape failed"
python -I $PIPE/ca_down_check.py --out-root "$(w $OUT)" --scoring-dir "$SCORE" --day1 "$(w $BASE/analysis/ca_day2)"     --out "$(w $BASE/analysis/ca_response_shape_day2)" || echo "*** DOWN speed check failed"
# held-out responsive-cell change (R5), trial common mode (R6), population test (R7), calibrated responsive cells (R8)
for scr in ca_responsive_change ca_trial_common_mode ca_population_response_test ca_responsive_calibrated; do
  python -I $PIPE/$scr.py --out-root "$(w $OUT)" --scoring-dir "$SCORE" --out "$(w $BASE/analysis/ca_responsive_change_day2)"       || echo "*** $scr failed"
done
# the same event analyses on the demixing-free ROI traces (5_analysis/traces_roi)
python -I $PIPE/ca_day1_analysis.py --out-root "$(w $OUT)" --scoring-dir "$SCORE" --out "$(w $BASE/analysis/ca_day2_roi)" --traces traces_roi --z-norm noise || echo "*** day-1 analysis (ROI traces) failed"
python -I $PIPE/ca_down_check.py --out-root "$(w $OUT)" --scoring-dir "$SCORE" --day1 "$(w $BASE/analysis/ca_day2_roi)" --out "$(w $BASE/analysis/ca_response_shape_day2)" --traces traces_roi || echo "*** R4 (ROI traces) failed"
for scr in ca_population_response_test ca_responsive_calibrated; do
  python -I $PIPE/$scr.py --out-root "$(w $OUT)" --scoring-dir "$SCORE" --out "$(w $BASE/analysis/ca_responsive_change_day2)" --traces traces_roi || echo "*** $scr (ROI traces) failed"
done
python -I $PIPE/ca_raw_roi_check.py --out-root "$(w $OUT)" --scoring-dir "$SCORE" --day1 "$(w $BASE/analysis/ca_day2)" --out "$(w $BASE/analysis/ca_response_shape_day2)" || echo "*** R9 failed"
python -I $PIPE/ca_field_change.py --out-root "$(w $OUT)" --scoring-dir "$SCORE" --out "$(w $BASE/analysis/ca_response_shape_day2)" || echo "*** R10 failed"
# which undecided cells move the main result most (GUI review order)
# 5941: do the rescued cells carry the signal of the cells kept in the 5-min test?
for m in 5941 5937 5939 1191; do
  for s in pre post; do
    [ -f $OUT/${m}_d2/5_analysis/traces/${m}_d2_${s}_traces.npz ] || continue
    python -I $PIPE/ca_behaviour_video.py --day-root "$(w $OUT/${m}_d2)" --rid ${m}_d2_$s --scoring-dir "$SCORE" \
        --out-dir "$(w $BASE/analysis/ca_videos_day2)" --auto --dur 40 \
        --event-tests "$(w $BASE/analysis/ca_day2/ca_d2_cells_event_tests.csv)" || echo "*** ${m}_d2_$s: video failed"
  done
done
# the Day-1 <-> Day-2 pairs of ca_day2_decisions.py (Day-2 evidence; the fingerprint matcher is circular with the
# Day-1 forced seeds), refreshed from the current review tables (your exclusions / unpairings included)
for m in 5941 5939 5937 1191; do
  [ -f $OUT/${m}_d2/4_cell_review/${m}_d2_cell_review.csv ] && python -I $PIPE/ca_day2_decisions.py --mouse $m --refresh-pairs
done
python -I $PIPE/ca_longitudinal.py --day1-root "$(w $BASE/mini1p_out/20261019)" --day1-scoring "C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\scoring_mini1p\20261019_final" \
    --day2-root "$(w $OUT)" --day2-scoring "$SCORE" --out "$(w $BASE/analysis/ca_longitudinal)" || echo "*** longitudinal failed"
echo "#### cohort done $(date '+%F %T') ####"
python -I $PIPE/ca_cell_event_matrix.py --base "$(w $BASE)" || echo "*** cell-event matrix failed"
