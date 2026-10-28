#!/bin/bash
# RUN_day2_roi_rerun_20261022.sh - the Day-2 cohort parts that use the demixing-free ROI traces (and R9), again after
# 1191's ROI traces were made (its first try failed on 2 frames the camera dropped; ca_roi_traces / ca_raw_roi_check fixed)
PIPE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_pipeline
BASE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553
OUT=$BASE/mini1p_out/20261021
SCORE='C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\scoring_mini1p\20261021_final'
export CA_DAY=2
w() { cygpath -w "$1"; }
python -I $PIPE/ca_day1_analysis.py --out-root "$(w $OUT)" --scoring-dir "$SCORE" --out "$(w $BASE/analysis/ca_day2_roi)" --traces traces_roi --z-norm noise || echo "*** ROI analysis failed"
python -I $PIPE/ca_down_check.py --out-root "$(w $OUT)" --scoring-dir "$SCORE" --day1 "$(w $BASE/analysis/ca_day2_roi)" --out "$(w $BASE/analysis/ca_response_shape_day2)" --traces traces_roi || echo "*** R4 ROI failed"
for scr in ca_population_response_test ca_responsive_calibrated; do
  python -I $PIPE/$scr.py --out-root "$(w $OUT)" --scoring-dir "$SCORE" --out "$(w $BASE/analysis/ca_responsive_change_day2)" --traces traces_roi || echo "*** $scr ROI failed"
done
python -I $PIPE/ca_raw_roi_check.py --out-root "$(w $OUT)" --scoring-dir "$SCORE" --day1 "$(w $BASE/analysis/ca_day2)" --out "$(w $BASE/analysis/ca_response_shape_day2)" || echo "*** R9 failed"
echo "roi rerun done $(date '+%F %T')"
