#!/bin/bash
# RUN_day2_robust_decks_20261022.sh - Day-2 R1/R2 robustness again (its first run was before 1191's ROI traces existed, so the
# ROI variant had 3 mice and no group value), the Day-2 main-findings figure, then the Day-2 and Day 1+2 decks (EN + KR)
PIPE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_pipeline
BASE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553
OUT=$BASE/mini1p_out/20261021
L=$OUT/logs
SCORE='C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\scoring_mini1p\20261021_final'
w() { cygpath -w "$1"; }
echo "start $(date '+%F %T')"
CA_DAY=2 python -I $PIPE/ca_robustness.py --out-root "$(w $OUT)" --scoring-dir "$SCORE" --out "$(w $BASE/analysis/ca_robustness_day2)" \
    > $L/robustness_day2_r2.log 2>&1 || echo "*** robustness failed"
CA_DAY=2 python -I $PIPE/ca_main_findings.py --base "$(w $BASE)" || echo "*** main findings failed"
echo "robustness done $(date '+%F %T')"
bash $PIPE/RUN_all_decks.sh day2 > $L/decks_day2_r2.log 2>&1
echo "day2 decks done $(date '+%F %T')"
bash $PIPE/RUN_all_decks.sh long > $L/decks_long_r2.log 2>&1
echo "all done $(date '+%F %T')"
