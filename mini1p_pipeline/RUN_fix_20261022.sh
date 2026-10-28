#!/bin/bash
# RUN_fix_20261022.sh - your corrections of 2026-10-22 morning, end to end:
#   A  1191 Day 1: #1 + #2 = one neuron (merge_split_soma + ca_day1_merge_apply), then the Day-1 post of 1191
#   B  Day 2, 5939 / 5941 / 1191: the not-found Day-1 neurons put back with their Day-1 footprint (rescue_footprints,
#      joint least squares, new review numbers; ca_day2_recover_apply)
#   C  Day-2 post of all 4 mice (new rule for Day-1 rule-offs, forced pairs: recovered + your 1191 pair; the Day-2
#      behaviour with the logger re-triggers dropped and your grooming answers)
#   D  Day-1 cohort (1191 changed) and Day-2 cohort + longitudinal, the main-findings figures
#   E  every deck, EN + KR (Day 1, Day 2, Day 1+2)
# Max 2 heavy jobs at a time.
PIPE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_pipeline
BASE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553
L=$BASE/mini1p_out/20261021/logs
MATLAB="/c/Program Files/MATLAB/R2024b/bin/matlab.exe"
w() { cygpath -w "$1"; }
echo "start $(date '+%F %T')"
[ -f $BASE/SUMMARY_20261021_KR.md ] && cp $BASE/SUMMARY_20261021_KR.md $BASE/SUMMARY_20261021_KR_before_1022.md
jobA() {
  "$MATLAB" -batch "addpath('$(w $PIPE)'); merge_split_soma('$(w $BASE/mini1p_out/20261019/1191_d1/3_extract/1191_d1_extract.mat)', 1, 2);" \
      < /dev/null > $L/fix_A_merge.log 2>&1
  grep -q "merge_split_soma: done" $L/fix_A_merge.log || { echo "*** A: merge failed"; return 1; }
  python -I $PIPE/ca_day1_merge_apply.py --mouse 1191 --keep 1 --merge 2 >> $L/fix_A_merge.log 2>&1
  bash $PIPE/ca_post_mouse.sh 1191 > $L/fix_A_post_1191_d1.log 2>&1
  echo "A done $(date '+%F %T')"
}
jobB() {
  for m in 5939 5941 1191; do
    "$MATLAB" -batch "M='$m'; run('$(w $PIPE/RUN_recover_day2_20261022.m)')" < /dev/null > $L/fix_B_recover_$m.log 2>&1
    if grep -q "#### recover ${m}_d2 done" $L/fix_B_recover_$m.log; then
      python -I $PIPE/ca_day2_recover_apply.py --mouse $m >> $L/fix_B_recover_$m.log 2>&1 || echo "*** B: apply $m failed"
    else
      echo "*** B: recover $m failed"
    fi
  done
  echo "B done $(date '+%F %T')"
}
jobA & jobB & wait
echo "A + B done $(date '+%F %T')"
post2() { for m in "$@"; do bash $PIPE/ca_post_mouse_day2.sh $m > $L/fix_C_post_${m}_d2.log 2>&1; echo "C: $m done $(date '+%T')"; done; }
post2 5939 5941 & post2 1191 5937 & wait
echo "C done $(date '+%F %T')"
CA_COHORT_RELEASE=1 CA_FORCE_COHORT=1 bash $PIPE/ca_post_cohort.sh > $L/fix_D_cohort_day1.log 2>&1 &
CA_COHORT_RELEASE=1 CA_FORCE_COHORT=1 bash $PIPE/ca_post_cohort_day2.sh > $L/fix_D_cohort_day2.log 2>&1 &
wait
CA_DAY=2 python -I $PIPE/ca_main_findings.py --base "$(w $BASE)" > /dev/null 2>&1 || echo "*** main findings day 2 failed"
python -I $PIPE/ca_tracking_validation.py > $L/fix_D_validation.log 2>&1 || echo "*** tracking validation failed"
for m in 5941 5937 5939 1191; do python -I $PIPE/ca_confirm_sheet.py --mouse $m >> $L/fix_D_confirm.log 2>&1 || echo "*** confirm $m failed"; done
echo "D done $(date '+%F %T')"
bash $PIPE/RUN_decks_day2_long_20261022.sh Day1_PI_summary Day1_per_mouse Day1_FULL Day2_PI_summary Day2_per_mouse Day2_FULL \
    Day1+2_longitudinal > $L/fix_E_decks.log 2>&1
echo "all done $(date '+%F %T')"
