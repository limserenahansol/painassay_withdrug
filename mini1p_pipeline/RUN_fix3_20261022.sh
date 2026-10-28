#!/bin/bash
# RUN_fix3_20261022.sh - 1191's Day-2 post again (the Day-1 post had relabelled every Day-1 decision 'user'; the Day-2
# decisions now read the frozen provenance analysis\crossday\<m>_day1_provenance.csv), in parallel with the 5939 -> 5937
# lane of RUN_fix2 that is still running; then the cohorts (D) and every deck (E).
PIPE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_pipeline
BASE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553
L=$BASE/mini1p_out/20261021/logs
w() { cygpath -w "$1"; }
echo "start $(date '+%F %T')"
bash $PIPE/ca_post_mouse_day2.sh 1191 > $L/fix_C_post_1191_d2.log 2>&1; echo "C: 1191 done $(date '+%T')"
while ps -ef | grep -v grep | grep -qE "ca_post_mouse_day2.sh (5939|5937)|RUN_fix2_20261022"; do sleep 20; done
grep -q "#### post 5937_d2 done" $L/fix_C_post_5937_d2.log || bash $PIPE/ca_post_mouse_day2.sh 5937 > $L/fix_C_post_5937_d2.log 2>&1
for m in 5941 5939 1191 5937; do grep -q "#### post ${m}_d2 done" $L/fix_C_post_${m}_d2.log && echo "C: $m ok" || echo "*** C: post $m did not finish"; done
echo "C done $(date '+%F %T')"
CA_COHORT_RELEASE=1 CA_FORCE_COHORT=1 bash $PIPE/ca_post_cohort.sh > $L/fix_D_cohort_day1.log 2>&1 &
CA_COHORT_RELEASE=1 CA_FORCE_COHORT=1 bash $PIPE/ca_post_cohort_day2.sh > $L/fix_D_cohort_day2.log 2>&1 &
wait
CA_DAY=2 python -I $PIPE/ca_main_findings.py --base "$(w $BASE)" > /dev/null 2>&1 || echo "*** main findings day 2 failed"
python -I $PIPE/ca_tracking_validation.py > $L/fix_D_validation.log 2>&1 || echo "*** tracking validation failed"
for m in 5941 5937 5939 1191; do python -I $PIPE/ca_confirm_sheet.py --mouse $m >> $L/fix_D_confirm.log 2>&1 || echo "*** confirm $m failed"; done
echo "D done $(date '+%F %T')"
bash $PIPE/RUN_decks_day2_long_20261022.sh Day2_PI_summary Day2_per_mouse Day2_FULL Day1+2_longitudinal \
    Day1_PI_summary Day1_per_mouse Day1_FULL > $L/fix_E_decks.log 2>&1
echo "all done $(date '+%F %T')"
