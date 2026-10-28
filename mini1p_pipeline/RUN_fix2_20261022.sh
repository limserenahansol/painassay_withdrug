#!/bin/bash
# RUN_fix2_20261022.sh - the rest of RUN_fix_20261022.sh after the Day-2 posts of 5939 / 1191 / 5937 stopped on a
# transient PNG write error (OSError Errno 22; figure saves now retry, safe_io.py). 5941's post was still running.
#   C  Day-2 post of 5939, 1191, 5937 (lane 1 waits for 5941's post to finish, max 2 at a time)
#   D  Day-1 cohort (1191 changed) + Day-2 cohort + longitudinal, main findings, tracking validation, confirm sheets
#   E  every deck, EN + KR
PIPE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_pipeline
BASE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553
L=$BASE/mini1p_out/20261021/logs
w() { cygpath -w "$1"; }
echo "start $(date '+%F %T')"
post2() { for m in "$@"; do bash $PIPE/ca_post_mouse_day2.sh $m > $L/fix_C_post_${m}_d2.log 2>&1; echo "C: $m done $(date '+%T')"; done; }
lane1() {
  while ps -ef | grep -v grep | grep -q "ca_post_mouse_day2.sh 5941"; do sleep 20; done
  echo "C: 5941 done (earlier run) $(date '+%T')"
  post2 1191
}
lane1 & post2 5939 5937 & wait
for m in 5941 5939 1191 5937; do grep -q "#### post ${m}_d2 done" $L/fix_C_post_${m}_d2.log || echo "*** C: post $m did not finish"; done
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
