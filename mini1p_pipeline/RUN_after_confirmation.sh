#!/bin/bash
# RUN_after_confirmation.sh [--day1 <mice> --] <mice ...> - after your confirmation (2026-10-22 version):
#   1. python ca_apply_confirmation.py --confirm 5941 5939 1191 5937   (your answers in CONFIRM_DAY2_<date>\CONFIRM_LIST_DAY2.xlsx)
#      -> it prints the command for this script
#   2. this script: Day-1 post of the mice you changed on Day 1 (--day1 ... --), the Day-2 post of every mouse given (your
#      reviewed rows are kept; dedup; pairs refreshed; keep + merge, traces, figures, maps, videos) + the map redrawn from
#      the final table, the Day-1 / Day-2 cohorts + longitudinal, behaviour crossover, main findings, tracking validation,
#      confirmation sheets + list, every old-style deck and every new-style deck (EN + KR, PSTH cache remade).
# The PROVISIONAL mark goes away when all four mice are confirmed (ALL_CONFIRMED.flag).
PIPE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_pipeline
BASE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553
OUT=$BASE/mini1p_out/20261021
L=$OUT/logs
w() { cygpath -w "$1"; }
D1=""
if [ "$1" = "--day1" ]; then
  shift
  while [ -n "$1" ] && [ "$1" != "--" ]; do D1="$D1 $1"; shift; done
  shift
fi
st=$(date +%H%M)
for m in $D1; do
  echo "[$(date '+%T')] Day-1 post $m"
  bash $PIPE/ca_post_mouse.sh $m > $L/post_${m}_d1_rerun_$st.log 2>&1
done
for m in "$@"; do
  echo "[$(date '+%T')] Day-2 post $m"
  bash $PIPE/ca_post_mouse_day2.sh $m > $L/post_${m}_d2_rerun_$st.log 2>&1
  python -I $PIPE/ca_day2_decisions.py --mouse $m --replot >> $L/post_${m}_d2_rerun_$st.log 2>&1
done
n=$(ls $OUT/*_d2_CONFIRMED.flag 2> /dev/null | wc -l)
[ "$n" -eq 4 ] && touch $OUT/ALL_CONFIRMED.flag
echo "[$(date '+%T')] cohorts"
[ -n "$D1" ] && CA_COHORT_RELEASE=1 CA_FORCE_COHORT=1 bash $PIPE/ca_post_cohort.sh > $L/cohort_day1_rerun_$st.log 2>&1 &
CA_COHORT_RELEASE=1 CA_FORCE_COHORT=1 bash $PIPE/ca_post_cohort_day2.sh > $L/cohort_day2_rerun_$st.log 2>&1 &
wait
python -I $PIPE/behavior_crossover_analysis.py > /dev/null 2>&1
CA_DAY=2 python -I $PIPE/ca_main_findings.py --base "$(w $BASE)" > /dev/null 2>&1 || echo "*** main findings failed"
python -I $PIPE/ca_tracking_validation.py > $L/validation_rerun_$st.log 2>&1
for m in 5941 5937 5939 1191; do python -I $PIPE/ca_confirm_sheet.py --mouse $m > /dev/null 2>&1; done
python -I $PIPE/ca_confirm_list.py > /dev/null 2>&1
echo "[$(date '+%T')] decks"
DECKS="Day2_PI_summary Day2_per_mouse Day2_FULL Day1+2_longitudinal"
[ -n "$D1" ] && DECKS="$DECKS Day1_PI_summary Day1_per_mouse Day1_FULL"
bash $PIPE/RUN_decks_day2_long_20261022.sh $DECKS > $L/decks_rerun_$st.log 2>&1
bash $PIPE/RUN_new_decks_20261022.sh --fresh-cache > $L/new_decks_rerun_$st.log 2>&1
echo "[$(date '+%T')] done"
