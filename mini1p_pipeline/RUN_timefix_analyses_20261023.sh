#!/bin/bash
# RUN_timefix_analyses_20261023.sh - every analysis that uses the withdrawal / flinch / delivery times, again from the
# timing-corrected scoring copies (scoring_timefix.py; user 10-23). The current outputs are first copied to
# <folder>_before_timefix_20261023 (nothing is lost). Two streams at most (behaviour | event analysis -> validation -> sex).
PIPE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_pipeline
BASE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553
A=$BASE/analysis
L=$A/timefix_logs
w() { cygpath -w "$1"; }
mkdir -p $L
echo "start $(date '+%F %T')"
for d in event_analysis behavior_crossover behavior_day1 behavior_day2 sex_analysis; do
  [ -d $A/${d}_before_timefix_20261023 ] || cp -r $A/$d $A/${d}_before_timefix_20261023
done
export SCORING_SUFFIX=_timefix
behaviour() {
  python -I $PIPE/behavior_crossover_analysis.py > $L/behavior_crossover.log 2>&1 || echo "*** crossover failed"
  python -I $PIPE/behavior_day_analysis.py --final-dir "$(w $BASE/scoring_mini1p/20261019_final_timefix)" --day d1 \
      --out-dir "$(w $A/behavior_day1)" > $L/behavior_day1.log 2>&1 || echo "*** day1 failed"
  python -I $PIPE/behavior_day_analysis.py --final-dir "$(w $BASE/scoring_mini1p/20261021_final_timefix)" --day d2 \
      --out-dir "$(w $A/behavior_day2)" --arms 5941=vehicle,5939=vehicle,5937=SBI-553,1191=SBI-553 > $L/behavior_day2.log 2>&1 || echo "*** day2 failed"
  echo "[$(date '+%T')] behaviour done"
}
calcium() {
  python -I $PIPE/ca_event_analysis.py --step all > $L/event_analysis.log 2>&1 || echo "*** event analysis failed"
  echo "[$(date '+%T')] event analysis done"
  python -I $PIPE/ca_event_validate.py --step all > $L/event_validate.log 2>&1 || echo "*** validation failed"
  echo "[$(date '+%T')] validation done"
  python -I $PIPE/ca_event_p2_versions.py > $L/p2_versions.log 2>&1 || echo "*** P2 versions failed"
  python -I $PIPE/ca_sex_analysis.py > $L/sex_analysis.log 2>&1 || echo "*** sex analysis failed"
  echo "[$(date '+%T')] sex done"
}
behaviour > $L/_behaviour.log 2>&1 &
calcium > $L/_calcium.log 2>&1
wait
cat $L/_behaviour.log $L/_calcium.log
grep -l Traceback $L/*.log && echo "*** tracebacks above"
echo "all done $(date '+%F %T')"
