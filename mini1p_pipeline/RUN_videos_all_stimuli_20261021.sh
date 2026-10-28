#!/bin/bash
# RUN_videos_all_stimuli_20261021.sh - behaviour + Ca videos for every stimulus and for spontaneous
# movement (no-stimulus start), all 8 Day-1 sessions (Heat videos exist already). 2 at a time.
PIPE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_pipeline
BASE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553
OUT=$BASE/mini1p_out/20261019
SCORE='C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\scoring_mini1p\20261019_final'
w() { cygpath -w "$1"; }
one() {  # mouse session stimulus
  python -I $PIPE/ca_behaviour_video.py --day-root "$(w $OUT/${1}_d1)" --rid ${1}_d1_$2 --scoring-dir "$SCORE" \
      --out-dir "$(w $BASE/analysis/ca_videos)" --auto --dur 40 --stim "$3" \
      --event-tests "$(w $BASE/analysis/ca_day1/ca_d1_cells_event_tests.csv)" 2>&1 | grep -v Warning | tail -1 \
      || echo "*** ${1}_d1_$2 $3: video failed"
}
echo "#### videos start $(date '+%F %T') ####"
for st in "Light touch" "Mild touch" "Pin prick" "movement onset"; do
  ( for m in 5941 5939; do for s in pre post; do one $m $s "$st"; done; done ) &
  ( for m in 5937 1191; do for s in pre post; do one $m $s "$st"; done; done ) &
  wait
done
echo "#### videos done $(date '+%F %T') ####"
