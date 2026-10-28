#!/bin/bash
# RUN_day_v2_20261021.sh - whole-day re-detection of 5939, 1191, 5937 with the settings you confirmed on
# the 1-min previews (2026-10-21), then the per-mouse post-processing and the cohort analyses.
#   5939 and 1191 in parallel (2 mice at a time), 5937 after 1191; 5941 is not re-run.
# Logs: mini1p_out/20261019/logs_v2_20261021/
PIPE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_pipeline
OUT=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_out/20261019
MATLAB="/c/Program Files/MATLAB/R2024b/bin/matlab.exe"
L=$OUT/logs_v2_20261021; mkdir -p $L
w() { cygpath -w "$1"; }
detect() {
  echo "detect $1 start $(date '+%F %T')"
  "$MATLAB" -batch "DO = {'$1'}; run('$(w $PIPE/RUN_ca_day_v2_20261021.m)')" < /dev/null > $L/detect_$1.log 2>&1
  echo "detect $1 end $(date '+%F %T')"
}
post() {
  if [ -f $OUT/${1}_d1_v2_detect_done.flag ]; then
    bash $PIPE/ca_post_mouse.sh $1 > $L/post_$1.log 2>&1
    echo "post $1 end $(date '+%F %T')"
  else
    echo "*** $1: detection failed - see $L/detect_$1.log"
  fi
}
echo "#### day v2 start $(date '+%F %T') ####"
( detect 5939; post 5939 ) &
( detect 1191; post 1191; detect 5937; post 5937 ) &
wait
bash $PIPE/ca_post_cohort.sh > $L/cohort.log 2>&1
echo "#### day v2 done $(date '+%F %T') ####"
