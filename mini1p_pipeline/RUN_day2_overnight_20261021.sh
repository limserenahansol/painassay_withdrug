#!/bin/bash
# RUN_day2_overnight_20261021.sh - Day 2 (2026-10-21) Ca, all 4 mice, overnight (your request 10-21 evening:
# "do all this ca data for whole pipeline and day1-2 tracking"). Four workers, handing the mice over by flags:
#   copy    (MATLAB)  lossless .isxd -> h5, verified          RUN_ca_overnight_20261021.m WORKER='copy'
#   mc      (MATLAB)  motion correction, QC, raw deleted      RUN_ca_overnight_20261021.m WORKER='mc'
#   detect  (bash)    Day-1 neurons -> Day-2 seeds (ca_day2_seeds.py), then EXTRACT + review (MATLAB, one mouse)
#   post    (bash)    ca_post_mouse_day2.sh: Day-1 decisions + cross-day tracking, traces, figures, maps, videos
# then the Day-2 cohort analyses + the real longitudinal analysis (ca_post_cohort_day2.sh).
# Selections are PROVISIONAL (<tag>_PROVISIONAL.flag): run through, labelled, waiting for your confirmation.
# Order 5941, 5937, 5939, 1191. Logs: mini1p_out/20261021/logs/
PIPE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_pipeline
OUT=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_out/20261021
MATLAB="/c/Program Files/MATLAB/R2024b/bin/matlab.exe"
L=$OUT/logs; mkdir -p $L
w() { cygpath -w "$1"; }
RUNM=$(w $PIPE/RUN_ca_overnight_20261021.m)
MICE="5941 5937 5939 1191"
failed() { ls $OUT/${1}_*_failed.flag > /dev/null 2>&1; }
echo "#### day 2 overnight start $(date '+%F %T') ####"
"$MATLAB" -batch "WORKER='copy'; run('$RUNM')" < /dev/null > $L/overnight_copy.log 2>&1 &
sleep 20
"$MATLAB" -batch "WORKER='mc'; run('$RUNM')" < /dev/null > $L/overnight_mc.log 2>&1 &
(
  for m in $MICE; do
    tag=${m}_d2
    until [ -f $OUT/${tag}_mc_ready.flag ]; do
      failed $tag && { echo "*** $tag: copy / mc failed - detection skipped"; continue 2; }
      sleep 60
    done
    echo "detect $tag start $(date '+%F %T')"
    python -I $PIPE/ca_day2_seeds.py --mouse $m > $L/seeds_$tag.log 2>&1 || echo "*** $tag: Day-1 seeds failed (automatic seeds only)"
    "$MATLAB" -batch "WORKER='detect'; M='$m'; run('$RUNM')" < /dev/null > $L/detect_$tag.log 2>&1
    echo "detect $tag end $(date '+%F %T')"
  done
) > $L/overnight_detect.log 2>&1 &
(
  for m in $MICE; do
    tag=${m}_d2
    until [ -f $OUT/${tag}_detect_done.flag ]; do
      failed $tag && { echo "*** $tag: an earlier step failed - post skipped"; continue 2; }
      sleep 60
    done
    echo "post $tag start $(date '+%F %T')"
    bash $PIPE/ca_post_mouse_day2.sh $m > $L/post_$tag.log 2>&1
    echo "post $tag end $(date '+%F %T')"
  done
  CA_FORCE_COHORT=1 bash $PIPE/ca_post_cohort_day2.sh > $L/cohort_day2.log 2>&1
  echo "cohort end $(date '+%F %T')"
) > $L/overnight_post.log 2>&1 &
wait
echo "#### day 2 overnight done $(date '+%F %T') ####"
