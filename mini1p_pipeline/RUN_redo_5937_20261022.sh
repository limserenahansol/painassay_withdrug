#!/bin/bash
# RUN_redo_5937_20261022.sh - 5937 Day 2 again with the corrected (sensor-pattern-proof) registration of the seeds.
# Waits until the 5939 detection is done (so that at most 2 heavy jobs run: this + MC 1191 or the 1191 detection),
# keeps the first run (_v1_sensorreg), re-registers the seeds, re-detects (pass 1 re-used), then the post-processing.
PIPE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_pipeline
OUT=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_out/20261021
MATLAB="/c/Program Files/MATLAB/R2024b/bin/matlab.exe"
L=$OUT/logs
w() { cygpath -w "$1"; }
until [ -f $OUT/5939_d2_detect_done.flag ] || [ -f $OUT/5939_d2_detect_failed.flag ]; do sleep 60; done
touch $OUT/WAIT_5937.flag      # the cohort waits for this mouse
echo "5937 redo start $(date '+%F %T')"
R=$OUT/5937_d2
for d in 3_extract 4_cell_review 5_analysis; do
  [ -d $R/$d ] && [ ! -d $R/${d}_v1_sensorreg ] && mv $R/$d $R/${d}_v1_sensorreg
done
mkdir -p $R/3_extract/pass1
rm -f $OUT/5937_d2_detect_done.flag $OUT/5937_d2_post_done.flag
for f in pairs pairs_all day1_not_found tracking tracking_map confirm_items; do
  for g in /c/Users/hsollim/Documents/HEAL_mini1p_SBI553/analysis/crossday/5937_${f}*; do [ -f "$g" ] && mv "$g" "${g%.*}_v1_sensorreg.${g##*.}"; done
done
touch $OUT/5937_d2_detect_running.flag
python -I $PIPE/ca_day2_seeds.py --mouse 5937 > $L/seeds_5937_d2_redo.log 2>&1
cat $L/seeds_5937_d2_redo.log
"$MATLAB" -batch "run('$(w $PIPE/RUN_ca_redo_5937_20261022.m)')" < /dev/null > $L/detect_5937_d2_redo.log 2>&1
grep -a -E "STOPPED|done in|candidates" $L/detect_5937_d2_redo.log | tail -3
if [ -f $OUT/5937_d2_detect_done.flag ]; then
  bash $PIPE/ca_post_mouse_day2.sh 5937 > $L/post_5937_d2_redo.log 2>&1
  echo "5937 redo post end $(date '+%F %T')"
fi
rm -f $OUT/WAIT_5937.flag
