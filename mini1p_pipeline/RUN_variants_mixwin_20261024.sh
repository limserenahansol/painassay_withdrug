#!/bin/bash
# RUN_variants_mixwin_20261024.sh - the two deck variants the user asked for (10-24): a response window PER EVENT, 0-0.5 or 0-1 s,
# whichever gave the clearer vehicle vs SBI-553 difference in % UP (Fisher; no correction; stated on the slides):
#   V1 new (timing-corrected) marks   -> analysis\_variants\mix_new, NEW_STYLE_mixnew
#   V2 old (original) marks           -> analysis\_variants\mix_old, NEW_STYLE_mixold
# The state before (timing fix, window 0-0.5 s) is kept in analysis\_variants\timefix_w05; at the end the canonical analysis
# folders hold V1 again. Final assembly (with each variant's checked slide texts) is done afterwards by
# RUN_variants_assemble_20261024.sh.
PIPE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_pipeline
B="C:/Users/hsollim/Documents/HEAL_mini1p_SBI553"
A=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/analysis
V=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/lab_meeting/decks_Day1_Day2/v2_20261022
VA=$A/_variants
w() { cygpath -w "$1"; }
SETS="event_analysis tracked_crossover sex_analysis pi_extra pipeline_figs behavior_crossover behavior_day1 behavior_day2"
MIX_NEW='{"Light touch":[0.5,0,0,1],"Mild touch":[0.5,0,0,1],"Pin prick":[0.5,0,0,0.5],"Heat":[0.5,0,0,1],"withdrawal":[1,0,0,1],"flinch":[1,0,0,1],"movement onset":[1,0,0,1],"escape/rearing onset":[1,0,0,0.5],"grooming onset":[1,0,0,0.5],"attending onset":[1,0,0,0.5]}'
MIX_OLD='{"Light touch":[0.5,0,0,1],"Mild touch":[0.5,0,0,0.5],"Pin prick":[0.5,0,0,1],"Heat":[0.5,0,0,0.5],"withdrawal":[1,0,0,1],"flinch":[1,0,0,1],"movement onset":[1,0,0,1],"escape/rearing onset":[1,0,0,0.5],"grooming onset":[1,0,0,0.5],"attending onset":[1,0,0,0.5]}'
archive() { mkdir -p $VA/$1; for d in $SETS; do rm -rf $VA/$1/$d; cp -r $A/$d $VA/$1/$d; done; echo "[$(date '+%T')] archived -> $1"; }
restore() { for d in $SETS; do rm -rf $A/$d; cp -r $VA/$1/$d $A/$d; done; echo "[$(date '+%T')] restored <- $1"; }
analyses() {   # $1 = log dir, $2 = extra arg for the session script
  L=$1; mkdir -p $L
  python -I $PIPE/ca_event_analysis.py --step rest > $L/event_analysis.log 2>&1 || echo "*** event analysis failed"
  python -I -X utf8 $PIPE/ca_event_p2_versions.py > $L/p2_versions.log 2>&1 || echo "*** P2 versions failed"
  python -I $PIPE/ca_event_validate.py --step all > $L/validate.log 2>&1 || echo "*** validate failed"
  python -I $PIPE/ca_event_motion_check.py > $L/motion.log 2>&1 || echo "*** motion check failed"
  python -I $PIPE/ca_sex_analysis.py > $L/sex.log 2>&1 || echo "*** sex failed"
  python -I -X utf8 $PIPE/ca_p7_by_condition.py > $L/p7c.log 2>&1 || echo "*** P7c failed"
  python -I -X utf8 $PIPE/ca_tracked_crossover.py > $L/tracked.log 2>&1 || echo "*** tracked failed"
  python -I -X utf8 $PIPE/crop_split.py > $L/crop.log 2>&1 || echo "*** crop failed"
  for s in ca_flow_classes ca_pop_pooled ca_responsive_traces ca_space_stability ca_nature_style; do
    python -I -X utf8 $PIPE/$s.py > $L/$s.log 2>&1 || echo "*** $s failed"
  done
  python -I -X utf8 $PIPE/ca_session_blocks.py $2 > $L/ca_session_blocks.log 2>&1 || echo "*** session blocks failed"
  python -I -X utf8 $PIPE/make_pipeline_figs.py > $L/pipeline_figs.log 2>&1 || echo "*** pipeline figs failed"
  grep -l Traceback $L/*.log && echo "*** tracebacks in $L"
  echo "[$(date '+%T')] analyses done ($L)"
}
decks() {      # $1 = NEW_STYLE folder, $2 = trace cache
  NS=$V/$1; PC="$(w $2)"; mkdir -p $NS/EN $NS/KR $NS/logs
  CA_DAY=2 python -I $PIPE/make_new_deck.py --base "$B" --out "$(w $NS/EN/Day1+2_FULL.pptx)" --days 1 2 --lang en --psth-cache "$PC" \
      > $NS/logs/Day1+2_FULL.log 2>&1 || echo "*** Day1+2_FULL failed"
  one() {
    lg=$1; sub=EN; [ $lg = kr ] && sub=KR
    for spec in "Day1+2_FULL|1 2|" "Day1+2_PI|1 2|--short" "Day1_FULL|1|" "Day1_PI|1|--short" "Day2_FULL|2|" "Day2_PI|2|--short"; do
      IFS='|' read name days extra <<< "$spec"
      out=$name; [ $lg = kr ] && out=${name}_KR
      [ $lg = en ] && [ $name = Day1+2_FULL ] && continue
      CA_DAY=2 python -I $PIPE/make_new_deck.py --base "$B" --out "$(w $NS/$sub/$out.pptx)" --days $days --lang $lg --psth-cache "$PC" $extra \
          > $NS/logs/$out.log 2>&1 || echo "*** $out failed"
    done
    out=PerMouse_Day1+2; [ $lg = kr ] && out=PerMouse_Day1+2_KR
    CA_DAY=2 python -I $PIPE/make_per_mouse_v2.py --lang $lg --out "$(w $NS/$sub/$out.pptx)" > $NS/logs/$out.log 2>&1 || echo "*** $out failed"
    out=Sex_analysis; [ $lg = kr ] && out=Sex_analysis_KR
    CA_DAY=2 python -I $PIPE/make_sex_deck.py --lang $lg --out "$(w $NS/$sub/$out.pptx)" > $NS/logs/$out.log 2>&1 || echo "*** $out failed"
  }
  one en > $NS/logs/_en.log 2>&1 &
  one kr > $NS/logs/_kr.log 2>&1
  wait
  grep -l Traceback $NS/logs/*.log && echo "*** tracebacks in $NS"
  echo "[$(date '+%T')] decks done ($1)"
}
echo "start $(date '+%F %T')"
[ -d $VA/timefix_w05/event_analysis ] || archive timefix_w05
export WIN_MODE=mix
# ---------------- V1: new marks, mixed windows
echo "#### V1 new marks $(date '+%T')"
export SCORING_SUFFIX=_timefix CA_WIN_SET="$MIX_NEW"; unset PSTH_CACHE
python -I -X utf8 $PIPE/behavior_crossover_analysis.py > $A/timefix_logs/newmarks_crossover.log 2>&1 || echo "*** crossover failed"
python -I $PIPE/behavior_day_analysis.py --final-dir "$(w $B/scoring_mini1p/20261019_final_timefix)" --day d1 --out-dir "$(w $A/behavior_day1)" > /dev/null 2>&1
python -I $PIPE/behavior_day_analysis.py --final-dir "$(w $B/scoring_mini1p/20261021_final_timefix)" --day d2 --out-dir "$(w $A/behavior_day2)"     --arms 5941=vehicle,5939=vehicle,5937=SBI-553,1191=SBI-553 > /dev/null 2>&1
analyses $A/timefix_logs/mix_new ""
decks NEW_STYLE_mixnew $A/new_decks/psth_cache_win3_tf.npz
archive mix_new
# ---------------- V2: old marks, mixed windows
echo "#### V2 old marks $(date '+%T')"
export SCORING_SUFFIX="" CA_WIN_SET="$MIX_OLD" PSTH_CACHE="$(w $A/new_decks/psth_cache_win3.npz)"
python -I $PIPE/behavior_crossover_analysis.py > $A/timefix_logs/oldmarks_crossover.log 2>&1 || echo "*** crossover failed"
python -I $PIPE/behavior_day_analysis.py --final-dir "$(w $B/scoring_mini1p/20261019_final)" --day d1 --out-dir "$(w $A/behavior_day1)" > /dev/null 2>&1
python -I $PIPE/behavior_day_analysis.py --final-dir "$(w $B/scoring_mini1p/20261021_final)" --day d2 --out-dir "$(w $A/behavior_day2)" \
    --arms 5941=vehicle,5939=vehicle,5937=SBI-553,1191=SBI-553 > /dev/null 2>&1
cp $A/event_analysis_before_timefix_20261023/window_search.csv $A/event_analysis/window_search.csv
analyses $A/timefix_logs/mix_old "--rebuild"
decks NEW_STYLE_mixold $A/new_decks/psth_cache_win3.npz
archive mix_old
# ---------------- canonical = V1 again
restore mix_new
echo "all done $(date '+%F %T')"
