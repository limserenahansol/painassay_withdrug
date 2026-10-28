#!/bin/bash
# RUN_variant_mix4_20261025.sh - the third deck variant the user chose (10-25): marks AND window chosen per event (original or
# timing-corrected marks x 0-0.5 or 0-1 s, the combination with the most significant single-neuron + population tests; exploratory):
#   heat: original marks, 0-1 s | light, mild: corrected, 0-1 s | pin: corrected, 0-0.5 s | withdrawal: original, 0-1 s |
#   flinch: corrected, 0-1 s | movement, attending: 0-0.5 s | escape, grooming: 0-1 s
# scoring copies: scoring_mini1p\2026xxxx_final_mix4 (scoring_mix4.py); outputs -> analysis\_variants\mix4, NEW_STYLE_mix4;
# the canonical analysis folders hold mix_new again at the end. Final assembly: RUN_variants_assemble_20261024.sh mix4
PIPE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_pipeline
B="C:/Users/hsollim/Documents/HEAL_mini1p_SBI553"
A=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/analysis
V=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/lab_meeting/decks_Day1_Day2/v2_20261022
VA=$A/_variants
w() { cygpath -w "$1"; }
SETS="event_analysis tracked_crossover sex_analysis pi_extra pipeline_figs behavior_crossover behavior_day1 behavior_day2"
MIX4='{"Light touch":[0.5,0,0,1],"Mild touch":[0.5,0,0,1],"Pin prick":[0.5,0,0,0.5],"Heat":[0.5,0,0,1],"withdrawal":[1,0,0,1],"flinch":[1,0,0,1],"movement onset":[1,0,0,0.5],"escape/rearing onset":[1,0,0,1],"grooming onset":[1,0,0,1],"attending onset":[1,0,0,0.5]}'
archive() { mkdir -p $VA/$1; for d in $SETS; do rm -rf $VA/$1/$d; cp -r $A/$d $VA/$1/$d; done; echo "[$(date '+%T')] archived -> $1"; }
restore() { for d in $SETS; do rm -rf $A/$d; cp -r $VA/$1/$d $A/$d; done; echo "[$(date '+%T')] restored <- $1"; }
echo "start $(date '+%F %T')"
export WIN_MODE=mix SCORING_SUFFIX=_mix4 CA_WIN_SET="$MIX4"
CACHE=$A/new_decks/psth_cache_win3_mix4.npz
export PSTH_CACHE="$(w $CACHE)"
python -I $PIPE/scoring_mix4.py > $A/timefix_logs/mix4_scoring.log 2>&1 || { echo "*** scoring_mix4 failed"; exit 1; }
# the trace cache of the mixed marks (the analysis scripts read it before the decks are built)
python -I -X utf8 -c "import sys; sys.path.insert(0, r'$(w $PIPE)'); import make_new_deck as N; N.psth_all(r'$B', N.ALL_EV, r'$(w $CACHE)'); print('cache ok')" \
    > $A/timefix_logs/mix4_cache.log 2>&1 || { echo "*** cache failed"; exit 1; }
python -I $PIPE/behavior_crossover_analysis.py > $A/timefix_logs/mix4_crossover.log 2>&1 || echo "*** crossover failed"
python -I $PIPE/behavior_day_analysis.py --final-dir "$(w $B/scoring_mini1p/20261019_final_mix4)" --day d1 --out-dir "$(w $A/behavior_day1)" > /dev/null 2>&1
python -I $PIPE/behavior_day_analysis.py --final-dir "$(w $B/scoring_mini1p/20261021_final_mix4)" --day d2 --out-dir "$(w $A/behavior_day2)" \
    --arms 5941=vehicle,5939=vehicle,5937=SBI-553,1191=SBI-553 > /dev/null 2>&1
L=$A/timefix_logs/mix4; mkdir -p $L
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
python -I -X utf8 $PIPE/ca_session_blocks.py --rebuild > $L/ca_session_blocks.log 2>&1 || echo "*** session blocks failed"
python -I -X utf8 $PIPE/make_pipeline_figs.py > $L/pipeline_figs.log 2>&1 || echo "*** pipeline figs failed"
grep -l Traceback $L/*.log && echo "*** tracebacks in $L"
echo "[$(date '+%T')] analyses done"
NS=$V/NEW_STYLE_mix4; PC="$(w $CACHE)"; mkdir -p $NS/EN $NS/KR $NS/logs
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
echo "[$(date '+%T')] decks done"
archive mix4
restore mix_new
echo "all done $(date '+%F %T')"
