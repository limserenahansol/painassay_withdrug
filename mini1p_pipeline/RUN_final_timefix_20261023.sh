#!/bin/bash
# RUN_final_timefix_20261023.sh - the final decks after the timing fix and the 10-23 evening requests: event-analysis figures
# redrawn (figure-review fixes), split parts re-cut, tracked / by-condition figures, pipeline figures (new behaviour clips), every
# new-style deck into NEW_STYLE_v5 (v4 stays), then ALL_PI / ALL_FULL into FINAL_20261023_timefix (FINAL_20261023 stays), the
# rule audit, 100-dpi previews of both PI decks and the slide-by-slide comparison with the 10-23 morning decks.
PIPE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_pipeline
B="C:/Users/hsollim/Documents/HEAL_mini1p_SBI553"
A=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/analysis
V=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/lab_meeting/decks_Day1_Day2/v2_20261022
NS=$V/NEW_STYLE_v5
F=$V/FINAL_20261023_timefix
PC="$(cygpath -w $A/new_decks/psth_cache_win3_tf.npz)"
w() { cygpath -w "$1"; }
export SCORING_SUFFIX=_timefix
mkdir -p $NS/EN $NS/KR $NS/logs $F/previews
echo "start $(date '+%F %T')"
python -I $PIPE/ca_event_analysis.py --step plots > $NS/logs/ea_plots.log 2>&1 || echo "*** event plots failed"
python -I -X utf8 $PIPE/ca_p7_by_condition.py > $NS/logs/p7c.log 2>&1 || echo "*** P7c failed"
python -I -X utf8 $PIPE/ca_tracked_crossover.py > $NS/logs/tracked.log 2>&1 || echo "*** tracked failed"
python -I -X utf8 $PIPE/crop_split.py > $NS/logs/crop_split.log 2>&1 || echo "*** crop_split failed"
python -I -X utf8 $PIPE/make_pipeline_figs.py > $NS/logs/pipeline_figs.log 2>&1 || echo "*** pipeline figs failed"
echo "[$(date '+%T')] Day1+2_FULL"
CA_DAY=2 python -I $PIPE/make_new_deck.py --base "$B" --out "$(w $NS/EN/Day1+2_FULL.pptx)" --days 1 2 --lang en \
    --psth-cache "$PC" > $NS/logs/Day1+2_FULL.log 2>&1 || echo "*** Day1+2_FULL failed"
one_lang() {
  lg=$1; sub=EN; [ $lg = kr ] && sub=KR
  for spec in "Day1+2_FULL|1 2|" "Day1+2_PI|1 2|--short" "Day1_FULL|1|" "Day1_PI|1|--short" "Day2_FULL|2|" "Day2_PI|2|--short"; do
    IFS='|' read name days extra <<< "$spec"
    out=$name; [ $lg = kr ] && out=${name}_KR
    [ $lg = en ] && [ $name = Day1+2_FULL ] && continue
    echo "[$(date '+%T')] $out"
    CA_DAY=2 python -I $PIPE/make_new_deck.py --base "$B" --out "$(w $NS/$sub/$out.pptx)" --days $days --lang $lg \
        --psth-cache "$PC" $extra > $NS/logs/$out.log 2>&1 || echo "*** $out failed"
  done
  out=PerMouse_Day1+2; [ $lg = kr ] && out=PerMouse_Day1+2_KR
  CA_DAY=2 python -I $PIPE/make_per_mouse_v2.py --lang $lg --out "$(w $NS/$sub/$out.pptx)" > $NS/logs/$out.log 2>&1 || echo "*** $out failed"
  out=Sex_analysis; [ $lg = kr ] && out=Sex_analysis_KR
  CA_DAY=2 python -I $PIPE/make_sex_deck.py --lang $lg --out "$(w $NS/$sub/$out.pptx)" > $NS/logs/$out.log 2>&1 || echo "*** $out failed"
}
one_lang en > $NS/logs/_en.log 2>&1 &
one_lang kr > $NS/logs/_kr.log 2>&1
wait
cat $NS/logs/_en.log $NS/logs/_kr.log
grep -l "Traceback" $NS/logs/*.log 2>/dev/null && echo "*** tracebacks above"
echo "[$(date '+%T')] assemble"
export NEW_STYLE_DIR=NEW_STYLE_v5 OUT_DIR=FINAL_20261023_timefix
for lg in en kr; do
  python -I -X utf8 $PIPE/assemble_master.py --lang $lg 2>&1 | tail -4 | cut -c1-1200
done
python -I -X utf8 $PIPE/audit_rules.py $(for f in ALL_PI_EN ALL_PI_KR ALL_FULL_EN ALL_FULL_KR; do echo "$(w $F/$f.pptx)"; done) \
  --out "$(w $F/EVALUATION_KR.md)" --lang kr > $NS/logs/audit.log 2>&1 || echo "*** audit failed"
python -I -X utf8 $PIPE/pptx_preview.py "$(w $F/ALL_PI_EN.pptx)" "$(w $F/previews/PI_EN)" --dpi 100 > $NS/logs/prev_en.log 2>&1 &
python -I -X utf8 $PIPE/pptx_preview.py "$(w $F/ALL_PI_KR.pptx)" "$(w $F/previews/PI_KR)" --dpi 100 > $NS/logs/prev_kr.log 2>&1
wait
O=$V/FINAL_20261023
for k in PI_EN FULL_EN; do
  python -I -X utf8 $PIPE/deck_diff.py "$(w $O/ALL_$k.pptx)" "$(w $F/ALL_$k.pptx)" --out "$(w $F/DIFF_$k.csv)" > $NS/logs/diff_$k.log 2>&1
  cat $NS/logs/diff_$k.log
done
ls -la $F/*.pptx
echo "all done $(date '+%F %T')"
