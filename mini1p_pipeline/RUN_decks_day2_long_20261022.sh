#!/bin/bash
# RUN_decks_day2_long_20261022.sh [deck ...] - the Day-2 decks (PI, per mouse, FULL) and the Day 1+2 deck, EN + KR, with the
# Korean previews of these decks only (RUN_all_decks.sh re-renders every KR deck after each call). Missing Korean
# templates of each deck -> logs/missing_kr_<deck>.json. Arguments: which decks (default all four); "kr:<deck>" = only
# translate the existing EN deck again and re-render its Korean previews.
PIPE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_pipeline
B="C:/Users/hsollim/Documents/HEAL_mini1p_SBI553"
D=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/lab_meeting/decks_Day1_Day2
w() { cygpath -w "$1"; }
kr() {
  local name=$1
  python -I $PIPE/pptx_translate.py --in "$(w $D/EN/$name.pptx)" --out "$(w $D/KR/${name}_KR.pptx)" \
      --dump "$(w $D/logs/missing_kr_$name.json)" | head -1
  rm -f $D/KR/previews/${name}_KR_slide*.png
  python -I $PIPE/pptx_preview.py "$(w $D/KR/${name}_KR.pptx)" "$(w $D/KR/previews)" --dpi 60 > /dev/null 2>&1
}
build() {
  local day=$1 mod=$2 name=$3
  echo "[$(date '+%T')] $name"
  CA_DAY=$day python -I $PIPE/$mod.py --base "$B" --out "$(w $D/EN/$name.pptx)" > $D/logs/$name.log 2>&1 \
    || { echo "*** $name failed (see $D/logs/$name.log)"; return 1; }
  kr $name
}
ALL="Day2_PI_summary Day2_per_mouse Day2_FULL Day1+2_longitudinal"
for x in ${@:-$ALL}; do
  case $x in
    kr:*) echo "[$(date '+%T')] ${x#kr:} (Korean only)"; kr ${x#kr:} ;;
    Day1_PI_summary) build 1 make_day1_full_deck Day1_PI_summary ;;
    Day1_per_mouse) build 1 make_day1_per_mouse_deck Day1_per_mouse ;;
    Day1_FULL) build 1 make_day1_complete_deck Day1_FULL ;;
    Day2_PI_summary) build 2 make_day1_full_deck Day2_PI_summary ;;
    Day2_per_mouse) build 2 make_day1_per_mouse_deck Day2_per_mouse ;;
    Day2_FULL) build 2 make_day1_complete_deck Day2_FULL ;;
    Day1+2_longitudinal) build 2 make_longitudinal_deck Day1+2_longitudinal ;;
    *) echo "unknown deck $x" ;;
  esac
done
echo "all done $(date '+%F %T')"
