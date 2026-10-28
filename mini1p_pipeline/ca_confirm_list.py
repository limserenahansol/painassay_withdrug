"""ca_confirm_list.py - ONE folder with everything for your Day-2 confirmation, and ONE list to fill in.

  CONFIRM_DAY2_<date>\\
    CONFIRM_LIST_DAY2.xlsx      sheet README + one sheet per mouse: every item of the confirmation sheets, with the label
                                you see on the map (r160 = recovered Day-1 #160, d12 = left out by my Day-1 rule,
                                23 = Day-1 #23, n40 = new on Day 2), the page / position on <m>_CONFIRM_p<k>.png, the
                                current state (kept / deleted), the question, and an empty column 'your_answer'
    CONFIRM_LIST_<m>.csv        the same per mouse
    <m>_tracking_map.png, <m>_CONFIRM_p*.png, <m>_tracking_pairs_p*.png, <m>_new_candidates_p*.png,
    <m>_d2_maxproj_decisions_*.png (quadrants, for neurons nobody found), 5939_local_registration.png
    README_KR.md
Answers: keep / delete (or 유지 / 삭제); 'not same' (다른 뉴런) for a pair; empty = as it is now.

  python ca_confirm_list.py
"""
import datetime
import glob
import os
import shutil

import numpy as np
import pandas as pd

BASE = r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553"
MO = ["5941", "5939", "1191", "5937"]
PER_PAGE, NCOL = 24, 4
Q = {"RECOVERED": ("same neuron as Day 1? keep / delete", "Day 1과 같은 뉴런? 유지 / 삭제"),
     "D1 OFF": ("keep on Day 2? (add on Day 1 too?)", "Day 2에 유지? (Day 1에도 추가?)"),
     "CHECK": ("same neuron? keep / delete / not same", "같은 뉴런? 유지 / 삭제 / 다른 뉴런"),
     "DELETED": ("add back? keep / delete", "다시 넣을까? 유지 / 삭제"),
     "NOT FOUND": ("no Day-2 component here (information)", "Day 2 성분 없음 (참고)"),
     "NEW KEPT": ("new on Day 2: keep / delete", "Day 2 새 뉴런: 유지 / 삭제"),
     "NEW MAYBE": ("new on Day 2, dropped by the strict rule: add? keep / delete", "Day 2 새 후보 (엄격 규칙 탈락): 추가? 유지 / 삭제"),
     "PAIR": ("two overlapping neighbours, opposite traces: one neuron split? keep both / delete one", "겹치는 이웃 두 개, 반대 trace: 한 뉴런이 쪼개짐? 둘 다 유지 / 하나 삭제"),
     "SPLIT": ("bright body cut into deleted pieces: one neuron? add / leave", "밝은 몸체가 지워진 조각들: 한 뉴런? 추가 / 그대로"),
     "REDO": ("your Day-1 delete looks like a neuron on Day 2: still delete?", "Day 1에 지우신 것이 Day 2에서 뉴런처럼 보임: 계속 삭제?")}


def label(r):
    k, d1, d2 = r.kind, int(r.day1_id), int(r.day2_id)
    if k == "RECOVERED":
        return f"r{d1}"
    if k == "D1 OFF":
        return f"d{d1}"
    if k in ("NEW KEPT", "NEW MAYBE"):
        return f"n{d2}"
    if k == "NOT FOUND":
        return f"x{d1}"
    if k in ("REDO",) or d1 == 0:
        return f"Day-2 #{d2}"
    return str(d1)


def main():
    cd = os.path.join(BASE, "analysis", "crossday")
    od = os.path.join(BASE, f"CONFIRM_DAY2_{datetime.date.today():%Y%m%d}")
    os.makedirs(od, exist_ok=True)
    sheets = {}
    for m in MO:
        C = pd.read_csv(os.path.join(cd, f"{m}_confirm_items.csv"))
        R = pd.read_csv(os.path.join(BASE, "mini1p_out", "20261021", f"{m}_d2", "4_cell_review", f"{m}_d2_cell_review.csv")).set_index("id")
        rows = []
        for i, r in enumerate(C.itertuples()):
            pg, pos = i // PER_PAGE + 1, i % PER_PAGE
            st = ("no Day-2 component" if r.kind == "NOT FOUND" else
                  ("kept" if int(R.keep.get(int(r.day2_id), 0)) == 1 else "deleted"))
            rows.append(dict(mouse=m, item=i + 1, kind=r.kind, map_label=label(r), day1_id=int(r.day1_id) or "",
                             day2_id=int(r.day2_id) or "", your_yellow_mark=int(r.mark) if pd.notna(getattr(r, "mark", np.nan)) and r.mark else "",
                             now=st, sheet=f"{m}_CONFIRM_p{pg}.png", row_on_sheet=pos // NCOL + 1, column_on_sheet=pos % NCOL + 1,
                             question=Q.get(r.kind, ("", ""))[0], 질문=Q.get(r.kind, ("", ""))[1], why=r.why, your_answer=""))
        T = pd.DataFrame(rows)
        T.to_csv(os.path.join(od, f"CONFIRM_LIST_{m}.csv"), index=False, encoding="utf-8-sig")
        sheets[m] = T
        pats = [f"{m}_tracking_map.png", f"{m}_CONFIRM_p*.png", f"{m}_tracking_pairs_p*.png", f"{m}_new_candidates_p*.png"]
        for old in glob.glob(os.path.join(od, f"{m}_*.png")):
            os.remove(old)
        for pat in pats:
            for p in glob.glob(os.path.join(cd, pat)):
                if "_v1_" not in os.path.basename(p):          # first-run files kept for reference only
                    shutil.copy(p, od)
        for p in glob.glob(os.path.join(BASE, "mini1p_out", "20261021", f"{m}_d2", "5_analysis", f"{m}_d2_maxproj_decisions_*.png")):
            shutil.copy(p, od)
    p = os.path.join(cd, "5939_local_registration.png")
    if os.path.isfile(p):
        shutil.copy(p, od)
    readme = pd.DataFrame(dict(how_to=[
        "1. Open <mouse>_tracking_map.png (whole field) and <mouse>_CONFIRM_p*.png (each item, Day 1 left | Day 2 right).",
        "2. In the mouse's sheet of this file, write your_answer only where you disagree: keep / delete / not same (or 유지 / 삭제 / 다른 뉴런).",
        "   Empty = as it is now (column 'now'). map_label = the number on the map: r = recovered, d = left out by my Day-1 rule, n = new on Day 2.",
        "3. Neurons nobody found: write them as a new row (map_label = where, e.g. 'near 23'), or tell me in chat.",
        "4. Save the file; I apply it (ca_apply_confirmation.py) and re-run every analysis and deck without the PROVISIONAL mark.",
        "",
        "1. <마우스>_tracking_map.png (전체)와 <마우스>_CONFIRM_p*.png (항목별, 왼쪽 Day 1 | 오른쪽 Day 2)를 보세요.",
        "2. 이 파일의 마우스 시트에서 다르게 생각하는 항목만 your_answer에 유지 / 삭제 / 다른 뉴런 을 적으세요. 빈칸 = 지금 상태 그대로.",
        "   map_label = 지도 위 번호: r = 복구, d = 제 Day 1 규칙이 뺐던 것, n = Day 2 새 뉴런.",
        "3. 아무도 못 찾은 뉴런은 새 줄로 적거나 채팅으로 알려 주세요.",
        "4. 저장하면 제가 반영(ca_apply_confirmation.py)하고 모든 분석과 덱을 PROVISIONAL 없이 다시 만듭니다."]))
    with pd.ExcelWriter(os.path.join(od, "CONFIRM_LIST_DAY2.xlsx"), engine="openpyxl") as xw:
        readme.to_excel(xw, sheet_name="README", index=False)
        summ = pd.concat([T.groupby(["kind", "now"]).size().rename(m) for m, T in sheets.items()], axis=1).fillna(0).astype(int)
        summ.to_excel(xw, sheet_name="counts")
        for m, T in sheets.items():
            T.to_excel(xw, sheet_name=m, index=False)
            ws = xw.sheets[m]
            for col, wdt in zip("ABCDEFGHIJKLMNOP", (7, 5, 11, 11, 8, 8, 8, 18, 19, 7, 7, 46, 40, 70, 14)):
                ws.column_dimensions[col].width = wdt
            ws.freeze_panes = "A2"
        xw.sheets["README"].column_dimensions["A"].width = 150
    n = {m: len(T) for m, T in sheets.items()}
    print(f"-> {od}: CONFIRM_LIST_DAY2.xlsx ({', '.join(f'{m} {k}' for m, k in n.items())} items) + maps, sheets, quadrants")


if __name__ == "__main__":
    main()
