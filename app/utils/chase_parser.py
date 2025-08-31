from typing import List, Dict, Optional
from datetime import datetime
import re

from app.utils.transaction_utils import auto_categorize, parse_amount

CHASE_PERIOD = re.compile(
    r'\b([A-Za-z]{3,9})\s+(\d{1,2}),\s*(\d{4})\s*(?:through|to|–|-)\s*([A-Za-z]{3,9})\s+(\d{1,2}),\s*(\d{4})',
    re.IGNORECASE
)
CHASE_HEADER_MARKERS = [
    "JPMorgan Chase Bank", "Chase", "Total fees for this period"
]
CHASE_DATE_LINE = re.compile(r'^\s*(\d{2}/\d{2})\b')

MONTHS = {
    # full
    "january":1,"february":2,"march":3,"april":4,"may":5,"june":6,
    "july":7,"august":8,"september":9,"october":10,"november":11,"december":12,
    # abbrev
    "jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
    "jul":7,"aug":8,"sep":9,"sept":9,"oct":10,"nov":11,"dec":12,
}
MONEY_TOKEN = re.compile(r'[-+()]?\s*\$?\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})')
DATE_TOKEN = re.compile(r'\b\d{2}/\d{2}(?:/\d{2,4})?\b')


def is_chase_statement(text: str) -> bool:
    header = text[:4000]
    return any(m.lower() in header.lower() for m in CHASE_HEADER_MARKERS) and bool(CHASE_PERIOD.search(text))

def _chase_period_map(text: str) -> Optional[Dict]:
    m = CHASE_PERIOD.search(text)
    if not m:
        return None
    sm, sd, sy, em, ed, ey = m.groups()
    sm_i = MONTHS.get(sm.lower()); em_i = MONTHS.get(em.lower())
    if not sm_i or not em_i:
        return None
    return {
        "start": (int(sy), sm_i, int(sd)),
        "end": (int(ey), em_i, int(ed)),
        "year_by_month": {f"{sm_i:02d}": sy, f"{em_i:02d}": ey},
    }

def _infer_iso_date_mmdd(mmdd: str, ymap: Dict[str, str]) -> datetime:
    mm, dd = mmdd.split("/")
    yyyy = ymap.get(mm)
    if yyyy is None:
        yyyy = max(ymap.values()) if ymap else str(datetime.now().year)
    return datetime.strptime(f"{yyyy}-{mm}-{dd}", "%Y-%m-%d")

def _tail_amounts(block: str):
    """
    Return (amount, balance) from a full transaction block (date + wraps).

    Robust picking rule for Chase:
      - Strip dates so 04/09 doesn't look like money.
      - Consider ONLY the rightmost window (last 240 chars) which visually
        contains the amount & balance columns most of the time.
      - Find *all* money tokens there with positions (finditer).
      - Take the two tokens with the greatest .end() positions (the two rightmost).
      - amount  = the LEFT one of those two
        balance = the RIGHT one
      - If only one token exists in the window, treat it as the amount.
      - If none in the window, try the entire cleaned block once.
    """
    cleaned = DATE_TOKEN.sub("", block)
    window = cleaned[-240:] if len(cleaned) > 240 else cleaned

    # primary pass: right-edge window
    matches = list(MONEY_TOKEN.finditer(window))
    if not matches:
        # fallback: whole block (rarely needed)
        matches = list(MONEY_TOKEN.finditer(cleaned))
        if not matches:
            return None, None

    # pick two rightmost by end-index
    matches.sort(key=lambda m: m.end())
    if len(matches) == 1:
        amt_raw = matches[0].group()
        return parse_amount(amt_raw), None

    rightmost = matches[-1]
    second_rightmost = matches[-2]

    # amount is the left one of the two rightmost;
    # balance is the rightmost (Chase prints AMOUNT left of BALANCE)
    left = second_rightmost if second_rightmost.end() <= rightmost.end() else rightmost
    right = rightmost if rightmost.end() >= second_rightmost.end() else second_rightmost

    return parse_amount(left.group()), parse_amount(right.group())

SKIP_KEYWORDS = (
    "beginning balance", "ending balance", "available balance", "balance summary",
    "total", "summary", "fees charged", "overdraft protection", "daily balance",
    "checking summary", "transaction detail", "date description amount balance",
    "deposits and additions", "atm & debit card withdrawals", "electronic withdrawals",
    "page of"
)

def parse_chase_transactions(text: str) -> List[Dict]:
    print(f"🔍 Chase parser called")
    mapping = _chase_period_map(text)
    if not mapping:
        print(f"🔍 Chase period not found in text")
        raise ValueError("Chase period not found")
    print(f"🔍 Chase period mapping: {mapping}")
    ymap = mapping["year_by_month"]

    txs: List[Dict] = []
    cur = None
    line_count = 0

    for raw in text.splitlines():
        line = raw.rstrip().replace("\u00a0", " ")
        line_count += 1
        if not line:
            continue

        m = CHASE_DATE_LINE.match(line)
        if m:
            if line_count <= 50:  # Only debug first 50 lines
                print(f"🔍 Found date line {line_count}: {line[:100]}")
            mmdd = m.group(1)
            rest = line[m.end():].strip()

            if cur and cur.get("amount") is not None:
                txs.append(cur)
            cur = {"mmdd": mmdd, "desc": rest, "amount": None}

            # Use just the current description for the block on first line
            block_text = cur["desc"]
            amt, _bal = _tail_amounts(block_text)
            if amt is not None:
                if line_count <= 50:
                    print(f"🔍 Found amount on first line: {amt}")
                cur["amount"] = amt
                txs.append(cur)
                cur = None
        else:
            if cur:
                cur["desc"] += " " + line.strip()
                block_text = cur["desc"]
                amt, _bal = _tail_amounts(block_text)
                if amt is not None:
                    if line_count <= 50:
                        print(f"🔍 Found amount on continuation line: {amt}")
                    cur["amount"] = amt
                    txs.append(cur)
                    cur = None

    if cur and cur.get("amount") is not None:
        txs.append(cur)
    
    print(f"🔍 Raw transactions found: {len(txs)}")
    for i, tx in enumerate(txs[:5]):  # Show first 5
        print(f"🔍 Raw tx {i}: {tx}")

    normalized: List[Dict] = []
    for i, t in enumerate(txs):
        date_obj = _infer_iso_date_mmdd(t["mmdd"], ymap)
        desc = " ".join(t["desc"].split())
        amt = t["amount"]
        category = auto_categorize(desc)
        desc_l = desc.lower()
        
        # Debug income transactions specifically
        if category.lower() == "income":
            print(f"🔍 INCOME tx {i}: desc='{desc}', raw_amt={amt}, category={category}")
        
        if any(k in desc_l for k in SKIP_KEYWORDS):
            if i < 5:
                print(f"🔍 Skipping tx {i} due to keyword: {desc}")
            continue
        if not desc or desc == "-":
            if i < 5:
                print(f"🔍 Skipping tx {i} due to empty desc: {desc}")
            continue
            
        if i < 5:
            print(f"🔍 Normalized tx {i}: date={date_obj}, desc={desc}, amt={amt}, cat={category}")
        if not re.search(r"[A-Za-z]", desc):
            continue
            
        # Debug the final amount for income
        if category.lower() == "income":
            print(f"🔍 INCOME final: desc='{desc}', final_amt={amt}")
            
        normalized.append({
            "date": date_obj,
            "description": desc,
            "amount": amt,
            "category": category,
        })
    
    print(f"🔍 Final normalized transactions: {len(normalized)}")
    return normalized