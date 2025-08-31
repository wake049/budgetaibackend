from typing import List, Dict, Tuple
from datetime import datetime
import re
import fitz  # PyMuPDF

from app.utils.common_parsing import parse_amount, auto_categorize

# Date like 04/09 (Chase rows usually omit year; statement period gives year)
DATE_TOKEN = re.compile(r"\b(\d{2}/\d{2})\b")
# Money (require decimals or $/commas/() to avoid 04 / 05)
MONEY_TOKEN = re.compile(r"[-+()]?\s*\$?\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})")

# Words that indicate non-transaction summary lines
SKIP_KEYWORDS = (
    "beginning balance", "ending balance", "total", "summary",
    "fees charged", "overdraft protection", "daily balance"
)

MONTHS = {
    "january":1,"february":2,"march":3,"april":4,"may":5,"june":6,
    "july":7,"august":8,"september":9,"october":10,"november":11,"december":12
}
CHASE_PERIOD = re.compile(
    r'([A-Za-z]+)\s+(\d{2}),\s*(\d{4})\s*through\s*([A-Za-z]+)\s+(\d{2}),\s*(\d{4})',
    re.IGNORECASE
)

def _chase_year_map(full_text: str):
    m = CHASE_PERIOD.search(full_text)
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

def _infer_date(mmdd: str, ymap: Dict[str, str]) -> datetime:
    mm, dd = mmdd.split("/")
    yyyy = ymap.get(mm)
    if yyyy is None:
        yyyy = max(ymap.values()) if ymap else str(datetime.now().year)
    return datetime.strptime(f"{yyyy}-{mm}-{dd}", "%Y-%m-%d")

def _clean_spaces(s: str) -> str:
    # normalize weird spaces from PDFs
    return " ".join(s.replace("\u00a0", " ").split())

def _group_lines_by_y(spans: List[Tuple[float, float, str]], y_tol: float = 2.0):
    """
    spans: list of (y, x, text) -> returns list of lines, each line is list of (x, text) sorted by x.
    """
    rows: List[List[Tuple[float, str]]] = []
    spans_sorted = sorted(spans, key=lambda t: (round(t[0],1), t[1]))

    current_y = None
    current_row: List[Tuple[float, str]] = []
    for y, x, txt in spans_sorted:
        if current_y is None or abs(y - current_y) <= y_tol:
            current_y = y if current_y is None else min(current_y, y)
            current_row.append((x, txt))
        else:
            rows.append(sorted(current_row, key=lambda t: t[0]))
            current_row = [(x, txt)]
            current_y = y
    if current_row:
        rows.append(sorted(current_row, key=lambda t: t[0]))
    return rows

def parse_chase_transactions_layout(pdf_bytes: bytes, full_text: str) -> List[Dict]:
    """
    Robust to bad line wraps: reads coordinates and reconstructs rows.
    Strategy:
      - For each visual line, find a date token within the left 120px.
      - For amounts: use the rightmost 120px window to find money tokens.
      - If 2+ money tokens: second-last = amount, last = running balance.
      - If 1 token: it's the amount (no visible balance on that line).
      - Concatenate middle text as description; carry-over to next row if needed.
    """
    ymap_obj = _chase_year_map(full_text)
    if not ymap_obj:
        return []
    ymap = ymap_obj["year_by_month"]

    txs: List[Dict] = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    for page in doc:
        page_dict = page.get_text("dict")
        spans_with_pos: List[Tuple[float, float, str]] = []

        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    s = _clean_spaces(span.get("text", ""))
                    if not s:
                        continue
                    # y = baseline y, x = left
                    y = line["bbox"][1]
                    x = span["bbox"][0]
                    spans_with_pos.append((y, x, s))

        rows = _group_lines_by_y(spans_with_pos)

        for row in rows:
            # Build full line text with x ordering
            parts = [txt for _x, txt in row]
            full_line = _clean_spaces(" ".join(parts))
            if not full_line:
                continue

            # Find a date token near the left side
            left_text = _clean_spaces(" ".join(txt for x, txt in row if x <= (row[0][0] + 120)))
            date_m = DATE_TOKEN.search(left_text) or DATE_TOKEN.search(full_line)
            if not date_m:
                continue  # not a transaction row

            mmdd = date_m.group(1)

            # Right window for money
            max_x = max(x for x, _ in row)
            right_text = _clean_spaces(" ".join(txt for x, txt in row if max_x - x <= 120))
            # If right window is too small or empty, just use full line tail
            tail = right_text if right_text else full_line[-120:]

            # Strip date from the tail before money scan
            tail = DATE_TOKEN.sub("", tail)
            monies = [t.strip() for t in MONEY_TOKEN.findall(tail) if t.strip()]
            if not monies:
                # Try scanning entire line tail if right window failed
                tail2 = DATE_TOKEN.sub("", full_line[-160:])
                monies = [t.strip() for t in MONEY_TOKEN.findall(tail2) if t.strip()]
            if not monies:
                continue

            if len(monies) >= 2:
                amt_raw = monies[-2]
            else:
                amt_raw = monies[-1]

            amt = parse_amount(amt_raw)

            # Description = everything minus the left date chunk and the rightmost money tokens
            # Simple heuristic: take middle part
            # Remove the leading token containing the date
            # (Find the span that contained date_m and build desc from others)
            desc = full_line
            # prune obvious summary linesfrom typing import List, Dict, Tuple, Optional
from datetime import datetime
import re
import fitz  # PyMuPDF

from app.utils.transaction_utils import auto_categorize, parse_amount

# --- regex helpers ---
DATE_TOKEN  = re.compile(r"\b(\d{2}/\d{2})\b")
MONEY_TOKEN = re.compile(r"[-+()]?\s*\$?\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})")
NBSP = "\u00a0"

# --- period (month map supports short & long) ---
MONTHS = {
    "january":1,"february":2,"march":3,"april":4,"may":5,"june":6,
    "july":7,"august":8,"september":9,"october":10,"november":11,"december":12,
    "jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
    "jul":7,"aug":8,"sep":9,"sept":9,"oct":10,"nov":11,"dec":12,
}
CHASE_PERIOD = re.compile(
    r'\b([A-Za-z]{3,9})\s+(\d{1,2}),\s*(\d{4})\s*(?:through|to|–|-)\s*([A-Za-z]{3,9})\s+(\d{1,2}),\s*(\d{4})',
    re.IGNORECASE
)

def _year_map(full_text: str) -> Optional[Dict[str, str]]:
    m = CHASE_PERIOD.search(full_text.replace(NBSP, " "))
    if not m: return None
    sm, sd, sy, em, ed, ey = m.groups()
    sm_i = MONTHS.get(sm.lower()); em_i = MONTHS.get(em.lower())
    if not sm_i or not em_i: return None
    return {f"{sm_i:02d}": sy, f"{em_i:02d}": ey}

def _infer_date(mmdd: str, ymap: Dict[str, str]) -> datetime:
    mm, dd = mmdd.split("/")
    yyyy = ymap.get(mm) or max(ymap.values()) if ymap else str(datetime.now().year)
    return datetime.strptime(f"{yyyy}-{mm}-{dd}", "%Y-%m-%d")

def _clean(s: str) -> str:
    return " ".join(s.replace(NBSP, " ").split())

# --- column detection using header positions ---
def _find_column_x_ranges(page: fitz.Page) -> Optional[Dict[str, Tuple[float, float]]]:
    """
    Find x ranges for columns by locating spans with header words:
      DATE | DESCRIPTION | AMOUNT | BALANCE
    Returns dict of column -> (x_left, x_right)
    """
    pd = page.get_text("dict")
    headers = {}
    for block in pd.get("blocks", []):
        if block.get("type") != 0: continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                txt = _clean(span.get("text", "")).lower()
                if txt in ("date", "description", "amount", "balance"):
                    x0, y0, x1, y1 = span["bbox"]
                    headers[txt] = (x0, x1)

    if not all(k in headers for k in ("date", "description", "amount", "balance")):
        return None

    # Build boundaries halfway between adjacent headers
    # Ensure ascending by left edge
    ordered = sorted(headers.items(), key=lambda kv: kv[1][0])  # [(name,(x0,x1)),...]
    centers = [(name, (x0+x1)/2) for name, (x0,x1) in ordered]

    # boundaries between centers
    bounds = []
    for i in range(len(centers)-1):
        mid = (centers[i][1] + centers[i+1][1]) / 2
        bounds.append(mid)

    # Ranges:
    # DATE: (-inf, bounds[0])
    # DESCRIPTION: (bounds[0], bounds[1])
    # AMOUNT: (bounds[1], bounds[2])
    # BALANCE: (bounds[2], +inf)
    x_ranges = {
        centers[0][0]: (-float("inf"), bounds[0]),
        centers[1][0]: (bounds[0], bounds[1]),
        centers[2][0]: (bounds[1], bounds[2]),
        centers[3][0]: (bounds[2], float("inf")),
    }
    # Normalize keys to title-case expected externally
    return {
        "DATE": x_ranges.get("date") or x_ranges.get("Date"),
        "DESCRIPTION": x_ranges.get("description") or x_ranges.get("Description"),
        "AMOUNT": x_ranges.get("amount") or x_ranges.get("Amount"),
        "BALANCE": x_ranges.get("balance") or x_ranges.get("Balance"),
    }

def _group_rows(spans: List[Tuple[float, float, str]], y_tol: float = 2.0):
    """Group spans into visual rows by y, then sort each row by x."""
    rows = []
    spans_sorted = sorted(spans, key=lambda t: (round(t[0],1), t[1]))
    cur_y = None; cur = []
    for y, x, txt in spans_sorted:
        if cur_y is None or abs(y - cur_y) <= y_tol:
            cur_y = y if cur_y is None else min(cur_y, y)
            cur.append((x, txt))
        else:
            rows.append(sorted(cur, key=lambda t: t[0])); cur = [(x, txt)]; cur_y = y
    if cur: rows.append(sorted(cur, key=lambda t: t[0]))
    return rows

def parse_chase_transactions_amount_only(pdf_bytes: bytes, full_text: str) -> List[Dict]:
    """
    Extract ONLY the AMOUNT column values. BALANCE is ignored completely.
    We:
      1) Detect columns by header x positions.
      2) For each visual row with a DATE token in the DATE column, read:
         - description from DESCRIPTION column spans
         - amount from money tokens whose span-x falls in AMOUNT column range (rightmost token on that row)
      3) Keep the sign as printed; do NOT flip by category.
    """
    ymap = _year_map(full_text) or {}
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    txs: List[Dict] = []

    for page in doc:
        # 1) detect columns on this page
        xcols = _find_column_x_ranges(page)
        if not xcols:      # if header not present on this page, skip it
            continue
        x_date_l, x_date_r = xcols["DATE"]
        x_desc_l, x_desc_r = xcols["DESCRIPTION"]
        x_amt_l,  x_amt_r  = xcols["AMOUNT"]

        # 2) collect spans with coordinates
        pd = page.get_text("dict")
        spans = []
        for block in pd.get("blocks", []):
            if block.get("type") != 0: continue
            for line in block.get("lines", []):
                y = line["bbox"][1]
                for span in line.get("spans", []):
                    txt = _clean(span.get("text",""))
                    if not txt: continue
                    x0 = span["bbox"][0]
                    spans.append((y, x0, txt))

        # 3) group into rows
        rows = _group_rows(spans, y_tol=2.0)

        for row in rows:
            # split row spans by column via x ranges
            date_txts = [t for x,t in row if x_date_l <= x < x_date_r]
            desc_txts = [t for x,t in row if x_desc_l <= x < x_desc_r]
            amt_txts  = [t for x,t in row if x_amt_l  <= x < x_amt_r]

            # must have a date token in DATE col to be a transaction row
            date_blob = _clean(" ".join(date_txts))
            m = DATE_TOKEN.search(date_blob)
            if not m:
                continue
            mmdd = m.group(1)

            # description (for category/skips)
            desc = _clean(" ".join(desc_txts))

            # amount: find the rightmost money token inside the AMOUNT column only
            amt_candidates = list(MONEY_TOKEN.finditer(" ".join(amt_txts)))
            if not amt_candidates:
                # if nothing in AMOUNT col, skip this row (do NOT fall back to balance)
                continue
            amt_raw = amt_candidates[-1].group()
            try:
                amt = parse_amount(amt_raw)
            except Exception:
                continue

            # basic skip rules
            if not desc or not re.search(r"[A-Za-z]", desc):
                continue
            dl = desc.lower()
            if any(k in dl for k in (
                "beginning balance", "ending balance", "available balance", "balance summary",
                "total", "summary", "fees charged", "overdraft protection", "daily balance",
                "checking summary", "transaction detail", "date description amount balance",
                "deposits and additions", "atm & debit card withdrawals", "electronic withdrawals",
                "page of"
            )):
                continue

            txs.append({
                "date": _infer_date(mmdd, ymap),
                "description": desc,
                "amount": amt,                 # ← AMOUNT column only
                "category": auto_categorize(desc),
            })

    return txs
