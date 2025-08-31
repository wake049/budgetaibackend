# app/utils/chase_amount_layout.py
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import re
import fitz  # PyMuPDF

from app.utils.transaction_utils import parse_amount
from app.rl.auto_categorize import load_model, categorize_with_confidence, _clean_text
from app.rl.normalize_transaction import normalize_bank_memo



CONF_THRESHOLD = 0.35
_MODEL = None

def ensure_loaded():
    global _MODEL
    if _MODEL is None:
        _MODEL = load_model()
    return _MODEL
def ml_categorize(desc: str) -> str:
    if not desc:
        return "Uncategorized"
    model = ensure_loaded()
    core = normalize_bank_memo(desc)
    if not core:
        core = desc
    core = _clean_text(core)
    label, conf, _ = categorize_with_confidence(core, model, top_k=3)
    return label if conf >= CONF_THRESHOLD else "Uncategorized"


NBSP = "\u00a0"
DATE_TOKEN  = re.compile(r"\b(\d{2}/\d{2})\b")
MONEY_TOKEN = re.compile(r"[-+()]?\s*\$?\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})")

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

def _clean(s: str) -> str:
    return " ".join(s.replace(NBSP, " ").split())

def _year_map(full_text: str) -> Optional[Dict[str, str]]:
    m = CHASE_PERIOD.search(_clean(full_text))
    if not m: return None
    sm, sd, sy, em, ed, ey = m.groups()
    sm_i = MONTHS.get(sm.lower()); em_i = MONTHS.get(em.lower())
    if not sm_i or not em_i: return None
    return {f"{sm_i:02d}": sy, f"{em_i:02d}": ey}

def _infer_date(mmdd: str, ymap: Optional[Dict[str, str]]) -> datetime:
    mm, dd = mmdd.split("/")
    yyyy = (ymap or {}).get(mm) or (max((ymap or {}).values()) if ymap else str(datetime.now().year))
    return datetime.strptime(f"{yyyy}-{mm}-{dd}", "%Y-%m-%d")

def _group_rows(spans: List[Tuple[float, float, str]], y_tol: float = 2.0):
    rows, cur, cur_y = [], [], None
    spans_sorted = sorted(spans, key=lambda t: (round(t[0], 1), t[1]))
    for y, x, txt in spans_sorted:
        if cur_y is None or abs(y - cur_y) <= y_tol:
            cur_y = y if cur_y is None else min(cur_y, y)
            cur.append((x, txt))
        else:
            rows.append(sorted(cur, key=lambda t: t[0]))
            cur, cur_y = [(x, txt)], y
    if cur: rows.append(sorted(cur, key=lambda t: t[0]))
    return rows

def _find_column_x_ranges(page: fitz.Page):
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
    if not all(k in headers for k in ("date","description","amount","balance")):
        return None
    ordered = sorted(headers.items(), key=lambda kv: kv[1][0])   # by left x
    centers = [(name, (x0+x1)/2) for name,(x0,x1) in ordered]
    bounds = [ (centers[i][1]+centers[i+1][1])/2 for i in range(len(centers)-1) ]
    return {
        "DATE":        (-float("inf"), bounds[0]),
        "DESCRIPTION": (bounds[0], bounds[1]),
        "AMOUNT":      (bounds[1], bounds[2]),
        "BALANCE":     (bounds[2], float("inf")),
    }

def parse_chase_transactions_amount_only(pdf_bytes: bytes, full_text: str) -> List[Dict]:
    """
    Reads *only the AMOUNT column* using x-coordinates.
    BALANCE is ignored completely.
    """
    ymap = _year_map(full_text)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    txs: List[Dict] = []

    for page in doc:
        xcols = _find_column_x_ranges(page)
        if not xcols:
            continue
        x_date_l, x_date_r = xcols["DATE"]
        x_desc_l, x_desc_r = xcols["DESCRIPTION"]
        x_amt_l,  x_amt_r  = xcols["AMOUNT"]

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

        for row in _group_rows(spans):
            date_txts = [t for x,t in row if x_date_l <= x < x_date_r]
            desc_txts = [t for x,t in row if x_desc_l <= x < x_desc_r]
            amt_txts  = [t for x,t in row if x_amt_l  <= x < x_amt_r]

            date_blob = _clean(" ".join(date_txts))
            m = DATE_TOKEN.search(date_blob)
            if not m:
                continue
            mmdd = m.group(1)

            desc = _clean(" ".join(desc_txts))
            if not desc or not re.search(r"[A-Za-z]", desc):
                continue

            amt_matches = list(MONEY_TOKEN.finditer(" ".join(amt_txts)))
            if not amt_matches:
                continue  # Nothing in AMOUNT col; *do not* fall back to balance
            amt_raw = amt_matches[-1].group()
            try:
                amt = parse_amount(amt_raw)
            except Exception:
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
                "amount": amt,              # ← AMOUNT column only
                "category": ml_categorize(desc),
            })

    return txs
