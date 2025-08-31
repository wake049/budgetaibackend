import re

_STATES = r"\b(AL|AK|AS|AZ|AR|CA|CO|CT|DC|DE|FL|GA|GU|HI|IA|ID|IL|IN|KS|KY|LA|MA|MD|ME|MI|MN|MO|MP|MS|MT|NC|ND|NE|NH|NJ|NM|NV|NY|OH|OK|OR|PA|PR|RI|SC|SD|TN|TX|UT|VA|VI|VT|WA|WI|WV|WY)\b"

_PATTERNS = [
    (re.compile(r"\b(card|recurring)\s+purchase(s)?(\s+with\s+pin)?\b", re.I), " "),
    (re.compile(r"\bweb\s*id:\s*\S+\b", re.I), " "),
    (re.compile(r"\bppd\s*id:\s*\S+\b", re.I), " "),
    (re.compile(r"\bending\s+in\s+\d{3,4}\b", re.I), " "),
    (re.compile(r"\bcard\s*\d{3,4}\b", re.I), " "),
    (re.compile(r"\b\d{2}/\d{2}(/\d{2,4})?\b"), " "),          # dates like 05/06 or 04/24/2025
    (re.compile(r"\b\d{3}[-\s]?\d{3}[-\s]?\d{4}\b"), " "),     # phone-like
    (re.compile(_STATES), " "),                                # state abbrev
    (re.compile(r"\bhttps?://\S+|\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b", re.I), " "),
    (re.compile(r"\bconroe\b|\bhouston\b|\baustin\b|\bdallas\b|\bsan\s+antonio\b", re.I), " "),
    (re.compile(r"[^a-z0-9\s\.\-&/]", re.I), " "),             # keep letters/digits and a few separators
]

def normalize_bank_memo(s: str) -> str:
    s = (s or "").lower()
    for rx, repl in _PATTERNS:
        s = rx.sub(repl, s)
    s = re.sub(r"\s+", " ", s).strip()
    # Collapse common Chase prefixes that still leak through
    s = s.replace("purchase ", " ").replace("recurring ", " ").replace("payment ", " ").strip()
    return s