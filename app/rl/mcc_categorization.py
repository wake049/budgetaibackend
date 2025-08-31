from typing import Dict, Tuple, Optional, Callable, Any
import re

ALLOWED = {
    "Income", "Food", "Transportation", "Entertainment",
    "Shopping", "Utilities", "Housing", "Debt Payment", "Other"
}

# ---------- MCC -> your 9 categories ----------
# Keep this small & practical; expand as you see real data.
MCC_TO_CATEGORY: Dict[int, str] = {
    # Food & drink
    5411: "Food",   # Grocery Stores/Supermarkets
    5812: "Food",   # Restaurants
    5813: "Entertainment",  # Bars
    5814: "Food",   # Fast Food

    # Transportation (gas, rides, parking, transit)
    4111: "Transportation", # Commuter/Local transit
    4121: "Transportation", # Taxi/Limo
    4789: "Transportation", # Transportation Svcs
    5532: "Transportation", # Auto stores
    5533: "Transportation", # Auto parts
    5541: "Transportation", # Service stations
    5542: "Transportation", # Automated fuel
    7523: "Transportation", # Parking

    # Shopping (general retail)
    5300: "Shopping", # Wholesale clubs
    5311: "Shopping", # Department stores
    5399: "Shopping", # Misc general merch
    5651: "Shopping", # Clothing
    5661: "Shopping", # Shoes
    5699: "Shopping", # Apparel/misc
    5732: "Shopping", # Electronics
    5999: "Shopping", # Misc specialty retail

    # Utilities
    4812: "Utilities", # Telecom equip/phones
    4814: "Utilities", # Telecom services
    4900: "Utilities", # Utilities (electric/gas/water)

    # Housing (rent/real estate)
    6513: "Housing",  # Real estate agents/managers (rents)

    # Entertainment (media, theaters, recreation)
    5735: "Entertainment", # Record/video/streaming
    5942: "Entertainment", # Books
    7800: "Entertainment", # Amusement parks
    7832: "Entertainment", # Movie theaters
    7922: "Entertainment", # Theatrical/Orchestras
    7996: "Entertainment", # Amusement/game supplies
    7999: "Entertainment", # Recreation services

    # Travel → (you don’t have Travel; map to Transportation or Other)
    4411: "Transportation", # Cruise lines (closest fit)
    4511: "Transportation", # Airlines
    7011: "Other",          # Lodging/Hotels (no Housing—rent—so put to Other)
    7032: "Other",          # Camps

    # Financial institutions & fees → often Debt Payment or Other
    6011: "Other",          # ATM
    6012: "Debt Payment",   # Financial institutions – likely paydowns/transfers
    6300: "Other",          # Insurance (no Health/Fees category here)
    9399: "Other",          # Government services (fees/fines)
}

# Helpful broad ranges if no exact hit:
MCC_RANGES: Tuple[Tuple[int, int, str], ...] = (
    (4000, 4799, "Transportation"),  # transit/transport services
    (5000, 5599, "Shopping"),
    (5600, 5699, "Shopping"),
    (5700, 5799, "Shopping"),
    (5800, 5899, "Food"),
    (7800, 7999, "Entertainment"),
)

def _from_mcc(mcc: Optional[int]) -> Optional[str]:
    if not mcc:
        return None
    if mcc in MCC_TO_CATEGORY:
        return MCC_TO_CATEGORY[mcc]
    for a, b, cat in MCC_RANGES:
        if a <= mcc <= b:
            return cat
    return None

# ---------- Basic description cleaning (optional for ML) ----------
_PATTERNS = [
    (re.compile(r"\b(card|recurring)\s+purchase(s)?(\s+with\s+pin)?\b", re.I), " "),
    (re.compile(r"\bweb\s*id:\s*\S+\b", re.I), " "),
    (re.compile(r"\bppd\s*id:\s*\S+\b", re.I), " "),
    (re.compile(r"\bending\s+in\s+\d{3,4}\b", re.I), " "),
    (re.compile(r"\bcard\s*\d{3,4}\b", re.I), " "),
    (re.compile(r"\b\d{1,2}/\d{1,2}\b"), " "),
    (re.compile(r"[^a-z0-9\s]", re.I), " "),
    (re.compile(r"\s{2,}"), " "),
]
def clean_desc(s: str) -> str:
    s = (s or "").lower()
    for pat, repl in _PATTERNS:
        s = pat.sub(repl, s)
    return s.strip()

# ---------- ML adapter ----------
ML_PREDICTOR: Optional[Callable[[str], Tuple[str, float]]] = None
_SK_PIPE: Any = None

def set_predictor(fn: Callable[[str], Tuple[str, float]]):
    global ML_PREDICTOR, _SK_PIPE
    ML_PREDICTOR, _SK_PIPE = fn, None

def set_sklearn_pipeline(pipe: Any):
    global _SK_PIPE, ML_PREDICTOR
    _SK_PIPE, ML_PREDICTOR = pipe, None

def _sk_predict(desc: str) -> Tuple[str, float]:
    try:
        proba = _SK_PIPE.predict_proba([desc])[0]
        classes = list(_SK_PIPE.classes_)
        i = int(proba.argmax())
        return str(classes[i]), float(proba[i])
    except Exception:
        return str(_SK_PIPE.predict([desc])[0]), 0.70

def _predict(desc: str) -> Tuple[str, float, str]:
    if ML_PREDICTOR:
        lab, p = ML_PREDICTOR(desc); return lab, float(p), "ml_predict"
    if _SK_PIPE:
        lab, p = _sk_predict(desc);  return lab, float(p), "sklearn"
    return "Other", 0.0, "no-ml"

# ---------- Map any ML label → your 9 categories ----------
ML_TO_ALLOWED: Dict[str, str] = {
    # Common synonyms / Plaid-like labels → your set
    "FOOD_AND_DRINK": "Food",
    "GROCERIES": "Food",
    "RESTAURANT": "Food",
    "FAST_FOOD": "Food",

    "TRANSPORTATION": "Transportation",
    "RIDE_SHARING": "Transportation",
    "GAS": "Transportation",
    "AUTO_AND_TRANSPORT": "Transportation",

    "ENTERTAINMENT": "Entertainment",
    "STREAMING": "Entertainment",
    "MOVIES": "Entertainment",
    "GAMES": "Entertainment",

    "SHOPPING": "Shopping",
    "GENERAL_MERCHANDISE": "Shopping",
    "CLOTHING": "Shopping",
    "ELECTRONICS": "Shopping",

    "UTILITIES": "Utilities",
    "PHONE_INTERNET": "Utilities",
    "ELECTRICITY": "Utilities",
    "WATER": "Utilities",

    "HOUSING": "Housing",
    "RENT": "Housing",

    # Debt‑ish predictions → Debt Payment
    "DEBT": "Debt Payment",
    "LOAN_PAYMENT": "Debt Payment",
    "CREDIT_CARD_PAYMENT": "Debt Payment",

    # Income
    "INCOME": "Income",
    "PAYROLL": "Income",
    "DIRECT_DEPOSIT": "Income",
}

def normalize_to_allowed(label: str) -> str:
    if label in ALLOWED:
        return label
    up = (label or "").strip().upper().replace(" ", "_")
    return ML_TO_ALLOWED.get(up, "Other")

# ---------- Debt Payment heuristic (when MCC/ML are ambiguous) ----------
_DEBT_HINTS = (
    "payment", "pmt", "paydown", "credit card", "cc payment",
    "loan", "auto loan", "student loan", "capital one", "discover",
    "american express", "amex", "chase", "citibank", "wells fargo",
    "synchrony", "navient", "nelnet"
)

def maybe_debt_payment(desc_raw: str) -> bool:
    d = (desc_raw or "").lower()
    return any(h in d for h in _DEBT_HINTS)

# ---------- Main entry ----------
def auto_categorize(plaid_txn: dict,
                    *, use_cleaning: bool = True,
                    min_confidence: float = 0.70) -> Dict[str, object]:
    """
    Returns: {category, source, confidence, reason}
    Only categories from your 9-item set are returned.
    """
    mcc = plaid_txn.get("mcc")
    desc_raw = (
        plaid_txn.get("merchant_name")
        or plaid_txn.get("name")
        or plaid_txn.get("authorized_description")
        or ""
    )
    desc = clean_desc(desc_raw) if use_cleaning else (desc_raw or "")

    # 1) MCC first
    cat = _from_mcc(mcc)
    if cat:
        return {"category": cat, "source": "mcc", "confidence": 0.95,
                "reason": f"mcc {mcc} → {cat}"}

    # 2) Your ML on description
    label, prob, why = _predict(desc)
    cat = normalize_to_allowed(label)
    conf = float(prob or 0.0)

    # 2a) If ML is confident → use it
    if conf >= min_confidence and cat in ALLOWED:
        return {"category": cat, "source": "ml_description",
                "confidence": round(conf, 2),
                "reason": f"{why}; p={round(conf,3)}; desc='{desc_raw[:80]}'"}

    # 3) If ML is low‑confidence but description strongly hints at a payoff
    if maybe_debt_payment(desc_raw):
        return {"category": "Debt Payment", "source": "heuristic_debt_desc",
                "confidence": 0.72, "reason": "debt keywords in description"}

    # 4) Fallback
    return {"category": "Other", "source": "fallback",
            "confidence": round(conf, 2),
            "reason": f"{why}; low confidence or unmapped label='{label}'"}