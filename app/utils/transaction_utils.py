# Shared utilities for transaction processing

import re


CATEGORY_KEYWORDS = {
    "Food": ["mcdonald", "burger", "pizza", "chipotle", "starbucks", "walmart", "kroger", "heb", "whole foods"],
    "Transportation": ["uber", "lyft", "shell", "chevron", "gas"],
    "Entertainment": ["netflix", "spotify", "hulu", "cinema", "theater"],
    "Shopping": ["amazon", "target", "best buy", "mall"],
    "Utilities": ["comcast", "spectrum", "at&t", "verizon", "t-mobile"],
    "Housing": ["property", "landlord", "rent"],
    "Income": ["payroll", "deposit", "payment from"],
    "Other": []
}


def parse_amount(raw: str) -> float:
    """
    Parses amounts like:
      $1,234.56   1234.56   -$45.00   (45.00)   - 5.41   + 12.00
    and returns a signed float. Whitespace is stripped aggressively.
    """
    s = str(raw)
    # Normalize spaces & symbols
    s = s.replace("\u00a0", " ")         # NBSP → space
    s = s.replace(",", "").replace("$", "")
    s = s.strip()
    s = re.sub(r"\s+", "", s)            # <<< remove ALL internal spaces

    neg = False
    # Parentheses indicate negative
    m = re.match(r"^\((.*)\)$", s)
    if m:
        neg = True
        s = m.group(1)

    # Leading sign
    if s.startswith("+"):
        s = s[1:]
    elif s.startswith("-"):
        neg = True
        s = s[1:]
    print(f"🔍 Parsed amount: {s}")
    # Guard: must still look like a number
    if not re.match(r"^\d+(?:\.\d+)?$", s):
        raise ValueError(f"Unrecognized amount format: {raw!r}")

    val = float(s)
    return -val if neg else val


def auto_categorize(description: str) -> str:
    """Auto-categorize a transaction based on description keywords"""
    desc_lower = description.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in desc_lower for keyword in keywords):
            return category
    return "Other"
