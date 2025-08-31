from fastapi import APIRouter, UploadFile, File, HTTPException, Request, Form, Depends
from typing import List, Dict
from datetime import datetime
from io import BytesIO
import re

import fitz  # PyMuPDF
from sqlalchemy.orm import Session

from app.db.dependency import get_db
from app.models.transaction import Transaction
from app.models.debt_account import DebtAccount
from app.models.user import User
from app.utils.auth import get_current_user
from app.utils.chase_parser import is_chase_statement
from app.utils.chase_amount_layout import parse_chase_transactions_amount_only
from app.utils.transaction_utils import parse_amount
from app.rl.auto_categorize import load_model, categorize_with_confidence, _clean_text
from app.rl.normalize_transaction import normalize_bank_memo

router = APIRouter()


CONF_THRESHOLD = 0.35  # good default for your current dataset
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
# ----------------------------------------------------------------
# Helpers: text extraction, balance
# ----------------------------------------------------------------
def extract_text_from_pdf(file_bytes: bytes) -> str:
    text = ""
    with fitz.open(stream=BytesIO(file_bytes), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

# Unified balance rule everywhere:
#  - debit_card:  balance += amount   (expense negative ↓, income positive ↑)
#  - credit/loan: balance -= amount   (expense negative ↑ debt, payment positive ↓)
def apply_card_balance(card: DebtAccount, amount: float) -> None:
    before = float(card.balance or 0)
    amt = float(amount or 0)
    if card.type == "debit_card":
        card.balance = before + amt
    elif card.type in ["credit_card", "loan"]:
        card.balance = before - amt

LINE_PATTERN = r"(\d{2}/\d{2}/\d{4})\s+([^\n]+?)\s+(\(?\s*(?:[+-]?\s*\$|\$?\s*[+-]?)?\s*\d[\d,]*(?:\.\d{2})?\s*\)?)"

def _parse_transactions_generic(text: str) -> List[Dict]:
    transactions: List[Dict] = []
    for match in re.finditer(LINE_PATTERN, text):
        date_str, description, amount_str = match.groups()
        amt = parse_amount(amount_str)
        description_clean = description.strip()
        category = ml_categorize(description_clean)

        try:
            date_obj = datetime.strptime(date_str, "%m/%d/%Y")
        except ValueError:
            date_obj = datetime.now()

        transactions.append({
            "date": date_obj,
            "description": description_clean or "Unlabeled",
            "amount": amt,
            "category": category
        })

    return transactions

def parse_transactions(text: str) -> List[Dict]:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    print(f"🔍 Parsing transactions, text length after cleanup: {len(text)}")
    
    try:
        if is_chase_statement(text):
            print(f"🔍 Detected as Chase statement")
            return parse_chase_transactions_amount_only(text)
        else:
            print(f"🔍 Not detected as Chase statement, using generic parser")
    except Exception as e:
        print(f"🔍 Chase parser failed: {e}, falling back to generic")
        
    result = _parse_transactions_generic(text)
    print(f"🔍 Generic parser found {len(result)} transactions")
    return result

# -------------
# Debug endpoint
# -------------
@router.post("/debug")
async def debug_upload(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    print("🔍 Debug upload endpoint called")
    try:
        content_type = request.headers.get("content-type", "unknown")
        print(f"🔍 Content-Type: {content_type}")
        form = await request.form()
        print(f"🔍 Form keys: {list(form.keys())}")
        for key, value in form.items():
            if hasattr(value, "filename"):
                print(f"  📁 {key}: {value.filename} ({getattr(value, 'content_type', 'unknown')})")
            else:
                s = str(value)
                print(f"  📄 {key}: {s[:120]}{'...' if len(s) > 120 else ''}")
        return {"status": "ok", "form_keys": list(form.keys()), "content_type": content_type}
    except Exception as e:
        print(f"💥 Debug error: {e}")
        return {"error": str(e)}

# ----------------------
# Main PDF upload route
# ----------------------
@router.post("/")
async def upload_pdf(
    file: UploadFile = File(...),
    debt_id: int = Form(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    print("🔍 PDF Upload request received")
    try:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        if not contents.startswith(b"%PDF"):
            raise HTTPException(status_code=400, detail="File is not a valid PDF")

        text = extract_text_from_pdf(contents)
        print(f"🔍 Extracted text length: {len(text)}")
        print(f"🔍 First 500 chars of extracted text:")
        print(repr(text[:500]))
        normalized = text.replace("\u00a0", " ")
        normalized = re.sub(r"[ \t]+", " ", normalized)
        print(f"🔍 Text contains Chase markers: {any(marker.lower() in normalized.lower() for marker in ['JPMorgan Chase Bank', 'Chase', 'Total fees for this period'])}")

        if is_chase_statement(normalized):
            print("🔍 Using Chase amount-only layout parser")
            txs = parse_chase_transactions_amount_only(contents, normalized)
            if not txs:
                print("🔍 Chase amount-only parser returned 0; falling back to generic")
                txs = _parse_transactions_generic(normalized)
        else:
            txs = _parse_transactions_generic(normalized)
            print(f"🔍 Parsed {len(txs)} transactions from PDF")

        if not txs:
            return {"message": "PDF processed but no transactions found", "count": 0, "transactions": []}

        # fetch card
        card = db.query(DebtAccount).filter_by(id=debt_id, user_id=current_user.id).first()
        if not card:
            raise HTTPException(status_code=404, detail="Card not found")

        added: List[Transaction] = []
        for tx in txs:
            # optional: dedupe
            exists = db.query(Transaction).filter_by(
                user_id=current_user.id,
                debt_id=card.id,
                amount=tx["amount"],
                timestamp=tx["date"],
                description=tx["description"],
            ).first()
            if exists:
                continue

            db_tx = Transaction(
                amount=tx["amount"],              # signed
                category=tx["category"],
                description=tx["description"],
                timestamp=tx["date"],
                user_id=current_user.id,
                debt_id=card.id,
            )
            db.add(db_tx)

            # ✅ Update balance with unified rule
            apply_card_balance(card, tx["amount"])
            added.append(db_tx)

        db.add(card)  # ensure SQLAlchemy tracks modified card
        db.commit()

        print(f"✅ Saved {len(added)} transactions; card {card.id} new balance={card.balance}")

        return {
            "message": "PDF processed successfully",
            "count": len(added),
            "transactions": [
                {
                    "amount": t.amount,
                    "category": t.category,
                    "description": t.description,
                    "timestamp": t.timestamp.isoformat() if t.timestamp else None,
                }
                for t in added
            ],
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"💥 PDF processing error: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {e}")
