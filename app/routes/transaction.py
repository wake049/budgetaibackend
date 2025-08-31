from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.schemas.transaction import TransactionCreate, TransactionOut, TransactionListResponse
from app.schemas.debt_account import DebtAccountOut, DebtAccountWithTransactions, DebtAccountCreate
from app.utils.auth import get_current_user
from app.db.dependency import get_db
from app.models.user import User
from typing import List
from app.models.transaction import Transaction
from fastapi import Query
from app.models.debt_account import DebtAccount 

router = APIRouter()

@router.get("/test")
def test_endpoint():
    """Simple test endpoint to verify the router is working"""
    return {"message": "Transaction router is working!", "timestamp": "2025-01-20"}

@router.get("/auth-test")
def test_auth(current_user: User = Depends(get_current_user)):
    """Test endpoint to verify authentication is working"""
    return {
        "message": "Authentication working!", 
        "user_id": current_user.id,
        "user_email": current_user.email
    }

@router.post("/cards/{card_id}", response_model=TransactionOut)
def create_transaction_for_card(
    card_id: int,
    transaction: TransactionCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    card = db.query(DebtAccount).filter_by(id=card_id, user_id=current_user.id).first()
    if not card:
        raise HTTPException(status_code=404, detail="Card not found")

    new_tx = Transaction(
        **transaction.dict(),
        user_id=current_user.id,
        debt_id=card.id,
    )
    if card.type == "debit_card":
        card.balance += transaction.amount
    elif card.type in ["credit_card", "loan"]:
        card.balance -= transaction.amount

    db.add(new_tx)
    db.add(card)                 # ensure SQLAlchemy tracks the change
    db.commit()
    db.refresh(new_tx)
    db.refresh(card)             # optional: immediately get the new balance
    return new_tx

@router.get("/cards/{card_id}", response_model=List[TransactionOut])
def get_transactions_for_card(
    card_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    card = db.query(DebtAccount).filter_by(id=card_id, user_id=current_user.id).first()
    if not card:
        raise HTTPException(status_code=404, detail="Card not found")

    transactions = db.query(Transaction).filter_by(user_id=current_user.id, debt_id=card.id).order_by(Transaction.timestamp.desc()).all()
    return transactions

@router.get("/cards", response_model=List[DebtAccountWithTransactions])
def get_debit_cards(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Get all cards/debt accounts for the current user with their transactions"""
    print(f"🔍 Fetching cards with transactions for user ID: {current_user.id}, email: {current_user.email}")
    
    cards = (
        db.query(DebtAccount)
        .filter(DebtAccount.user_id == current_user.id)
        .all()
    )
    
    print(f"🔍 Found {len(cards)} cards for user")
    
    result = []
    for card in cards:
        print(f"🔍 Processing card: ID={card.id}, Name={card.name}, Type={card.type}, Balance={card.balance}")
        
        # Get transactions for this card - handle case where debt_id might be None
        transactions = (
            db.query(Transaction)
            .filter(
                Transaction.user_id == current_user.id,
                Transaction.debt_id == card.id
            )
            .order_by(Transaction.timestamp.desc())
            .all()
        )
        
        print(f"🔍 Found {len(transactions)} transactions for card {card.name}")
        
        # Format transactions
        formatted_transactions = []
        for txn in transactions:
            formatted_transactions.append({
                "id": txn.id,
                "amount": txn.amount,
                "category": txn.category,
                "description": txn.description,
                "timestamp": txn.timestamp.isoformat() if txn.timestamp else None,
            })
        
        # Add card with transactions
        result.append({
            "id": card.id,
            "name": card.name,
            "balance": card.balance,
            "interest_rate": card.interest_rate,
            "type": card.type,
            "user_id": card.user_id,
            "transactions": formatted_transactions
        })
    
    print(f"🔍 Returning {len(result)} cards with transactions")
    return result

@router.get("/debug/cards")
def debug_get_cards(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Debug endpoint - get cards without response model"""
    print(f"🔍 DEBUG: Fetching cards for user ID: {current_user.id}")
    
    cards = db.query(DebtAccount).filter(DebtAccount.user_id == current_user.id).all()
    
    result = {
        "user_id": current_user.id,
        "user_email": current_user.email,
        "cards_count": len(cards),
        "cards": []
    }
    
    for card in cards:
        result["cards"].append({
            "id": card.id,
            "name": card.name,
            "balance": card.balance,
            "interest_rate": card.interest_rate,
            "type": card.type,
            "user_id": card.user_id
        })
    
    return result

@router.delete("/{tx_id}")
def delete_transaction(
    tx_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    tx = db.query(Transaction).filter(
        Transaction.id == tx_id, Transaction.user_id == current_user.id
    ).first()
    if not tx:
        raise HTTPException(status_code=404, detail="Transaction not found")

    card = db.query(DebtAccount).filter_by(id=tx.debt_id, user_id=current_user.id).first()
    if card:
        if card.type == "debit_card":
            card.balance -= tx.amount        # revert +=
        elif card.type in ["credit_card", "loan"]:
            card.balance += tx.amount        # revert -=
        db.add(card)

    db.delete(tx)
    db.commit()
    return {"detail": "Deleted successfully"}

@router.post("/cards/{card_id}/import_plaid")
def import_plaid_transactions(
    card_id: int,
    transactions: List[TransactionCreate],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    card = db.query(DebtAccount).filter_by(id=card_id, user_id=current_user.id).first()
    if not card:
        raise HTTPException(status_code=404, detail="Card not found")

    imported = 0
    for tx_data in transactions:
        exists = db.query(Transaction).filter_by(
            user_id=current_user.id,
            debt_id=card.id,
            amount=tx_data.amount,
            timestamp=tx_data.timestamp,
            description=tx_data.description,
        ).first()
        if exists:
            continue

        tx = Transaction(
            amount=tx_data.amount,
            category=tx_data.category,
            description=tx_data.description,
            timestamp=tx_data.timestamp or datetime.utcnow(),
            user_id=current_user.id,
            debt_id=card.id,
        )
        db.add(tx)

        # ✅ same rule
        if card.type == "debit_card":
            card.balance += tx.amount
        elif card.type in ["credit_card", "loan"]:
            card.balance -= tx.amount

        imported += 1

    db.add(card)
    db.commit()
    return {"detail": f"{imported} transactions imported"}

@router.post("/cards", response_model=DebtAccountOut)
def create_card(
    card: DebtAccountCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    new_card = DebtAccount(
        name=card.name,
        type=card.type,
        balance=card.balance,
        interest_rate=card.interest_rate,
        user_id=current_user.id
    )
    db.add(new_card)
    db.commit()
    db.refresh(new_card)
    return new_card