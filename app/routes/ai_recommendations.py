# app/routers/ai_recommendations.py

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.db.dependency import get_db
from app.utils.auth import get_current_user
from app.models.user import User
from app.models.transaction import Transaction
from app.models.debt_account import DebtAccount
from app.rl.agent import get_recommendation

router = APIRouter()

# --- helpers: month bounds (local naive; adjust if you store UTC) ---
def month_bounds(dt: Optional[datetime] = None):
    now = dt or datetime.utcnow()
    start = datetime(now.year, now.month, 1)
    if now.month == 12:
        end = datetime(now.year + 1, 1, 1)
    else:
        end = datetime(now.year, now.month + 1, 1)
    return start, end


class RecommendationRequest(BaseModel):
    surplus: Optional[float] = None
    year: Optional[int] = None
    month: Optional[int] = None
    months_emergency: int = 3


class RecommendationResponse(BaseModel):
    action: str
    confidence: float
    current_state: dict
    projected_state: dict
    reasoning: str
    surplus_allocated: float
    target_savings: float
    diagnostics: dict


@router.post("/", response_model=RecommendationResponse)
def get_ai_recommendation(
    request: RecommendationRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Compute monthly spending/income server-side and produce an RL recommendation
    with target_savings = 3 × monthly_spending (or months_emergency × spending).
    """

    # ---- Choose month window ----
    if request.year and request.month:
        start = datetime(request.year, request.month, 1)
        if request.month == 12:
            end = datetime(request.year + 1, 1, 1)
        else:
            end = datetime(request.year, request.month + 1, 1)
    else:
        start, end = month_bounds()

    # ---- Aggregate transactions for this user and month ----
    # Income: positive amounts in "Income"
    income_val = (
        db.query(func.coalesce(func.sum(Transaction.amount), 0.0))
        .filter(
            Transaction.user_id == current_user.id,
            Transaction.category == "Income",
            Transaction.timestamp >= start,
            Transaction.timestamp < end,
        )
        .scalar()
        or 0.0
    )

    # Expenses: absolute value of amounts for non-Income categories
    expenses_val = (
        db.query(func.coalesce(func.sum(func.abs(Transaction.amount)), 0.0))
        .filter(
            Transaction.user_id == current_user.id,
            Transaction.category != "Income",
            Transaction.timestamp >= start,
            Transaction.timestamp < end,
        )
        .scalar()
        or 0.0
    )

    surplus = request.surplus if request.surplus is not None else (income_val - expenses_val)
    surplus = float(max(0.0, surplus))

    # ---- Current state: debt/savings/investment ----
    # Debt = sum of credit card (and loan) balances
    debt_total = (
        db.query(func.coalesce(func.sum(DebtAccount.balance), 0.0))
        .filter(
            DebtAccount.user_id == current_user.id,
            DebtAccount.type.in_(["credit_card", "loan"]),
        )
        .scalar()
        or 0.0
    )

    # Savings: from user column; fallback to 0 if missing
    savings_total = float(getattr(current_user, "current_savings", 0.0) or 0.0)

    investment_total = 0.0

    monthly_spending = float(expenses_val)
    months_emergency = int(max(1, request.months_emergency))

    # ---- Call RL policy ----
    rec = get_recommendation(
        debt=int(round(debt_total)),
        savings=int(round(savings_total)),
        investment=int(round(investment_total)),
        surplus=int(round(surplus)),
        monthly_spending=int(round(monthly_spending)),
        months_emergency=months_emergency,
    )

    # ---- Add some diagnostics so you can see inputs while tuning ----
    rec["target_savings"] = rec.get("target_savings", months_emergency * monthly_spending)
    rec["diagnostics"] = {
        "period": {"start": start.isoformat(), "end": end.isoformat()},
        "income": round(income_val, 2),
        "expenses": round(expenses_val, 2),
        "computed_surplus": round(income_val - expenses_val, 2),
        "used_surplus": round(surplus, 2),
        "debt_total": round(debt_total, 2),
        "savings_total": round(savings_total, 2),
        "investment_total": round(investment_total, 2),
        "months_emergency": months_emergency,
    }

    return RecommendationResponse(**rec)
