from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime
import calendar
import pandas as pd
import numpy as np

from app.db.dependency import get_db
from app.models import Transaction 
from app.utils.auth import get_current_user

router = APIRouter()

# Tunables
HIST_MONTHS   = 3
WEEKEND_BOOST = 1.15
SMOOTHING     = 0.2

# Bills we want to spike on due days (optional)
BILL_CATS = {"Rent", "Utilities", "Internet", "Phone", "Insurance"}

@router.get("/summary")
def viz_summary(db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    user_id = current_user.id
    today = datetime.today()
    first = datetime(today.year, today.month, 1)
    dim = calendar.monthrange(today.year, today.month)[1]
    days = pd.date_range(first, periods=dim, freq="D")

    lookback_start = pd.Timestamp(today.year, today.month, 1) - pd.DateOffset(months=HIST_MONTHS)

    # 👉 Pull ONLY this user's rows
    rows = (
        db.query(Transaction)
        .filter(Transaction.user_id == user_id)
        .filter(Transaction.timestamp >= lookback_start)
        .filter(Transaction.timestamp <= today)
        .all()
    )
    if not rows:
        return {
            "categories": [],
            "daily": [],
            "ideal": [],
            "meta": {
                "days_in_month": dim,
                "today": today.date().isoformat(),
                "user_id": user_id,
                "count_rows": 0,
                "income": 0.0,
                "bills": 0.0,
                "surplus": 0.0,
                "debt_goal": 0.0,
                "savings_goal": 0.0,
                "discretionary_pool": 0.0,
                "mtd_spent": 0.0,
            },
        }

    df = pd.DataFrame(
        {
            "date": [pd.to_datetime(t.timestamp) for t in rows],
            "category": [(t.category or "Uncategorized").strip() for t in rows],
            "amount": [float(t.amount) for t in rows],
            "debt_id": [t.debt_id for t in rows],
        }
    )
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    
    # Debug: Check timestamp parsing
    print(f"Raw timestamps from DB:")
    for i, row in enumerate(rows[:5]):  # First 5 rows
        print(f"Row {i}: timestamp={row.timestamp}, amount={row.amount}")
    print(f"Parsed dates in DataFrame:")
    print(df[["date", "amount"]].head())

    # Split month
    df_cur = df[(df["date"] >= first) & (df["date"] <= today)].copy()
    df_hist = df[(df["date"] < first)].copy()
    
    # Debug: Check the date filtering
    print(f"Date filtering debug:")
    print(f"first (start of month): {first}")
    print(f"today: {today}")
    print(f"Total transactions: {len(df)}")
    print(f"Current month transactions: {len(df_cur)}")
    print(f"Historical transactions: {len(df_hist)}")
    print(f"All transaction dates in df:")
    print(df[["date", "amount", "category"]].sort_values("date"))

    # ==== Derive types from sign/categories (no 'type' column available) ====
    # income: amount > 0
    # outflow: amount < 0 (spend/bill/debt/savings)
    df_cur["is_income"] = df_cur["amount"] > 0
    df_cur["is_outflow"] = df_cur["amount"] < 0
    df_cur["is_bill"] = df_cur["category"].isin(BILL_CATS) & df_cur["is_outflow"]
    df_cur["is_debt"] = df_cur["debt_id"].notna() & df_cur["is_outflow"]
    
    # Debug: Check the outflow identification
    print(f"Debug: Outflow identification...")
    print(f"Total rows in df_cur: {len(df_cur)}")
    print(f"Rows with amount < 0: {(df_cur['amount'] < 0).sum()}")
    print(f"Rows marked as is_outflow: {df_cur['is_outflow'].sum()}")
    print(f"df_cur dtypes:")
    print(df_cur.dtypes)

    # ---- Surplus pieces ----
    income = df_cur.loc[df_cur["is_income"], "amount"].sum()
    bills = df_cur.loc[df_cur["is_bill"], "amount"].abs().sum()
    debt_goal = df_cur.loc[df_cur["is_debt"], "amount"].abs().sum()
    # If you have savings, treat category == "Savings" as savings:
    savings_goal = df_cur.loc[(df_cur["category"] == "Savings") & df_cur["is_outflow"], "amount"].abs().sum()

    surplus = income - bills
    discretionary_pool = max(0.0, surplus - debt_goal - savings_goal)

    # ---- Category targets from history (use magnitudes for outflows only) ----
    disc_hist = df_hist[df_hist["amount"] < 0].copy()
    if disc_hist.empty:
        weights = {"Groceries": 0.4, "Transport": 0.2, "Dining": 0.25, "Entertainment": 0.15}
    else:
        by_cat = (
            disc_hist.assign(amount_abs=lambda x: x["amount"].abs())
            .groupby("category", as_index=False)["amount_abs"]
            .sum()
        )
        total = by_cat["amount_abs"].sum() or 1.0
        weights = {r["category"]: (r["amount_abs"] / total) for _, r in by_cat.iterrows()}

    median_by_cat = (
        disc_hist.assign(amount_abs=lambda x: x["amount"].abs())
        .groupby("category")["amount_abs"]
        .median()
        if not disc_hist.empty
        else pd.Series(dtype=float)
    )

    disc_targets = []
    for cat, w in weights.items():
        target = discretionary_pool * w
        cap = median_by_cat.get(cat, np.inf)
        target = float(min(target, cap if np.isfinite(cap) else target))
        disc_targets.append((cat, target))
    disc_targets_df = pd.DataFrame(disc_targets, columns=["category", "disc_target"])

    # ---- Category summary (outflows only, positive magnitudes) ----
    spent_cur = (
        df_cur[df_cur["amount"] < 0]
        .assign(amount_pos=lambda x: x["amount"].abs())
        .groupby("category", as_index=False)["amount_pos"].sum()
        .rename(columns={"amount_pos": "spent_to_date"})
    )
    cat_summary = spent_cur.merge(disc_targets_df, on="category", how="outer").fillna(0.0)
    cat_summary["variance_vs_target"] = cat_summary["spent_to_date"] - cat_summary["disc_target"]

    # ---- Daily cumulative (outflows only, positive magnitudes) ----
    print(f"Debug: Daily calculation...")
    outflow_for_daily = df_cur.loc[df_cur["is_outflow"]].copy()
    print(f"Outflows for daily calc: {len(outflow_for_daily)}")
    print(outflow_for_daily[["date", "amount", "is_outflow"]])
    
    # Normalize dates to remove time component for proper grouping
    outflow_for_daily["date_normalized"] = outflow_for_daily["date"].dt.normalize()
    
    dailies = (
        outflow_for_daily
        .assign(amount_pos=lambda x: x["amount"].abs())
        .groupby("date_normalized", as_index=False)["amount_pos"]
        .sum()
        .rename(columns={"date_normalized": "date", "amount_pos": "actual_day"})
    )
    print(f"Dailies after groupby:")
    print(dailies)
    
    # Create daily DataFrame with normalized dates for proper merging
    daily = pd.DataFrame({"date": days.normalize()}).merge(dailies, on="date", how="left").fillna({"actual_day": 0.0})
    daily["cum_spend"] = daily["actual_day"].cumsum()
    
    print(f"Daily dataframe with cumulative:")
    print(daily[daily["actual_day"] > 0])
    print(f"Full daily dataframe (first 10 rows):")
    print(daily.head(10))

    # ---- Ideal curve: bill spikes + smoothed discretionary ----
    daily["fixed"] = 0.0
    for _, r in df_cur.loc[df_cur["is_bill"]].iterrows():
        d = pd.Timestamp(r["date"]).normalize()
        daily.loc[daily["date"] == d, "fixed"] += abs(float(r["amount"]))

    disc_total = float(disc_targets_df["disc_target"].sum())
    if disc_total > 0:
        w = pd.Series(1.0, index=daily.index)
        dow = daily["date"].dt.dayofweek
        w = w * np.where(dow >= 5, WEEKEND_BOOST, 1.0)
        w = w.ewm(alpha=SMOOTHING).mean()
        w = w / w.sum() * disc_total
        daily["disc"] = w.values
    else:
        daily["disc"] = 0.0

    daily["ideal_cum"] = (daily["fixed"] + daily["disc"]).cumsum()

    # ---- MTD spent (authoritative) ----
    # Debug: Let's see what transactions we have this month
    print(f"Debug: Current month transactions:")
    print(f"df_cur shape: {df_cur.shape}")
    print(f"Outflow transactions this month:")
    outflow_debug = df_cur[df_cur["amount"] < 0][["date", "category", "amount"]]
    print(outflow_debug)
    
    # Calculate MTD spent directly from current month outflows
    print(f"Debug: Calculating MTD spent...")
    outflows_this_month = df_cur[df_cur["amount"] < 0]
    print(f"Outflows this month count: {len(outflows_this_month)}")
    print(f"Outflows this month details:")
    print(outflows_this_month[["date", "category", "amount"]])
    
    # Calculate step by step
    negative_amounts = df_cur[df_cur["amount"] < 0]["amount"]
    print(f"Negative amounts: {negative_amounts.tolist()}")
    
    absolute_amounts = negative_amounts.abs()
    print(f"Absolute amounts: {absolute_amounts.tolist()}")
    
    mtd_spent_direct = absolute_amounts.sum()
    print(f"Direct MTD calculation step by step: {mtd_spent_direct}")
    
    # Alternative calculation for verification
    mtd_alternative = df_cur.loc[df_cur["amount"] < 0, "amount"].abs().sum()
    print(f"Alternative MTD calculation: {mtd_alternative}")
    
    # Get the current cumulative spending up to today
    today_normalized = pd.Timestamp(today).normalize()
    print(f"Today normalized: {today_normalized}")
    print(f"Daily data:")
    print(daily[["date", "actual_day", "cum_spend"]])
    
    current_row = daily.loc[daily["date"] <= today_normalized]
    if not current_row.empty:
        mtd_spent = float(current_row["cum_spend"].iloc[-1])
    else:
        mtd_spent = 0.0
    
    print(f"Debug MTD calculation: today={today_normalized}, mtd_spent={mtd_spent}, direct_calc={mtd_spent_direct}")
    
    # Use the direct calculation as it's more reliable
    mtd_spent = float(mtd_spent_direct)
    
    return {
        "categories": cat_summary.sort_values("variance_vs_target", ascending=False).to_dict(orient="records"),
        "daily": daily.assign(date=daily["date"].dt.strftime("%Y-%m-%d"))[["date", "cum_spend"]].to_dict(orient="records"),
        "ideal": daily.assign(date=daily["date"].dt.strftime("%Y-%m-%d"))[["date", "ideal_cum"]].to_dict(orient="records"),
        "meta": {
            "days_in_month": dim,
            "today": today.date().isoformat(),
            "user_id": user_id,
            "count_rows": int(len(rows)),
            "income": float(income),
            "bills": float(bills),
            "surplus": float(surplus),
            "debt_goal": float(debt_goal),
            "savings_goal": float(savings_goal),
            "discretionary_pool": float(discretionary_pool),
            "mtd_spent": mtd_spent,
        },
    }
