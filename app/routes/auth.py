from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.db.dependency import get_db
from app.models.user import User
from app.schemas.user import UserCreate, UserLogin, UserOut
from app.utils.auth import hash_password, verify_password, create_access_token
from app.models.debt_account import DebtAccount
from app.models.investment_account import InvestmentAccount
from app.schemas.auth import RegisterRequest

router = APIRouter()

@router.post("/register")
def register_user(
    payload: RegisterRequest,
    db: Session = Depends(get_db)
):
    print(f"🔍 Registration attempt for email: {payload.email}")
    print(f"🔍 Payload: savings={payload.savings}, debts count={len(payload.debts)}, investments count={len(payload.investments)}")
    
    try:
        # Check if user already exists
        print("🔍 Checking if user exists...")
        existing_user = db.query(User).filter_by(email=payload.email).first()
        if existing_user:
            print("❌ User already exists")
            raise HTTPException(status_code=400, detail="Email already registered")

        # Create new user
        print("🔍 Creating new user...")
        user = User(email=payload.email)
        print("🔍 Setting password...")
        user.set_password(payload.password)
        user.current_savings = payload.savings or 0
        print(f"🔍 User created with savings: {user.current_savings}")

        print("🔍 Adding user to database...")
        db.add(user)
        db.commit()
        db.refresh(user)
        print(f"✅ User created with ID: {user.id}")
        
        # Create default debit card
        print("🔍 Creating default debit card...")
        default_debit_card = DebtAccount(
            user_id=user.id,
            name="Default Debit Card",
            balance=0,
            interest_rate=0.0,
            type="debit_card"
        )
        db.add(default_debit_card)
        print("✅ Default debit card created")

        # Add user's initial debts
        print(f"🔍 Processing {len(payload.debts)} initial debts...")
        for i, debt in enumerate(payload.debts):
            print(f"🔍 Creating debt {i+1}: {debt.name}, balance={debt.balance}, rate={debt.interest_rate}, type={debt.type}")
            new_debt = DebtAccount(
                user_id=user.id,
                name=debt.name,
                balance=debt.balance,
                interest_rate=debt.interest_rate,
                type=debt.type
            )
            db.add(new_debt)
            print(f"✅ Debt {i+1} created")

        # Add user's initial investments
        print(f"🔍 Processing {len(payload.investments)} initial investments...")
        for i, inv in enumerate(payload.investments):
            print(f"🔍 Creating investment {i+1}: {inv.name}, balance={inv.balance}")
            new_inv = InvestmentAccount(
                user_id=user.id,
                name=inv.name,
                balance=inv.balance
            )
            db.add(new_inv)
            print(f"✅ Investment {i+1} created")

        print("🔍 Final commit...")
        db.commit()
        print("🎉 Registration successful!")
        return {"message": "User registered successfully"}
        
    except HTTPException as http_ex:
        # Re-raise HTTP exceptions (like email already registered)
        print(f"❌ HTTP Exception: {http_ex.detail}")
        raise http_ex
    except Exception as e:
        # Log unexpected errors and return generic message
        print(f"💥 Unexpected registration error: {str(e)}")
        print(f"💥 Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@router.post("/login")
def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if not db_user or not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    access_token = create_access_token(data={"sub": db_user.email})

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "savings": db_user.current_savings
    }