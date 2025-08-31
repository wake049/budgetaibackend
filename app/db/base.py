from app.db.session import Base
# Import all models to ensure they are registered with Base
from app.models.user import User
from app.models.transaction import Transaction
from app.models.debt_account import DebtAccount
from app.models.investment_account import InvestmentAccount