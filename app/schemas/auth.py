from pydantic import BaseModel, EmailStr
from typing import List, Optional

from enum import Enum

class DebtType(str, Enum):
    credit_card = "credit_card"
    loan = "loan"
    debit_card = "debit_card"

class InitialDebt(BaseModel):
    name: str
    balance: int
    interest_rate: float
    type: DebtType = DebtType.credit_card

class InitialInvestment(BaseModel):
    name: str
    balance: int

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    savings: Optional[int] = 0
    debts: Optional[List[InitialDebt]] = []
    investments: Optional[List[InitialInvestment]] = []