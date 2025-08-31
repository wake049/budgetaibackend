from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class TransactionInCard(BaseModel):
    id: int
    amount: float
    category: Optional[str] = None
    description: Optional[str] = None
    timestamp: Optional[datetime] = None

class DebtAccountBase(BaseModel):
    name: str
    balance: int
    interest_rate: float
    type: str

class DebtAccountCreate(DebtAccountBase):
    pass

class DebtAccountOut(DebtAccountBase):
    id: int
    user_id: int

    class Config:
        from_attributes = True

class DebtAccountWithTransactions(DebtAccountBase):
    id: int
    user_id: int
    transactions: List[TransactionInCard] = []

    class Config:
        from_attributes = True
