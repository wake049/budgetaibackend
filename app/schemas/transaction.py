from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class TransactionBase(BaseModel):
    amount: float
    category: Optional[str] = None
    description: Optional[str] = None

class TransactionCreate(TransactionBase):
    timestamp: Optional[datetime] = None

class TransactionOut(TransactionBase):
    id: int
    timestamp: datetime 

    class Config:
        from_attributes = True

class TransactionListResponse(BaseModel):
    transactions: List[TransactionOut]
    has_more: bool
    total: int