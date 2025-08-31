from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import relationship
from app.db.session import Base

class DebtAccount(Base):
    __tablename__ = "debt_accounts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    name = Column(String, nullable=False)
    balance = Column(Integer, default=0)
    interest_rate = Column(Float, default=0.0)
    type = Column(String, default="credit_card")

    user = relationship("User", back_populates="debts")
    transactions = relationship("Transaction", back_populates="debt", cascade="all, delete-orphan")

