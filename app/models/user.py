from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.orm import relationship
from app.db.session import Base
from datetime import datetime
from app.utils.crypto import hash_password

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    transactions = relationship("Transaction", back_populates="user", cascade="all, delete")
    current_savings = Column(Integer, default=0)
    debts = relationship("DebtAccount", back_populates="user", cascade="all, delete-orphan")
    investments = relationship("InvestmentAccount", back_populates="user", cascade="all, delete-orphan")

    def set_password(self, password: str):
        """Hash and set the user's password."""
        self.hashed_password = hash_password(password)