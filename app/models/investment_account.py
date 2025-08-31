from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship
from app.db.session import Base

class InvestmentAccount(Base):
    __tablename__ = "investment_accounts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    name = Column(String, nullable=False)
    balance = Column(Integer, default=0)

    user = relationship("User", back_populates="investments")