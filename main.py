from fastapi import FastAPI
from app.routes import auth, charts, pdf_upload, transaction, plaid
from app.routes import ai_recommendations
from fastapi.middleware.cors import CORSMiddleware
from app.db.base import Base
from app.db.session import engine

app = FastAPI()
Base.metadata.create_all(bind=engine)

app.include_router(auth.router, tags=["Auth"])
app.include_router(transaction.router, prefix="/transactions", tags=["Transactions"])
app.include_router(pdf_upload.router, prefix="/pdf-upload", tags=["PDF Upload"])
app.include_router(ai_recommendations.router, prefix="/ai-recommendations", tags=["AI Recommendations"])
app.include_router(charts.router, prefix="/viz", tags=["Visualizations"])
app.include_router(plaid.router, tags=["Plaid"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "BudgetAI backend is running"}

@app.get("/healthz")
def healthz():
    return {"ok": True}