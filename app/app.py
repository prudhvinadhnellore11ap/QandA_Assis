import os
import sys
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from src.query_direct import run_rag_query

# ==========================================================
# ⚙️ FastAPI setup
# ==========================================================
load_dotenv()
app = FastAPI(title="PruQandA RAG API", version="2.0")

# ==========================================================
# 🧩 Models
# ==========================================================
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

# ==========================================================
# 🌐 Routes
# ==========================================================
@app.get("/")
async def home():
    return {"message": "PruQandA RAG API is running successfully 🚀"}

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    try:
        answer = run_rag_query(request.question)
        return QueryResponse(answer=answer)
    except Exception as e:
        return QueryResponse(answer=f"❌ Internal error: {e}")

# ==========================================================
# 🏃 Run locally:
# uvicorn app.app:app --reload
# ==========================================================
