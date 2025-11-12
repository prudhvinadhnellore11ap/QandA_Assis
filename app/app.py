import os
import sys
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

# -------------------------------------
# Ensure src folder is importable
# -------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

# -------------------------------------
# Import the simplified direct RAG function
# -------------------------------------
from src.query_direct import run_rag_query

# -------------------------------------
# FastAPI setup
# -------------------------------------
load_dotenv()
app = FastAPI(title="PruQandA RAG API", version="2.0")

# -------------------------------------
# Models
# -------------------------------------
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

# -------------------------------------
# Routes
# -------------------------------------
@app.get("/")
async def home():
    return {"message": "PruQandA RAG API is running successfully 🚀"}

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """Run direct RAG query using Azure Search + Azure OpenAI."""
    query = request.question.strip()
    try:
        answer = run_rag_query(query)
        return QueryResponse(answer=answer)
    except Exception as e:
        return QueryResponse(answer=f"❌ Error: {str(e)}")

# -------------------------------------
# Run locally:
# uvicorn app.app:app --reload
# -------------------------------------
