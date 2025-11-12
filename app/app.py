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

try:
    from src.query_langchain import qa_chain  # reuse your existing chain setup
except Exception as e:
    print(f"⚠️ Could not import qa_chain: {e}")
    qa_chain = None

# -------------------------------------
# FastAPI setup
# -------------------------------------
load_dotenv()
app = FastAPI(title="PruQandA RAG API", version="1.0")

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
    """Run LangChain RAG query and return only the answer."""
    query = request.question.strip()
    try:
        if not qa_chain:
            return QueryResponse(answer="❌ QA chain not loaded.")

        result = qa_chain.invoke({"query": query})
        answer = result.get("result", "No answer generated.")
        return QueryResponse(answer=answer)

    except Exception as e:
        return QueryResponse(answer=f"❌ Error: {str(e)}")

# -------------------------------------
# Run from project root:
# uvicorn app.app:app --reload
# -------------------------------------
