import os
import requests
from dotenv import load_dotenv
from openai import AzureOpenAI

# -------------------------
# Setup
# -------------------------
load_dotenv()

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX")

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
)

# -------------------------
# Simple RAG Function
# -------------------------
def run_rag_query(question: str):
    """Hybrid retrieval from Azure Search + reasoning via Azure OpenAI"""
    
    # 1️⃣ Retrieve top docs from Azure Cognitive Search
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_SEARCH_API_KEY
    }
    payload = {
        "search": question,
        "top": 5,
        "queryType": "semantic"
    }

    search_url = f"{AZURE_SEARCH_ENDPOINT}/indexes/{AZURE_SEARCH_INDEX}/docs/search?api-version=2021-04-30-Preview"
    resp = requests.post(search_url, headers=headers, json=payload)
    docs = resp.json().get("value", [])

    context = "\n\n".join([doc.get("content", "") for doc in docs])

    # 2️⃣ Send to Azure OpenAI for answer generation
    messages = [
        {"role": "system", "content": "You are an assistant that answers based on user messages and logical reasoning."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer with reasoning:"}
    ]

    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_CHAT_MODEL"),
        messages=messages,
        temperature=0.2
    )

    answer = response.choices[0].message.content.strip()
    return answer
