import os
import requests
from dotenv import load_dotenv
from openai import AzureOpenAI

# ==========================================================
# 🔧 Render-safe setup
# ==========================================================
load_dotenv()

# Remove proxies injected by Render (causes Client.__init__() error)
for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(var, None)

# Ensure OpenAI client doesn't re-introduce proxy args
import openai
openai.default_http_client = None

# ==========================================================
# 🔑 Environment variables
# ==========================================================
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX")

AZURE_CHAT_MODEL = os.getenv("AZURE_OPENAI_CHAT_MODEL")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

# ==========================================================
# 🧠 Azure OpenAI client
# ==========================================================
client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    api_version=AZURE_API_VERSION,
)

# ==========================================================
# 🔍 Direct RAG function
# ==========================================================
def run_rag_query(question: str) -> str:
    """Hybrid retrieval from Azure Cognitive Search + reasoning via Azure OpenAI."""
    if not question.strip():
        return "Please provide a valid question."

    # 1️⃣ Retrieve top documents from Azure Cognitive Search
    headers = {"Content-Type": "application/json", "api-key": AZURE_SEARCH_API_KEY}
    payload = {"search": question, "top": 5, "queryType": "semantic"}
    url = f"{AZURE_SEARCH_ENDPOINT}/indexes/{AZURE_SEARCH_INDEX}/docs/search?api-version=2021-04-30-Preview"

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=10)
        resp.raise_for_status()
        docs = resp.json().get("value", [])
    except Exception as e:
        return f"❌ Azure Search error: {e}"

    context = "\n\n".join(doc.get("content", "") for doc in docs if doc.get("content"))

    # 2️⃣ Generate answer via Azure OpenAI
    try:
        messages = [
            {"role": "system", "content": "You are an assistant that answers logically using given context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer with reasoning:"},
        ]

        response = client.chat.completions.create(
            model=AZURE_CHAT_MODEL,
            messages=messages,
            temperature=0.2,
        )

        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ OpenAI error: {e}"
