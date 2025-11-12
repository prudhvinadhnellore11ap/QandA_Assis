import os
from dotenv import load_dotenv

# ===============================================================
# 🔧 Render-safe patch for AzureOpenAIEmbeddings ("proxies" bug)
# ===============================================================
from langchain_openai.embeddings import AzureOpenAIEmbeddings as _AzureEmb

class AzureEmbeddingsPatched(_AzureEmb):
    """Ignore unsupported kwargs like proxies/session in cloud envs."""
    def __init__(self, *args, **kwargs):
        for bad in ("proxies", "proxy", "session"):
            kwargs.pop(bad, None)
        super().__init__(*args, **kwargs)

# Replace the original class globally
import langchain_openai.embeddings as _emb_mod
_emb_mod.AzureOpenAIEmbeddings = AzureEmbeddingsPatched

# ===============================================================
# Imports (normal)
# ===============================================================
from langchain_openai import AzureChatOpenAI
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ===============================================================
# Environment setup
# ===============================================================
load_dotenv()

# -------------------- Embeddings --------------------
embeddings = AzureEmbeddingsPatched(
    deployment=os.getenv("AZURE_OPENAI_EMB_DEPLOYMENT"),
    model="text-embedding-3-small",
    api_key=os.getenv("AZURE_OPENAI_EMB_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_EMB_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_EMB_API_VERSION", "2024-12-01-preview"),
)

# -------------------- Azure Search --------------------
vector_store = AzureSearch(
    azure_search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    azure_search_key=os.getenv("AZURE_SEARCH_API_KEY"),
    index_name=os.getenv("AZURE_SEARCH_INDEX"),
    embedding_function=embeddings.embed_query,
    content_field="content",
    vector_field="content_vector",
    hybrid_fields=["content"],
    hybrid_weight=0.5,
)

# -------------------- Retriever --------------------
retriever = vector_store.as_retriever(search_type="hybrid")
retriever.search_kwargs = {}
retriever.search_kwargs["filters"] = None
retriever.k = 40

# -------------------- Chat Model --------------------
llm = AzureChatOpenAI(
    deployment_name=os.getenv("AZURE_OPENAI_CHAT_MODEL"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
    temperature=0.2,
)

# -------------------- Prompt --------------------
prompt_template = """
You are a professional assistant analyzing member messages to answer natural questions.

Read all messages carefully.
Infer logical answers from repeated behaviors or clues.
Only say "I don't know" if no connection can be inferred.

Question: {question}

Messages:
{context}

Answer with reasoning:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# -------------------- RetrievalQA --------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True,
)
