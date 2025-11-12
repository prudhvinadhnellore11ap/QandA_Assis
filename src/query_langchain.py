import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
# ---------------------------
# Load environment variables
# ---------------------------
load_dotenv()

# ---------------------------
# Initialize Embeddings
# ---------------------------
embeddings = AzureOpenAIEmbeddings(
    deployment=os.getenv("AZURE_OPENAI_EMB_DEPLOYMENT"),
    model="text-embedding-3-small",  # must match your Azure embedding deployment
    api_key=os.getenv("AZURE_OPENAI_EMB_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_EMB_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_EMB_API_VERSION", "2024-12-01-preview")
)

# ---------------------------
# Connect to Azure Cognitive Search
# ---------------------------
vector_store = AzureSearch(
    azure_search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    azure_search_key=os.getenv("AZURE_SEARCH_API_KEY"),
    index_name=os.getenv("AZURE_SEARCH_INDEX"),
    embedding_function=embeddings.embed_query,
    content_field="content",        # ✅ matches your schema
    vector_field="content_vector",  # ✅ vector field
    hybrid_fields=["content"],        # explicitly include textual relevance
    hybrid_weight=0.5    
)

# ---------------------------
# Retriever setup
# ---------------------------
retriever = vector_store.as_retriever(search_type="hybrid")

# make sure no duplicate args are passed internally
retriever.search_kwargs = {}  
retriever.search_kwargs["filters"] = None
retriever.k = 50   # controls number of top results to retrieve


# ---------------------------
# Initialize Chat Model
# ---------------------------
llm = AzureChatOpenAI(
    deployment_name=os.getenv("AZURE_OPENAI_CHAT_MODEL"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
    temperature=0.2,
)

# ---------------------------
# Build RetrievalQA Chain
# ---------------------------

# Custom reasoning-aware prompt
prompt_template = """
You are a professional assistant who analyzes member messages to answer natural questions.

You must:
- Read all related messages carefully.
- Infer likely answers from context, tone, and repeated patterns.
- If multiple clues suggest a behavior or pattern (e.g., frequent car hires), summarize that logically.
- Only say "I don't know" if absolutely no connection can be inferred.

Question: {question}

Messages:
{context}

Answer with reasoning:
"""



prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True,
)

# ---------------------------
# Run a Query
# ---------------------------
"""query = input("Enter your question: ").strip()
print(f"\n🔍 Question: {query}")

try:
    result = qa_chain.invoke({"query": query})
    print("\n💬 Answer:\n", result.get("result", "No answer generated."))

    print("\n📚 Top Sources:")
    for doc in result.get("source_documents", []):
        meta = getattr(doc, "metadata", {})
        text = doc.page_content or meta.get("content") or "(no text)"
        print(f"- {meta.get('user_name', 'Unknown')}: {text[:150]}...")
except Exception as e:
    print(f"❌ Error running query: {e}")"""

