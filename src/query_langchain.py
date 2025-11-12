import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ===============================================================
# Environment setup
# ===============================================================
load_dotenv()

# -------------------- Embeddings --------------------
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_OPENAI_EMB_DEPLOYMENT"),
    openai_api_key=os.getenv("AZURE_OPENAI_EMB_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_EMB_ENDPOINT"),
    openai_api_version=os.getenv("AZURE_OPENAI_EMB_API_VERSION", "2024-12-01-preview"),
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
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_MODEL"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
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
