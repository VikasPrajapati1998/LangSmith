# pip install -U langchain langchain-openai langchain-community faiss-cpu pypdf python-dotenv

import os
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv(find_dotenv())
os.environ['LANGCHAIN_PROJECT'] = "RAG_v1"

PDF_PATH = "ISLR.pdf"

# ========== 1: Load PDF ==========
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()

# ========== 2: Chunk ==========
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
splits = splitter.split_documents(docs)

# ========== 3: Embed + Index ==========
embedding = OllamaEmbeddings(model="nomic-embed-text")
vector_store = FAISS.from_documents(splits, embedding)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# ========== 4: Prompt ==========
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. If not found, say you don't know."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

# ========== 5: Chain ==========
llm = ChatOllama(
    model="qwen2.5:0.5b",   # llama3.2:1b, qwen2.5:0.5b
    temperature=0.7
)

def format_docs(docs): 
    return "\n\n".join(doc.page_content for doc in docs)

parallel = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

chain = parallel | prompt | llm | StrOutputParser()

# ========== 6: Invoke ==========
print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")
q = input("\nQ: ")
ans = chain.invoke(q.strip())
print("\nA:", ans)
