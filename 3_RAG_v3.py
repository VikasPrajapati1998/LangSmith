# pip install -U langchain langchain-openai langchain-community faiss-cpu pdfplumber python-dotenv

import os
import pickle
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv(find_dotenv())
os.environ['LANGCHAIN_PROJECT'] = "RAG_v1"

PDF_PATH = "ISLR.pdf"
VECTOR_STORE_DIR = "vector_stores"
METADATA_FILE = "pdf_metadata.pkl"

def get_vector_store_path(pdf_path):
    """Generate a unique vector store path based on PDF filename."""
    pdf_name = Path(pdf_path).stem
    return os.path.join(VECTOR_STORE_DIR, pdf_name)

def load_metadata():
    """Load metadata about previously processed PDFs."""
    metadata_path = os.path.join(VECTOR_STORE_DIR, METADATA_FILE)
    if os.path.exists(metadata_path):
        with open(metadata_path, 'rb') as f:
            return pickle.load(f)
    return {}

def save_metadata(metadata):
    """Save metadata about processed PDFs."""
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    metadata_path = os.path.join(VECTOR_STORE_DIR, METADATA_FILE)
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)

def should_regenerate_vector_store(pdf_path):
    """Check if vector store needs to be regenerated."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    vector_store_path = get_vector_store_path(pdf_path)
    
    # Check if vector store exists
    if not os.path.exists(vector_store_path):
        return True
    
    # Check metadata
    metadata = load_metadata()
    pdf_mtime = os.path.getmtime(pdf_path)
    
    if pdf_path not in metadata:
        return True
    
    # Check if PDF has been modified since last processing
    if metadata[pdf_path]['mtime'] != pdf_mtime:
        return True
    
    return False

def create_and_save_vector_store(pdf_path, embedding):
    """Create vector store from PDF and save it."""
    print(f"Processing PDF: {pdf_path}")
    
    # Load PDF
    loader = PDFPlumberLoader(pdf_path)
    docs = loader.load()
    print(f"Loaded {len(docs)} pages")
    
    # Chunk
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = splitter.split_documents(docs)
    print(f"Created {len(splits)} chunks")
    
    # Embed + Index
    print("Creating embeddings and vector store...")
    vector_store = FAISS.from_documents(splits, embedding)
    
    # Save vector store
    vector_store_path = get_vector_store_path(pdf_path)
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    vector_store.save_local(vector_store_path)
    print(f"Vector store saved to: {vector_store_path}")
    
    # Update metadata
    metadata = load_metadata()
    metadata[pdf_path] = {
        'mtime': os.path.getmtime(pdf_path),
        'num_chunks': len(splits)
    }
    save_metadata(metadata)
    
    return vector_store

def load_vector_store(pdf_path, embedding):
    """Load existing vector store."""
    vector_store_path = get_vector_store_path(pdf_path)
    print(f"Loading existing vector store from: {vector_store_path}")
    vector_store = FAISS.load_local(
        vector_store_path, 
        embedding,
        allow_dangerous_deserialization=True
    )
    return vector_store

# ========== Initialize Embedding Model ==========
embedding = OllamaEmbeddings(model="nomic-embed-text")

# ========== Load or Create Vector Store ==========
if should_regenerate_vector_store(PDF_PATH):
    print("=" * 50)
    print("Creating new vector store...")
    print("=" * 50)
    vector_store = create_and_save_vector_store(PDF_PATH, embedding)
else:
    print("=" * 50)
    print("Loading cached vector store...")
    print("=" * 50)
    vector_store = load_vector_store(PDF_PATH, embedding)

retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 20})

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
print("=" * 50)
print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")
print("=" * 50)
q = input("\nQ: ")
ans = chain.invoke(q.strip())
print("\nA:", ans)

