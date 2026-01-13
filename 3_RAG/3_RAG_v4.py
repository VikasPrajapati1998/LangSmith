import os
import json
import hashlib
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

from langsmith import traceable

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv(find_dotenv())

# Configure LangSmith tracing
# os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = 'PDF-RAG-Pipeline'
# os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'

# Configuration
PDF_PATH = "ISLR.pdf"  # change to your file
INDEX_ROOT = Path(".indices")
INDEX_ROOT.mkdir(exist_ok=True)

# ----------------- helpers (traced) -----------------
@traceable(name="Load PDF", tags=["ingestion"])
def load_pdf(path: str):
    """Load PDF document from the given path."""
    print(f"Loading PDF from: {path}")
    try:
        loader = PyPDFLoader(path)
        docs = loader.load()
        print(f"Loaded {len(docs)} pages")
        return docs
    except Exception as e:
        print(f"Error loading PDF: {e}")
        raise

@traceable(name="Split Documents", tags=["ingestion"])
def split_documents(docs, chunk_size=1000, chunk_overlap=150):
    """Split documents into chunks for embedding."""
    print(f"Splitting {len(docs)} documents into chunks (size={chunk_size}, overlap={chunk_overlap})")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    splits = splitter.split_documents(docs)
    print(f"Created {len(splits)} chunks")
    return splits

@traceable(name="Build Vectorstore", tags=["ingestion", "embeddings"])
def build_vectorstore(splits, embed_model_name: str):
    """Generate embeddings and build FAISS vectorstore."""
    print(f"Generating embeddings using model: {embed_model_name}")
    print(f"Processing {len(splits)} chunks...")
    embedding = OllamaEmbeddings(model=embed_model_name)
    vectorstore = FAISS.from_documents(splits, embedding)
    print("Vectorstore created successfully")
    return vectorstore

# ----------------- cache key / fingerprint -----------------
@traceable(name="Calculate File Fingerprint", tags=["cache"])
def _file_fingerprint(path: str) -> dict:
    """Calculate SHA256 hash and metadata for cache validation."""
    print(f"Calculating fingerprint for: {path}")
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    fingerprint = {
        "sha256": h.hexdigest(), 
        "size": p.stat().st_size, 
        "mtime": int(p.stat().st_mtime)
    }
    print(f"Fingerprint: {fingerprint['sha256'][:16]}...")
    return fingerprint

@traceable(name="Generate Index Key", tags=["cache"])
def _index_key(pdf_path: str, chunk_size: int, chunk_overlap: int, embed_model_name: str) -> str:
    """Generate unique cache key based on PDF and configuration."""
    print("Generating index key...")
    meta = {
        "pdf_fingerprint": _file_fingerprint(pdf_path),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_model": embed_model_name,
        "format": "v1",
    }
    key = hashlib.sha256(json.dumps(meta, sort_keys=True).encode("utf-8")).hexdigest()
    print(f"Index key: {key[:16]}...")
    return key

# ----------------- explicitly traced load/build runs -----------------
@traceable(name="Load Cached Index", tags=["index", "cache"])
def load_index_run(index_dir: Path, embed_model_name: str):
    """Load pre-built FAISS index from disk."""
    print(f"Loading cached index from: {index_dir}")
    embedding = OllamaEmbeddings(model=embed_model_name)
    vectorstore = FAISS.load_local(
        str(index_dir),
        embedding,
        allow_dangerous_deserialization=True
    )
    print("Index loaded successfully from cache")
    return vectorstore

@traceable(name="Build New Index", tags=["index", "build"])
def build_index_run(pdf_path: str, index_dir: Path, chunk_size: int, chunk_overlap: int, embed_model_name: str):
    """Build new FAISS index from PDF."""
    print(f"Building new index at: {index_dir}")
    
    # Load and process PDF (child traces)
    docs = load_pdf(pdf_path)
    splits = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    vs = build_vectorstore(splits, embed_model_name)
    
    # Save index and metadata
    index_dir.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(index_dir))
    
    meta_data = {
        "pdf_path": os.path.abspath(pdf_path),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_model": embed_model_name,
        "num_chunks": len(splits),
        "num_pages": len(docs),
    }
    (index_dir / "meta.json").write_text(json.dumps(meta_data, indent=2))
    print(f"Index built and saved successfully")
    
    return vs

# ----------------- dispatcher (traced) -----------------
@traceable(name="Load or Build Index", tags=["index"])
def load_or_build_index(
    pdf_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    embed_model_name: str = "nomic-embed-text",
    force_rebuild: bool = False,
):
    """Load cached index or build new one if needed."""
    print("\n" + "="*80)
    print("INDEXING PHASE")
    print("="*80)
    
    key = _index_key(pdf_path, chunk_size, chunk_overlap, embed_model_name)
    index_dir = INDEX_ROOT / key
    cache_hit = index_dir.exists() and not force_rebuild
    
    if cache_hit:
        print("✓ Cache HIT - loading existing index")
        return load_index_run(index_dir, embed_model_name)
    else:
        print("✗ Cache MISS - building new index")
        return build_index_run(pdf_path, index_dir, chunk_size, chunk_overlap, embed_model_name)

# ----------------- model, prompt, and pipeline -----------------
@traceable(name="Initialize LLM", tags=["setup"])
def initialize_llm(model_name: str = "qwen2.5:0.5b", temperature: float = 0.7):
    """Initialize the language model."""
    print(f"Initializing LLM: {model_name}")
    return ChatOllama(model=model_name, temperature=temperature)

# Create prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer ONLY based on the provided context. "
               "If the answer is not in the context, clearly state that you don't know."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

@traceable(name="Format Retrieved Documents", tags=["retrieval"])
def format_docs(docs):
    """Format retrieved documents into a single context string."""
    formatted = "\n\n".join(f"[Document {i+1}]\n{d.page_content}" for i, d in enumerate(docs))
    print(f"Formatted {len(docs)} documents into context")
    return formatted

@traceable(name="Setup Retriever", tags=["setup", "retrieval"])
def setup_retriever(vectorstore, k: int = 4):
    """Configure the retriever with search parameters."""
    print(f"Setting up retriever with k={k}")
    return vectorstore.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": k}
    )

@traceable(name="Build RAG Chain", tags=["setup"])
def build_rag_chain(retriever, llm):
    """Construct the complete RAG chain."""
    print("Building RAG chain...")
    
    parallel = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    })
    
    chain = parallel | prompt | llm | StrOutputParser()
    print("RAG chain built successfully")
    return chain

@traceable(name="Query RAG System", tags=["query"])
def query_rag_system(chain, question: str, k: int = 4):
    """Execute a query through the RAG chain."""
    print(f"\nQuerying: {question}")
    result = chain.invoke(
        question,
        config={
            "run_name": "rag_query_execution", 
            "tags": ["qa"], 
            "metadata": {"k": k, "question_length": len(question)}
        }
    )
    print(f"Query completed")
    return result

@traceable(name="PDF RAG Full Pipeline")
def setup_pipeline_and_query(
    pdf_path: str,
    question: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    embed_model_name: str = "nomic-embed-text",
    llm_model_name: str = "qwen2.5:0.5b",
    llm_temperature: float = 0.7,
    retrieval_k: int = 4,
    force_rebuild: bool = False,
):
    """
    Complete RAG pipeline: index PDF, retrieve relevant chunks, and generate answer.
    
    Args:
        pdf_path: Path to the PDF file
        question: User's question
        chunk_size: Size of text chunks for splitting
        chunk_overlap: Overlap between chunks
        embed_model_name: Name of the embedding model
        llm_model_name: Name of the LLM model
        llm_temperature: Temperature for LLM generation
        retrieval_k: Number of documents to retrieve
        force_rebuild: Force rebuild of index
        
    Returns:
        str: Generated answer
    """
    print("\n" + "="*80)
    print("PDF RAG PIPELINE STARTED")
    print("="*80)
    
    # Phase 1: Indexing
    vectorstore = load_or_build_index(
        pdf_path=pdf_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embed_model_name=embed_model_name,
        force_rebuild=force_rebuild,
    )
    
    # Phase 2: Setup
    print("\n" + "="*80)
    print("SETUP PHASE")
    print("="*80)
    llm = initialize_llm(llm_model_name, llm_temperature)
    retriever = setup_retriever(vectorstore, k=retrieval_k)
    chain = build_rag_chain(retriever, llm)
    
    # Phase 3: Query
    print("\n" + "="*80)
    print("QUERY PHASE")
    print("="*80)
    answer = query_rag_system(chain, question, k=retrieval_k)
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETED")
    print("="*80)
    
    return answer

# ----------------- Main -----------------
@traceable(name="Main")
def main():
    # Verify LangSmith configuration
    if not os.getenv('LANGCHAIN_API_KEY'):
        print("⚠️  Warning: LANGCHAIN_API_KEY not found in environment variables.")
        print("LangSmith tracing will not work without a valid API key.")
        print("Set it in your .env file:\n")
        print("LANGCHAIN_TRACING_V2=true")
        print("LANGCHAIN_API_KEY=your_api_key_here")
        print("LANGCHAIN_PROJECT=PDF-RAG-Pipeline\n")
    else:
        print("✓ LangSmith tracing enabled")
        print(f"✓ Project: {os.getenv('LANGCHAIN_PROJECT')}\n")
    
    print("="*80)
    print("PDF RAG System Ready")
    print("="*80)
    print(f"PDF File: {PDF_PATH}")
    print("Ask a question (or Ctrl+C to exit)\n")
    
    try:
        question = input("Q: ").strip()
        if not question:
            print("No question provided. Exiting.")
            exit(0)
            
        answer = setup_pipeline_and_query(
            pdf_path=PDF_PATH,
            question=question,
            chunk_size=1000,
            chunk_overlap=150,
            embed_model_name="nomic-embed-text",
            llm_model_name="qwen2.5:0.5b",
            llm_temperature=0.7,
            retrieval_k=4,
            force_rebuild=False
        )
        
        print("\n" + "="*80)
        print("ANSWER:")
        print("="*80)
        print(answer)
        print("="*80)
        
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except FileNotFoundError:
        print(f"\n❌ Error: PDF file '{PDF_PATH}' not found!")
        print("Please update PDF_PATH variable with your PDF file path.")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__": 
    main()

