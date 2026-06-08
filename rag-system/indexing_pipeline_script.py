# =============================
# MAIN PIPELINE (only runs if checks pass)
# ======================

# Imports
from dotenv import load_dotenv
import getpass
import os
import sys
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredMarkdownLoader
from langchain_text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

if not os.getenv("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"]=getpass.getpass("Enter your Gemini API key")
    
print(f"GEMINI KEY {os.getenv("GEMINI_API_KEY")}")

def run_rag_pipeline(file_path):
    
    # STEP 1: Load documents
    print(f"\n[1/4] Loading document: {file_path}")
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.txt'):
        loader = TextLoader(file_path)
    elif file_path.endswith('.md'):
        loader = UnstructuredMarkdownLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    docs = loader.load()
    print(f"✅ Loaded {len(docs)} document(s)")
   
    # STEP 2: Chunk documents
    print(f"\n[2/4] Chunking documents")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(docs)
    print(f"✅ Create {len(chunks)} chunks")
    
    # STEP 3: Initialize embeddings
    print(f"\n[3/4] Initializing embedding model")
    embeddings=GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
    print(f"✅ Embedding model ready")
 
    # STEP 4: Create vector store
    print(f"\n[4/4] Creating vector store")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embeddings=embeddings,
        persist_directory="./chroma_db"
    )
    print(f"✅ Vector store ready")
    
    # Step 5: Test a query
    query = "What is this document about?"
    results = vectorstore.similiary_search(query)
    print(f"Query: '{query}'")
    print(f"Found {len(results)} relevant chunks")
    
    for i, result in enumerate(results, 1):
        print(f"\n Result {i}")
        print(f"    {result.page_content[:150]}...")

# =================
# MAIN EXECUTION
# =================

if __name__ == "__main__":
    
    file_path="test_doc.txt"
    vectorstore = run_rag_pipeline(file_path)