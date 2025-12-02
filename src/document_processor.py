import os
import glob
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    Docx2txtLoader,
    TextLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Map file extensions to their specific Loaders
LOADER_MAPPING = {
    ".pdf": PyMuPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
}

def load_and_split_docs(directory_path: str, chunk_size: int, chunk_overlap: int):
    """
    Loads documents from a directory (PDF, DOCX, TXT, MD), splits them,
    and returns a list of chunks.
    """
    print(f"--- Loading documents from {directory_path} ---")
    
    # Get all files in the directory
    all_files = glob.glob(f"{directory_path}/*")
    
    documents = []
    
    for file_path in all_files:
        # Get the file extension (e.g., '.pdf')
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext in LOADER_MAPPING:
            try:
                loader_class = LOADER_MAPPING[ext]
                # Initialize the specific loader for this file type
                loader = loader_class(file_path)
                
                print(f"Loading {ext} file: {os.path.basename(file_path)}")
                docs = loader.load()
                documents.extend(docs)
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        else:
            # Skip unsupported files (like images or system files)
            print(f"Skipping unsupported file type: {file_path}")

    print(f"Total loaded pages/documents: {len(documents)}")

    if not documents:
        print("No documents were loaded. Exiting.")
        return []

    # 2. Split Documents
    print(f"--- Splitting Documents (Size: {chunk_size}, Overlap: {chunk_overlap}) ---")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    doc_chunks = text_splitter.split_documents(documents)
    print(f"Total {len(doc_chunks)} chunks created.")
    
    return doc_chunks