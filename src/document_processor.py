import glob
from langchain_community.document_loaders import PyMuPDFLoader
file_path = "./example_data/layout-parser-paper.pdf"
loader = PyMuPDFLoader(file_path)
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_split_docs( directory_path: str,chunk_size: int,chunk_overlap: int) :
    """
    Loads all PDFs from a directory, splits them into chunks, and returns
    a list of document chunks.
    """
    print(f"--- Loading PDFs from {directory_path} ---")
    pdf_files = glob.glob(f"{directory_path}/*.pdf")
    if not pdf_files:
        print(f"No PDF files found in {directory_path}.")
        return []
        
    print(f"Found {len(pdf_files)} PDFs.")
    
    # 1. Load Documents
    documents = []
    for file in pdf_files:
        try:
            loader = PyMuPDFLoader(file) 
            docs = loader.load()
            documents.extend(docs)
        except Exception as e:
            print(f"Error loading {file}: {e}")
    print(f"Loaded {len(documents)} document pages.")

    # 2. Split Documents
    print(f"--- Splitting Documents into Chunks (Size: {chunk_size}, Overlap: {chunk_overlap}) ---")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    doc_chunks = text_splitter.split_documents(documents)
    print(f"Total {len(doc_chunks)} chunks created.")
    
    return doc_chunks