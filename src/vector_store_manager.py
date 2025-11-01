import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

def get_or_create_vector_store( index_path: str, embedding_model_name: str, doc_chunks) -> FAISS:
    """
    Loads the vector store from disk if it exists, otherwise creates it
    from the document chunks and saves it to disk.
    """
    # Initialize the embedding model
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    if os.path.exists(index_path):
        print(f"--- Loading Existing Vector Store from {index_path} ---")
        vectorstore = FAISS.load_local(
            index_path, 
            embeddings, 
            allow_dangerous_deserialization=True  # Required for FAISS
        )
        print("Vector store loaded successfully.")
    else:
        print("--- Creating New Vector Store ---")
        if not doc_chunks:
            raise ValueError("No document chunks found to create a new vector store.")
            
        vectorstore = FAISS.from_documents(
            documents=doc_chunks,
            embedding=embeddings
        )
        print("Saving vector store to disk...")
        vectorstore.save_local(index_path)
        print(f"Vector store created and saved to {index_path}.")
        
    return vectorstore