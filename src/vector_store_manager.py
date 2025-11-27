import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import config

def get_or_create_vector_store(doc_chunks) -> FAISS:
    """
    Loads the vector store from disk if it exists, otherwise creates it
    from the document chunks and saves it to disk.
    """
    # Initialize the embedding model
    embeddings = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL_NAME,)

    if os.path.exists(config.FAISS_INDEX_PATH):
        print(f"--- Loading Existing Vector Store from {config.FAISS_INDEX_PATH} ---")
        vectorstore = FAISS.load_local(
                folder_path=config.FAISS_INDEX_PATH, 
                embeddings=embeddings, 
                allow_dangerous_deserialization=True 
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
        vectorstore.save_local(config.FAISS_INDEX_PATH)
        print(f"Vector store created and saved to {config.FAISS_INDEX_PATH}.")
        
    return vectorstore