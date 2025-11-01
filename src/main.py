import config
from document_processor import load_and_split_docs
from vector_store_manager import get_or_create_vector_store
from rag_pipeline import create_rag_chain

def main():
    # 1. Load and split documents (only needed if vector store doesn't exist)
    # We pass the chunks to the next step, which will only use them
    # if the index needs to be built.
    doc_chunks = load_and_split_docs(
        directory_path=config.PDF_DIRECTORY_PATH,
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )

    # 2. Get or create the vector store (with persistence)
    vectorstore = get_or_create_vector_store(
        index_path=config.FAISS_INDEX_PATH,
        embedding_model_name=config.EMBEDDING_MODEL_NAME,
        doc_chunks=doc_chunks
    )

    # 3. Create the RAG chain
    rag_chain = create_rag_chain(
        vectorstore=vectorstore,
        k_val=config.RETRIEVER_K,
        llm_repo_id=config.LLM_REPO_ID,
        llm_task=config.LLM_TASK,
        llm_token=config.HUGGINGFACE_API_TOKEN
    )

    # 4. Start the chat loop
    print("\n--- 6. Chatbot is Ready! Type 'exit' to quit. ---")
    while True:
        try:
            user_query = input("\nYou: ")
            if user_query.lower() == 'exit':
                print("Goodbye!")
                break

            # Invoke the RAG chain to get a response
            response = rag_chain.invoke(user_query)
            print(f"Bot: {response}")

        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()