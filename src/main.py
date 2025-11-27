import config
from document_processor import load_and_split_docs
from vector_store_manager import get_or_create_vector_store
from rag_pipeline import create_rag_chain
from langchain_core.messages import HumanMessage, AIMessage

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
    # 4. Initialize conversation history
    chat_history = []

    # 5. Start the chat loop
    print("\n--- Chatbot is Ready! Type 'exit' to quit, 'clear' to reset conversation history. ---")
    while True:
        try:
            user_query = input("\nYou: ").strip()
            if user_query.lower() == 'exit':
                print("Goodbye!")
                break
            
            if user_query.lower() == 'clear':
                chat_history = []
                print("Conversation history cleared!")
                continue    

           # Invoke the RAG chain with the question and chat history
            response = rag_chain.invoke({
                "question": user_query,
                "chat_history": chat_history
            })
            print(f"Bot: {response}")

            # Update chat history
            chat_history.append(HumanMessage(content=user_query))
            chat_history.append(AIMessage(content=response))
            
            # Optional: Limit history to last N exchanges to avoid context overflow
            MAX_HISTORY_LENGTH = 10  # Keep last 10 messages (5 exchanges)
            if len(chat_history) > MAX_HISTORY_LENGTH:
                chat_history = chat_history[-MAX_HISTORY_LENGTH:]
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Continuing conversation...")

if __name__ == "__main__":
    main()