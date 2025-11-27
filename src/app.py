import streamlit as st
import config  
from document_processor import load_and_split_docs
from vector_store_manager import get_or_create_vector_store
from rag_pipeline import create_rag_chain
from langchain_core.messages import HumanMessage, AIMessage

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="UniBot DZ",
    page_icon="🎓",
    layout="wide"
)
st.title("🎓 UniBot DZ - Your University Assistant")
st.write("Ask me anything about your university documents!")

# --- 2. Load and Initialize the RAG Chain ---
@st.cache_resource
def load_rag_chain():
    """
    Loads and splits docs, creates the vector store,
    and returns the RAG chain.
    """
    print("--- Initializing RAG Chain (this should run once) ---")
    doc_chunks = load_and_split_docs(
        directory_path=config.PDF_DIRECTORY_PATH,
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    
    vectorstore = get_or_create_vector_store(
        doc_chunks=doc_chunks
    )
    
    rag_chain = create_rag_chain(
        vectorstore=vectorstore,
        k_val=config.RETRIEVER_K,
        llm_model_name=config.LLM_MODEL_NAME,
        google_api_key=config.GOOGLE_API_KEY
    )
    print("--- RAG Chain Loaded Successfully ---")
    return rag_chain

rag_chain = load_rag_chain()

# --- 3. Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 4. Display Chat History ---
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# --- 5. Chat Input and RAG Chain Invocation ---
if prompt := st.chat_input("Ask me anything..."):
    
    # 5.1. Add user's message to state and display it
    user_message = HumanMessage(content=prompt)
    st.session_state.messages.append(user_message)
    with st.chat_message("user"):
        st.markdown(prompt)

    # 5.2. Get the bot's response
    with st.spinner("Thinking..."):
        try:
            # Send the question AND the chat history to the chain
            response = rag_chain.invoke({
                "question": prompt,
                "chat_history": st.session_state.messages
            })
            
            bot_response = AIMessage(content=response)
        
        except Exception as e:
            st.error(f"An error occurred: {e}")
            bot_response = AIMessage(content="Sorry, I ran into an error.")

    # 5.3. Add bot's response to state and display it
    st.session_state.messages.append(bot_response)
    with st.chat_message("assistant"):
        st.markdown(bot_response.content)

# Add a "Clear History" button
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun() # Rerun the app to clear the screen