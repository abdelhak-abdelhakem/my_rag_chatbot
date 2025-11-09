# --- LangChain Core Imports ---
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# --- LangChain Hugging Face Integrations ---
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

def format_docs(docs):
    """
    Formats the retrieved documents into a single string.
    """
    return "\n\n".join(doc.page_content for doc in docs)

def format_chat_history(chat_history):
    """
    Formats the chat history into a readable string with clear labels.
    """
    if not chat_history:
        return "No previous conversation."
    
    formatted = []
    for i, message in enumerate(chat_history):
        if isinstance(message, HumanMessage):
            formatted.append(f"User said: {message.content}")
        elif isinstance(message, AIMessage):
            formatted.append(f"Assistant replied: {message.content}")
    
    return "\n".join(formatted)

def create_rag_chain(vectorstore, k_val, llm_repo_id, llm_task, llm_token, llm_max_tokens=256, llm_temp=0.5):
    """
    Creates and returns the RAG (Retrieval-Augmented Generation) chain with conversation memory.
    """
    print("--- Setting up LLM and Retriever ---")
    
    # 1. Set up the Retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k_val}
    )
    
    # 2. Set up the LLM (Zephyr-7B-beta)
    llm_endpoint = HuggingFaceEndpoint(
        repo_id=llm_repo_id,
        task=llm_task,
        max_new_tokens=llm_max_tokens,
        temperature=llm_temp,
        huggingfacehub_api_token=llm_token,
    )
    llm = ChatHuggingFace(llm=llm_endpoint)
    print("LLM and retriever are ready.")
    
    # 3. Define the Prompt Template with Chat History
    template = """You are a helpful assistant. Answer the question based on:
1. The conversation history (MOST IMPORTANT - check this first!)
2. The retrieved context from documents
3. Your general knowledge

IMPORTANT: If the answer is in the conversation history, use it! Don't say you don't know if it was mentioned before.

Conversation History:
{chat_history}

Retrieved Context from Documents:
{context}

Current Question: {question}

Answer (keep it concise and natural):"""
    
    prompt = PromptTemplate.from_template(template)
    
    # 4. Create the RAG Chain with Memory Support
    # This chain expects a dict with 'question' and 'chat_history' keys
    rag_chain = (
        {
            "context": lambda x: format_docs(retriever.invoke(x["question"])),
            "question": lambda x: x["question"],
            "chat_history": lambda x: format_chat_history(x.get("chat_history", []))
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("--- RAG Chain with Memory Created Successfully ---")
    return rag_chain