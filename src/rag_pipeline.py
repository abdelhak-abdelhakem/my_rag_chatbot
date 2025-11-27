# --- LangChain Core Imports ---
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# --- LangChain Google’s Generative AI models Integrations ---
from langchain_google_genai import ChatGoogleGenerativeAI
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

def create_rag_chain(vectorstore, k_val, google_api_key, llm_model_name, llm_max_tokens=2048, llm_temp=0.0):
    """
    Creates and returns the RAG (Retrieval-Augmented Generation) chain with conversation memory.
    """
    print("--- Setting up LLM and Retriever ---")
    
    # 1. Set up the Retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k_val}
    )
    # 2. Set up the LLM (Gemini 2.5 Flash)
    llm = ChatGoogleGenerativeAI(
        model=llm_model_name,
        temperature=llm_temp,
        max_output_tokens=llm_max_tokens,
        google_api_key=google_api_key,
    )

    print(f"LLM ready: {llm_model_name}")
    
    # 3. Define the Prompt Template with Chat History
    template = """You are UniBot DZ, the specialized AI assistant for Djillali Liabes University.
Your goal is to provide accurate, administrative, and academic assistance to students.

### INSTRUCTIONS:
1. **Source of Truth:** Answer strictly based on the "Retrieved Context" below. Do not invent administrative rules, dates, or procedures.
2. **Uncertainty:** If the answer is not in the context or chat history, strictly state: "I cannot find this information in the official documents provided."
3. **Formatting:** Use Markdown to make answers readable. Use **bold** for dates/deadlines and bullet points for lists (e.g., required files, modules).
4. **Language:** Answer in the same language the student used (French, English, or Arabic).
5. **Tone:** Professional, encouraging, and direct.

### CONTEXT:
Conversation History:
{chat_history}

Retrieved Official Documents:
{context}

### INPUT:
Student Question: {question}

Answer:"""
    
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