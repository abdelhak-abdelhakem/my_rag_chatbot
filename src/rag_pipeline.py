# --- LangChain Core Imports ---
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- LangChain Hugging Face Integrations ---
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

def format_docs(docs):
    """
    Formats the retrieved documents into a single string.
    """
    return "\n\n".join(doc.page_content for doc in docs)

def create_rag_chain(vectorstore, k_val, llm_repo_id, llm_task, llm_token, llm_max_tokens=256, llm_temp=0.5):
    """
    Creates and returns the RAG (Retrieval-Augmented Generation) chain.
    """
    print("--- Setting up LLM and Retriever ---")
    
    # 1. Set up the Retriever
    # Changed to k=k_val (which will be 3 from config)
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

    # 3. Define the Prompt Template
    template = """
You are a helpful assistant. Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know. Keep the answer concise.

Context: {context}
Question: {question}
Helpful Answer:
"""
    prompt = PromptTemplate.from_template(template)

    # 4. Create the RAG Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("--- RAG Chain Created Successfully ---")
    return rag_chain