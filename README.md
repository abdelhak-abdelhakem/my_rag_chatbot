# My First LangChain RAG Chatbot

This is my **first Retrieval-Augmented Generation (RAG)** chatbot project — built using **LangChain**, **Hugging Face**, and **FAISS**.  
It can read your PDF documents, store their contents as vector embeddings, and answer your questions accurately based on the document context.

---

## Project Overview

This chatbot was designed as my **first hands-on project** in the RAG ecosystem — a foundational step in my path toward becoming an **LLM Engineer**.

The chatbot:
- Loads and processes PDFs from a local `docs/` directory.  
- Splits the text into chunks for better retrieval.  
- Generates embeddings using a **Sentence Transformers model**.  
- Stores them in a **FAISS** vector database for efficient similarity search.  
- Uses **Zephyr-7B-Beta** via the Hugging Face API to generate context-aware answers.

---

## Project Structure
```
my_rag_chatbot/
│
├── docs/ # Folder containing your PDF documents
├── my_faiss_index/ # Saved FAISS vector index
├── src/
│ ├── config.py # Configuration (models, paths, parameters)
│ ├── document_processor.py # Document loading & text splitting
│ ├── vector_store_manager.py # Vector store creation & persistence
│ ├── rag_pipeline.py # RAG chain (retriever + LLM)
│ └── main.py # Entry point (chat loop)
│
├── requirements.txt # Project dependencies
├── README.md # Project documentation
└── .env # API keys (not uploaded for security)
```

---

## Technologies Used

- **LangChain** (core RAG framework)  
- **Hugging Face Hub** (Zephyr-7B-Beta LLM)  
- **FAISS** (vector similarity search)  
- **Sentence Transformers** (embeddings)  
- **Python Dotenv** (environment variable management)

---

## How to Run the Project

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/my_rag_chatbot.git
cd my_rag_chatbot
```

###  2. Create and activate a virtual environment
```bash
python -m venv rag_env
source rag_env/bin/activate   # On Windows: rag_env\Scripts\activate
```

###  3. Install dependencies
```bash
pip install -r requirements.txt
```

###  4. Add your environment variables
Create a .env file in the project root with:
```bash
HUGGINGFACEHUB_API_TOKEN=your_api_key_here
```

###  5. Run the chatbot
```bash
python src/main.py
```

### 6. Chat with your bot
Type your questions in the terminal.
To exit, type exit.

## Example
```bash
You: What is the full name of Abdelhak?
Bot: Abdelhak Abdelhakem is a fourth-year AI Engineering student specializing in LLMs and NLP...
```

## What I Learned
* How to implement document-based retrieval with FAISS
* How RAG pipelines combine retrieval + generation
* How to structure modular AI projects in Python
* Basics of integrating LangChain with Hugging Face APIs

## Author
**Abdelhak Abdelhakem**  
🎓 AI Engineering Student | Future LLM Engineer  
📍 Algeria  
📧 [abdelhakemabdelhak@gmail.com](mailto:abdelhakemabdelhak@gmail.com)  
🔗 [LinkedIn Profile](https://www.linkedin.com/in/abdelhak-abdelhakem/)

## Future Improvements

* Add support for multiple document types
* Improve prompt templates for precision
* Build a Streamlit interface
* Integrate metadata-based filtering
