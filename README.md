# 🎓 UniBot DZ - AI University Assistant

**UniBot DZ** is a Retrieval-Augmented Generation (RAG) chatbot designed to assist students of **Djillali Liabes University**. It uses official university documents (PDFs) to answer academic and administrative questions accurately.

The bot utilizes **OpenAI** for high-quality text embeddings and **Google Gemini (2.5 Flash)** for fast, intelligent response generation.

## 🚀 Features

* **RAG Architecture:** Answers based strictly on provided official PDF documents to minimize hallucinations.
* **Hybrid Power:** Combines OpenAI Embeddings (`text-embedding-3-small`) with Google Gemini (`gemini-2.5-flash`).
* **Conversational Memory:** Remembers context from previous messages in the chat session.
* **Dual Interface:**
    * 🖥️ **Command Line Interface (CLI):** For quick testing and debugging.
    * 🌐 **Web Interface (Streamlit):** A user-friendly web app for students.
* **Optimized Indexing:** Checks for an existing FAISS vector index to avoid reprocessing PDFs on every restart.

## 📂 Project Structure

```text
my_rag_chatbot/
│
├── docs/                   # 📄 Place your PDF documents here
├── my_faiss_index/         # 💾 Saved FAISS vector index (auto-generated)
├── src/
│   ├── config.py           # ⚙️ Configuration (Models, API Keys, Constants)
│   ├── document_processor.py # 🛠️ PDF loading & Text splitting logic
│   ├── vector_store_manager.py # 🗄️ Vector store creation & saving
│   ├── rag_pipeline.py     # 🧠 RAG Chain construction & Prompting
│   ├── main.py             # 🖥️ CLI Chatbot entry point
│   └── app.py              # 🌐 Streamlit Web App entry point
│
├── requirements.txt        # 📦 Python dependencies
├── README.md               # 📖 Documentation
└── .env                    # 🔑 API Keys (Keep secret!)
```
## 🛠️ Installation & Setup
### 1. Clone the Repository
```bash
git clone https://github.com/abdelhak-abdelhakem/my_rag_chatbot/
cd my_rag_chatbot
```
### 2. Create a Virtual Environment
It is recommended to use a virtual environment.
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
### 4. Configure API Keys
```bash
OPENAI_API_KEY=sk-proj-your-openai-key-here...
GOOGLE_API_KEY=AIzaSy-your-google-key-here...
```
### 5. Add Documents
Place your university PDF documents inside the docs/ folder. The system will automatically read all .pdf files found there.

## 🏃‍♂️ Usage
### Option A: Run the Web Interface (Streamlit)
This is the main interface for end-users.
```bash
streamlit run src/app.py
```
* Open your browser at http://localhost:8501.

* Note: The first time you run this, it will take a moment to process the PDFs and build the my_faiss_index. Subsequent runs will be instant.

### Option B: Run the Terminal Chatbot (CLI)
Useful for quick debugging or testing the pipeline without a UI.
```bash
python src/main.py
```
* Type your question and press Enter.
* Type exit to quit or clear to reset memory.

## 🧠 How It Works (Technical)

1.  **Ingestion:** `document_processor.py` loads PDFs using **PyMuPDF** and splits them into chunks (Size: 1000, Overlap: 200).
2.  **Embedding:** `vector_store_manager.py` converts these chunks into vectors using **OpenAI's `text-embedding-3-small`**.
3.  **Storage:** The vectors are stored locally using **FAISS** in the `my_faiss_index/` directory.
4.  **Retrieval:** When a user asks a question, the system searches the top 15 most similar chunks (`config.RETRIEVER_K = 15`).
5.  **Generation:** `rag_pipeline.py` sends the retrieved context + chat history + user question to **Gemini 2.5 Flash**, which generates the final answer.

## ⚙️ Customization

You can tweak the system parameters in `src/config.py`:

* **Chunk Size:** Adjust `CHUNK_SIZE` and `CHUNK_OVERLAP` for better context handling.
* **Retriever K:** Change `RETRIEVER_K` to retrieve more or fewer documents.
* **LLM Model:** Switch `LLM_MODEL_NAME` if you want to use a different Gemini version .

## 🤝 Contributing

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature-branch`).
3.  Commit your changes.
4.  Push to the branch and open a Pull Request.

## 👤 Author

**Abdelhak Abdelhakem**

* 🎓 AI Engineering Student | Future LLM Engineer
* 📍 Algeria
* 📧 [abdelhakemabdelhak@gmail.com](mailto:abdelhakemabdelhak@gmail.com)
* 🔗 [LinkedIn Profile](https://www.linkedin.com/in/abdelhak-abdelhakem/)
