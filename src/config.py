import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- API Keys ---
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# --- Model Definitions ---
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_REPO_ID = "HuggingFaceH4/zephyr-7b-beta"
LLM_TASK = "conversational"

# --- Text Splitting ---
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# --- Retriever ---
RETRIEVER_K = 3  

# --- File Paths ---
PDF_DIRECTORY_PATH = "docs"
FAISS_INDEX_PATH = "my_faiss_index" # Directory to save/load the index