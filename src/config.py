import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- API Keys ---
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Model Definitions ---
#EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
LLM_MODEL_NAME = "gemini-2.5-flash"
#LLM_REPO_ID = "HuggingFaceH4/zephyr-7b-beta"
#LLM_TASK = "conversational"

# --- Text Splitting ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- Retriever ---
RETRIEVER_K = 5  

# --- File Paths ---
PDF_DIRECTORY_PATH = "docs"
FAISS_INDEX_PATH = "my_faiss_index" # Directory to save/load the index