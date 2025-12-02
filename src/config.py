import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY not found! "
        "Please add it to your .env file."
    )
if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY not found! "
        "Please add it to your .env file."
    )

if not OPENAI_API_KEY.startswith("sk-"):
    raise ValueError("Invalid OPENAI_API_KEY format!")

if not GOOGLE_API_KEY.startswith("AIzaSy"):
    raise ValueError("Invalid GOOGLE_API_KEY format!")

# --- Model Definitions ---
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
LLM_MODEL_NAME = "gemini-2.5-flash"

# --- Text Splitting ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- Retriever ---
RETRIEVER_K = 15  

# --- File Paths ---
DIRECTORY_PATH = "docs"
FAISS_INDEX_PATH = "my_faiss_index" # Directory to save/load the index
