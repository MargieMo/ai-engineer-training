import os

from dotenv import load_dotenv
from llama_index.core.settings import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI


# Load environment variables from a .env file.
load_dotenv()

# The OpenAI API key (and optional base URL) must be set in the environment / .env file.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
if not OPENAI_API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

# --- Model & embedding configuration ---
# Use OpenAI's text-embedding-3-small as the embedding model (1536 dimensions).
EMBED_MODEL = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY,
    api_base=OPENAI_API_BASE,
)

# Use GPT-5 as the LLM for answer synthesis over retrieved results.
# Note: gpt-5 is a reasoning model and only supports the default temperature.
LLM = OpenAI(model="gpt-5", api_key=OPENAI_API_KEY, api_base=OPENAI_API_BASE)

# --- LlamaIndex global settings ---
Settings.embed_model = EMBED_MODEL
Settings.llm = LLM
Settings.chunk_size = 512
Settings.chunk_overlap = 20

# --- Data & index path configuration ---
DATA_DIR = "data"
FAQ_FILE = os.path.join(DATA_DIR, "faqs.csv")
INDEX_DIR = "vector_index"  # Where Milvus Lite data is stored.

# --- Milvus configuration ---
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
MILVUS_URI = "./milvus_demo.db"  # Milvus Lite uses a local file.
COLLECTION_NAME = "faq_collection"
DIMENSION = 1536  # Dimension of OpenAI's text-embedding-3-small model.
