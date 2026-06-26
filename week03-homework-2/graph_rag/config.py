import os

from dotenv import load_dotenv
from llama_index.core.settings import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI


# Load environment variables from a .env file.
load_dotenv()

# --- API keys and credentials ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
# Neo4j connection defaults match the bundled docker-compose.yml.
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")

if not OPENAI_API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

# --- LlamaIndex global settings ---
# Use OpenAI GPT-5 as the LLM (reasoning model; only the default temperature is supported).
Settings.llm = OpenAI(model="gpt-5", api_key=OPENAI_API_KEY, api_base=OPENAI_API_BASE)
# Use OpenAI's text-embedding-3-small as the embedding model.
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small", api_key=OPENAI_API_KEY, api_base=OPENAI_API_BASE
)

# --- Data paths ---
DATA_DIR = "data"
COMPANY_DOC_PATH = os.path.join(DATA_DIR, "companies.txt")
SHAREHOLDER_CSV_PATH = os.path.join(DATA_DIR, "shareholders.csv")

# --- Vector index settings ---
INDEX_DIR = "graph_rag_index"

# --- Neo4j graph settings ---
NEO4J_DATABASE = "neo4j"
