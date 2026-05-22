import os
from dotenv import load_dotenv

# Explicitly point to the .env file in the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(BASE_DIR, '.env')
load_dotenv(dotenv_path=env_path)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-004")
TOP_K = int(os.getenv("TOP_K", "5"))

if not TELEGRAM_BOT_TOKEN:
    print("Warning: TELEGRAM_BOT_TOKEN is not set.")
if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY is not set.")
