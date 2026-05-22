# 🦁 Mandai Zoo AI Chatbot

A **Retrieval-Augmented Generation (RAG)** Telegram chatbot that answers visitor queries about Singapore Zoo and the Mandai Wildlife Reserve. Every response is grounded in content scraped from [mandai.com](https://www.mandai.com) — no hallucinations, always cited.

## How It Works

```
User asks a question on Telegram
        ↓
Query is embedded using Gemini Embedding API
        ↓
ChromaDB retrieves the most relevant content chunks
        ↓
Gemini LLM generates an answer grounded in that context
        ↓
User receives a cited, accurate response
```

The bot scrapes key pages from mandai.com (visitor info, tickets, animals, FAQs, dining), chunks the content into semantically meaningful pieces, embeds them into a vector store, and retrieves relevant chunks at query time to augment the LLM prompt.

## Architecture

| Component | Technology | Role |
|---|---|---|
| Chat Interface | Telegram via `python-telegram-bot` | Receives user messages, sends responses |
| Orchestration | Python 3.11+ | Coordinates the RAG pipeline |
| LLM | Google Gemini (`gemini-2.5-flash`) | Generates grounded natural-language answers |
| Embeddings | Gemini `gemini-embedding-001` | Converts text into vector representations |
| Vector Store | ChromaDB (persistent, local) | Stores and retrieves document embeddings |
| Scraper | `requests` + `BeautifulSoup4` | Crawls mandai.com, cleans HTML, extracts content |

## Project Structure

```
mandaizoo-chatbot/
├── src/
│   ├── bot.py              # Telegram bot setup and message handling
│   ├── rag.py              # RAG pipeline: embed → retrieve → prompt → generate
│   ├── embeddings.py       # Gemini embedding API wrapper
│   ├── vector_store.py     # ChromaDB operations (init, query, upsert)
│   └── config.py           # Environment variable loading and constants
├── scraper/
│   ├── crawl.py            # Web scraper (requests + BeautifulSoup)
│   ├── chunker.py          # Text chunking logic
│   └── ingest.py           # Orchestrator: crawl → clean → chunk → embed → store
├── data/
│   └── seed_urls.txt       # URLs to scrape
├── chroma_db/              # ChromaDB persistent storage (gitignored)
├── design.md               # Full design document
├── requirements.txt
├── .env.example
└── .gitignore
```

## Setup

### Prerequisites

- Python 3.11+
- A [Google Gemini API key](https://aistudio.google.com)
- A [Telegram Bot Token](https://t.me/BotFather)

### Installation

```bash
git clone https://github.com/liewlucas/mandaizoo-chatbot.git
cd mandaizoo-chatbot
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Configuration

Copy the example environment file and fill in your keys:

```bash
cp .env.example .env
```

Edit `.env`:

```
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
GEMINI_API_KEY=your_gemini_api_key
CHROMA_DB_PATH=./chroma_db
GEMINI_MODEL=gemini-2.5-flash
EMBEDDING_MODEL=gemini-embedding-001
TOP_K=5
```

### Data Ingestion

Scrape mandai.com and populate the vector store:

```bash
python -m scraper.ingest
```

This crawls the seed URLs, chunks the content, generates embeddings, and stores everything in ChromaDB. Re-run anytime to refresh the data.

### Run the Bot

```bash
python -m src.bot
```

The bot will start polling Telegram for messages.

## Bot Commands

| Command | Description |
|---|---|
| `/start` | Welcome message explaining the bot's capabilities |
| `/help` | Example questions you can ask |
| `/sources` | List of pages the bot's knowledge is based on |

Any other message is treated as a question and routed through the RAG pipeline.

## Design Decisions

Detailed rationale is documented in [`design.md`](design.md). Key choices include:

- **Semantic-first chunking** with a ~1000 character cap, splitting on natural boundaries (paragraphs, headings) before resorting to character limits. FAQ pages are split by Q&A pair to keep answers intact.
- **Asymmetric embeddings** using Gemini's `task_type` parameter — `RETRIEVAL_DOCUMENT` for indexing, `RETRIEVAL_QUERY` for search — which improves retrieval quality.
- **Strict grounding** in the system prompt ensures the bot never fabricates information. If the answer isn't in the knowledge base, it says so and links to the official website.
- **Source citations** in every response so users can verify information and read further.
- **ChromaDB** for its simplicity — no server setup, persistent to disk, with built-in metadata filtering.

## Prompt Engineering

The bot uses a two-part prompt strategy:

1. **System prompt** establishes the persona (friendly zoo guide), enforces grounding rules (only answer from context), and defines fallback behavior.
2. **User message template** injects retrieved chunks with clear source demarcation, enabling the LLM to attribute information and handle contradictions between chunks.

See the full prompts in [`design.md`](design.md#5-prompt-engineering).

## Data Freshness

The scraper is idempotent — re-running `python -m scraper.ingest` re-crawls all pages and upserts into ChromaDB using URL-based IDs, so stale content is overwritten. Each chunk carries a `scraped_at` timestamp for staleness detection.

## Dependencies

```
python-telegram-bot>=20.0
google-genai>=1.0.0
chromadb>=0.5.0
beautifulsoup4>=4.12.0
requests>=2.31.0
python-dotenv>=1.0.0
```
