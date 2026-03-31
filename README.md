# ArXiv Research Assistant 🔬

An AI-powered research assistant that lets you chat with scientific papers from ArXiv using RAG (Retrieval Augmented Generation) + OpenAI.

Ask questions in natural language and get answers grounded in real papers, with citations.

![Python](https://img.shields.io/badge/Python-3.11-blue) ![LangChain](https://img.shields.io/badge/LangChain-0.2-green) ![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-orange) ![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)

---

## What it does

1. **Searches ArXiv** for papers on any topic you choose
2. **Downloads and indexes** the PDFs into a local vector database (ChromaDB)
3. **Answers your questions** using only the content of those papers — no hallucinations, sources always cited

---

## Architecture

```
arxiv-research-assistant/
├── src/
│   ├── arxiv_client.py   # ArXiv API connector — search + download PDFs
│   ├── ingestion.py      # PDF → text → chunks → embeddings → ChromaDB
│   ├── retriever.py      # Semantic search over the vector store
│   ├── chain.py          # RAG chain: retriever + prompt + GPT-4o
│   └── app.py            # Streamlit web interface
├── data/
│   ├── papers/           # Downloaded PDFs (auto-generated)
│   └── chroma_db/        # Vector database (auto-generated)
├── requirements.txt
└── README.md
```

### RAG pipeline

```
[ArXiv API] → PDF files → text extraction → chunks (800 chars, 150 overlap)
    → OpenAI embeddings → ChromaDB (indexed once)

[User question] → embedding → semantic search → top-5 chunks
    → prompt + context → GPT-4o → answer with citations
```

---

## Stack

| Component | Technology | Role |
|---|---|---|
| Data source | ArXiv API | Fetch real scientific papers |
| Text extraction | PyMuPDF | PDF → plain text |
| Chunking | LangChain RecursiveCharacterTextSplitter | Split text into overlapping chunks |
| Embeddings | OpenAI text-embedding-3-small | Semantic vector representation |
| Vector DB | ChromaDB | Local persistent vector store |
| Orchestration | LangChain | RAG chain assembly |
| LLM | GPT-4o | Answer generation |
| Interface | Streamlit | Web UI |

---

## Quick start

### Prerequisites
- Python 3.10+
- OpenAI API key

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/arxiv-research-assistant.git
cd arxiv-research-assistant

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Set up your API key
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Run the app

```bash
streamlit run src/app.py
```

Open `http://localhost:8501` in your browser.

---

## Usage

1. **Enter a topic** in the sidebar (e.g. "retrieval augmented generation", "transformer attention")
2. **Set the number of papers** with the slider (3–10 recommended)
3. **Click "Search & Index"** — the app downloads PDFs and builds the vector store
4. **Ask questions** in the chat input

### Example questions

```
"What are the main components of a RAG system?"
"Compare the retrieval approaches described in the papers"
"What limitations do the authors acknowledge?"
"What evaluation metrics are used and why?"
```

---

## How it works — key concepts

**Chunking:** Papers are split into overlapping 800-character segments. Overlap (150 chars) ensures ideas that span chunk boundaries are not lost.

**Embeddings:** Each chunk is converted into a ~1500-dimensional vector by OpenAI's embedding model. Semantically similar texts produce numerically similar vectors.

**Semantic search:** When you ask a question, it's also embedded into a vector. ChromaDB finds the 5 chunks whose vectors are closest (cosine similarity) — regardless of exact word match.

**Grounded generation:** GPT-4o receives your question plus the retrieved chunks as context. The system prompt instructs it to answer only from that context and cite its sources.

---

## Configuration

Key parameters you can tune in the source files:

| Parameter | File | Default | Effect |
|---|---|---|---|
| `chunk_size` | ingestion.py | 800 | Larger = more context per chunk, less precision |
| `chunk_overlap` | ingestion.py | 150 | Larger = fewer missed boundary ideas, more redundancy |
| `k` (chunks retrieved) | retriever.py | 5 | More = richer context, higher cost per query |
| `temperature` | chain.py | 0.2 | Lower = more factual, higher = more creative |
| `model` | chain.py | gpt-4o | Swap for gpt-4o-mini to reduce cost |

---

## Extending the project

- **Different data sources:** Replace `arxiv_client.py` with any document source (local PDFs, web scraping, SEC filings, internal docs)
- **Different embedding models:** Swap `text-embedding-3-small` for a local model via `sentence-transformers` (free, no API needed)
- **Agentic mode:** Add LangChain agents so the assistant decides whether to search ArXiv for more papers before answering
- **Conversation memory:** Add `ConversationBufferMemory` to maintain context across turns

---

## License

MIT
