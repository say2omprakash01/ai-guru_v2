
# ai-guru_v2
# AI Guru v2

Emotion-aware, RAG-grounded Bhagavad Gita guidance system built with:

- Semantic retrieval (`sentence-transformers` + FAISS)
- Optional emotion classification (`transformers`)
- OpenAI-compatible LLM generation (OpenAI/OpenRouter)
- FastAPI backend + React (Vite + TypeScript) frontend

---

## What this project does

Given a user message (for example, "I feel anxious and hopeless"), the system:

1. Embeds the message in vector space
2. Retrieves top-k relevant verses from a FAISS index
3. Optionally predicts emotion using a transformer classifier
4. Sends grounded context to an LLM
5. Returns structured guidance JSON (plus retrieved verses/citations)

This is retrieval-grounded guidance, not free-form generation.

---

## Tech stack

- **Backend:** Python, FastAPI, Pydantic
- **Retrieval:** `sentence-transformers`, FAISS
- **Emotion model:** Hugging Face `transformers`
- **LLM:** OpenAI-compatible API (`OPENAI_BASE_URL` + `OPENAI_API_KEY`)
- **Frontend:** React 18, TypeScript, Vite

---

## Project structure

```text
ai-guru_v2/
├── api/                    # FastAPI app (/health, /v1/guidance)
├── src/                    # Core library (embeddings, retriever, pipeline, llm, emotion)
├── scripts/                # CLI tools (build index, retrieval/rag demos)
├── frontend/               # React + Vite web app
├── data/processed/         # CSV corpus
├── models/index/gita/      # FAISS artifacts (generated, gitignored)
├── docs/                   # Architecture + phase learning docs
├── requirements.txt
└── .env.example
```

---

## Prerequisites

- Python 3.10+
- Node.js 20+ (recommended; in WSL use Linux Node via `nvm`)
- Optional NVIDIA GPU (CUDA) for faster embedding/emotion inference

---

## Quick start (backend first, then frontend)

### 1) Install backend dependencies

```bash
cd /home/say2omprakash01/ai-guru_v2
pip install -r requirements.txt
```

### 2) Configure environment

```bash
cp .env.example .env
```

Set your key in `.env`:

```bash
OPENAI_BASE_URL=https://openrouter.ai/api/v1   # optional if using OpenRouter
OPENAI_API_KEY=your-key-here
LLM_MODEL=openai/gpt-4o-mini
```

Optional (GPU query embedding):

```bash
AI_GURU_EMBED_DEVICE=cuda
```

### 3) Build FAISS index (one-time or when changing model/data)

```bash
export PYTHONPATH=.
python3 scripts/build_faiss_index.py --query "I feel anxious about my future"
```

Example stronger model:

```bash
export PYTHONPATH=.
python3 scripts/build_faiss_index.py \
  --model BAAI/bge-base-en-v1.5 \
  --device cuda \
  --query "I feel sad and hopeless"
```

### 4) Run API

```bash
PYTHONPATH=. uvicorn api.main:app --host 127.0.0.1 --port 8000
```

- Swagger: `http://127.0.0.1:8000/docs`
- Health: `GET /health`
- Guidance: `POST /v1/guidance`

### 5) Run frontend (new terminal)

```bash
cd /home/say2omprakash01/ai-guru_v2
./scripts/run_frontend.sh
```

Open `http://localhost:5173`

---

## API example

```bash
curl -s -X POST http://127.0.0.1:8000/v1/guidance \
  -H "Content-Type: application/json" \
  -d '{"message":"I feel anxious about my future","top_k":5,"use_emotion_model":true}'
```

---

## Useful scripts

- `scripts/build_faiss_index.py` — build/rebuild vector index
- `scripts/query_semantic.py` — retrieval only test
- `scripts/run_pipeline_retrieval.py` — retrieval via pipeline facade
- `scripts/run_rag_turn.py` — full retrieval + emotion + LLM turn
- `scripts/run_frontend.sh` — frontend dev server helper (WSL-friendly)

---

## Security notes

- Never commit `.env` or real API keys
- Rotate keys if exposed accidentally
- Generated artifacts (`models/index/`, `frontend/node_modules/`, `frontend/dist/`) are gitignored

---

## Documentation

- Main docs index: `docs/README.md`
- Architecture: `docs/ARCHITECTURE.md`
- Quick run guide: `docs/QUICKSTART_RUN.md`
- Frontend phase guide: `docs/PHASE_7_README.md`
- Frontend deep dive: `docs/frontend/README.md`


