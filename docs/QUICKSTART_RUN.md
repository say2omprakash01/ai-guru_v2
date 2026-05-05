# Quickstart — run the full pipeline

## Words first (30 seconds)

This project **does not train** the embedding or emotion networks from scratch. It uses **pretrained** models from Hugging Face. What you “train” in practice is **building the vector index** (encoding every verse once into FAISS). After that, each request is **inference only**: embed query → search → optional emotion → LLM.

---

## One-time setup

```bash
cd /path/to/ai-guru_v2

# Python 3.10+ recommended. Prefer a venv:
#   sudo apt install python3.12-venv   # Debian/Ubuntu if needed
#   python3 -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt
# If your OS blocks global pip: pip install --break-system-packages -r requirements.txt
```

Ensure `data/processed/gita_verses.csv` exists (719 rows if you used the ingest script).

---

## Step 1 — Build the “brain” (FAISS index)

Encodes all verses and saves `models/index/gita/`.

```bash
export PYTHONPATH=.
python scripts/build_faiss_index.py --query "I feel anxious about my future"
```

You should see **Top-5** hits and **Saved FAISS store**.

### Optional — stronger embeddings + GPU (better similarity / ranking)

The default model (`all-MiniLM-L6-v2`) is small and fast. For **richer** semantic matches (e.g. emotional wording vs formal translations), rebuild the index with a **larger** English retrieval model, then use the **same** index at runtime (the manifest stores the model id).

**Example (fits many 6 GB laptop GPUs, e.g. RTX 3050):** `BAAI/bge-base-en-v1.5`

```bash
export PYTHONPATH=.
python scripts/build_faiss_index.py \
  --model BAAI/bge-base-en-v1.5 \
  --device cuda \
  --query "I feel sad and hopeless"
```

If you hit **CUDA out-of-memory**, close other GPU apps, or build on **CPU** (`--device cpu`) — slower once, runtime still fast if you set `AI_GURU_EMBED_DEVICE=cuda` after.

**Use GPU for query encoding** (API and scripts that read `PipelineConfig`):

- Add to **`.env`**: `AI_GURU_EMBED_DEVICE=cuda`
- Or CLI: `PYTHONPATH=. python scripts/run_rag_turn.py -q "..." --device cuda`

Restart **`uvicorn`** after changing `.env`.

---

## Step 2 — Retrieval only (no API key)

```bash
PYTHONPATH=. python scripts/run_pipeline_retrieval.py -q "duty without attachment to results"
```

Or:

```bash
PYTHONPATH=. python scripts/query_semantic.py -q "your question" -k 5
```

---

## Where to put your API key (do **not** paste into code)

**Never** commit keys or put them in chat logs. Use one of these:

### Option A — `.env` file (recommended)

1. Copy the template: `cp .env.example .env`
2. Edit **`.env`** in the project root (`ai-guru_v2/.env`).

**OpenRouter** (if your key is from [openrouter.ai](https://openrouter.ai/)):

```bash
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_API_KEY=your-openrouter-secret-key
LLM_MODEL=openai/gpt-4o-mini
```

Use a real **OpenRouter model id** for `LLM_MODEL` (see their Models page). The variable is still named `OPENAI_API_KEY` because the Python client is OpenAI-compatible.

**OpenAI** directly: leave out `OPENAI_BASE_URL` and use your OpenAI key + e.g. `LLM_MODEL=gpt-4o-mini`.

3. `.env` is **gitignored**; only you keep the real file locally.
4. Install deps so `python-dotenv` loads it: `pip install -r requirements.txt`
5. Run the scripts below from the **repo root** — they auto-load `.env`.

### Option B — shell only (this terminal session)

**OpenRouter:**

```bash
export OPENAI_BASE_URL=https://openrouter.ai/api/v1
export OPENAI_API_KEY=your-openrouter-key
export LLM_MODEL=openai/gpt-4o-mini
```

**OpenAI:**

```bash
export OPENAI_API_KEY=sk-your-openai-key
```

---

## Step 3 — Full stack (retrieval + emotion + LLM)

Needs **`OPENAI_API_KEY`** (or OpenRouter + `OPENAI_BASE_URL`). Use Option A or B above.

```bash
PYTHONPATH=. python scripts/run_rag_turn.py -q "I feel anxious about my future"
```

---

## Step 4 — HTTP API

With `.env` or `export` set as above:

```bash
PYTHONPATH=. uvicorn api.main:app --host 127.0.0.1 --port 8000
# Browser: http://127.0.0.1:8000/docs
```

---

## Step 5 — Web UI (Vite + React)

Two terminals from the repo root:

1. **API** (same as Step 4): `PYTHONPATH=. uvicorn api.main:app --host 127.0.0.1 --port 8000`
2. **Frontend**: `./scripts/run_frontend.sh` (WSL: uses Linux Node via nvm) or `cd frontend && nvm use && npm run dev` — open the URL Vite prints (usually `http://localhost:5173`). First time in `frontend/`: `npm install`.

The UI calls `POST /v1/guidance`. CORS allows `localhost` / `127.0.0.1` on port **5173** by default; override with `CORS_ORIGINS` in `.env` if you use another origin.

Optional: create `frontend/.env.local` with `VITE_API_URL=http://127.0.0.1:8000` if the API is not on that host/port.

See [frontend/README.md](../frontend/README.md) for build details.

---

## Optional: Hugging Face token

If downloads are slow or rate-limited:

```bash
export HF_TOKEN=hf_...
```

---

When these work, use [LEARNING_PATH.md](LEARNING_PATH.md) to go deeper.
