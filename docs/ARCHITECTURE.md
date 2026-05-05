# AI Guru v2 — Architecture & System Flow

**Product:** Emotion-aware spiritual guidance (grounded in scripture, not free-form invention).  
**This document:** End-to-end architecture, technology choices, and phased build order.  
**Context:** v1 used TF–IDF + regression (baseline). **v2 is a new stack**: dense embeddings, vector search, transformer emotion, and LLM reasoning **only** over retrieved text.

---

## 1. Goals and non-goals

### Goals

- Accept natural language (e.g. *“I feel anxious about my future”*).
- **Detect emotion** with a **transformer** model (not classical ML on bag-of-words).
- **Retrieve** the most relevant verses by **semantic similarity** (not keyword TF–IDF).
- Use **Top-K** passages (e.g. K=5) as the **only** scripture the LLM may cite or paraphrase as authoritative text.
- Produce a **fixed, structured** response: emotion, insight, explanation, practical guidance, reflection question, disclaimer.
- Expose the system via a **FastAPI** service suitable for production-style deployment.

### Non-goals (for initial v2)

- Replacing human spiritual authority or clinical care (disclaimer is mandatory).
- Letting the LLM invent verses, chapter numbers, or text not present in retrieval.
- Keyword-only search as the primary retrieval mechanism.

---

## 2. High-level system flow (conceptual)

Each stage has a single responsibility. Data moves forward as **structured objects** (types), not opaque strings.

```mermaid
flowchart LR
  U[User input] --> E[Emotion model\nTransformers]
  E --> Q[Query prep\noptional expansion]
  Q --> V[Embedder\nsentence-transformers]
  V --> I[FAISS index\nTop-K vectors]
  I --> R[Retriever\nrank + metadata join]
  R --> L[LLM\nRAG generator]
  L --> O[Structured JSON\nresponse]
```



**Mental model:**  
v1 asked *“which words overlap?”* (TF–IDF) and *“what label fits?”* (regression).  
v2 asks *“what does this mean in vector space?”* (embeddings) and *“what should we say given **these** exact passages?”* (LLM with retrieval boundary).

---

## 3. End-to-end pipeline (step-by-step)


| Step  | What happens             | Technology                                                       | Output                                                                  |
| ----- | ------------------------ | ---------------------------------------------------------------- | ----------------------------------------------------------------------- |
| **1** | User sends text          | FastAPI                                                          | Raw query string                                                        |
| **2** | Emotion classification   | Hugging Face `transformers` (sequence classification or similar) | Label + confidence (and optional multi-label)                           |
| **3** | Query text for retrieval | Optional LLM or light rules                                      | Single string (original, expanded, or concatenation of both for search) |
| **4** | Encode query to vector   | `sentence-transformers`                                          | Normalized embedding vector                                             |
| **5** | Similarity search        | FAISS (inner product or L2, depending on index config)           | Top-K indices + scores                                                  |
| **6** | Resolve rows             | Your CSV/DB keyed by stable `id`                                 | List of verse records: text, meaning, reference, themes, tags           |
| **7** | Prompt construction      | Template + strict rules                                          | Prompt containing **only** retrieved blocks                             |
| **8** | Generation               | OpenAI or OpenRouter API                                         | Structured fields (JSON), no new scripture                              |
| **9** | Validate / package       | Optional checks + FastAPI schema                                 | HTTP response                                                           |


**Critical invariant:** Steps 7–8 must not receive full scripture corpora—only the K retrieved snippets plus fixed instructions.

---

## 4. Repository layout (target)

This matches the modular product structure you defined; each box maps to deployable/testable code.

```text
ai-guru-v2/
├── data/
│   ├── raw/                      # Immutable sources
│   └── processed/
│       └── gita_verses.csv       # scripture_name, chapter, verse, shloka_text, meaning, themes, emotion_tags
├── src/
│   ├── emotion/
│   │   └── emotion_model.py      # Load HF model, predict emotion
│   ├── embeddings/
│   │   └── embedder.py           # Encode text → vectors (same model for index + query)
│   ├── vector_db/
│   │   └── faiss_store.py        # Build, persist, load, search FAISS; id ↔ metadata map
│   ├── retriever/
│   │   └── semantic_retriever.py # Top-K orchestration, optional filters/rerank hooks
│   ├── llm/
│   │   └── rag_generator.py      # Grounded prompts, provider client, structured output
│   ├── pipeline/
│   │   └── pipeline.py          # Wires all steps; single entry for API/tests
│   └── utils/                    # Config, logging helpers, IO
├── api/
│   └── main.py                   # FastAPI app, routes, dependencies
├── frontend/                     # Vite + React + TS — HTTP client to POST /v1/guidance (see docs/PHASE_7_README.md)
├── scripts/                      # CLI: build index, demos (not imported by api at request time)
├── models/                       # Downloaded HF weights, local checkpoints, FAISS indices (gitignored artifacts)
├── notebooks/                    # EDA, index build experiments
├── docs/
│   └── ARCHITECTURE.md           # This file
├── requirements.txt
└── README.md
```

**Why this split:** Swapping FAISS for another engine later should not rewrite the API; swapping LLM provider should not touch embedding code.

**Browser client:** `frontend/` is optional for learning and demos; it does **not** import `src/`. It calls the same JSON API as `curl` (see **`docs/flow/FRONTEND.md`** for ports, CORS, and env).

---

## 5. Data model and indexing strategy

### 5.1 Source columns (your contract)


| Column             | Role                                            |
| ------------------ | ----------------------------------------------- |
| `scripture_name`   | Provenance                                      |
| `chapter`, `verse` | Stable citation                                 |
| `shloka_text`      | Display / authenticity                          |
| `meaning`          | Plain-language sense; strong for semantic match |
| `themes`           | Retrieval signal + UI                           |
| `emotion_tags`     | Metadata for filtering/reranking later          |


#### Are `themes` and `emotion_tags` required?

**No.** They are **optional metadata**, not the mechanism that makes open-ended user queries work.

- **Primary retrieval in v2** is **semantic search**: the embedder maps the user’s free-text question and each verse’s text into the **same continuous vector space**. Similar *meaning* yields high similarity even when the user never says a tag you listed in the CSV. You do **not** need to enumerate “all topics” in columns for that to work.
- `**meaning` (and optionally `shloka_text`)** is enough to build a strong index. If `themes` / `emotion_tags` are missing or sparse, retrieval still works; you simply lose **extra signals** below.
- **What tags are good for:** UI (“verses about duty”), **optional** post-retrieval filters or reranking (e.g. boost verses whose `emotion_tags` overlap the **detected** user emotion), analytics, and human curation. They are **not** a replacement for embeddings.

**Can “the model” decide themes/tags?**

- **Embedding model:** It does not output human-readable themes; it outputs a vector. That is what handles arbitrary user wording.
- **Emotion classifier (transformer):** It labels **the user’s** utterance, not each verse—unless you train or run a separate **verse-tagging** pipeline. The LLM after retrieval can **describe** themes in natural language, but that is **interpretation**, not a substitute for curated verse metadata when you want consistent filters.
- **Risk of LLM-only tagging for the corpus:** Automatic tagging is possible (batch job: “label each verse”), but labels become **model-dependent** and need QA; many teams keep optional manual `themes` / `emotion_tags` for control.

**Bottom line:** Your worry (“user can say anything; we can’t list all topics”) is exactly why v2 uses **dense retrieval**, not a closed tag vocabulary. Keep `themes` / `emotion_tags` if they help product features; omit or leave empty if you do not need them yet.

### 5.2 What gets embedded (recommended)

- **Minimum viable retrieval text:** `meaning` (plus optional `shloka_text` if it helps your audience). That alone supports arbitrary user queries via embeddings.
- **Optional enrichment:** append `themes` and `emotion_tags` to the embedded string if present—they can slightly sharpen similarity when they align with query wording, but they are not required.

Example concatenation when all fields exist:

  `meaning + themes + emotion_tags` (and optionally a short prefix like scripture name).

- Keep `shloka_text` as **display-only** if you want the LLM to explain *around* the verse without polluting the vector with archaic phrasing that mismatches modern queries—or include it if your users query in Sanskrit/transliteration often.

**Rule:** Whatever you embed at build time, you must embed queries with the **same** model and compatible preprocessing.

#### What exactly is compared at search time?

Retrieval is **vector vs vector**, not “Sanskrit line vs English sentence” as raw text:

1. **Query side:** The user’s message (e.g. English: *“I feel anxious about my future”*) is passed through the **same** embedder → one **query vector**.
2. **Corpus side:** Each verse row has a **precomputed document vector** built from whatever you chose to embed (typically `**meaning`** in the same language as most queries, plus optional fields).
3. **Similarity:** FAISS finds the Top-K corpus vectors **closest** to the query vector (cosine / inner product / L2, depending on index setup).

So for **English (or Hindi) user questions**, comparing directly to **Sanskrit `shloka_text` alone** is usually a **bad default**: the wording does not align with how people ask, and many general sentence encoders are **weaker across a big Sanskrit–English gap** than **English meaning ↔ English query**. Your instinct is correct: **primary similarity should align the query with plain-language `meaning` (and themes/tags if you add them), not rely on matching the original śloka.**

**After** search returns the best row IDs, you still **attach `shloka_text`** (and chapter/verse) from metadata for **display and grounding**—the user sees the authentic verse; the **index** did not have to “match Sanskrit to English” to find it.

**If users often type Sanskrit or transliteration:** use a **strong multilingual** embedding model and/or **include `shloka_text` in the embedded string** (sometimes alongside `meaning`) so queries in that form still land near the right vectors.

**Summary:** Compare **embedded user query** to **embedded retrieval text** (usually **meaning**, not śloka-only for modern-language queries). `**shloka_text` remains the source of truth for what verse was retrieved**, not the main similarity signal unless you explicitly design for Sanskrit queries.

### 5.3 Stable IDs

- Assign each CSV row a `**verse_id`** (integer or UUID) at ingest time.
- FAISS row `i` maps to `verse_id` via a sidecar list or table **never** inferred by CSV row order alone after sorting/filtering.

---

## 6. Component responsibilities (deep but short)

### 6.1 `embedder.py`

- Loads a **single** sentence-transformer model (local cache under `models/`).
- `encode(texts: list[str]) -> ndarray` with batching for throughput.
- Optional L2 normalization if using inner-product index.

### 6.2 `faiss_store.py`

- **Build:** vectors (N × dim) → train/add to index (IndexFlatIP / IndexFlatL2 or IVF later for scale).
- **Persist:** FAISS binary + manifest (model name, dim, metric, version, timestamp).
- **Search:** `query_vector` → top-K (ids, scores).
- **No scripture strings inside FAISS**—only vectors and integer ids; text lives in CSV/DB keyed by id.

### 6.3 `semantic_retriever.py`

- Calls embedder on the **query string**.
- Calls FAISS search.
- Joins results to metadata records.
- **Extension hooks:** filter by scripture, rerank with emotion-tag overlap, MMR for diversity across K.

### 6.4 `emotion_model.py` (Phase 5 in build order, but architected now)

- HF tokenizer + model; returns label + scores.
- Runs **in parallel** or **before** retrieval depending on latency budget; architecturally it only **annotates** unless you add reranking.

### 6.5 `rag_generator.py`

- Builds a prompt with: user message, emotion summary, **verbatim retrieved passages** with citations.
- System instructions: *Do not output scripture outside the provided block; if unsure, say so.*
- Requests **JSON** (or tool/schema) matching your API: `emotion`, `insight`, `explanation`, `practical_guidance`, `reflection_question`, `disclaimer`.
- **Optional hardening:** reject or repair outputs that reference chapter/verse not in the retrieved set.

### 6.6 `pipeline.py`

- The **orchestrator**: one function `run_turn(user_text: str) -> GuidanceResponse`.
- Owns config: model names, K, temperature, API keys from environment.
- Easier testing: mock retriever or LLM per unit test.

### 6.7 `api/main.py`

- FastAPI routes: health, `/guidance` (POST).
- Pydantic request/response models.
- Timeouts on LLM HTTP calls; structured error responses.

---

## 7. Technology stack (mandatory set) and rationale


| Layer         | Choice                         | Rationale                                                                     |
| ------------- | ------------------------------ | ----------------------------------------------------------------------------- |
| Language      | Python 3.11+                   | Ecosystem for ML + API                                                        |
| API           | FastAPI                        | Typed routes, async-friendly, OpenAPI docs                                    |
| Embeddings    | `sentence-transformers`        | Production-grade sentence/paragraph encoders on CPU/GPU                       |
| Vector search | FAISS                          | Fast, local, no server dependency for v2; scales to large N with IVF/PQ later |
| Emotion       | Hugging Face `transformers`    | Real sequence models vs v1 regression                                         |
| LLM           | OpenAI / OpenRouter            | Reasoning + JSON mode; OpenRouter for model routing                           |
| Config        | Env vars + small config module | No secrets in code; reproducible deploys                                      |


**Modern alternatives (optional later):** Qdrant/Weaviate for heavy metadata filters; ONNX/TensorRT for latency; streaming SSE for long LLM tokens.

---

## 8. Phased implementation (strict order)

This is the **delivery sequence**; each phase is testable before the next.


| Phase | Deliverable                                            | Success criteria                                                                       |
| ----- | ------------------------------------------------------ | -------------------------------------------------------------------------------------- |
| **1** | Embedding pipeline + FAISS + semantic search           | Given a query, return sensible Top-K ids from `gita_verses.csv`; index save/load works |
| **2** | Wire **semantic retriever** as the only retrieval path | No TF–IDF in the hot path                                                              |
| **3** | **K=5** (or configured) multi-verse retrieval          | Responses always include multiple candidates in the prompt builder input               |
| **4** | **RAG prompt + LLM**                                   | JSON output; scripture content matches retrieved snippets only                         |
| **5** | **Transformer emotion**                                | Emotion field driven by HF model, not regression                                       |
| **6** | **Output quality & structure**                         | Validation, disclaimers, optional reranking, eval notebook                             |


Skipping phases hides bugs (e.g. building LLM before retrieval guarantees confusion about grounding).

---

## 9. Grounding, safety, and quality

- **Grounding:** Retrieved text is the **single source of truth** for scriptural claims in the LLM turn.
- **Prompting:** Explicit “use only the CONTEXT” and “do not quote new verses.”
- **Programmatic:** Structured output + optional citation whitelist check.
- **Product:** Persistent disclaimer; not a substitute for medical or professional mental health care.
- **Privacy:** Minimize logging of user messages; rotate API keys; no training on user data without consent.

---

## 10. Configuration and versioning (optimised ops)

- **Embedding model version** and **index artifact** version pinned together (manifest file next to the `.index`).
- **Deterministic ingest:** script that rebuilds index from `gita_verses.csv` hash.
- **Dependencies:** Pinned ranges in `requirements.txt` for reproducible builds.

---

## 11. Testing strategy (lightweight, high leverage)

- **Phase 1:** Unit tests: embedder shape; FAISS round-trip; known query hits a known verse id.
- **Phase 4:** Golden-file tests: mock LLM; assert prompt contains exactly retrieved shloka/meaning snippets.
- **Phase 5:** Emotion model tests on a tiny labeled snippet set.

---

## 12. Mental summary (v1 → v2)


| Concern   | v1                             | v2                                                    |
| --------- | ------------------------------ | ----------------------------------------------------- |
| Retrieval | TF–IDF (lexical)               | Dense vectors + FAISS (semantic)                      |
| Emotion   | Regression on shallow features | Transformer classification                            |
| Knowledge | Same corpus idea               | Same corpus, **stricter** grounding via Top-K context |
| Reasoning | Limited                        | LLM **bounded** by retrieval                          |


---

## 13. Next step after this document

Proceed with **Phase 1** implementation: `embedder.py` → `faiss_store.py` → small ingest path that reads `data/processed/gita_verses.csv` and proves Top-K search end-to-end **without** LLM or emotion yet.

---

*Document version: 1.0 — aligned with AI Guru v2 modular layout and mandatory stack.*