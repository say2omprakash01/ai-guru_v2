# AI Guru v2 — learning guides (docs)

**Confused which script does what?** [flow/README.md](flow/README.md) — build-time vs runtime, `build_faiss_index.py` vs `semantic_retriever.py`, who calls whom.

**Run it first:** [QUICKSTART_RUN.md](QUICKSTART_RUN.md) — build index + retrieval + RAG + API in order.

**Then learn systematically:** [LEARNING_PATH.md](LEARNING_PATH.md) — syllabus from **basic → advanced**, checkpoints, and how to know you are ready for the next project phase.

Each phase doc below explains **what you are building**, **why it exists**, and **how to run it** so you learn the system, not only the file tree.

| Doc | Topic |
|-----|--------|
| [PHASE_0_README.md](PHASE_0_README.md) | Data: CSV schema, sourcing all ślokas, validation |
| [PHASE_1_README.md](PHASE_1_README.md) | Embeddings + FAISS: semantic vectors, index, smoke test |
| [PHASE_2_README.md](PHASE_2_README.md) | Semantic retriever: load index + model, `retrieve(query, k)` |
| [PHASE_3_README.md](PHASE_3_README.md) | Pipeline: default Top-K=5, `RetrievalPhaseResult`, context formatter |
| [PHASE_4_README.md](PHASE_4_README.md) | RAG: OpenAI-compatible LLM, `GuidanceResponse`, `run_guidance_turn` |
| [PHASE_5_README.md](PHASE_5_README.md) | HF transformer emotion + RAG prompt + enforced `emotion` field |
| [PHASE_6_README.md](PHASE_6_README.md) | FastAPI: `/health`, `/v1/guidance`, lifespan, thread pool |
| [PHASE_7_README.md](PHASE_7_README.md) | Web UI: Vite + React + TS, `fetch`, CORS, `VITE_API_URL` |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Whole-system architecture (reference) |

**Browser ↔ API integration (diagrams + mental model):** [flow/FRONTEND.md](flow/FRONTEND.md).

**Frontend deep dive (every file, hooks vs classes, why `node_modules` is huge):** [frontend/README.md](frontend/README.md).

**Interview prep:** Phase guides end with **“Sources and references”** and **“Useful commands”** where applicable.
