# AI Guru — web UI

React + Vite + TypeScript. Calls `POST /v1/guidance` on the FastAPI app.

**Learning (step-by-step, like backend phases):** [docs/PHASE_7_README.md](../docs/PHASE_7_README.md). **Full flow + integration:** [docs/flow/FRONTEND.md](../docs/flow/FRONTEND.md). **Every file + hooks vs classes:** [docs/frontend/README.md](../docs/frontend/README.md).

## Prerequisites

- Node.js 18+ (20+ recommended), **Linux binaries in WSL** — not Windows `C:\Program Files\nodejs` (that breaks `npm install` on `/home/...` paths).
- **WSL one-time:** install [nvm](https://github.com/nvm-sh/nvm), then `nvm install` inside `frontend/` (uses `.nvmrc` → Node 20). New terminals: `source ~/.bashrc` or run `export NVM_DIR="$HOME/.nvm" && . "$NVM_DIR/nvm.sh"` before `npm`.
- API running (see repo root `docs/QUICKSTART_RUN.md`)

## Setup

```bash
cd frontend
# If you use nvm (recommended on WSL):
#   nvm install && nvm use
npm install
```

Optional: copy `.env.example` to `.env.local` and set `VITE_API_URL` if the API is not at `http://127.0.0.1:8000`.

## Run

From repo root (loads nvm and runs Vite):

```bash
./scripts/run_frontend.sh
```

Or from `frontend/` after `nvm use`:

```bash
npm run dev
```

Open the printed URL (default `http://localhost:5173`). The API must allow this origin via CORS (defaults in `api/main.py` include port 5173).

## Build

```bash
npm run build
```

Static output is in `dist/`. Serve it behind any static host; set `VITE_API_URL` at build time to the public API URL.
