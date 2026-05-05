#!/usr/bin/env bash
# Start Vite dev server using Linux Node (nvm), not Windows Node under WSL.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
if [[ -s "$NVM_DIR/nvm.sh" ]]; then
  # shellcheck source=/dev/null
  . "$NVM_DIR/nvm.sh"
fi
if ! command -v node >/dev/null || [[ "$(command -v node)" == *"/mnt/c/"* ]]; then
  echo "Install nvm and Node 20 in WSL (one-time):"
  echo "  curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash"
  echo "  source ~/.bashrc   # or: export NVM_DIR=\"\$HOME/.nvm\" && . \"\$NVM_DIR/nvm.sh\""
  echo "  nvm install 20"
  exit 1
fi
cd "$ROOT/frontend"
if [[ -f .nvmrc ]]; then nvm use 2>/dev/null || true; fi
exec npm run dev
