#!/usr/bin/env bash
# quark-on-spark -- one command to serve Quark behind a public OpenAI-compatible URL.
#
#   ./setup.sh            start the API and a cloudflared quick tunnel, print URL + token
#   ./setup.sh stop       stop both
#   ./setup.sh status     show health, URL and token
#   ./setup.sh logs       tail both logs
#
# Environment: PORT, MAX_CONCURRENT, MAX_BATCH, KV_BUDGET_GB, MODEL_ID, NO_TUNNEL=1
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8000}"
MAX_CONCURRENT="${MAX_CONCURRENT:-64}"
MAX_BATCH="${MAX_BATCH:-$MAX_CONCURRENT}"
KV_BUDGET_GB="${KV_BUDGET_GB:-80}"
NO_TUNNEL="${NO_TUNNEL:-0}"
MODEL_ID="${MODEL_ID:-empero-ai/Quark-150M-128K-Zero}"
VENV="./venv"
PY="$VENV/bin/python"
mkdir -p logs bin

say() { printf '%s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

stop_all() {
  for f in server.pid cloudflared.pid; do
    if [ -f "$f" ]; then
      pid="$(cat "$f")"
      kill "$pid" 2>/dev/null || true
      rm -f "$f"
    fi
  done
  pkill -f "uvicorn server:app --host 127.0.0.1 --port $PORT" 2>/dev/null || true
  pkill -f "cloudflared tunnel --no-autoupdate --url http://127.0.0.1:$PORT" 2>/dev/null || true
  sleep 1
}

url_of_tunnel() {
  grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' logs/cloudflared.log 2>/dev/null | tail -1
}

case "${1:-start}" in
  stop) stop_all; say "stopped"; exit 0 ;;
  logs) tail -n 40 -f logs/server.log logs/cloudflared.log; exit 0 ;;
  status)
    say "health: $(curl -s "http://127.0.0.1:$PORT/health" || echo 'not running')"
    say "url:    $(url_of_tunnel || true)"
    say "token:  $(cat token.txt 2>/dev/null || echo 'not generated yet')"
    exit 0 ;;
  start) ;;
  *) die "unknown command '$1' (use start, stop, status or logs)" ;;
esac

command -v python3 >/dev/null || die "python3 not found"
command -v curl >/dev/null || die "curl not found"
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' \
  || die "python 3.10 or newer is required"

if curl -s "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  die "something is already listening on port $PORT (run ./setup.sh stop first, or set PORT)"
fi

# 1. virtualenv + dependencies
if [ ! -x "$PY" ]; then
  say "==> creating virtualenv"
  python3 -m venv "$VENV" || die "python3 -m venv failed (install python3-venv)"
fi
if [ ! -f ".deps-installed" ]; then
  say "==> installing dependencies (this takes a few minutes the first time)"
  "$PY" -m pip install --upgrade pip -q
  "$PY" -m pip install -q -r requirements.txt
  touch .deps-installed
fi

# 2. bearer token
if [ ! -f token.txt ]; then
  "$PY" -c 'import secrets; print(secrets.token_urlsafe(32))' > token.txt
  chmod 600 token.txt
  say "==> generated token.txt"
fi
TOKEN="$(cat token.txt)"

# 3. model weights
if [ ! -f "model/config.json" ]; then
  say "==> downloading $MODEL_ID (~660 MB)"
  "$PY" - <<PY
from huggingface_hub import snapshot_download
snapshot_download("$MODEL_ID", local_dir="model")
PY
fi

# 4. cloudflared
if [ "$NO_TUNNEL" != "1" ] && [ ! -x bin/cloudflared ]; then
  case "$(uname -m)" in
    aarch64|arm64) CF=cloudflared-linux-arm64 ;;
    x86_64|amd64)  CF=cloudflared-linux-amd64 ;;
    *) die "unsupported architecture $(uname -m)" ;;
  esac
  say "==> downloading cloudflared"
  curl -fsSL -o bin/cloudflared \
    "https://github.com/cloudflare/cloudflared/releases/latest/download/$CF" \
    || die "could not download cloudflared"
  chmod +x bin/cloudflared
fi

# 5. API server
say "==> starting the API on 127.0.0.1:$PORT"
PORT="$PORT" MAX_CONCURRENT="$MAX_CONCURRENT" MAX_BATCH="$MAX_BATCH" \
KV_BUDGET_GB="$KV_BUDGET_GB" MODEL_ID="$MODEL_ID" \
  nohup "$VENV/bin/uvicorn" server:app --host 127.0.0.1 --port "$PORT" --workers 1 \
  >> logs/server.log 2>&1 &
echo $! > server.pid

for _ in $(seq 1 90); do
  if curl -s "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then break; fi
  sleep 2
done
curl -s "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 \
  || { say "the API did not come up; last lines of logs/server.log:"; tail -n 20 logs/server.log; exit 1; }
say "    $(curl -s "http://127.0.0.1:$PORT/health")"

# 6. tunnel
BASE="http://127.0.0.1:$PORT"
if [ "$NO_TUNNEL" = "1" ]; then
  say ""
  say "Base URL: $BASE/v1   (tunnel disabled, local only)"
  say "Token:    $TOKEN"
  exit 0
fi
say "==> opening a cloudflared quick tunnel"
nohup ./bin/cloudflared tunnel --no-autoupdate --url "$BASE" >> logs/cloudflared.log 2>&1 &
echo $! > cloudflared.pid

URL=""
for _ in $(seq 1 30); do
  URL="$(url_of_tunnel || true)"
  [ -n "$URL" ] && break
  sleep 2
done
[ -n "$URL" ] || { say "no tunnel URL yet; check logs/cloudflared.log"; exit 1; }

say ""
say "  Base URL: $URL/v1"
say "  Token:    $TOKEN"
say ""
say "  curl $URL/v1/chat/completions \\"
say "    -H \"Authorization: Bearer $TOKEN\" \\"
say "    -H \"Content-Type: application/json\" \\"
say "    -d '{\"model\":\"$(basename "$MODEL_ID")\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}'"
say ""
say "  Stop with ./setup.sh stop. Quick-tunnel URLs change on every restart."
