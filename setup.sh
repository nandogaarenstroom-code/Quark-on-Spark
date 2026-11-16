#!/usr/bin/env bash
# quark-on-spark -- one command to serve Quark behind a public OpenAI-compatible URL.
#
#   ./setup.sh            install, start the API + a quick tunnel, print URL and token
#   ./setup.sh stop       stop everything
#   ./setup.sh status     show health, current URL and token
#   ./setup.sh logs       tail the logs
#
# Environment: PORT, MAX_CONCURRENT, MAX_BATCH, KV_BUDGET_GB, MAX_DEFAULT_OUTPUT, MODEL_ID, NO_TUNNEL=1
set -euo pipefail
cd "$(dirname "$0")"

# remember the port used, so stop/status/logs work without repeating PORT=...
if [ -f .run-port ] && [ -z "${PORT:-}" ]; then
  PORT="$(cat .run-port)"
fi
PORT="${PORT:-8000}"
MAX_CONCURRENT="${MAX_CONCURRENT:-64}"
MAX_BATCH="${MAX_BATCH:-$MAX_CONCURRENT}"
KV_BUDGET_GB="${KV_BUDGET_GB:-48}"
MAX_DEFAULT_OUTPUT="${MAX_DEFAULT_OUTPUT:-1024}"
NO_TUNNEL="${NO_TUNNEL:-0}"
MODEL_ID="${MODEL_ID:-empero-ai/Quark-150M-128K-Zero}"
VENV="./venv"
PY="$VENV/bin/python"
mkdir -p logs bin

say() { printf '%s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

stop_all() {
  if [ -f supervisor.pid ]; then
    kill "$(cat supervisor.pid)" 2>/dev/null || true
    rm -f supervisor.pid
  fi
  sleep 1
  for f in server.pid cloudflared.pid; do
    if [ -f "$f" ]; then
      kill "$(cat "$f")" 2>/dev/null || true
      rm -f "$f"
    fi
  done
  pkill -f "uvicorn server:app --host 127.0.0.1 --port $PORT" 2>/dev/null || true
  pkill -f "cloudflared tunnel --no-autoupdate --url http://127.0.0.1:$PORT" 2>/dev/null || true
  sleep 1
}

current_url() { cat current-url.txt 2>/dev/null || true; }

case "${1:-start}" in
  stop) stop_all; rm -f .run-port; say "stopped"; exit 0 ;;
  logs) tail -n 40 -f logs/supervisor.log server.log cloudflared.log; exit 0 ;;
  status)
    say "health: $(curl -s "http://127.0.0.1:$PORT/health" || echo 'not running')"
    say "url:    $(current_url)"
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
  say "==> installing dependencies (a few minutes the first time)"
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

# 5. supervisor: starts the API and the tunnel, and restarts either if it dies
printf '%s\n' "$PORT" > .run-port
say "==> starting the API and tunnel under the supervisor"
PORT="$PORT" MAX_CONCURRENT="$MAX_CONCURRENT" MAX_BATCH="$MAX_BATCH" \
KV_BUDGET_GB="$KV_BUDGET_GB" MAX_DEFAULT_OUTPUT="$MAX_DEFAULT_OUTPUT" \
  nohup bash supervise.sh >> logs/supervisor.log 2>&1 &
echo $! > supervisor.pid

for _ in $(seq 1 90); do
  if curl -s "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then break; fi
  sleep 2
done
curl -s "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 \
  || { say "the API did not come up; last lines of logs/supervisor.log and server.log:"; tail -n 20 logs/supervisor.log server.log 2>/dev/null; exit 1; }
say "    $(curl -s "http://127.0.0.1:$PORT/health")"

if [ "$NO_TUNNEL" = "1" ]; then
  say ""
  say "Base URL: http://127.0.0.1:$PORT/v1   (tunnel disabled, local only)"
  say "Token:    $TOKEN"
  exit 0
fi

URL=""
for _ in $(seq 1 30); do
  URL="$(current_url)"
  [ -n "$URL" ] && break
  sleep 2
done
[ -n "$URL" ] || { say "no tunnel URL yet; check cloudflared.log"; exit 1; }

say ""
say "  Base URL: $URL/v1"
say "  Token:    $TOKEN"
say ""
say "  curl $URL/v1/chat/completions \\"
say "    -H \"Authorization: Bearer $TOKEN\" \\"
say "    -H \"Content-Type: application/json\" \\"
say "    -d '{\"model\":\"$(basename "$MODEL_ID")\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}'"
say ""
say "  Stop with ./setup.sh stop. If cloudflared restarts, the URL changes:"
say "  read current-url.txt (or ./setup.sh status) for the live one."
