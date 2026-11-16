#!/usr/bin/env bash
# Keep the Quark API and its cloudflared quick tunnel alive.
# A process can be alive yet wedged, so the API is judged by its /health probe,
# with SIGTERM then SIGKILL escalation. The live public URL lands in current-url.txt.
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p logs

PORT="${PORT:-8000}"
MAX_CONCURRENT="${MAX_CONCURRENT:-64}"
MAX_BATCH="${MAX_BATCH:-64}"
KV_BUDGET_GB="${KV_BUDGET_GB:-48}"
MAX_DEFAULT_OUTPUT="${MAX_DEFAULT_OUTPUT:-1024}"
LOGS=logs/supervisor.log

log() { printf '%s %s\n' "$(date -Is)" "$*" >> "$LOGS"; }

# keep the logs from growing without bound: one rotated copy, then truncate in place
# (truncating rather than replacing keeps the file descriptor uvicorn/cloudflared hold)
rotate_logs() {
  local f size
  for f in server.log cloudflared.log; do
    [ -f "$f" ] || continue
    size=$(stat -c %s "$f" 2>/dev/null || echo 0)
    if [ "$size" -gt 20971520 ]; then
      cp "$f" "$f.1" 2>/dev/null || true
      : > "$f"
      log "rotated $f (was $((size / 1024 / 1024)) MB)"
    fi
  done
}
pid_of() { cat "$1" 2>/dev/null; }

alive() {           # pid file -> running and not a zombie
  local f="$1" pid st
  pid=$(pid_of "$f") || return 1
  [ -n "${pid:-}" ] || return 1
  [ -d "/proc/$pid" ] || return 1
  st=$(awk '{print $3}' "/proc/$pid/stat" 2>/dev/null)
  [ "${st:-Z}" != "Z" ] || return 1
  return 0
}

healthy() { curl -sf -m 10 -o /dev/null "http://127.0.0.1:$PORT/health"; }

last_url() { grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' cloudflared.log 2>/dev/null | tail -1; }

record_url() {
  local u
  u=$(last_url)
  [ -n "${u:-}" ] || return 0
  if [ "$u" != "$(cat current-url.txt 2>/dev/null)" ]; then
    printf '%s\n' "$u" > current-url.txt
    log "public url $u"
  fi
}

start_server() {
  if healthy; then log "port $PORT already answering, not spawning a duplicate"; return; fi
  PORT="$PORT" MAX_CONCURRENT="$MAX_CONCURRENT" MAX_BATCH="$MAX_BATCH" \
  KV_BUDGET_GB="$KV_BUDGET_GB" MAX_DEFAULT_OUTPUT="$MAX_DEFAULT_OUTPUT" \
    nohup "$PWD/venv/bin/uvicorn" server:app --host 127.0.0.1 --port "$PORT" --workers 1 \
    >> server.log 2>&1 &
  echo $! > server.pid
  log "started api pid $(cat server.pid)"
}

stop_server() {
  local pid
  pid=$(pid_of server.pid); [ -n "${pid:-}" ] || return 0
  log "api pid $pid unresponsive, sending SIGTERM"
  kill "$pid" 2>/dev/null
  for _ in $(seq 1 10); do sleep 2; alive server.pid || return 0; done
  log "api pid $pid survived SIGTERM for 20s, sending SIGKILL"
  kill -9 "$pid" 2>/dev/null
  sleep 2
}

start_tunnel() {
  local before u
  before=$(last_url)
  nohup ./bin/cloudflared tunnel --no-autoupdate --url "http://127.0.0.1:$PORT" \
    >> cloudflared.log 2>&1 &
  echo $! > cloudflared.pid
  log "started tunnel pid $(cat cloudflared.pid)"
  # a new tunnel prints its hostname only once it is registered: wait for a *new* one
  for _ in $(seq 1 30); do
    sleep 2
    u=$(last_url)
    if [ -n "${u:-}" ] && [ "$u" != "${before:-}" ]; then
      printf '%s\n' "$u" > current-url.txt
      log "public url $u"
      return 0
    fi
  done
  log "tunnel started but no new url appeared within 60s"
}

log "supervisor up (pid $$)"
if ! alive server.pid || ! healthy; then stop_server; start_server; fi
alive cloudflared.pid || start_tunnel
record_url

fails=0
tick=0
while true; do
  sleep 15
  tick=$((tick + 1))
  if healthy; then fails=0; else fails=$((fails + 1)); fi
  if [ "$fails" -ge 3 ]; then
    log "api unresponsive for 3 probes, restarting"
    stop_server
    start_server
    fails=0
  fi
  if ! alive cloudflared.pid; then
    log "tunnel gone, restarting"
    start_tunnel
  fi
  if [ $((tick % 20)) -eq 0 ]; then
    rotate_logs
    record_url
    log "heartbeat api=$(alive server.pid && echo up || echo down) health=$([ "$fails" -eq 0 ] && echo ok || echo "fail:$fails") url=$(cat current-url.txt 2>/dev/null)"
  fi
done
