# quark-on-spark

Serve [Quark](https://huggingface.co/empero-ai/Quark-150M-128K-Zero), a 165M-parameter hybrid
model with a 128K context window, as an OpenAI-compatible API on a single NVIDIA box, behind a
public `trycloudflare.com` URL, with one command.

```bash
./setup.sh
```

That installs dependencies, downloads the weights, starts the API, opens a quick tunnel, and
prints the base URL and bearer token:

```
  Base URL: https://random-words-here.trycloudflare.com/v1
  Token:    <generated token>
```

## Requirements

- Linux on aarch64 or x86_64, Python 3.10+, `curl`
- An NVIDIA GPU. The weights are float32 and load into ~0.7 GB of VRAM; a 128K-token session
  adds ~1.1 GB of KV cache, so a 16 GB card is comfortable for a handful of long sessions.
  Measured on a GB10 (130 GB unified memory, 20 cores).
- No Hugging Face token needed: the weights are public. `git`, `docker` and root are not used.

## What you get

| Endpoint | Notes |
| --- | --- |
| `POST /v1/chat/completions` | streaming and non-streaming |
| `POST /v1/responses` | OpenAI Responses API, streaming events and non-streaming |
| `POST /v1/completions` | legacy text completions |
| `GET /v1/models` | advertises `context_length: 131072`, `max_output_tokens: 65536` |
| `GET /health` | in-flight count, reserved KV cache, batch size |
| `GET /` | small status page |

Every `/v1` route needs `Authorization: Bearer <token>`. `/health` and `/` are open.

```bash
curl "$BASE/chat/completions" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"Quark-150M-128K-Zero","messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
```

Anything that speaks the OpenAI API works: point a client's `base_url` at the printed URL and
use the token as the API key.

## Limits and overload behaviour

- Context is 131072 tokens per request. Output is capped at 65536, and **defaults to 1024
  when a client does not send `max_tokens`** -- asking for the ceiling by accident used to let
  one request hold the GPU for twenty minutes. Over-long prompts are rejected with
  `400 context_length_exceeded` before they touch the GPU.
- 64 requests in flight, plus an 80 GB KV-cache budget as a backstop. Either limit answers
  `429` with an OpenAI-style `overloaded_error` and `Retry-After: 2`. There is no queue, so a
  rejected request fails fast instead of hanging.
- Context is per request: the server keeps no conversation state, so clients resend history.
- A batch is bounded by tokens (`MAX_BATCH_TOKENS`) as well as rows, so 64 sessions at 128k
  context split into several passes instead of trying to prefill millions of tokens at once.
- Batches are time-sliced: a job decodes at most 64 to 2048 tokens per pass (scaled with
  context length), then goes back in the queue. A long generation cannot block newcomers:
  measured on this box, 16-token requests answer in ~1.5 s while a 2000-token job is running.
- A job holds its slot until the work actually stops. If a client disconnects, the slot is
  returned by whichever side finishes first, so a dropped connection cannot leak capacity.
- Request bodies are capped at 8 MB (`MAX_BODY_MB`); larger ones get `413 request_too_large`.

## How it serves 64 sessions on one GPU

One process, one GPU, a small batched engine. Requests are gathered for 20 ms, clustered by
output budget and prompt length, left-padded, and decoded as one batch with per-row temperature
and top_p; each row streams its own SSE deltas. Batching is what makes concurrency work: a
naive one-`generate()`-per-request server collapses to 0.6 tok/s per stream at 64 concurrent.

Each pass re-prefills the batch, which is what makes the slicing fair, and the slice grows
with context length to keep that tax small. Measured with the same 256-token answer: 84 tok/s
on a short prompt, 43 tok/s at ~10k tokens of context, 27 tok/s at ~25k. Raise or lower the
`width // 40` divisor in `run_batch` to trade long-context speed against how quickly a newcomer
gets its turn.

Measured here, 192-token replies, one GB10:

| concurrent | aggregate | per stream |
| --- | --- | --- |
| 1 | 68 tok/s | 68 tok/s |
| 8 | 370 tok/s | 46 tok/s |
| 32 | 309 tok/s | 10 tok/s |
| 64 | 711 tok/s | 14 tok/s |

## Keeping it alive

`./setup.sh` starts `supervise.sh`, which loops every 15 seconds. The API is judged by its
`/health` probe rather than by the process existing, because a process can be alive and wedged;
three failed probes trigger SIGTERM, then SIGKILL after 20 seconds. The tunnel is judged by pid.
Whenever the tunnel restarts, the new URL is written to `current-url.txt`, because a quick
tunnel never reuses a hostname.

That matters in practice: a quick tunnel that dies leaves Cloudflare answering `502` until
something brings it back. Nothing here survives a reboot, so run `./setup.sh` again after one.

`/health` returns `503` with `"status": "degraded"` if the batch engine stops beating, which is
how the supervisor notices a wedged engine rather than a dead process. For a deployment that
should survive reboots and logouts, wrap it in a systemd user unit instead of a bare shell:

```ini
[Unit]
Description=quark-on-spark
[Service]
WorkingDirectory=/path/to/quark-on-spark
ExecStart=/bin/bash supervise.sh
Restart=always
KillMode=control-group
[Install]
WantedBy=default.target
```

Then `systemctl --user enable --now quark-on-spark` and `loginctl enable-linger $USER`.

## Per-client limits

Two guards stop one caller from owning the whole endpoint. `PER_IP_MAX` caps how many
requests a single client holds at once, so with 64 slots and a cap of 24 one client cannot
starve everyone else. `RATE_LIMIT_RPM` is a token bucket per client: `RATE_BURST` requests at
once, refilling at `RATE_LIMIT_RPM` per minute. Both answer `429` with distinct error codes
(`client_concurrency_limit`, `rate_limit_exceeded`) and `Retry-After`.

The client is identified by `CF-Connecting-IP`, falling back to `X-Real-IP`, `X-Forwarded-For`
and finally the socket address. That is why the server must stay bound to `127.0.0.1` behind
cloudflared: those headers are trusted, and only the tunnel should be able to set them.

Set `PER_IP_MAX` at or above `MAX_CONCURRENT` to let a single client use every slot.

## Shutdown behaviour

On SIGTERM the server keeps serving in-flight requests for `DRAIN_GRACE` seconds, then stops;
anything cut short comes back with `finish_reason: "length"` rather than pretending it finished.
This is what stops a stopping process from hanging on a long generation, which once kept a
wedged server holding 11 GB of GPU memory.

## Configuration

Set these as environment variables before `./setup.sh`:

| Variable | Default | Meaning |
| --- | --- | --- |
| `PORT` | `8000` | local port the API binds to (127.0.0.1 only) |
| `MAX_CONCURRENT` | `64` | in-flight request slots |
| `MAX_BATCH` | same as above | rows per forward pass |
| `KV_BUDGET_GB` | `48` | KV-cache budget, the memory backstop |
| `MAX_DEFAULT_OUTPUT` | `1024` | output budget when a client omits `max_tokens` |
| `MAX_BODY_MB` | `8` | largest accepted request body |
| `MAX_BATCH_TOKENS` | `262144` | tokens per forward pass; splits oversized batches |
| `PER_IP_MAX` | `24` | requests one client may have in flight (0 disables) |
| `RATE_LIMIT_RPM` | `180` | sustained requests per minute per client |
| `RATE_BURST` | `32` | instantaneous burst per client (keep above `PER_IP_MAX`) |
| `DRAIN_GRACE` | `20` | seconds in-flight work is allowed to finish on shutdown |
| `MODEL_ID` | `empero-ai/Quark-150M-128K-Zero` | any compatible checkpoint |
| `NO_TUNNEL` | `0` | `1` keeps it local, no cloudflared |

```bash
PORT=8100 MAX_CONCURRENT=16 ./setup.sh
./setup.sh status     # health, URL, token
./setup.sh stop
```

The port is remembered in `.run-port`, so `status`, `stop` and `logs` work without repeating
`PORT=`.

## Caveats worth reading

- Quick tunnels are for testing: no uptime guarantee, a new URL on every restart, and Cloudflare
  documents a 200 in-flight ceiling (SSE is documented as unsupported, though it works in
  practice here). For anything permanent, use a named Cloudflare Tunnel with your own hostname.
- Anyone with the URL and token can use the endpoint. Rotate by deleting `token.txt` and
  restarting.
- Quark is a research artifact, not a product: it is a 165M-parameter model, so expect 165M
  quality. It has no tool-calling support here, and no safety training beyond a small refusal
  set.
- `./setup.sh` leaves two detached processes behind (uvicorn and cloudflared) with logs in
  `logs/`. Nothing else on the machine is touched.

## Files

| File | What it is |
| --- | --- |
| `server.py` | FastAPI app: auth, limits, batch engine, OpenAI-compatible routes |
| `setup.sh` | install, start, stop, status |
| `requirements.txt` | torch, transformers>=5.23, fastapi, uvicorn |
| `supervise.sh` | keep-alive: restarts the API or tunnel if either exits or wedges |
| `token.txt` | generated on first run, git-ignored |
| `current-url.txt` | the live tunnel URL, rewritten whenever the tunnel restarts |

## License

Apache-2.0.
