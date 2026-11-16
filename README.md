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

- Context is 131072 tokens per request, output is capped at 65536. Over-long prompts are
  rejected with `400 context_length_exceeded` before they touch the GPU.
- 64 requests in flight, plus an 80 GB KV-cache budget as a backstop. Either limit answers
  `429` with an OpenAI-style `overloaded_error` and `Retry-After: 2`. There is no queue, so a
  rejected request fails fast instead of hanging.
- Context is per request: the server keeps no conversation state, so clients resend history.

## How it serves 64 sessions on one GPU

One process, one GPU, a small batched engine. Requests are gathered for 20 ms, clustered by
output budget and prompt length, left-padded, and decoded as one batch with per-row temperature
and top_p; each row streams its own SSE deltas. Batching is what makes concurrency work: a
naive one-`generate()`-per-request server collapses to 0.6 tok/s per stream at 64 concurrent.

Measured here, 192-token replies, one GB10:

| concurrent | aggregate | per stream |
| --- | --- | --- |
| 1 | 68 tok/s | 68 tok/s |
| 8 | 370 tok/s | 46 tok/s |
| 32 | 309 tok/s | 10 tok/s |
| 64 | 711 tok/s | 14 tok/s |

## Configuration

Set these as environment variables before `./setup.sh`:

| Variable | Default | Meaning |
| --- | --- | --- |
| `PORT` | `8000` | local port the API binds to (127.0.0.1 only) |
| `MAX_CONCURRENT` | `64` | in-flight request slots |
| `MAX_BATCH` | same as above | rows per forward pass |
| `KV_BUDGET_GB` | `80` | KV-cache budget, the memory backstop |
| `MODEL_ID` | `empero-ai/Quark-150M-128K-Zero` | any compatible checkpoint |
| `NO_TUNNEL` | `0` | `1` keeps it local, no cloudflared |

```bash
PORT=8100 MAX_CONCURRENT=16 ./setup.sh
./setup.sh status     # health, URL, token
./setup.sh stop
```

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
| `token.txt` | generated on first run, git-ignored |

## License

Apache-2.0.
