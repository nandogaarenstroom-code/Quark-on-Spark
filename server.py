#!/usr/bin/env python3
"""OpenAI-compatible API server for Quark on a single GPU.

One process, one GPU, a small batched engine: requests are gathered for 20 ms,
clustered by budget and prompt length, and decoded as one padded batch.
"""
import asyncio
import json
import os
import secrets
import queue
import threading
import time
import uuid

import torch
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LogitsProcessor
from transformers.generation.streamers import BaseStreamer

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(ROOT, "model"))
MODEL_ID = os.environ.get("MODEL_ID", "empero-ai/Quark-150M-128K-Zero")
MODEL_NAME = os.environ.get("MODEL_NAME", MODEL_ID.split("/")[-1])
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", MODEL_ID)
MAX_CONTEXT = int(os.environ.get("MAX_CONTEXT", "131072"))
MAX_OUTPUT = int(os.environ.get("MAX_OUTPUT", "65536"))
MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", "64"))
KV_BYTES_PER_TOKEN = 8192  # 4 global layers x 4 KV heads x 64 dim x 2 (K,V) x 4 bytes
KV_BUDGET_BYTES = int(float(os.environ.get("KV_BUDGET_GB", "80")) * 1e9)
TOKEN_PATH = os.environ.get("TOKEN_PATH", os.path.join(ROOT, "token.txt"))
if os.path.exists(TOKEN_PATH):
    TOKEN = open(TOKEN_PATH, "r", encoding="utf-8").read().strip()
else:
    TOKEN = secrets.token_urlsafe(32)
    with open(TOKEN_PATH, "w", encoding="utf-8") as fh:
        fh.write(TOKEN + "\n")
    os.chmod(TOKEN_PATH, 0o600)
STARTED = time.time()

slots = threading.Semaphore(MAX_CONCURRENT)
active = 0
kv_reserved = 0
active_lock = threading.Lock()
stats = {"requests": 0, "rejected": 0, "prompt_tokens": 0, "completion_tokens": 0}

app = FastAPI(title=MODEL_NAME, version="1.0", docs_url=None, redoc_url=None, openapi_url=None)
tok = None
model = None


# ---------------------------------------------------------------- streaming
# --------------------------------------------------------------- batch engine
MAX_BATCH = int(os.environ.get("MAX_BATCH", str(MAX_CONCURRENT)))
EOS_IDS = None


class Job:
    """One in-flight generation, fed by the engine thread."""

    __slots__ = ("prompt_ids", "budget", "temperature", "top_p", "loop", "queue", "parts", "count")

    def __init__(self, prompt_ids, budget, temperature, top_p, loop):
        self.prompt_ids = prompt_ids
        self.budget = budget
        self.temperature = temperature
        self.top_p = top_p
        self.loop = loop
        self.queue = asyncio.Queue()
        self.parts = []
        self.count = 0

    def emit(self, text):
        self.parts.append(text)
        self.loop.call_soon_threadsafe(self.queue.put_nowait, ("delta", text))

    def finish(self):
        self.loop.call_soon_threadsafe(self.queue.put_nowait, ("done", None))

    def fail(self, message):
        self.loop.call_soon_threadsafe(self.queue.put_nowait, ("error", message))


class PerRowSampler(LogitsProcessor):
    """Per-request temperature / top_p inside one batched forward pass."""

    def __init__(self, temps, top_ps):
        self.temps = temps
        self.top_ps = top_ps
        self._t = None
        self._p = None

    def __call__(self, input_ids, scores):
        if self._t is None or self._t.device != scores.device or self._t.shape[0] != scores.shape[0]:
            self._t = torch.tensor(self.temps, dtype=torch.float32, device=scores.device).unsqueeze(-1)
            self._p = torch.tensor(self.top_ps, dtype=torch.float32, device=scores.device).unsqueeze(-1)
        greedy = self._t <= 0
        safe_t = torch.where(greedy, torch.ones_like(self._t), self._t)
        out = scores.float() / safe_t

        if float(self._p.min()) < 1.0:
            sorted_logits, sorted_idx = torch.sort(out, dim=-1, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(probs, dim=-1)
            cut = (cumulative - self._p) > 0
            cut[..., 1:] = cut[..., :-1].clone()
            cut[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(cut, float("-inf"))
            out = torch.empty_like(out).scatter_(-1, sorted_idx, sorted_logits)

        if bool(greedy.any()):
            keep = out.argmax(dim=-1, keepdim=True)
            only = torch.full_like(out, float("-inf")).scatter_(-1, keep, 0.0)
            out = torch.where(greedy, only, out)
        return out


class BatchedStreamer(BaseStreamer):
    """Splits one batched decode step into per-request text deltas."""

    def __init__(self, jobs, tokenizer, eos_ids):
        self.jobs = jobs
        self.tok = tokenizer
        self.eos = set(eos_ids or [])
        self.next_tokens_are_prompt = True
        self.pending = [[] for _ in jobs]
        self.done = [False] * len(jobs)
        self.count = [0] * len(jobs)

    def _flush(self, i):
        if self.pending[i]:
            text = self.tok.decode(self.pending[i], skip_special_tokens=True)
            self.pending[i] = []
            if text:
                self.jobs[i].emit(text)

    def put(self, value):
        if self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return
        ids = value.reshape(-1).tolist() if hasattr(value, "reshape") else [int(value)]
        for i, job in enumerate(self.jobs):
            if self.done[i]:
                continue
            token = ids[i]
            if token in self.eos:
                self.done[i] = True
                self._flush(i)
                continue
            self.count[i] += 1
            self.pending[i].append(token)
            if self.count[i] >= job.budget:
                self.done[i] = True
                self._flush(i)
                continue
            text = self.tok.decode(self.pending[i], skip_special_tokens=True)
            if text.endswith("\ufffd"):
                continue
            self.pending[i] = []
            if text:
                job.emit(text)

    def end(self):
        for i in range(len(self.jobs)):
            self._flush(i)
            self.jobs[i].count = self.count[i]


pending_jobs = queue.Queue()


def engine_loop():
    while True:
        first = pending_jobs.get()
        jobs = [first]
        deadline = time.time() + 0.02
        while len(jobs) < MAX_BATCH:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                jobs.append(pending_jobs.get(timeout=remaining))
            except queue.Empty:
                break
        # cluster similar budgets and prompt lengths so padding waste stays small
        jobs.sort(key=lambda j: (j.budget // 256, int(j.prompt_ids.shape[1]) // 512))
        try:
            run_batch(jobs)
        except Exception as exc:
            for job in jobs:
                job.fail(str(exc))


def run_batch(jobs):
    width = max(int(j.prompt_ids.shape[1]) for j in jobs)
    rows = len(jobs)
    input_ids = torch.full((rows, width), tok.pad_token_id, dtype=torch.long)
    attention = torch.zeros((rows, width), dtype=torch.long)
    for i, job in enumerate(jobs):
        n = int(job.prompt_ids.shape[1])
        input_ids[i, width - n:] = job.prompt_ids[0]
        attention[i, width - n:] = 1
    streamer = BatchedStreamer(jobs, tok, model.generation_config.eos_token_id)
    with torch.inference_mode():
        model.generate(
            input_ids=input_ids.to("cuda"),
            attention_mask=attention.to("cuda"),
            max_new_tokens=max(j.budget for j in jobs),
            do_sample=True,
            logits_processor=[PerRowSampler([j.temperature for j in jobs], [j.top_p for j in jobs])],
            eos_token_id=model.generation_config.eos_token_id,
            pad_token_id=tok.pad_token_id,
            streamer=streamer,
        )
    streamer.end()
    for job in jobs:
        job.finish()


def submit(prompt_ids, budget, body, loop):
    temperature = body.get("temperature")
    temperature = 1.0 if temperature is None else float(temperature)
    top_p = body.get("top_p")
    top_p = 0.95 if top_p is None else float(top_p)
    job = Job(prompt_ids, budget, temperature, min(max(top_p, 1e-4), 1.0), loop)
    pending_jobs.put(job)
    return job


async def job_stream(job):
    while True:
        kind, payload = await job.queue.get()
        if kind == "delta":
            yield payload
        elif kind == "error":
            raise RuntimeError(payload)
        else:
            return


async def job_collect(job):
    parts = []
    async for delta in job_stream(job):
        parts.append(delta)
    return "".join(parts).strip(), job.count


# ------------------------------------------------------------------ helpers
def err(status, message, etype="invalid_request_error", code=None, param=None):
    body = {"error": {"message": message, "type": etype, "param": param, "code": code}}
    return JSONResponse(status_code=status, content=body)


def overloaded(reason="slots"):
    if reason == "memory":
        message = ("Server KV-cache budget is full (%.0f GB reserved of %.0f GB). Retry shortly."
                   % (kv_reserved / 1e9, KV_BUDGET_BYTES / 1e9))
    else:
        message = "Server is at its concurrency limit (%d in flight). Retry shortly." % MAX_CONCURRENT
    return JSONResponse(
        status_code=429,
        headers={"Retry-After": "2"},
        content={
            "error": {
                "message": message,
                "type": "overloaded_error",
                "param": None,
                "code": "overloaded",
            }
        },
    )


def check_auth(request: Request):
    header = request.headers.get("authorization") or ""
    if header.lower().startswith("bearer "):
        presented = header[7:].strip()
    else:
        presented = (request.headers.get("api-key") or request.headers.get("x-api-key") or "").strip()
    if presented and secrets.compare_digest(presented, TOKEN):
        return None
    return JSONResponse(
        status_code=401,
        headers={"WWW-Authenticate": 'Bearer realm="quark"'},
        content={
            "error": {
                "message": "Missing or invalid API token. Send 'Authorization: Bearer <token>'.",
                "type": "invalid_request_error",
                "param": None,
                "code": "invalid_api_key",
            }
        },
    )


def acquire(reserve_bytes=0):
    global active, kv_reserved
    with active_lock:
        if kv_reserved + reserve_bytes > KV_BUDGET_BYTES:
            stats["rejected"] += 1
            return "memory"
    if not slots.acquire(blocking=False):
        stats["rejected"] += 1
        return "slots"
    with active_lock:
        active += 1
        kv_reserved += reserve_bytes
    stats["requests"] += 1
    return None


def release(reserve_bytes=0):
    global active, kv_reserved
    with active_lock:
        active = max(0, active - 1)
        kv_reserved = max(0, kv_reserved - reserve_bytes)
    slots.release()


def messages_from_chat(body):
    msgs = body.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return None, "Field 'messages' must be a non-empty array."
    clean = []
    for m in msgs:
        if not isinstance(m, dict) or "role" not in m:
            return None, "Each message needs a 'role'."
        content = m.get("content")
        if isinstance(content, list):
            parts = []
            for p in content:
                if isinstance(p, dict) and p.get("type") in ("text", "input_text", "output_text"):
                    parts.append(str(p.get("text", "")))
                elif isinstance(p, str):
                    parts.append(p)
            content = "".join(parts)
        elif content is None:
            content = ""
        else:
            content = str(content)
        clean.append({"role": str(m["role"]), "content": content})
    return clean, None


def messages_from_responses(body):
    instructions = body.get("instructions")
    raw = body.get("input")
    msgs = []
    if isinstance(instructions, str) and instructions.strip():
        msgs.append({"role": "system", "content": instructions})
    if isinstance(raw, str):
        msgs.append({"role": "user", "content": raw})
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, str):
                msgs.append({"role": "user", "content": item})
                continue
            if not isinstance(item, dict):
                return None, "Unsupported item in 'input'."
            itype = item.get("type")
            if itype in (None, "message"):
                role = item.get("role") or "user"
                if role == "developer":
                    role = "system"
                content = item.get("content", "")
                if isinstance(content, list):
                    parts = []
                    for p in content:
                        if isinstance(p, dict):
                            parts.append(str(p.get("text", "")))
                        elif isinstance(p, str):
                            parts.append(p)
                    content = "".join(parts)
                msgs.append({"role": str(role), "content": str(content)})
            elif itype == "input_text":
                msgs.append({"role": "user", "content": str(item.get("text", ""))})
            else:
                return None, "Unsupported input item type '%s'." % itype
    elif raw is None:
        return None, "Field 'input' is required."
    else:
        return None, "Field 'input' must be a string or an array."
    if not msgs:
        return None, "No usable input messages."
    return msgs, None


def build_prompt(msgs):
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    return ids


def budget_for(body, prompt_len, max_field_names):
    requested = None
    for name in max_field_names:
        if body.get(name) is not None:
            requested = int(body[name])
            break
    if requested is not None and requested < 1:
        return None, "max tokens must be >= 1."
    if requested is not None and requested > MAX_OUTPUT:
        return None, "Requested max tokens %d exceeds the limit of %d." % (requested, MAX_OUTPUT)
    room = MAX_CONTEXT - prompt_len
    if room < 1:
        return None, (
            "Prompt is %d tokens, over the %d-token context limit." % (prompt_len, MAX_CONTEXT)
        )
    budget = min(requested or MAX_OUTPUT, room, MAX_OUTPUT)
    return budget, None


# ------------------------------------------------------------------ endpoints
@app.on_event("startup")
def load_model():
    global tok, model
    asyncio.get_running_loop().set_default_executor(
        ThreadPoolExecutor(max_workers=MAX_CONCURRENT + 8, thread_name_prefix="gen")
    )
    source = MODEL_DIR if os.path.isdir(MODEL_DIR) else MODEL_ID
    if source == MODEL_ID:
        print("model not found in %s, downloading %s" % (MODEL_DIR, MODEL_ID), flush=True)
        from huggingface_hub import snapshot_download
        source = snapshot_download(MODEL_ID, local_dir=MODEL_DIR)
    tok = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        source, trust_remote_code=True, dtype=torch.float32, device_map="cuda"
    )
    model.eval()
    threading.Thread(target=engine_loop, name="engine", daemon=True).start()
    print("model loaded:", MODEL_NAME, "batch engine started", flush=True)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "max_context": MAX_CONTEXT,
        "max_output": MAX_OUTPUT,
        "max_concurrent": MAX_CONCURRENT,
        "in_flight": active,
        "kv_reserved_gb": round(kv_reserved / 1e9, 2),
        "kv_budget_gb": round(KV_BUDGET_BYTES / 1e9, 1),
        "max_batch": MAX_BATCH,
        "uptime_s": round(time.time() - STARTED, 1),
    }


@app.get("/v1/models")
def list_models(request: Request):
    denied = check_auth(request)
    if denied:
        return denied
    created = int(STARTED)
    base = {
        "object": "model",
        "created": created,
        "owned_by": "empero-ai",
        "context_length": MAX_CONTEXT,
        "max_output_tokens": MAX_OUTPUT,
        "max_completion_tokens": MAX_OUTPUT,
        "per_request_context_limit": MAX_CONTEXT,
        "capabilities": {
            "chat_completions": True,
            "responses": True,
            "completions": True,
            "streaming": True,
            "tool_calls": False,
        },
    }
    return {
        "object": "list",
        "data": [
            dict(base, id=MODEL_NAME, root=MODEL_ALIAS),
            dict(base, id=MODEL_ALIAS, root=MODEL_ALIAS),
        ],
    }


@app.get("/v1/models/{model_id:path}")
def get_model(model_id: str, request: Request):
    denied = check_auth(request)
    if denied:
        return denied
    listing = list_models(request).get("data", [])
    for entry in listing:
        if entry["id"] == model_id:
            return entry
    return err(404, "Model '%s' not found." % model_id, code="model_not_found")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    denied = check_auth(request)
    if denied:
        return denied
    try:
        body = await request.json()
    except Exception:
        return err(400, "Body must be valid JSON.")
    if not isinstance(body, dict):
        return err(400, "Body must be a JSON object.")

    msgs, problem = messages_from_chat(body)
    if problem:
        return err(400, problem)
    if body.get("n") not in (None, 1):
        return err(400, "Only n=1 is supported.")

    try:
        prompt_ids = build_prompt(msgs)
    except Exception as exc:
        return err(400, "Could not apply the chat template: %s" % exc)

    prompt_len = int(prompt_ids.shape[1])
    budget, problem = budget_for(body, prompt_len, ("max_completion_tokens", "max_tokens"))
    if problem:
        return err(400, problem, code="context_length_exceeded")

    reserve = (prompt_len + budget) * KV_BYTES_PER_TOKEN
    reason = acquire(reserve)
    if reason:
        return overloaded(reason)

    cid = "chatcmpl-" + uuid.uuid4().hex[:24]
    created = int(time.time())
    streaming = bool(body.get("stream"))
    include_usage = bool((body.get("stream_options") or {}).get("include_usage"))

    if not streaming:
        try:
            text, completion = await job_collect(submit(prompt_ids, budget, body, asyncio.get_running_loop()))
        except Exception as exc:
            release(reserve)
            return err(500, "Generation failed: %s" % exc, etype="server_error")
        release(reserve)
        stats["prompt_tokens"] += prompt_len
        stats["completion_tokens"] += completion
        return {
            "id": cid,
            "object": "chat.completion",
            "created": created,
            "model": body.get("model") or MODEL_NAME,
            "system_fingerprint": None,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_len,
                "completion_tokens": completion,
                "total_tokens": prompt_len + completion,
            },
        }

    job = submit(prompt_ids, budget, body, asyncio.get_running_loop())

    async def events():
        try:
            head = {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": body.get("model") or MODEL_NAME,
                "system_fingerprint": None,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "logprobs": None, "finish_reason": None}],
            }
            yield "data: " + json.dumps(head) + "\n\n"
            async for payload in job_stream(job):
                chunk = {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": body.get("model") or MODEL_NAME,
                    "system_fingerprint": None,
                    "choices": [{"index": 0, "delta": {"content": payload}, "logprobs": None, "finish_reason": None}],
                }
                yield "data: " + json.dumps(chunk) + "\n\n"
            tail = {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": body.get("model") or MODEL_NAME,
                "system_fingerprint": None,
                "choices": [{"index": 0, "delta": {}, "logprobs": None, "finish_reason": "stop"}],
            }
            yield "data: " + json.dumps(tail) + "\n\n"
            if include_usage:
                stats["prompt_tokens"] += prompt_len
                stats["completion_tokens"] += job.count
                yield "data: " + json.dumps({
                    "id": cid, "object": "chat.completion.chunk", "created": created,
                    "model": body.get("model") or MODEL_NAME, "choices": [],
                    "usage": {"prompt_tokens": prompt_len, "completion_tokens": job.count,
                              "total_tokens": prompt_len + job.count},
                }) + "\n\n"
            yield "data: [DONE]\n\n"
        finally:
            release(reserve)

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/v1/responses")
async def responses(request: Request):
    denied = check_auth(request)
    if denied:
        return denied
    try:
        body = await request.json()
    except Exception:
        return err(400, "Body must be valid JSON.")
    if not isinstance(body, dict):
        return err(400, "Body must be a JSON object.")

    msgs, problem = messages_from_responses(body)
    if problem:
        return err(400, problem)
    try:
        prompt_ids = build_prompt(msgs)
    except Exception as exc:
        return err(400, "Could not apply the chat template: %s" % exc)

    prompt_len = int(prompt_ids.shape[1])
    budget, problem = budget_for(body, prompt_len, ("max_output_tokens", "max_completion_tokens", "max_tokens"))
    if problem:
        return err(400, problem, code="context_length_exceeded")

    reserve = (prompt_len + budget) * KV_BYTES_PER_TOKEN
    reason = acquire(reserve)
    if reason:
        return overloaded(reason)

    rid = "resp_" + uuid.uuid4().hex[:24]
    mid = "msg_" + uuid.uuid4().hex[:24]
    created = int(time.time())
    model_name = body.get("model") or MODEL_NAME
    streaming = bool(body.get("stream"))

    def envelope(status, output, text, out_tokens, incomplete=None):
        return {
            "id": rid,
            "object": "response",
            "created_at": created,
            "status": status,
            "error": None,
            "incomplete_details": incomplete,
            "instructions": body.get("instructions"),
            "max_output_tokens": budget,
            "model": model_name,
            "output": output,
            "parallel_tool_calls": False,
            "previous_response_id": None,
            "store": False,
            "temperature": body.get("temperature", 1.0),
            "text": {"format": {"type": "text"}},
            "tool_choice": "auto",
            "tools": [],
            "top_p": body.get("top_p", 0.95),
            "truncation": "disabled",
            "usage": {
                "input_tokens": prompt_len,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": out_tokens,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": prompt_len + out_tokens,
            },
            "output_text": text,
        }

    def message_item(text, status="completed"):
        return {
            "id": mid,
            "type": "message",
            "status": status,
            "role": "assistant",
            "content": [{"type": "output_text", "text": text, "annotations": [], "logprobs": []}],
        }

    if not streaming:
        try:
            text, completion = await job_collect(submit(prompt_ids, budget, body, asyncio.get_running_loop()))
        except Exception as exc:
            release(reserve)
            return err(500, "Generation failed: %s" % exc, etype="server_error")
        release(reserve)
        stats["prompt_tokens"] += prompt_len
        stats["completion_tokens"] += completion
        return envelope("completed", [message_item(text)], text, completion)

    job = submit(prompt_ids, budget, body, asyncio.get_running_loop())
    seq = [0]

    def event(name, payload):
        seq[0] += 1
        return "event: %s\ndata: %s\n\n" % (name, json.dumps(dict(payload, sequence_number=seq[0])))

    async def events():
        acc = []
        try:
            yield event("response.created", {"type": "response.created", "response": envelope("in_progress", [], "", 0)})
            yield event("response.in_progress", {"type": "response.in_progress", "response": envelope("in_progress", [], "", 0)})
            yield event("response.output_item.added", {
                "type": "response.output_item.added", "output_index": 0,
                "item": {"id": mid, "type": "message", "status": "in_progress", "role": "assistant", "content": []},
            })
            yield event("response.content_part.added", {
                "type": "response.content_part.added", "item_id": mid, "output_index": 0, "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": [], "logprobs": []},
            })
            async for payload in job_stream(job):
                acc.append(payload)
                yield event("response.output_text.delta", {
                    "type": "response.output_text.delta", "item_id": mid, "output_index": 0,
                    "content_index": 0, "delta": payload, "logprobs": [],
                })
            text = "".join(acc).strip()
            yield event("response.output_text.done", {
                "type": "response.output_text.done", "item_id": mid, "output_index": 0,
                "content_index": 0, "text": text, "logprobs": [],
            })
            yield event("response.content_part.done", {
                "type": "response.content_part.done", "item_id": mid, "output_index": 0, "content_index": 0,
                "part": {"type": "output_text", "text": text, "annotations": [], "logprobs": []},
            })
            yield event("response.output_item.done", {
                "type": "response.output_item.done", "output_index": 0, "item": message_item(text),
            })
            stats["prompt_tokens"] += prompt_len
            stats["completion_tokens"] += job.count
            yield event("response.completed", {
                "type": "response.completed",
                "response": envelope("completed", [message_item(text)], text, job.count),
            })
        finally:
            release(reserve)

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/v1/completions")
async def completions(request: Request):
    denied = check_auth(request)
    if denied:
        return denied
    try:
        body = await request.json()
    except Exception:
        return err(400, "Body must be valid JSON.")
    prompt = body.get("prompt")
    if isinstance(prompt, list):
        prompt = prompt[0] if prompt else ""
    if not isinstance(prompt, str):
        return err(400, "Field 'prompt' must be a string.")
    msgs = [{"role": "user", "content": prompt}]
    prompt_ids = build_prompt(msgs)
    prompt_len = int(prompt_ids.shape[1])
    budget, problem = budget_for(body, prompt_len, ("max_tokens",))
    if problem:
        return err(400, problem, code="context_length_exceeded")
    reserve = (prompt_len + budget) * KV_BYTES_PER_TOKEN
    reason = acquire(reserve)
    if reason:
        return overloaded(reason)
    try:
        text, completion = await job_collect(submit(prompt_ids, budget, body, asyncio.get_running_loop()))
    except Exception as exc:
        release(reserve)
        return err(500, "Generation failed: %s" % exc, etype="server_error")
    release(reserve)
    stats["prompt_tokens"] += prompt_len
    stats["completion_tokens"] += completion
    return {
        "id": "cmpl-" + uuid.uuid4().hex[:24],
        "object": "text_completion",
        "created": int(time.time()),
        "model": body.get("model") or MODEL_NAME,
        "choices": [{"index": 0, "text": text, "logprobs": None, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt_len, "completion_tokens": completion,
                  "total_tokens": prompt_len + completion},
    }


@app.get("/", response_class=HTMLResponse)
def index():
    up = int(time.time() - STARTED)
    hours, rem = divmod(up, 3600)
    mins, secs = divmod(rem, 60)
    ready = model is not None
    seal = "#C8321E" if ready else "#8A8680"
    ticks = "".join(
        '<span style="display:inline-block;width:3px;height:%dpx;margin-right:5px;background:%s"></span>'
        % (14 + (i % 3) * 4, "#C8321E" if i < active else "#D8D3C8")
        for i in range(MAX_CONCURRENT)
    )
    return """<!doctype html><html><head><meta charset="utf-8"><title>%(name)s</title>
<style>
body{background:#F4F1EA;color:#111;font-family:"Zen Old Mincho",Georgia,serif;margin:0;
display:flex;min-height:100vh;align-items:center;justify-content:center}
main{width:34rem;padding:3rem}
h1{font-size:1.6rem;font-weight:400;letter-spacing:.02em;margin:0 0 .2rem}
.sub{color:#8A8680;font-size:.85rem;margin-bottom:2rem}
dl{display:grid;grid-template-columns:11rem 1fr;gap:.5rem 1rem;font-size:.9rem;margin:0}
dt{color:#8A8680}
dd{margin:0;font-family:ui-monospace,monospace}
.seal{width:52px;height:52px;border-radius:6px;background:%(seal)s;margin-bottom:1.5rem}
</style></head><body><main>
<div class="seal"></div>
<h1>%(name)s</h1>
<div class="sub">%(alias)s &middot; OpenAI-compatible endpoint</div>
<dl>
<dt>state</dt><dd>%(state)s</dd>
<dt>slots in use</dt><dd>%(ticks)s <span style="font-family:inherit;color:#8A8680">%(active)d / %(maxc)d</span></dd>
<dt>context limit</dt><dd>%(ctx)d tokens</dd>
<dt>max output</dt><dd>%(out)d tokens</dd>
<dt>uptime</dt><dd>%(h)02d:%(m)02d:%(s)02d</dd>
<dt>endpoints</dt><dd>/v1/chat/completions<br>/v1/responses<br>/v1/models</dd>
<dt>auth</dt><dd>Authorization: Bearer &lt;token&gt;</dd>
</dl>
</main></body></html>""" % {
        "name": MODEL_NAME, "alias": MODEL_ALIAS, "seal": seal,
        "state": "ready" if ready else "loading",
        "ticks": ticks, "active": active, "maxc": MAX_CONCURRENT,
        "ctx": MAX_CONTEXT, "out": MAX_OUTPUT, "h": hours, "m": mins, "s": secs,
    }
