"""Black-box measurements of an OpenAI-compatible DeepSeek-V4.1 server.

The same client measured both sides of the published comparison (this engine and the MiaAI-Lab vLLM kit,
commit 6f7d1590ad49), on the same two nodes. It needs only the server's HTTP API: /v1/models, /v1/completions,
/v1/chat/completions and vLLM's /tokenize.

Standard library only, so it runs on a bare node. The API key is read from a file (KEY=value lines or
a bare key) and never printed.

  python3 kit_bench.py --base http://127.0.0.1:8000 --model M --key-file <API_KEY_FILE> ready --timeout 3600
  python3 kit_bench.py ... decode     --out decode.json
  python3 kit_bench.py ... concurrent --streams 2,4 --out conc.json
  python3 kit_bench.py ... sustained --streams 4 --seconds 90 --out sustained.json   (N in flight for the window)
  python3 kit_bench.py ... prefill    --lengths 8192,32768,131072 --out prefill.json
  python3 kit_bench.py ... depth      --lengths 131072 --tokens 256 --out depth.json
  python3 kit_bench.py ... oracle     --out oracle.jsonl

Decode tok/s = (completion tokens - 1) / (last chunk time - first chunk time), first token excluded.
Prefill tok/s = prompt tokens / time to first token, with a fresh random prefix so no prefix cache hits
(--fixed-words: the same words every rep behind the fresh prefix, so reps after the first read warm Engram rows).
The oracle records, for a fixed prompt set, the greedy continuation (64 tokens, top-20 logprobs per
step) and the teacher-forced top-20 logprobs at every prompt+continuation position.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import threading
import time
import urllib.error
import urllib.request

DECODE_PROMPTS = [
    ("code", "Write a complete Python implementation of a thread-safe LRU cache with per-entry TTL expiry. "
             "Include type hints, docstrings, and a set of unittest test cases covering eviction and expiry."),
    ("prose", "Write a long, vivid essay about the history of lighthouses: how they were built, how their lamps "
              "and lenses evolved, and the lives of the keepers who tended them through storms."),
    ("structured", "Return a JSON array of 15 fictional employees. Each object has the fields id (int), name, "
                   "department, title, salary (int), start_date (YYYY-MM-DD), manager_id (int or null) and "
                   "skills (array of strings). Output only the JSON, pretty-printed with two-space indents."),
]

# A second decode set (--set b): a short coding task with tests, a 400-word essay and a counting task, the prompt classes
# of other published 2x GB10 numbers for this model (our own wording); measured with --tokens 384.
DECODE_PROMPTS_B = [
    ("code", "Write a Python class LRUCache with get(key) and put(key, value), both O(1), using a dict and a doubly "
             "linked list. Then write unittest test cases for it."),
    ("prose", "Write a 400-word essay about the invention of the printing press and how it changed Europe."),
    ("structured", "Count from 1 to 200. Write each number on its own line and nothing else."),
]
PROMPT_SETS = {"a": DECODE_PROMPTS, "b": DECODE_PROMPTS_B}

# Fixed oracle set: chat-templated prompts (thinking off) and raw-text prompts. Never change these in
# place; add a new set name instead, so earlier oracle files stay comparable.
ORACLE_SET_V1 = [
    ("chat", "What is the capital of Australia, and why was it chosen over Sydney and Melbourne?"),
    ("chat", "Explain the difference between a mutex and a semaphore with a short C example."),
    ("chat", "Write a haiku about autumn rain on a tin roof."),
    ("chat", "Solve step by step: a train leaves at 9:40 and travels 210 km at 84 km/h. When does it arrive?"),
    ("chat", "Translate into French and German: 'The library opens at eight, but the archive needs an appointment.'"),
    ("chat", "List five practical tips for reducing memory fragmentation in a long-running C++ server."),
    ("chat", "Give me a JSON object describing a book with title, author, year, isbn and a list of three themes."),
    ("chat", "Summarize the plot of Romeo and Juliet in exactly three sentences."),
    ("chat", "用中文简要解释什么是量子纠缠，并举一个日常类比。"),
    ("chat", "Write a SQL query that returns the top 3 customers by total order value in 2025, with ties broken by name."),
    ("chat", "What are the main causes of the French Revolution? Answer in a numbered list."),
    ("chat", "Refactor this into idiomatic Rust: for (int i = 0; i < n; i++) { if (a[i] % 2 == 0) sum += a[i]; }"),
    ("raw", "def quicksort(arr):\n    \"\"\"Sort a list using the quicksort algorithm.\"\"\"\n"),
    ("raw", "The mitochondrion is a double-membrane-bound organelle found in most eukaryotic organisms. It"),
    ("raw", "In 1969, the Apollo 11 mission"),
    ("raw", "#include <stdio.h>\n\nint main(void) {\n    int primes[10];\n"),
    ("raw", "Q: If x + 2y = 11 and 3x - y = 5, what are x and y?\nA:"),
    ("raw", "| Country | Capital | Population (millions) |\n|---|---|---|\n| Japan | Tokyo | 125 |\n|"),
    ("raw", "Once upon a time, in a village at the edge of a great forest, there lived an old clockmaker who"),
    ("raw", "SELECT name, COUNT(*) AS n\nFROM orders o\nJOIN customers c ON c.id = o.customer_id\n"),
    ("raw", "Die Relativitätstheorie von Albert Einstein besteht aus zwei Teilen:"),
    ("raw", "{\"name\": \"Ada Lovelace\", \"born\": 1815, \"known_for\": ["),
    ("raw", "The derivative of sin(x) * e^x with respect to x is"),
    ("raw", "Dear hiring committee,\n\nI am writing to apply for the position of"),
]

WORDS = ("time year people way day man thing woman life child world school state family student group country "
         "problem hand part place case week company system program question work government number night point "
         "home water room mother area money story fact month lot right study book eye job word business issue side "
         "kind head house service friend father power hour game line end member law car city community name "
         "president team minute idea kid body information back parent face others level office door health person "
         "art war history party result change morning reason research girl guy moment air teacher force education "
         "river mountain signal engine theory market garden window letter music answer bridge island harbor").split()


def load_key(path: str | None) -> str | None:
    if not path:
        return None
    for line in open(path).read().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        return line.split("=", 1)[1].strip().strip("'\"") if "=" in line else line
    return None


class Client:
    def __init__(self, base: str, model: str, key: str | None):
        self.base, self.model, self.key = base.rstrip("/"), model, key

    def _req(self, path: str, body: dict | None = None, timeout: float = 3600):
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        data = json.dumps(body).encode() if body is not None else None
        return urllib.request.urlopen(urllib.request.Request(self.base + path, data=data, headers=headers),
                                      timeout=timeout)

    def post(self, path: str, body: dict, timeout: float = 3600) -> dict:
        with self._req(path, body, timeout) as r:
            return json.loads(r.read())

    def get(self, path: str, timeout: float = 10) -> dict:
        with self._req(path, None, timeout) as r:
            return json.loads(r.read())

    def chat_ids(self, text: str) -> list[int]:
        out = self.post("/tokenize", {"model": self.model, "messages": [{"role": "user", "content": text}],
                                      "add_generation_prompt": True,
                                      "chat_template_kwargs": {"enable_thinking": False}})
        return out["tokens"]

    def text_ids(self, text: str, special: bool = True) -> list[int]:
        return self.post("/tokenize", {"model": self.model, "prompt": text, "add_special_tokens": special})["tokens"]

    def stream(self, body: dict, path: str) -> dict:
        body = dict(body, model=self.model, stream=True, stream_options={"include_usage": True})
        start = time.perf_counter()
        first = last = None
        usage, pieces, chunk_times = None, [], []
        with self._req(path, body) as resp:
            for raw in resp:
                line = raw.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[5:])
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices", []):
                    piece = choice.get("text") or (choice.get("delta") or {}).get("content") or ""
                    if piece:
                        now = time.perf_counter()
                        first = first if first is not None else now
                        last = now
                        chunk_times.append(now - start)
                        pieces.append(piece)
        end = time.perf_counter()
        n = int(usage["completion_tokens"]) if usage else None
        return {"start": start, "end": end, "ttft_s": (first - start) if first else None,
                "decode_s": (last - first) if first else None, "tokens": n,
                "prompt_tokens": int(usage["prompt_tokens"]) if usage else None,
                "decode_tps": (n - 1) / (last - first) if n and first and last > first else None,
                "chunks": len(chunk_times), "text": "".join(pieces)}


def decode_body(prompt: str, max_tokens: int) -> dict:
    return {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}


def cmd_ready(c: Client, a) -> dict:
    t0 = time.time()
    while time.time() - t0 < a.timeout:
        try:
            c.get("/v1/models")
            out = c.post("/v1/completions", {"model": c.model, "prompt": "Hello", "max_tokens": 1,
                                             "temperature": 0}, timeout=600)
            if out.get("choices"):
                return {"ready_after_s": time.time() - t0, "probe_started_unix": t0}
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError, json.JSONDecodeError):
            pass
        time.sleep(5)
    raise SystemExit("server not ready within timeout")


def cmd_decode(c: Client, a) -> list:
    rows = []
    for name, prompt in PROMPT_SETS[a.set]:
        c.stream(decode_body(prompt, 32), "/v1/chat/completions")  # warm-up
        runs = [c.stream(decode_body(prompt, a.tokens), "/v1/chat/completions") for _ in range(a.reps)]
        tps = [r["decode_tps"] for r in runs if r["decode_tps"]]
        row = {"set": a.set, "prompt": name, "max_tokens": a.tokens, "reps": a.reps,
               "decode_tps_median": statistics.median(tps), "decode_tps_all": [round(x, 3) for x in tps],
               "tokens_all": [r["tokens"] for r in runs], "chunks_all": [r["chunks"] for r in runs],
               "ttft_s_median": statistics.median(r["ttft_s"] for r in runs),
               "identical_texts": len({r["text"] for r in runs}) == 1, "sample": runs[0]["text"][:400]}
        print(json.dumps({k: row[k] for k in ("prompt", "decode_tps_median", "decode_tps_all", "tokens_all",
                                               "identical_texts")}), flush=True)
        rows.append(row)
    return rows


def cmd_concurrent(c: Client, a) -> list:
    rows = []
    for n in [int(x) for x in a.streams.split(",")]:
        for rep in range(a.reps):
            results: list = [None] * n
            barrier = threading.Barrier(n)

            def worker(i: int) -> None:
                prompts = PROMPT_SETS[a.set]
                name, prompt = prompts[i % len(prompts)]
                barrier.wait()
                results[i] = c.stream(decode_body(prompt + f" (variant {i})", a.tokens), "/v1/chat/completions")

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
            [t.start() for t in threads]
            [t.join() for t in threads]
            t_start = min(r["start"] for r in results)
            t_end = max(r["end"] for r in results)
            total = sum(r["tokens"] for r in results)
            row = {"streams": n, "rep": rep, "max_tokens": a.tokens, "total_tokens": total,
                   "wall_s": t_end - t_start, "aggregate_tps": total / (t_end - t_start),
                   "per_stream_decode_tps": [round(r["decode_tps"], 3) for r in results],
                   "per_stream_tokens": [r["tokens"] for r in results],
                   "ttft_s": [round(r["ttft_s"], 3) for r in results]}
            print(json.dumps({k: row[k] for k in ("streams", "rep", "aggregate_tps", "per_stream_decode_tps")}),
                  flush=True)
            rows.append(row)
    return rows


def cmd_sustained(c: Client, a) -> list:
    """Sustained load: exactly N requests in flight for a fixed window (a new request starts as each one ends, every
    prompt distinct). Aggregate tok/s over the window counts each reply's tokens spread evenly over its decode span and
    clipped to the window; per-request decode rate and TTFT as p50 / p95."""

    rows = []
    for n in [int(x) for x in a.streams.split(",")]:
        prompts = PROMPT_SETS[a.set]
        lock = threading.Lock()
        done: list = []
        counter = [0]
        t_start = time.perf_counter() + 0.5                       # every worker's first request starts here
        t_end = t_start + a.seconds

        def worker() -> None:
            while True:
                with lock:
                    k = counter[0]
                    counter[0] += 1
                now = time.perf_counter()
                if now < t_start:
                    time.sleep(t_start - now)
                elif now >= t_end:
                    return
                name, prompt = prompts[k % len(prompts)]
                r = c.stream(decode_body(prompt + f" (request {k})", a.tokens), "/v1/chat/completions")
                r["prompt"] = name
                with lock:
                    done.append(r)

        threads = [threading.Thread(target=worker) for _ in range(n)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        win = 0.0
        for r in done:
            first, last, tokens = r["start"] + (r["ttft_s"] or 0.0), r["start"] + (r["ttft_s"] or 0.0) + (r["decode_s"] or 0.0), r["tokens"] or 0
            if last <= first:
                win += tokens if t_start <= first <= t_end else 0
            else:
                win += tokens * max(0.0, min(last, t_end) - max(first, t_start)) / (last - first)
        rates = sorted(r["decode_tps"] for r in done if r["decode_tps"])
        ttfts = sorted(r["ttft_s"] for r in done if r["ttft_s"] is not None)

        def pct(v, q):
            return round(v[min(len(v) - 1, int(q * (len(v) - 1) + 0.5))], 3) if v else None

        row = {"streams": n, "seconds": a.seconds, "max_tokens": a.tokens, "set": a.set, "requests": len(done),
               "aggregate_tps": round(win / a.seconds, 2), "per_stream_tps_p50": pct(rates, 0.5),
               "per_stream_tps_p95": pct(rates, 0.95), "per_stream_tps_p05": pct(rates, 0.05),
               "ttft_s_p50": pct(ttfts, 0.5), "ttft_s_p95": pct(ttfts, 0.95)}
        print(json.dumps(row), flush=True)
        rows.append(row)
    return rows


def filler_ids(c: Client, length: int, seed: int, words_seed: int | None = None) -> list[int]:
    rng = random.Random(seed)
    salt = " ".join(f"{rng.randrange(10**9)}" for _ in range(8))
    if words_seed is not None:                 # the same words every rep (warm Engram rows), a fresh salt (no prefix hits)
        rng = random.Random(words_seed)
    words = [rng.choice(WORDS) for _ in range(int(length * 0.9) + 64)]
    text = f"Session {salt}. Read the following notes carefully.\n" + " ".join(words)
    ids = c.text_ids(text)
    while len(ids) < length:
        text += " " + " ".join(rng.choice(WORDS) for _ in range(length - len(ids) + 64))
        ids = c.text_ids(text)
    return ids[:length]


def cmd_prefill(c: Client, a) -> list:
    rows = []
    for length in [int(x) for x in a.lengths.split(",")]:
        for rep in range(a.reps):
            ids = filler_ids(c, length, seed=int(time.time() * 1000) ^ (length << 4) ^ rep,
                             words_seed=length if a.fixed_words else None)
            r = c.stream({"prompt": ids, "max_tokens": 1, "temperature": 0}, "/v1/completions")
            row = {"length": length, "rep": rep, "prompt_tokens": r["prompt_tokens"], "ttft_s": r["ttft_s"],
                   "prefill_tps": r["prompt_tokens"] / r["ttft_s"]}
            print(json.dumps(row), flush=True)
            rows.append(row)
    return rows


def cmd_depth(c: Client, a) -> list:
    """Decode speed after a long cold prompt: greedy, ignore_eos, a.tokens tokens."""

    rows = []
    for length in [int(x) for x in a.lengths.split(",")]:
        ids = filler_ids(c, length, seed=int(time.time() * 1000) ^ length)
        r = c.stream({"prompt": ids, "max_tokens": a.tokens, "temperature": 0, "ignore_eos": True}, "/v1/completions")
        row = {"length": length, "prompt_tokens": r["prompt_tokens"], "ttft_s": r["ttft_s"], "tokens": r["tokens"],
               "decode_tps": r["decode_tps"]}
        print(json.dumps(row), flush=True)
        rows.append(row)
    return rows


def cmd_oracle(c: Client, a) -> None:
    with open(a.out, "w") as f:
        for i, (kind, text) in enumerate(ORACLE_SET_V1):
            ids = c.chat_ids(text) if kind == "chat" else c.text_ids(text)
            gen = c.post("/v1/completions", {"model": c.model, "prompt": ids, "max_tokens": a.gen_tokens,
                                              "temperature": 0, "logprobs": a.topk, "return_tokens_as_token_ids": True})
            ch = gen["choices"][0]
            lp = ch.get("logprobs") or {}
            gen_ids = [int(t.split(":", 1)[1]) for t in lp.get("tokens", [])]
            step_top = [{k.split(":", 1)[1]: v for k, v in (d or {}).items()} for d in lp.get("top_logprobs", [])]
            tf = c.post("/v1/completions", {"model": c.model, "prompt": ids + gen_ids, "max_tokens": 1,
                                             "temperature": 0, "prompt_logprobs": a.topk})
            prompt_lp = tf.get("prompt_logprobs") or tf["choices"][0].get("prompt_logprobs")
            pos = []
            for d in prompt_lp or []:
                if d is None:
                    pos.append(None)
                    continue
                pos.append({str(k): (v["logprob"] if isinstance(v, dict) else v) for k, v in d.items()})
            rec = {"set": "v1", "index": i, "kind": kind, "text": text, "prompt_ids": ids, "gen_ids": gen_ids,
                   "gen_text": ch.get("text"), "gen_top": step_top, "finish": ch.get("finish_reason"),
                   "tf_top": pos}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(json.dumps({"index": i, "kind": kind, "prompt_len": len(ids), "gen_len": len(gen_ids),
                              "tf_positions": len(pos), "gen_head": (ch.get("text") or "")[:60]}, ensure_ascii=False),
                  flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--key-file")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("ready"); s.add_argument("--timeout", type=float, default=3600)
    s = sub.add_parser("decode"); s.add_argument("--tokens", type=int, default=512); s.add_argument("--reps", type=int, default=3)
    s.add_argument("--set", choices=sorted(PROMPT_SETS), default="a")
    s = sub.add_parser("concurrent"); s.add_argument("--streams", default="2,4"); s.add_argument("--tokens", type=int, default=256)
    s.add_argument("--set", choices=sorted(PROMPT_SETS), default="a")
    s.add_argument("--reps", type=int, default=2)
    s = sub.add_parser("sustained"); s.add_argument("--streams", default="4"); s.add_argument("--tokens", type=int, default=256)
    s.add_argument("--set", choices=sorted(PROMPT_SETS), default="b"); s.add_argument("--seconds", type=float, default=90)
    s = sub.add_parser("prefill"); s.add_argument("--lengths", default="8192,32768,131072"); s.add_argument("--reps", type=int, default=1)
    s.add_argument("--fixed-words", action="store_true",
                   help="the same words every rep of a length behind a fresh salt: later reps read warm Engram rows")
    s = sub.add_parser("depth"); s.add_argument("--lengths", default="131072"); s.add_argument("--tokens", type=int, default=256)
    s = sub.add_parser("oracle"); s.add_argument("--gen-tokens", type=int, default=64); s.add_argument("--topk", type=int, default=20)
    for s in sub.choices.values():
        s.add_argument("--out")
    a = p.parse_args()
    c = Client(a.base, a.model, load_key(a.key_file))
    if a.cmd == "oracle":
        cmd_oracle(c, a)
        return
    out = {"ready": cmd_ready, "decode": cmd_decode, "concurrent": cmd_concurrent, "sustained": cmd_sustained,
           "prefill": cmd_prefill, "depth": cmd_depth}[a.cmd](c, a)
    print(json.dumps(out)[:2000] if a.cmd == "ready" else "", flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"cmd": a.cmd, "base": a.base, "model": a.model, "unix": time.time(), "result": out}, f, indent=1)


if __name__ == "__main__":
    main()
