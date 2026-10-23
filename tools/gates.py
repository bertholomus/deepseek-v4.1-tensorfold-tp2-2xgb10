"""Acceptance gates against a running server (standard library only; the key is read from a file, never printed).

  python3 gates.py --base http://127.0.0.1:8000 --model DeepSeek-V4.1-Flash-TF --key-file <API_KEY_FILE> --out gates.json

Gates (each one recorded with ok true/false; the exit code is 1 if any gate fails):
  models             the served model id is listed by /v1/models
  chat_probe         greedy chat, thinking off: "17*19=" answers 323
  reasoning_stream   streamed chat with thinking on: reasoning_content and content both arrive
  tool_call          a tool call (plain and streamed, no tool markup leaking into content) and a reply that uses the
                     tool's result
  drafted_equals_serial
                     greedy (T=0) replies with speculative drafts equal the same request with "draft": false, token
                     for token (TensorFold's exactness check; return_token_ids). With --sampled also at T=0.6, seed
                     1234, top-k 20, top-p 0.95 (keyed sampling, the settings of results/spec_t06.json; measured 8/8
                     on the engine directly, this HTTP form is optional)
  burst4             four requests sent at once all complete (the engine serves one request at a time, so they queue;
                     the aggregate tok/s is reported, not judged)

The drafted == serial prompts are the 8 chat prompts of kit_bench.py's oracle set (tokenized by the server), or the
chat records of an oracle file written by `kit_bench.py oracle` (--oracle). The published results/release-20261003/
gates.json came from an earlier, less general form of this script that took the prompt ids from the kit's oracle file
(the same as --oracle results/kit-20261003T081806Z/oracle.jsonl) and had no --sampled mode.
"""

import argparse
import json
import sys
import threading
import time
import traceback
import urllib.request

# the first 8 chat prompts of kit_bench.py ORACLE_SET_V1 (keep in step with it)
CHAT_PROMPTS = [
    "What is the capital of Australia, and why was it chosen over Sydney and Melbourne?",
    "Explain the difference between a mutex and a semaphore with a short C example.",
    "Write a haiku about autumn rain on a tin roof.",
    "Solve step by step: a train leaves at 9:40 and travels 210 km at 84 km/h. When does it arrive?",
    "Translate into French and German: 'The library opens at eight, but the archive needs an appointment.'",
    "List five practical tips for reducing memory fragmentation in a long-running C++ server.",
    "Give me a JSON object describing a book with title, author, year, isbn and a list of three themes.",
    "Summarize the plot of Romeo and Juliet in exactly three sentences.",
]

TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"},
                                                    "unit": {"type": "string", "enum": ["c", "f"]}},
                   "required": ["city"]}}}]

NO_THINK = {"enable_thinking": False}


def load_key(path):
    if not path:
        return None
    for line in open(path).read().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line.split("=", 1)[1].strip().strip("'\"") if "=" in line else line
    return None


class Client:
    def __init__(self, base, model, key, timeout):
        self.base, self.model, self.key, self.timeout = base.rstrip("/"), model, key, timeout

    def req(self, path, body=None, stream=False):
        h = {"Content-Type": "application/json"}
        if self.key:
            h["Authorization"] = "Bearer " + self.key
        r = urllib.request.Request(self.base + path, data=json.dumps(body).encode() if body is not None else None,
                                   headers=h)
        resp = urllib.request.urlopen(r, timeout=self.timeout)
        if stream:
            return resp
        with resp:
            return json.loads(resp.read())

    def chat(self, body):
        return self.req("/v1/chat/completions", dict(body, model=self.model))

    def stream_chat(self, body):
        body = dict(body, stream=True, model=self.model)
        content, reasoning, calls, finish = "", "", {}, None
        with self.req("/v1/chat/completions", body, stream=True) as resp:
            for raw in resp:
                line = raw.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                ch = json.loads(line[5:])
                for c in ch.get("choices", []):
                    d = c.get("delta") or {}
                    content += d.get("content") or ""
                    reasoning += d.get("reasoning_content") or ""
                    for tc in d.get("tool_calls") or []:
                        e = calls.setdefault(tc.get("index", 0), {"name": "", "arguments": "", "id": None})
                        e["id"] = tc.get("id") or e["id"]
                        fn = tc.get("function") or {}
                        e["name"] += fn.get("name") or ""
                        e["arguments"] += fn.get("arguments") or ""
                    finish = c.get("finish_reason") or finish
        return {"content": content, "reasoning": reasoning, "calls": list(calls.values()), "finish": finish}


def gate_models(c, a):
    ids = [m["id"] for m in c.req("/v1/models")["data"]]
    return {"ok": c.model in ids, "ids": ids}


def gate_chat_probe(c, a):
    out = c.chat({"messages": [{"role": "user", "content": "17*19="}], "max_tokens": 32, "temperature": 0,
                  "chat_template_kwargs": NO_THINK})
    txt = out["choices"][0]["message"]["content"] or ""
    return {"ok": "323" in txt, "reply": txt[:80]}


def gate_reasoning_stream(c, a):
    s = c.stream_chat({"messages": [{"role": "user", "content": "Is 391 prime? Answer yes or no."}],
                       "max_tokens": 1024, "temperature": 0, "chat_template_kwargs": {"enable_thinking": True}})
    return {"ok": len(s["reasoning"]) > 0 and len(s["content"]) > 0, "reasoning_chars": len(s["reasoning"]),
            "content": s["content"][:120], "finish": s["finish"]}


def gate_tool_call(c, a):
    msgs = [{"role": "user", "content": "What's the weather in Paris in celsius? Use the tool."}]
    base = {"tools": TOOLS, "max_tokens": 256, "temperature": 0, "chat_template_kwargs": NO_THINK}
    t1 = c.chat(dict(base, messages=msgs))
    m1 = t1["choices"][0]["message"]
    calls = m1.get("tool_calls") or []
    res = {"finish": t1["choices"][0]["finish_reason"], "calls": calls, "content": (m1.get("content") or "")[:120]}
    ok_call = False
    if calls and calls[0]["function"]["name"] == "get_weather":
        try:
            ok_call = json.loads(calls[0]["function"]["arguments"]).get("city", "").lower().startswith("paris")
        except (ValueError, AttributeError):
            ok_call = False
    s1 = c.stream_chat(dict(base, messages=msgs))
    res["stream"] = {"finish": s1["finish"], "calls": s1["calls"], "content_leak": "DSML" in s1["content"]}
    uses_result = False
    if calls:
        msgs2 = msgs + [{"role": "assistant", "content": m1.get("content") or "", "tool_calls": calls},
                        {"role": "tool", "tool_call_id": calls[0]["id"],
                         "content": json.dumps({"city": "Paris", "temperature_c": 18, "sky": "light rain"})}]
        t2 = c.chat(dict(base, messages=msgs2))
        m2 = t2["choices"][0]["message"]
        uses_result = "18" in (m2.get("content") or "")
        res["continuation"] = {"finish": t2["choices"][0]["finish_reason"], "content": (m2.get("content") or "")[:200],
                               "uses_result": uses_result}
    res["ok"] = ok_call and uses_result and bool(s1["calls"]) and not res["stream"]["content_leak"]
    return res


def prompt_ids(c, a):
    if a.oracle:
        recs = [json.loads(line) for line in open(a.oracle) if line.strip()]
        return [r["prompt_ids"] for r in recs if r.get("kind") == "chat"][:8]
    return [c.req("/tokenize", {"model": c.model, "messages": [{"role": "user", "content": p}],
                                "add_generation_prompt": True, "chat_template_kwargs": NO_THINK})["tokens"]
            for p in CHAT_PROMPTS]


def gate_drafted_equals_serial(c, a):
    prompts = prompt_ids(c, a)
    res = {}
    modes = [("t0", {"temperature": 0})]
    if a.sampled:
        modes.append(("t06", {"temperature": 0.6, "top_k": 20, "top_p": 0.95, "seed": 1234}))
    for label, sampling in modes:
        rows = []
        for i, ids in enumerate(prompts):
            body = dict(sampling, model=c.model, prompt=ids, max_tokens=a.tokens, return_token_ids=True,
                        ignore_eos=True)
            d = c.req("/v1/completions", body)
            s = c.req("/v1/completions", dict(body, draft=False))
            got_d = (d.get("tensorfold") or {}).get("token_ids")
            got_s = (s.get("tensorfold") or {}).get("token_ids")
            if got_d is None or got_s is None:          # a server without token ids: compare the text
                got_d, got_s = d["choices"][0].get("text"), s["choices"][0].get("text")
            rows.append({"prompt": i, "equal": got_d == got_s})
        res[label] = {"equal": sum(r["equal"] for r in rows), "of": len(rows), "rows": rows}
    res["ok"] = all(res[k]["equal"] == res[k]["of"] > 0 for k, _ in modes)
    return res


def gate_burst4(c, a):
    prompts = ["Write a Python function that merges two sorted lists, with tests.",
               "Explain how a bloom filter works, with an example.",
               "List 20 countries and their capitals as JSON.",
               "Write a short story about a lighthouse keeper."]
    results, errors = [None] * 4, [None] * 4

    def work(i):
        t0 = time.time()
        try:
            o = c.chat({"messages": [{"role": "user", "content": prompts[i]}], "max_tokens": 256, "temperature": 0,
                        "chat_template_kwargs": NO_THINK})
            results[i] = {"tokens": o["usage"]["completion_tokens"], "seconds": round(time.time() - t0, 2)}
        except Exception as e:  # noqa: BLE001 - recorded, judged below
            errors[i] = repr(e)[:200]

    t0 = time.time()
    threads = [threading.Thread(target=work, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0
    done = [r for r in results if r]
    return {"ok": len(done) == 4, "wall_s": round(wall, 1),
            "aggregate_tps": round(sum(r["tokens"] for r in done) / wall, 2), "streams": results,
            "errors": [e for e in errors if e]}


GATES = {"models": gate_models, "chat_probe": gate_chat_probe, "reasoning_stream": gate_reasoning_stream,
         "tool_call": gate_tool_call, "drafted_equals_serial": gate_drafted_equals_serial, "burst4": gate_burst4}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="DeepSeek-V4.1-Flash-TF", help="the served model id (SERVED_NAME)")
    p.add_argument("--key-file", help="API key file (a bare key or KEY=value lines); omit for a server without a key")
    p.add_argument("--oracle", help="optional oracle.jsonl from `kit_bench.py oracle`: its chat prompt ids are used")
    p.add_argument("--tokens", type=int, default=128, help="tokens per reply in the drafted == serial gate")
    p.add_argument("--sampled", action="store_true", help="drafted == serial also at T=0.6 (seed 1234, top-k 20, top-p 0.95)")
    p.add_argument("--skip", default="", help="comma-separated gate names to skip")
    p.add_argument("--timeout", type=float, default=900, help="seconds per request")
    p.add_argument("--out")
    a = p.parse_args()
    c = Client(a.base, a.model, load_key(a.key_file), a.timeout)
    skip = {s.strip() for s in a.skip.split(",") if s.strip()}
    unknown = skip - set(GATES)
    if unknown:
        p.error(f"unknown gate(s): {', '.join(sorted(unknown))}")
    res = {"base": a.base, "model": a.model, "unix": time.time(), "gates": {}}
    for name, fn in GATES.items():
        if name in skip:
            continue
        t0 = time.time()
        try:
            r = fn(c, a)
        except Exception as e:  # noqa: BLE001 - a failing gate is a result, not a crash
            r = {"ok": False, "error": repr(e)[:300], "trace": traceback.format_exc(limit=3)[-600:]}
        r["seconds"] = round(time.time() - t0, 1)
        res["gates"][name] = r
        print(json.dumps({"gate": name, "ok": r["ok"], "seconds": r["seconds"]}), flush=True)
    failed = [k for k, v in res["gates"].items() if not v["ok"]]
    res["ok"], res["failed"] = not failed, failed
    print(json.dumps(res, indent=1, ensure_ascii=False))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1, ensure_ascii=False)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
