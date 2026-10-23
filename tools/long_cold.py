"""long_cold.py (TP2-25, thread "DS lanes 1016", 2026-10-23): long_check.py's cases once more, each one alone and cold,
against a long_check.py run's burst replies (standard library; the key is read from a file, never printed).

long_check.py's solo runs come after its burst, so each resumes its prompt from the kept state the burst left (its
cached_tokens). Here every case runs alone with a fresh cache_salt of its own, so no kept state is reused and each
prompt is prefilled whole with nothing else running; then prompt 0 greedy with drafting off, cold too. The prompts are
rebuilt as long_check.py builds them (the same seed, lengths and words, the lane's /tokenize): each rebuilt passphrase
must be the one the burst's reply named. Every cold reply's token ids must equal its burst reply's.

  python3 long_cold.py --base URL --model M [--key-file F] --burst LONG_JSON [--lengths ...] [--seed 11]
                       [--tokens 48] --out F
"""

import argparse
import json
import random
import time
import urllib.request
import uuid

# long_check.py's words, depths and request helper, copied unchanged (the prompts must be the same)
WORDS = ("time year people way day man thing woman life child world school state family student group country "
         "problem hand part place case week company system program question work government number night point "
         "home water room mother area money story fact month lot right study book eye job word business issue side "
         "kind head house service friend father power hour game line end member law car city community name "
         "president team minute idea kid body information back parent face others level office door health person "
         "art war history party result change morning reason research girl guy moment air teacher force education "
         "river mountain signal engine theory market garden window letter music answer bridge island harbor").split()
CODE_WORDS = "amber cobalt falcon granite harbor juniper kestrel lantern meadow nimbus orchid pepper quartz raven".split()
DEPTHS = [0.1, 0.5, 0.9, 0.3]


def post(base, path, body, key=None, timeout=3600):
    h = {"Content-Type": "application/json"}
    if key:
        h["Authorization"] = "Bearer " + key
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--key-file")
    p.add_argument("--burst", required=True)
    p.add_argument("--lengths", default="160000,160000,160000,160000")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--tokens", type=int, default=48)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    key = None
    if a.key_file:
        line = [x for x in open(a.key_file).read().splitlines() if x.strip() and not x.startswith("#")][0]
        key = line.split("=", 1)[1].strip().strip("'\"") if "=" in line else line.strip()
    burst = {(r["prompt"], r["sampling"]): r for r in json.load(open(a.burst))["replies"] if r["phase"] == "burst"}
    rng = random.Random(a.seed)
    lengths = [int(x) for x in a.lengths.split(",")]

    prompts = []              # long_check.py's loop, unchanged: the same rng calls in the same order
    for i, length in enumerate(lengths):
        phrase = f"{rng.choice(CODE_WORDS)}-{rng.choice(CODE_WORDS)}-{rng.randrange(1000, 9999)}"
        needle = f" The secret passphrase is {phrase}. Remember it. "
        depth = DEPTHS[i % 4]
        words = [rng.choice(WORDS) for _ in range(int(length * 0.75))]

        def build(ws):
            cut = int(len(ws) * depth)
            return "Read the notes below.\n" + " ".join(ws[:cut]) + needle + " ".join(ws[cut:])

        text = build(words)
        n = len(post(a.base, "/tokenize", {"model": a.model, "prompt": text}, key)["tokens"])
        while n < length - 300:
            words += [rng.choice(WORDS) for _ in range(int((length - n) * 0.7))]
            text = build(words)
            n = len(post(a.base, "/tokenize", {"model": a.model, "prompt": text}, key)["tokens"])
        q = text + "\n\nWhat is the secret passphrase mentioned in the notes? Reply with the passphrase only."
        ids = post(a.base, "/tokenize", {"model": a.model, "messages": [{"role": "user", "content": q}],
                                          "add_generation_prompt": True,
                                          "chat_template_kwargs": {"enable_thinking": False}}, key)["tokens"]
        prompts.append({"index": i, "phrase": phrase, "ids": ids})
        if phrase not in burst[(i, "greedy")]["text"]:
            raise SystemExit(f"prompt {i}: the rebuilt passphrase {phrase} is not the burst reply's")

    cases = [(i, "greedy", {"temperature": 0}) for i in range(len(lengths))]
    cases.append((0, "sampled", {"temperature": 0.6, "top_p": 0.95, "seed": 1000}))

    def run_case(case, phase, extra=None):
        idx, name, samp = case
        body = {"model": a.model, "prompt": prompts[idx]["ids"], "max_tokens": a.tokens, "ignore_eos": True,
                "return_token_ids": True, "cache_salt": "cold-" + uuid.uuid4().hex, **samp}
        if extra:
            body.update(extra)
        t0 = time.time()
        out = post(a.base, "/v1/completions", body, key)
        text = out["choices"][0]["text"] or ""
        rec = {"phase": phase, "prompt": idx, "sampling": name,
               "token_ids": (out.get("tensorfold") or {}).get("token_ids"), "text": text,
               "prompt_tokens": out.get("usage", {}).get("prompt_tokens"),
               "cached_tokens": (out.get("usage", {}).get("prompt_tokens_details") or {}).get("cached_tokens") or 0,
               "seconds": round(time.time() - t0, 1), "found": prompts[idx]["phrase"] in text}
        if extra:
            rec.update(extra)
        print(json.dumps({k: v for k, v in rec.items() if k != "token_ids"}), flush=True)
        return rec

    def same(x, y):                       # missing or empty token ids never count as equal
        return bool(x) and x == y

    t0 = time.time()
    cold = [run_case(c, "cold") for c in cases]
    cold_s = round(time.time() - t0, 1)
    t0 = time.time()
    serial = run_case(cases[0], "cold-serial", extra={"draft": False})
    serial_s = round(time.time() - t0, 1)
    summ = {"cold_equal_to_burst": sum(same(r["token_ids"], burst[(r["prompt"], r["sampling"])]["token_ids"])
                                       for r in cold),
            "of": len(cases),
            "cold_serial_equals_burst": same(serial["token_ids"], burst[(0, "greedy")]["token_ids"]),
            "needles_found_cold": sum(r["found"] for r in cold if r["sampling"] == "greedy"),
            "prompts": len(lengths),
            "cached_tokens_cold": [r["cached_tokens"] for r in cold] + [serial["cached_tokens"]],
            "cold_s": cold_s, "serial_s": serial_s}
    print(json.dumps(summ), flush=True)
    json.dump({"summary": summ, "replies": cold + [serial]}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
