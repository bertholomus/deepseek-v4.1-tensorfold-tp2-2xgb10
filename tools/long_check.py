"""Concurrent == solo on long prompts (standard library; the key is read from a file, never printed): each long chat
prompt is tokenized to ids and replayed against /v1/completions greedy (plus one seeded-sampled case on prompt 0),
first all cases at once (burst), then one after another (solo), then prompt 0 greedy alone with speculative drafting
off (serial). Every reply's token ids must equal its solo run's; long-prompt recall is checked on the greedy replies.

  python3 long_check.py --base URL --model M [--key-file F] [--lengths 160000,160000,160000,160000] [--seed 11]
                        [--tokens 48] --out F
"""

import argparse
import json
import random
import threading
import time
import urllib.request

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
    p.add_argument("--lengths", default="160000,160000,160000,160000")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--tokens", type=int, default=48)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    key = None
    if a.key_file:
        line = [x for x in open(a.key_file).read().splitlines() if x.strip() and not x.startswith("#")][0]
        key = line.split("=", 1)[1].strip().strip("'\"") if "=" in line else line.strip()
    rng = random.Random(a.seed)
    lengths = [int(x) for x in a.lengths.split(",")]

    prompts = []
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

    cases = [(i, "greedy", {"temperature": 0}) for i in range(len(lengths))]
    cases.append((0, "sampled", {"temperature": 0.6, "top_p": 0.95, "seed": 1000}))

    def case_key(c):
        return "%d/%s" % (c[0], c[1])

    def run_case(case, phase, extra=None):
        idx, name, samp = case
        body = {"model": a.model, "prompt": prompts[idx]["ids"], "max_tokens": a.tokens,
                "ignore_eos": True, "return_token_ids": True, **samp}
        if extra:
            body.update(extra)
        t0 = time.time()
        out = post(a.base, "/v1/completions", body, key)
        text = out["choices"][0]["text"] or ""
        rec = {"phase": phase, "prompt": idx, "sampling": name,
               "token_ids": (out.get("tensorfold") or {}).get("token_ids"),
               "text": text,
               "cached_tokens": (out.get("usage", {}).get("prompt_tokens_details") or {}).get("cached_tokens") or 0,
               "seconds": round(time.time() - t0, 1), "found": prompts[idx]["phrase"] in text}
        if extra:
            rec.update(extra)
        print(json.dumps({k: v for k, v in rec.items() if k != "token_ids"}), flush=True)
        return rec

    replies = []

    t0 = time.time()
    burst, errs = {}, []

    def worker(c):
        try:
            burst[case_key(c)] = run_case(c, "burst")
        except Exception as e:  # re-raised on the main thread after the join
            errs.append(e)

    ths = [threading.Thread(target=worker, args=(c,)) for c in cases]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    if errs:
        raise errs[0]
    burst_s = round(time.time() - t0, 1)
    replies += [burst[case_key(c)] for c in cases]

    t0 = time.time()
    solo = {}
    for c in cases:
        solo[case_key(c)] = run_case(c, "solo")
    solo_s = round(time.time() - t0, 1)
    replies += [solo[case_key(c)] for c in cases]

    t0 = time.time()
    serial = run_case(cases[0], "serial", extra={"draft": False})
    serial_s = round(time.time() - t0, 1)
    replies.append(serial)

    def same(x, y):                       # missing or empty token ids never count as equal
        return bool(x) and x == y

    summ = {"burst_equal_to_solo": sum(same(burst[k]["token_ids"], solo[k]["token_ids"]) for k in solo),
            "of": len(cases),
            "drafted_equals_serial": same(serial["token_ids"], solo[case_key(cases[0])]["token_ids"]),
            "needles_found_burst": sum(burst[case_key((i, "greedy", {"temperature": 0}))]["found"]
                                       for i in range(len(lengths))),
            "needles_found_solo": sum(solo[case_key((i, "greedy", {"temperature": 0}))]["found"]
                                      for i in range(len(lengths))),
            "prompts": len(lengths),
            "burst_s": burst_s, "solo_s": solo_s, "serial_s": serial_s,
            "cached_tokens_solo": [solo[case_key(c)]["cached_tokens"] for c in cases]}
    print(json.dumps(summ), flush=True)
    json.dump({"summary": summ, "replies": replies}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
