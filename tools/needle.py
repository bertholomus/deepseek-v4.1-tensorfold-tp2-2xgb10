"""Needle-in-a-haystack recall against an OpenAI chat server (stdlib only).

  python3 needle.py --base URL --model M [--key-file F] --lengths 8192,32768,131072 --depths 0.1,0.5,0.9 --out F
  python3 needle.py --base URL --model M --shared --lengths 131072 --depths 0.1,0.5,0.9 --out F

A random passphrase is placed at a relative depth of a filler text of about the given token length (sized with the
server's /tokenize), and the model is asked for it at the end, thinking off, greedy. Recall = the passphrase appears in
the reply. With --shared, one haystack per length carries one labelled vault needle per depth and each label is asked
for in its own follow-up request against the same text.
"""

import argparse
import json
import random
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
    p.add_argument("--lengths", default="8192,32768,131072")
    p.add_argument("--depths", default="0.1,0.5,0.9")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--shared", action="store_true")
    p.add_argument("--out")
    a = p.parse_args()
    key = None
    if a.key_file:
        line = [x for x in open(a.key_file).read().splitlines() if x.strip() and not x.startswith("#")][0]
        key = line.split("=", 1)[1].strip().strip("'\"") if "=" in line else line.strip()
    rng = random.Random(a.seed)
    rows = []
    if a.shared:
        depths = [float(x) for x in a.depths.split(",")]
        for length in [int(x) for x in a.lengths.split(",")]:
            phrases, needles = [], []
            for k in range(len(depths)):
                phrase = f"{rng.choice(CODE_WORDS)}-{rng.choice(CODE_WORDS)}-{rng.randrange(1000, 9999)}"
                phrases.append(phrase)
                needles.append(f" The secret passphrase for vault {chr(65 + k)} is {phrase}. Remember it. ")
            words = [rng.choice(WORDS) for _ in range(int(length * 0.75))]

            def build(ws):
                # each needle lands at its own relative depth of the word list (positions are taken on the
                # pre-insertion list and shifted by needles already inserted before them)
                parts = list(ws)
                order = sorted(range(len(depths)), key=lambda k: int(len(ws) * depths[k]))
                off = 0
                for k in order:
                    parts.insert(int(len(ws) * depths[k]) + off, needles[k])
                    off += 1
                return "Read the notes below.\n" + " ".join(parts)

            text = build(words)
            n = len(post(a.base, "/tokenize", {"model": a.model, "prompt": text}, key)["tokens"])
            while n < length - 300:
                words += [rng.choice(WORDS) for _ in range(int((length - n) * 0.7))]
                text = build(words)
                n = len(post(a.base, "/tokenize", {"model": a.model, "prompt": text}, key)["tokens"])
            for k, depth in enumerate(depths):
                label = chr(65 + k)
                q = text + f"\n\nWhat is the secret passphrase for vault {label} mentioned in the notes? Reply with the passphrase only."
                t0 = time.time()
                out = post(a.base, "/v1/chat/completions", {"model": a.model, "messages": [{"role": "user", "content": q}],
                                                             "max_tokens": 32, "temperature": 0,
                                                             "chat_template_kwargs": {"enable_thinking": False}}, key)
                reply = out["choices"][0]["message"]["content"] or ""
                row = {"length": length, "prompt_tokens": out.get("usage", {}).get("prompt_tokens"),
                       "cached_tokens": (out.get("usage", {}).get("prompt_tokens_details") or {}).get("cached_tokens") or 0,
                       "depth": depth, "label": label, "phrase": phrases[k], "found": phrases[k] in reply,
                       "reply": reply[:80], "seconds": round(time.time() - t0, 1), "shared": True}
                print(json.dumps(row), flush=True)
                rows.append(row)
    else:
        for length in [int(x) for x in a.lengths.split(",")]:
            for depth in [float(x) for x in a.depths.split(",")]:
                phrase = f"{rng.choice(CODE_WORDS)}-{rng.choice(CODE_WORDS)}-{rng.randrange(1000, 9999)}"
                needle = f" The secret passphrase is {phrase}. Remember it. "
                words = [rng.choice(WORDS) for _ in range(int(length * 0.75))]
                # grow to the target length by the tokenizer
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
                t0 = time.time()
                out = post(a.base, "/v1/chat/completions", {"model": a.model, "messages": [{"role": "user", "content": q}],
                                                             "max_tokens": 32, "temperature": 0,
                                                             "chat_template_kwargs": {"enable_thinking": False}}, key)
                reply = out["choices"][0]["message"]["content"] or ""
                row = {"length": length, "prompt_tokens": out.get("usage", {}).get("prompt_tokens"), "depth": depth,
                       "phrase": phrase, "found": phrase in reply, "reply": reply[:80], "seconds": round(time.time() - t0, 1)}
                print(json.dumps(row), flush=True)
                rows.append(row)
    summ = {"cases": len(rows), "found": sum(r["found"] for r in rows)}
    print(json.dumps({"summary": summ}), flush=True)
    if a.out:
        json.dump({"summary": summ, "rows": rows}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
