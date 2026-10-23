"""visionref.py --base URL --model M --key-file F --images DIR --out F (standard library; the key is never printed).

Image prompts for comparing two engines token for token: each PNG in DIR alone with one
question, then the first two together, as OpenAI chat image_url data URLs; greedy, thinking off, 48 tokens, token ids
returned (the tensorfold block's token_ids, with the reply text and the prompt length)."""
import argparse
import base64
import json
import urllib.error
import urllib.request
from pathlib import Path


def post(base, path, body, key, timeout=600):
    h = {"Content-Type": "application/json", "Authorization": "Bearer " + key}
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def part(path):
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + data}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--key-file", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    line = [x for x in open(a.key_file).read().splitlines() if x.strip() and not x.startswith("#")][0]
    key = line.split("=", 1)[1].strip().strip("'\"") if "=" in line else line.strip()
    pngs = sorted(Path(a.images).glob("*.png"))
    cases = [(f.name, [part(f)], "Describe this image in one sentence.") for f in pngs]
    cases.append((pngs[0].name + "+" + pngs[1].name, [part(pngs[0]), part(pngs[1])], "What differs between these two images?"))
    out = []
    for name, parts, q in cases:
        body = {"model": a.model, "messages": [{"role": "user", "content": parts + [{"type": "text", "text": q}]}],
                "max_tokens": 48, "temperature": 0, "return_token_ids": True,
                "chat_template_kwargs": {"enable_thinking": False}}
        try:
            r = post(a.base, "/v1/chat/completions", body, key)
            rec = {"case": name, "prompt_tokens": r["usage"]["prompt_tokens"],
                   "token_ids": (r.get("tensorfold") or {}).get("token_ids"),
                   "text": r["choices"][0]["message"].get("content")}
        except urllib.error.HTTPError as e:
            rec = {"case": name, "error": e.code, "body": e.read().decode(errors="replace")[:300]}
        print(json.dumps({k: v for k, v in rec.items() if k != "token_ids"}), flush=True)
        out.append(rec)
    json.dump(out, open(a.out, "w"))


if __name__ == "__main__":
    main()
