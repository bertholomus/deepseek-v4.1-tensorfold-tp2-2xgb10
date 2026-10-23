"""vconc.py --base URL --model M --key-file F --images DIR --ref LANE_REF --out OUT [--parallel 4] [--rounds 2]:
visionref.py's image requests (each gate picture alone, the first two together; greedy, max_tokens 48, token ids),
sent PARALLEL at a time for ROUNDS rounds, each reply's token ids against the lane's (LANE_REF: visionref.py's output
on the lane). Prints one summary line: equal / unequal / errors."""
import argparse
import base64
import json
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def post(base, path, body, key, timeout=900):
    h = {"Content-Type": "application/json", "Authorization": "Bearer " + key}
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def part(path):
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + data}}


def main():
    p = argparse.ArgumentParser()
    for k in ("--base", "--model", "--key-file", "--images", "--ref", "--out"):
        p.add_argument(k, required=True)
    p.add_argument("--parallel", type=int, default=4)
    p.add_argument("--rounds", type=int, default=2)
    a = p.parse_args()
    line = [x for x in open(a.key_file).read().splitlines() if x.strip() and not x.startswith("#")][0]
    key = line.split("=", 1)[1].strip().strip("'\"") if "=" in line else line.strip()
    pngs = sorted(Path(a.images).glob("*.png"))
    cases = [(f.name, [part(f)], "Describe this image in one sentence.") for f in pngs]
    cases.append((pngs[0].name + "+" + pngs[1].name, [part(pngs[0]), part(pngs[1])], "What differs between these two images?"))
    ref = {r["case"]: r.get("token_ids") for r in json.load(open(a.ref))}

    def one(c):
        name, parts, q = c
        body = {"model": a.model, "messages": [{"role": "user", "content": parts + [{"type": "text", "text": q}]}],
                "max_tokens": 48, "temperature": 0, "return_token_ids": True,
                "chat_template_kwargs": {"enable_thinking": False}}
        try:
            r = post(a.base, "/v1/chat/completions", body, key)
            ids = (r.get("tensorfold") or {}).get("token_ids")
            return {"case": name, "prompt_tokens": r["usage"]["prompt_tokens"], "token_ids": ids, "equal": ids == ref.get(name)}
        except urllib.error.HTTPError as e:
            return {"case": name, "error": e.code, "body": e.read().decode(errors="replace")[:300]}
        except Exception as e:  # a dropped connection or timeout
            return {"case": name, "error": repr(e)[:200]}

    out = []
    with ThreadPoolExecutor(a.parallel) as ex:
        for _ in range(a.rounds):
            out.extend(ex.map(one, cases))
    json.dump(out, open(a.out, "w"))
    eq = sum(1 for r in out if r.get("equal"))
    err = sum(1 for r in out if "error" in r)
    print(json.dumps({"requests": len(out), "equal": eq, "unequal": len(out) - eq - err, "errors": err,
                      "first_bad": next((r for r in out if not r.get("equal")), None)})[:600])


if __name__ == "__main__":
    main()
