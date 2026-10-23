#!/usr/bin/env python3
"""Prompt lengths against a server (standard library only). Each length L sends the first L ids of one long plain
text's tokens as a /v1/completions prompt (token ids as given), greedy, 8 tokens, return_token_ids, one at a time, and
records the reply's status, ids and text. The ids come from the server's /tokenize, or from an earlier run's output
(--ids-from), so two servers see the same prompts; --reference compares each length's ids with that run's.

  python3 tp4_lensweep.py --base http://127.0.0.1:18091 --model DeepSeek-V4.1-Flash-TP4 --out zig.json
  python3 tp4_lensweep.py --base ... --model ... --ids-from zig.json --reference zig.json --out served.json
"""

import argparse
import json
import time
import urllib.error
import urllib.request

LENGTHS = sorted(set(list(range(1, 21)) + [
    31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 2046, 2047, 2048, 2049,
    2050, 3071, 4095, 4096, 4097, 6143, 6144, 6145, 8191, 8192, 8193, 10000, 12287, 12289, 16383, 16384, 16385, 19999,
    20000]))
# "wide": every length to 160, each multiple of 64 to 1,024 and of 512 to 4,096 with its neighbours, each multiple of
# 2,048 to 20,480 with its neighbours (a chunk and one, two or three rows past it), and the default list
WIDE = sorted(set(LENGTHS) | set(range(1, 161))
              | {m + d for m in range(64, 1025, 64) for d in (-1, 0, 1, 2)}
              | {m + d for m in range(512, 4097, 512) for d in (-2, -1, 0, 1, 2, 3)}
              | {m + d for m in range(2048, 20481, 2048) for d in (-1, 0, 1, 2, 3)})


def post(base, path, body, timeout=900):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:  # noqa: BLE001
            return e.code, {"error": str(e)}
    except Exception as e:  # noqa: BLE001
        return 0, {"error": f"{type(e).__name__}: {e}"}


def text(n):
    kinds = ["rain", "snow", "sun", "wind"]
    return "\n".join(f"Entry {i}: the river at station {i % 97} rose {i % 13} cm while {kinds[i % 4]} moved east."
                     for i in range(n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids-from")
    ap.add_argument("--reference")
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--lengths", default="default", help="default, wide, or a comma-separated list")
    a = ap.parse_args()
    if a.ids_from:
        ids = json.load(open(a.ids_from))["ids"]
    else:
        st, tk = post(a.base, "/tokenize", {"model": a.model, "prompt": text(1500)})
        ids = tk.get("tokens") if st == 200 else None
        if not ids:
            raise SystemExit(f"tokenize failed: HTTP {st}: {str(tk)[:300]}")
    lengths = LENGTHS if a.lengths == "default" else WIDE if a.lengths == "wide" else sorted({int(x) for x in a.lengths.split(",")})
    if len(ids) < max(lengths):
        raise SystemExit(f"only {len(ids)} ids")
    ref = json.load(open(a.reference))["results"] if a.reference else None
    results, equal, errors = {}, 0, 0
    for n in lengths:
        t0 = time.time()
        st, r = post(a.base, "/v1/completions", {"model": a.model, "prompt": ids[:n], "max_tokens": a.max_tokens,
                                                 "temperature": 0, "return_token_ids": True})
        got = (r.get("tensorfold") or {}).get("token_ids") if st == 200 else None
        txt = ((r.get("choices") or [{}])[0].get("text")) if st == 200 else None
        rec = {"status": st, "token_ids": got, "text": txt, "seconds": round(time.time() - t0, 2)}
        if st != 200:
            rec["error"] = str(r.get("error") or r)[:300]
            errors += 1
        if ref is not None:
            rr = ref.get(str(n)) or {}
            rec["equal"] = got is not None and got == rr.get("token_ids")
            equal += int(rec["equal"])
        results[str(n)] = rec
        print(json.dumps({"length": n, **{k: v for k, v in rec.items() if k != "token_ids"}}), flush=True)
    summary = {"lengths": len(lengths), "errors": errors,
               "failed": [n for n in lengths if results[str(n)]["status"] != 200][:60]}
    if ref is not None:
        summary["equal_to_reference"] = equal
    json.dump({"base": a.base, "model": a.model, "ids": ids[:max(lengths)], "results": results, "summary": summary},
              open(a.out, "w"))
    print(json.dumps({"summary": summary}), flush=True)


if __name__ == "__main__":
    main()
