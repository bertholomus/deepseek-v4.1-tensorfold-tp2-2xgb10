"""vprobe_cmp.py LANE PORT: vprobe.py's cases compared (status, prompt tokens, token ids, tool calls' arguments and
the refusal's words with object addresses masked). One summary line, then each difference."""
import json
import re
import sys

lane, port = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
norm = lambda m: re.sub(r"0x[0-9a-f]+", "0x?", m or "")  # noqa: E731
same, diff = 0, []
for k, l in lane.items():
    p = port.get(k)
    if p is None:
        diff.append((k, "missing"))
        continue
    keys = ("status", "prompt_tokens", "token_ids") if l["status"] == 200 else ("status",)
    ok = all(l.get(x) == p.get(x) for x in keys)
    if l["status"] == 200:
        lt = [(c["function"]["name"], c["function"]["arguments"]) for c in (l.get("tool_calls") or [])]
        pt = [(c["function"]["name"], c["function"]["arguments"]) for c in (p.get("tool_calls") or [])]
        ok = ok and lt == pt
    else:
        ok = ok and norm(l.get("message")) == norm(p.get("message"))
    if ok:
        same += 1
    else:
        diff.append((k, {x: l.get(x) for x in ("status", "prompt_tokens", "message")}, {x: p.get(x) for x in ("status", "prompt_tokens", "message")}))
print(json.dumps({"equal": same, "of": len(lane), "differ": len(diff)}))
for d in diff:
    print(json.dumps(d)[:600])
