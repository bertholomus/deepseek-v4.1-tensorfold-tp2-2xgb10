"""Image input over the OpenAI chat API (standard library only; the key is read from a file, never printed).

  python3 vision_http.py --base http://127.0.0.1:8000 --model M --images DIR [--key-file F] [--out F]

DIR holds vision_red.png, vision_text.png, vision_tall.png (vision_check.py writes them). Checks: a solid colour, a few
words of text, two images in one turn, an image plus a tool call, an image inside a tool result, an image refused in an
assistant turn; each with thinking off and greedy decoding.
"""

import argparse
import base64
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def load_key(path):
    if not path:
        return None
    for line in open(path).read().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line.split("=", 1)[1].strip().strip("'\"") if "=" in line else line


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--key-file")
    p.add_argument("--out")
    a = p.parse_args()
    key = load_key(a.key_file)

    def post(body, timeout=600):
        h = {"Content-Type": "application/json"}
        if key:
            h["Authorization"] = "Bearer " + key
        req = urllib.request.Request(a.base.rstrip("/") + "/v1/chat/completions",
                                     data=json.dumps(dict(body, model=a.model)).encode(), headers=h)
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, json.loads(r.read()), time.time() - t0
        except urllib.error.HTTPError as e:
            return e.code, {"error": e.read().decode()[:300]}, time.time() - t0

    def img(name):
        data = (Path(a.images) / f"vision_{name}.png").read_bytes()
        return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(data).decode()}}

    base = {"max_tokens": 96, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    tools = [{"type": "function", "function": {
        "name": "record_color", "description": "Record the main colour seen in an image",
        "parameters": {"type": "object", "properties": {"color": {"type": "string"}}, "required": ["color"]}}},
        {"type": "function", "function": {
            "name": "take_screenshot", "description": "Take a screenshot of the user's screen",
            "parameters": {"type": "object", "properties": {}}}}]
    results = {}

    def check(name, body, ok, field="content"):
        status, out, sec = post({**base, **body})
        msg = (out.get("choices") or [{}])[0].get("message", {}) if status == 200 else {}
        text = msg.get("content") or ""
        calls = msg.get("tool_calls") or []
        passed = status == 200 and ok(text, calls, out)
        results[name] = {"ok": bool(passed), "status": status, "seconds": round(sec, 1), "reply": text[:300],
                         "tool_calls": calls, "prompt_tokens": (out.get("usage") or {}).get("prompt_tokens"),
                         "error": out.get("error")}
        print(json.dumps({name: results[name]}, ensure_ascii=False), flush=True)
        return msg

    check("solid_colour", {"messages": [{"role": "user", "content": [
        img("red"), {"type": "text", "text": "What colour is this image? Answer with one word."}]}]},
        lambda t, c, o: "red" in t.lower())
    check("reads_text", {"messages": [{"role": "user", "content": [
        img("text"), {"type": "text", "text": "What does the sign say? Reply with the exact words only."}]}]},
        lambda t, c, o: "open" in t.lower() and "24" in t and "hour" in t.lower())
    check("two_images", {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "Here are two images."}, img("red"), img("tall"),
        {"type": "text", "text": "Name the main colour of the first image and of the second image, in order, "
                                 "as two words separated by a comma."}]}]},
        lambda t, c, o: "red" in t.lower() and "blue" in t.lower() and t.lower().index("red") < t.lower().index("blue"))
    msg = check("image_then_tool_call", {"tools": tools, "messages": [{"role": "user", "content": [
        img("red"), {"type": "text", "text": "Look at the image and record its main colour with the record_color tool."}]}]},
        lambda t, c, o: bool(c) and c[0]["function"]["name"] == "record_color"
        and "red" in json.loads(c[0]["function"]["arguments"]).get("color", "").lower())
    check("image_in_tool_result", {"tools": tools, "messages": [
        {"role": "user", "content": "Take a screenshot and tell me what the sign on my screen says."},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {
            "name": "take_screenshot", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": [
            {"type": "text", "text": "Screenshot captured."}, img("text")]}]},
        lambda t, c, o: "open" in t.lower() and "24" in t)
    status, out, _ = post({**base, "messages": [
        {"role": "user", "content": "hi"}, {"role": "assistant", "content": [img("red")]},
        {"role": "user", "content": "what was that?"}]})
    results["image_in_assistant_refused"] = {"ok": status == 400, "status": status}
    print(json.dumps({"image_in_assistant_refused": results["image_in_assistant_refused"]}), flush=True)
    summary = {"passed": sum(r["ok"] for r in results.values()), "of": len(results)}
    print(json.dumps({"summary": summary}), flush=True)
    if a.out:
        json.dump({"summary": summary, "results": results}, open(a.out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
