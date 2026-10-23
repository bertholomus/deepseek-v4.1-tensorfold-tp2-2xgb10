"""vprobe.py --base URL --key-file F --images DIR --out OUT: the vision cases the port's server must answer as the
served lane does, run the same way against each (lane first, then the port; vprobe_cmp.py
compares): replies (greedy token ids) for pictures of many formats, message layouts (text parts around images, other
messages' parts, a long system prompt, a span across the 2,048-row prompt chunk, tool results), and every refusal
(status and words). One JSON object per case into OUT."""
import argparse
import base64
import json
import struct
import urllib.error
import urllib.request
import zlib
from pathlib import Path

PH = "<｜deepseek_image｜>"


def png_grey(w, h, v=128):
    raw = zlib.compress(b"".join(b"\x00" + bytes([v]) * w for _ in range(h)), 6)

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)) + chunk(b"IDAT", raw) + chunk(b"IEND", b"")


def url(data, mime="image/png"):
    return "data:%s;base64,%s" % (mime, base64.b64encode(data).decode())


def img(data):
    return {"type": "image_url", "image_url": {"url": url(data)}}


def T(s):
    return {"type": "text", "text": s}


def main():
    p = argparse.ArgumentParser()
    for k in ("--base", "--key-file", "--images", "--out"):
        p.add_argument(k, required=True)
    a = p.parse_args()
    line = [x for x in open(a.key_file).read().splitlines() if x.strip() and not x.startswith("#")][0]
    key = line.split("=", 1)[1].strip().strip("'\"") if "=" in line else line.strip()
    d = Path(a.images)
    pic = lambda name: img((d / name).read_bytes())  # noqa: E731
    small = (d / "a.png").read_bytes()

    def chat(messages, max_tokens=24, **extra):
        body = {"model": "DeepSeek-V4.1-Flash-TF", "messages": messages, "max_tokens": max_tokens, "temperature": 0,
                "return_token_ids": True, "chat_template_kwargs": {"enable_thinking": False}}
        body.update(extra)
        req = urllib.request.Request(a.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                j = json.loads(r.read())
            return {"status": 200, "prompt_tokens": j["usage"]["prompt_tokens"],
                    "token_ids": (j.get("tensorfold") or {}).get("token_ids"),
                    "tool_calls": j["choices"][0]["message"].get("tool_calls")}
        except urllib.error.HTTPError as e:
            raw = e.read().decode(errors="replace")
            try:
                msg = json.loads(raw)["error"]["message"]
            except Exception:
                msg = raw
            return {"status": e.code, "message": msg[:400]}

    q = "Describe this image in one sentence."
    cases = {}
    for name in ("jpeg_q90_s2.jpg", "jpeg_progressive.jpg", "jpeg_cmyk.jpg", "jpeg_exif6.jpg", "jpeg_grey.jpg",
                 "webp_lossy.webp", "webp_alpha_grad.webp", "webp_anim.webp", "gif_still.gif", "gif_anim.gif",
                 "gif_transparent.gif", "bmp_RGB.bmp", "tiff_tiff_lzw.tiff", "tiff_16.tiff", "avif.avif", "ico.ico",
                 "pbm.pbm", "tall.jpg", "wide.webp", "jp2.jp2", "qoi.qoi"):
        if (d / name).exists():
            cases["format:" + name] = [{"role": "user", "content": [pic(name), T(q)]}]
    S = pic("a.png")
    cases["parts:A,empty,B,IMG"] = [{"role": "user", "content": [T("Apples."), T(""), T("Bananas"), S]}]
    cases["parts:null text"] = [{"role": "user", "content": [T("Apples."), {"type": "text"}, S]}]
    cases["parts:IMG,IMG,text"] = [{"role": "user", "content": [S, pic("gif_still.gif"), T("Compare them.")]}]
    cases["parts:other message parts"] = [{"role": "user", "content": [T("Apples"), T("Bananas")]},
                                          {"role": "assistant", "content": "ok"}, {"role": "user", "content": [S, T(q)]}]
    cases["parts:system parts"] = [{"role": "system", "content": [T("Be brief."), T("Use plain words.")]},
                                   {"role": "user", "content": [S, T(q)]}]
    long_sys = " ".join("Rule %d: answer in plain words and keep the answer short." % i for i in range(70))
    cases["chunks:long system + image"] = [{"role": "system", "content": long_sys}, {"role": "user", "content": [S, T(q)]}]
    long_text = " ".join("Paragraph %d tells a short story about a red fox and a grey wolf in the forest." % i for i in range(90))
    big = png_grey(6000, 6000, 90)
    cases["chunks:span across 2048"] = [{"role": "user", "content": [T(long_text), img(big), T(q)]}]
    cases["tools:image in tool result"] = [
        {"role": "user", "content": "What does the screenshot show?"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "shot", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": [T("Screenshot taken."), S]}]
    cases["text:placeholder, no image"] = [{"role": "user", "content": "look " + PH + " here"}]
    # refusals
    cases["refuse:https"] = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.com/x.png"}}]}]
    cases["refuse:file path"] = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "/etc/hosts"}}]}]
    cases["refuse:base64 one over"] = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8hZ"}}]}]
    cases["refuse:base64 padding"] = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8"}}]}]
    cases["refuse:not base64"] = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png,abc"}}]}]
    cases["refuse:no comma"] = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64"}}]}]
    cases["refuse:empty"] = [{"role": "user", "content": [img(b"")]}]
    cases["refuse:garbage"] = [{"role": "user", "content": [img(b"not an image at all")]}]
    cases["refuse:truncated jpeg"] = [{"role": "user", "content": [img((d / "jpeg_q90_s2.jpg").read_bytes()[:300])]}]
    cases["refuse:truncated png"] = [{"role": "user", "content": [img(small[:200])]}]
    cases["refuse:bomb"] = [{"role": "user", "content": [img(png_grey(13401, 13401))]}]
    cases["refuse:system image"] = [{"role": "system", "content": [S]}, {"role": "user", "content": q}]
    cases["refuse:assistant image"] = [{"role": "user", "content": q}, {"role": "assistant", "content": [S]}, {"role": "user", "content": q}]
    cases["refuse:image_url string"] = [{"role": "user", "content": [{"type": "image_url", "image_url": url(small)}]}]
    cases["refuse:image_url no url"] = [{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}]
    cases["refuse:placeholder with image"] = [{"role": "user", "content": [S, T("look " + PH)]}]
    cases["refuse:placeholder elsewhere"] = [{"role": "user", "content": "x " + PH}, {"role": "assistant", "content": "ok"}, {"role": "user", "content": [S]}]
    cases["refuse:input_image part"] = [{"role": "user", "content": [{"type": "input_image", "image_url": url(small)}]}]
    out = {}
    for name, msgs in cases.items():
        out[name] = chat(msgs)
        print(name, json.dumps(out[name])[:160], flush=True)
    json.dump(out, open(a.out, "w"))


if __name__ == "__main__":
    main()
