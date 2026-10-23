#!/usr/bin/env python3
"""serving_1017.py MODEL_DIR [--template FILE] [--check]: make the public 1017 quant servable by the engine at
ENGINE_COMMIT. Standard library only, no GPU; it reads about 130 MB and changes no downloaded file.

MODEL_DIR is a download of bertholomus/DeepSeek-V4.1-Flash-1017-EXL3-2.5bpw, the quant this recipe serves (the
four-node recipe readies the 4.0 bpw quant the same way). The script adds two files to it:

- serving-1017.safetensors: six tensors under the names and types the engine reads.
  - mtp.2.markov_head.embed.weight (BF16) and mtp.2.markov_head.head.weight (F16): byte copies of the converter's
    mtp.2.markov_head.markov_w1.weight and markov_w2.weight.
  - layers.1 and layers.14 engram.q_weight and engram.k_weight, widened from BF16 to F32. The widening is exact, and
    the engine reads these four as F32.
  The engine maps every *.safetensors file in MODEL_DIR in byte order of the names, and a later file's tensor wins.
  This file sorts after the model-*.safetensors shards, so its four F32 tensors replace the BF16 ones.
- chat_template.jinja: the recipe's chat/chat_template.jinja. The quant ships no chat template, and the server needs
  one for chat requests.

--check changes nothing. It exits 0 when both files are in MODEL_DIR and equal what the script would write, and 1
otherwise. Either way the script prints the sha256 of both files.
"""
import argparse
import array
import hashlib
import json
import os
import re
import shutil
import struct
import sys

OUT = "serving-1017.safetensors"
TEMPLATE = "chat_template.jinja"
MARKOV = re.compile(r"^(mtp\.\d+)\.markov_head\.markov_w([12])\.weight$")
ENGRAM_QK = re.compile(r"^layers\.\d+\.engram\.[qk]_weight$")


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return 8 + n, h


def read_tensor(path, base, info):
    a, b = info["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + a)
        data = f.read(b - a)
    if len(data) != b - a:
        sys.exit(f"{path}: short read (an incomplete download?)")
    return data


def encode(tensors):
    """the file's bytes: tensors is a list of (name, dtype, shape, bytes) in file order"""
    hdr, off = {}, 0
    for name, dt, shape, data in tensors:
        hdr[name] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + len(data)]}
        off += len(data)
    hdr["__metadata__"] = {"format": "pt"}
    hb = json.dumps(hdr, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    return [struct.pack("<Q", len(hb)), hb] + [t[3] for t in tensors]


def widen_bf16(data):
    h = array.array("H")
    h.frombytes(data)
    if sys.byteorder != "little":
        h.byteswap()
    w = array.array("I", (x << 16 for x in h))
    if sys.byteorder != "little":
        w.byteswap()
    return w.tobytes()


def sha256_of(parts):
    s = hashlib.sha256()
    for p in parts:
        s.update(p)
    return s.hexdigest()


def sha256_file(path):
    s = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            s.update(block)
    return s.hexdigest()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model_dir")
    ap.add_argument("--template", default=os.path.join(here, "..", "chat", TEMPLATE))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    d = a.model_dir
    idx = os.path.join(d, "model.safetensors.index.json")
    if not os.path.isfile(idx):
        sys.exit(f"no {idx}: MODEL_DIR is a download of bertholomus/DeepSeek-V4.1-Flash-1017-EXL3-2.5bpw")
    wm = json.load(open(idx))["weight_map"]
    missing = sorted(set(f for f in wm.values() if not os.path.isfile(os.path.join(d, f))))
    if missing:
        sys.exit(f"{len(missing)} shards in the index are not in {d}: {', '.join(missing[:5])}")

    want = sorted(k for k in wm if MARKOV.match(k) or ENGRAM_QK.match(k))
    heads = {}
    tensors = []
    for k in want:
        p = os.path.join(d, wm[k])
        if p not in heads:
            heads[p] = header(p)
        base, h = heads[p]
        if k not in h:
            sys.exit(f"{k}: the index names {wm[k]}, whose header lacks it")
        v = h[k]
        data = read_tensor(p, base, v)
        m = MARKOV.match(k)
        if m:
            name = f"{m.group(1)}.markov_head.{'embed' if m.group(2) == '1' else 'head'}.weight"
            need = "BF16" if m.group(2) == "1" else "F16"
            if v["dtype"] != need:
                sys.exit(f"{k} is {v['dtype']}; the engine reads {name} as {need}")
            tensors.append((name, v["dtype"], v["shape"], data))
        elif v["dtype"] == "BF16":
            tensors.append((k, "F32", v["shape"], widen_bf16(data)))
        elif v["dtype"] != "F32":
            sys.exit(f"{k} is {v['dtype']}; the engine reads it as F32")
    names = sorted(t[0] for t in tensors)
    if not any(n.endswith(".embed.weight") for n in names) or not any(".engram." in n for n in names):
        sys.exit(f"found {names}: expected the markov head and the Engram q/k of the 1017 quant")
    tensors.sort(key=lambda t: t[0])
    parts = encode(tensors)
    out_sha = sha256_of(parts)
    tpl = open(a.template, "rb").read()
    tpl_sha = hashlib.sha256(tpl).hexdigest()

    out_p = os.path.join(d, OUT)
    tpl_p = os.path.join(d, TEMPLATE)
    if a.check:
        ok = True
        for p, sha in ((out_p, out_sha), (tpl_p, tpl_sha)):
            got = sha256_file(p) if os.path.isfile(p) else "absent"
            print(f"{os.path.basename(p)}: {got} ({'ok' if got == sha else 'expected ' + sha})")
            ok = ok and got == sha
        sys.exit(0 if ok else 1)
    tmp = out_p + ".partial"
    with open(tmp, "wb") as f:
        for p in parts:
            f.write(p)
    os.replace(tmp, out_p)
    if os.path.abspath(a.template) != os.path.abspath(tpl_p):
        shutil.copyfile(a.template, tpl_p)
    print(f"{OUT}: {', '.join(names)}")
    print(f"{OUT}: {out_sha}")
    print(f"{TEMPLATE}: {tpl_sha}")


if __name__ == "__main__":
    main()
