"""make_bias_vl.py MODEL_DIR OUT: the kit's vision/gate_bias_vl.safetensors from your own copy of
deepseek-ai/DeepSeek-V4.1-Flash (MIT), standard library only.

The EXL3 checkpoint leaves out the MoE gates' routing bias for image-span tokens (`*.ffn.gate.bias_vl`, 43 f32 vectors
of 384). This copies those tensors, byte for byte, from the original checkpoint's shards (found through its
model.safetensors.index.json; only the shards that hold them are opened, and only those bytes are read) into one small
safetensors file. The digest it prints is the one the released kit's README lists."""
import hashlib
import json
import os
import struct
import sys

META = {"source": "deepseek-ai/DeepSeek-V4.1-Flash (MIT): the ffn.gate.bias_vl tensors, byte for byte"}


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return 8 + n, json.loads(f.read(n))


def main():
    model, out = sys.argv[1:3]
    wm = json.load(open(os.path.join(model, "model.safetensors.index.json")))["weight_map"]
    names = sorted(k for k in wm if k.endswith(".ffn.gate.bias_vl"))
    if not names:
        sys.exit("no *.ffn.gate.bias_vl tensors in the index: is MODEL_DIR the original (not EXL3) checkpoint?")
    data, hdr, off = [], {}, 0
    for k in names:
        p = os.path.join(model, wm[k])
        base, h = header(p)
        t = h[k]
        a, b = t["data_offsets"]
        with open(p, "rb") as f:
            f.seek(base + a)
            raw = f.read(b - a)
        assert t["dtype"] == "F32" and len(raw) == b - a, (k, t)
        hdr[k] = {"dtype": t["dtype"], "shape": t["shape"], "data_offsets": [off, off + len(raw)]}
        data.append(raw)
        off += len(raw)
    hdr["__metadata__"] = META
    js = json.dumps(hdr).encode()
    js += b" " * (-len(js) % 8)
    blob = struct.pack("<Q", len(js)) + js + b"".join(data)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "wb") as f:
        f.write(blob)
    print(f"{len(names)} tensors, {len(blob)} bytes, sha256 {hashlib.sha256(blob).hexdigest()}")


if __name__ == "__main__":
    main()
