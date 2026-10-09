"""make_rope.py MODEL_DIR KIT_DIR: the kit's RoPE tables (rope-{plain,compressed}-{cos,sin}.f32), recomputed.

They are the fp32 (cos, sin) tables of both RoPE kinds as the DeepSeek-V4.1 Python family builds them (ops.rope_cs on
the CPU: torch's own pow and polar, whose last bits the Zig engine does not recompute itself), 1,114,112 rows each,
raw little-endian. Run it where that family is installed (github.com/bertholomus/TensorFold, branch deepseek-v41-tp2,
in nvcr.io/nvidia/pytorch:26.07-py3; no GPU needed). MODEL_DIR is the EXL3 checkpoint (its config.json). Each file's
sha256 is checked against KIT_DIR/rope.json, which the kit ships."""
import hashlib
import json
import os
import sys

import torch

from tensorfold.families.deepseek_v41.config import Cfg
from tensorfold.families.deepseek_v41.ops import RopeTables, rope_cs


def main():
    model, kit = sys.argv[1:3]
    want = json.load(open(os.path.join(kit, "rope.json")))
    cfg = Cfg.read(model)
    rt = RopeTables(cfg, device="cpu")
    assert want["half"] == cfg.rope_dim // 2, (want["half"], cfg.rope_dim)
    bad = 0
    for kind, compressed in (("plain", False), ("compressed", True)):
        cos, sin = rope_cs(cfg.rope_dim, want["rows"], *rt.args(compressed), device="cpu")
        for part, t in (("cos", cos), ("sin", sin)):
            raw = t.contiguous().view(torch.uint8).numpy().tobytes()
            name = f"rope-{kind}-{part}.f32"
            with open(os.path.join(kit, name), "wb") as f:
                f.write(raw)
            ok = hashlib.sha256(raw).hexdigest() == want[name]
            bad += not ok
            print(f"{name}: {len(raw)} bytes, {'sha256 ok' if ok else 'SHA256 DIFFERS from rope.json'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
