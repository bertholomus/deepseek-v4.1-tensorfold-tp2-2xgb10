#!/usr/bin/env python3
"""make_rope.py MODEL KIT_DIR [--check]: the kit's four RoPE tables (rope-{plain,compressed}-{cos,sin}.f32),
recomputed on the CPU, standalone (PyTorch only; no GPU).

They are the fp32 (cos, sin) tables of both rope kinds as the DeepSeek-V4.1 Python family builds them (ops.rope_cs:
rows below 2^19 from one freqs_cis call of 2^19 rows, the rest in blocks of 2^16 rows, torch's own pow and polar on the
CPU, whose last bits the Zig engine does not recompute itself), KIT_DIR/rope.json's row count each, raw little-endian.
The functions below are that family's (branch deepseek-v41-tp2, ops.py), copied so this script needs nothing else.
MODEL is the EXL3 checkpoint's folder or its config.json. Each table's sha256 is checked against KIT_DIR/rope.json;
exit 1 if any differs. --check: compute and compare without writing the tables."""
import hashlib
import json
import math
import os
import sys

import torch

F32 = torch.float32
ROPE_COMPAT = 1 << 19
ROPE_BLOCK = 1 << 16


def freqs_cis(dim, seqlen, orig_len, base, factor, beta_fast, beta_slow):
    """Complex rotations [seqlen, dim/2]; YaRN ramp when orig_len > 0 (no attention mscale)."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=F32) / dim))
    if orig_len > 0:
        def corr(rot):
            return dim * math.log(orig_len / (rot * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corr(beta_fast)), 0)
        high = min(math.ceil(corr(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=F32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    ang = torch.outer(torch.arange(seqlen, dtype=F32), freqs)
    return torch.polar(torch.ones_like(ang), ang)


def rope_block(dim, start, n, orig_len, base, factor, beta_fast, beta_slow):
    """freqs_cis's rotations at positions start .. start + n - 1: complex64 [n, dim/2] on the CPU."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=F32) / dim))
    if orig_len > 0:
        def corr(rot):
            return dim * math.log(orig_len / (rot * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corr(beta_fast)), 0)
        high = min(math.ceil(corr(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=F32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    ang = torch.outer(torch.arange(start, start + n, dtype=torch.int64).to(F32), freqs)  # exact below 2^24
    return torch.polar(torch.ones_like(ang), ang)


def rope_cs(dim, rows, orig_len, base, factor, beta_fast, beta_slow):
    """(cos, sin) fp32 [rows, dim/2]: rows below ROPE_COMPAT from freqs_cis(dim, ROPE_COMPAT, ...), the rest from
    rope_block in ROPE_BLOCK steps."""
    half = dim // 2
    cos = torch.empty((rows, half), dtype=F32)
    sin = torch.empty((rows, half), dtype=F32)
    f = freqs_cis(dim, ROPE_COMPAT, orig_len, base, factor, beta_fast, beta_slow)
    n = min(rows, ROPE_COMPAT)
    cos[:n].copy_(f.real[:n])
    sin[:n].copy_(f.imag[:n])
    del f
    for start in range(ROPE_COMPAT, rows, ROPE_BLOCK):
        f = rope_block(dim, start, ROPE_BLOCK, orig_len, base, factor, beta_fast, beta_slow)
        n = min(rows - start, ROPE_BLOCK)
        cos[start:start + n].copy_(f.real[:n])
        sin[start:start + n].copy_(f.imag[:n])
    return cos, sin


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    check = "--check" in sys.argv[1:]
    model, kit = args[:2]
    raw = json.load(open(model if os.path.isfile(model) else os.path.join(model, "config.json")))
    t = raw.get("text_config", raw)
    rs = t.get("rope_scaling") or {}
    dim = t["qk_rope_head_dim"]
    factor, beta_fast, beta_slow = rs.get("factor", 1.0), rs.get("beta_fast", 32), rs.get("beta_slow", 1)
    kinds = (("plain", (0, t["rope_theta"], factor, beta_fast, beta_slow)),
             ("compressed", (rs.get("original_max_position_embeddings", 0), t["compress_rope_theta"], factor, beta_fast,
                             beta_slow)))
    want = json.load(open(os.path.join(kit, "rope.json")))
    assert want["half"] == dim // 2, (want["half"], dim)
    bad = 0
    for kind, a in kinds:
        cos, sin = rope_cs(dim, want["rows"], *a)
        for part, tab in (("cos", cos), ("sin", sin)):
            data = tab.contiguous().view(torch.uint8).numpy().tobytes()
            name = f"rope-{kind}-{part}.f32"
            if not check:
                with open(os.path.join(kit, name), "wb") as fh:
                    fh.write(data)
            ok = hashlib.sha256(data).hexdigest() == want[name]
            bad += not ok
            print(f"{name}: {len(data)} bytes, {'sha256 ok' if ok else 'SHA256 DIFFERS from rope.json'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
