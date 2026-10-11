#!/usr/bin/env python3
"""Add general (no-alignment-assumption) `_index_score` variants to a DeepSeek-V4.1 TensorFold AOT kit.

Why: the kit's aot.json has, for most constexpr sets of `_index_score`, only variants that assume 16-byte alignment
(div16) of the VIS / KB / CAND / TMAX pointers and of n / cs / nt. Around 133k-136k prompt tokens a launch gets
unaligned buffers with one of those constexpr sets, `Set.find` (zig/src/cuda/aot.zig) finds no variant whose
assumptions all hold, even with its "assumes less" fallback, and the request fails with MissingTritonVariant.

What: for every distinct constexpr set of `_index_score` already in aot.json, compile one more variant that keeps
div16 only on the weight-like pointers Q / K / KS / Wt and assumes nothing on the rest. The engine prefers an exact
variant and falls back to these only when nothing more specialised fits, so normal launches are unchanged.

Run it INSIDE the engine-py image (Triton 3.7.1, e.g. dsv41-py:90b68e0), with the engine-py source mounted at the
dist-packages path the kit was captured from (the cubins embed it):

  docker run --rm --network none \
    -v <engine-py>/src/tensorfold:/usr/local/lib/python3.12/dist-packages/tensorfold:ro \
    -v <kit>/aot:/aot:ro -v <out>:/out -v $PWD/gen_general_index_score.py:/gen.py:ro \
    --entrypoint python3 dsv41-py:90b68e0 /gen.py --aot /aot/aot.json --out /out

  [--verify HASH_PREFIX]  first recompile one existing variant and compare its SASS with the kit's cubin (needs
                          cuobjdump in the image) - a check that the toolchain matches the kit's.

Output: <out>/aot.json (the kit's entries plus the new ones) and <out>/cubins/<hash>.cubin (the new cubins only).
Install on EVERY rank: back up <kit>/aot/aot.json, copy <out>/aot.json over it, copy <out>/cubins/*.cubin into
<kit>/aot/cubins/. On a launch that needs one, the engine logs once:
  warning: _index_score: a launch whose specialization no captured variant has runs variant <hash> (assumes less)
"""
import argparse, json, subprocess, sys
from pathlib import Path

sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from tensorfold.families.deepseek_v41.cuda import kernels as K

FN = K._index_score
TARGET = GPUTarget("cuda", 121, 32)  # GB10 (sm_121)
KEEP_ALIGNED = ("Q", "K", "KS", "Wt")


def compile_variant(params, consts, num_warps, pdl):
    sig, cexpr, attrs = {}, {}, {}
    runtime = {p["name"]: p for p in params}
    for i, name in enumerate(FN.arg_names):
        if name in runtime:
            sig[name] = runtime[name]["type"]
            if runtime[name]["div16"]:
                attrs[(i,)] = [["tt.divisibility", 16]]
        elif name in consts:
            sig[name], cexpr[(i,)] = "constexpr", consts[name]
        else:
            sig[name], cexpr[(i,)] = "constexpr", FN.params[i].default
    return triton.compile(ASTSource(FN, sig, cexpr, attrs), target=TARGET,
                          options={"num_warps": num_warps, "num_ctas": 1, "launch_pdl": pdl})


def entry(ck, params, consts):
    md = ck.metadata
    get = (lambda k, d=None: md.get(k, d)) if isinstance(md, dict) else (lambda k, d=None: getattr(md, k, d))
    return {"fn": "_index_score", "hash": ck.hash, "name": get("name"), "num_warps": get("num_warps"),
            "num_ctas": get("num_ctas", 1), "shared": get("shared", 0),
            "global_scratch": get("global_scratch_size", 0), "global_align": get("global_scratch_align", 1),
            "profile_scratch": get("profile_scratch_size", 0), "pdl": bool(get("launch_pdl", False)),
            "params": params, "consts": {n: {"int": int(v)} for n, v in consts.items()}}


def consts_of(e):
    return {n: v["int"] for n, v in e["consts"].items()}


def cuobjdump():
    import os, shutil
    tb = Path(triton.__file__).parent / "backends" / "nvidia" / "bin" / "cuobjdump"
    for c in (shutil.which("cuobjdump"), "/usr/local/cuda/bin/cuobjdump", str(tb)):
        if c and os.path.exists(c):
            return c
    sys.exit("cuobjdump not found: run without --verify")


def sass(path):
    r = subprocess.run([cuobjdump(), "-sass", str(path)], capture_output=True, text=True)
    lines = [l.split("*/", 1)[-1].split(";")[0].strip() for l in r.stdout.splitlines() if l.strip().startswith("/*") and "*/" in l]
    if r.returncode != 0 or not lines:  # an empty listing on both sides must not pass as "identical"
        sys.exit(f"cuobjdump gave no SASS for {path} (exit {r.returncode}): {r.stderr.strip()[:300]}")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aot", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--verify")
    a = ap.parse_args()
    aot = json.load(open(a.aot))
    out = Path(a.out); (out / "cubins").mkdir(parents=True, exist_ok=True)
    ks = [k for k in aot["kernels"] if k["fn"] == "_index_score"]
    print("triton", triton.__version__, "| _index_score variants in the kit:", len(ks))
    if a.verify:
        e = next((k for k in ks if k["hash"].startswith(a.verify)), None)
        if e is None:
            sys.exit(f"--verify {a.verify}: no _index_score variant in {a.aot} has a hash starting with it")
        ck = compile_variant(e["params"], consts_of(e), e["num_warps"], e["pdl"])
        mine = out / "verify.cubin"; mine.write_bytes(ck.asm["cubin"])
        theirs = sass(Path(a.aot).parent / "cubins" / f'{e["hash"]}.cubin')
        same = sass(mine) == theirs
        print("verify", e["hash"][:12], "SASS identical:", same, f"({len(theirs)} SASS lines)")
        if not same:
            sys.exit("toolchain does not reproduce the kit's SASS: stop here")
    new, seen = [], set()
    for e in ks:
        c = consts_of(e)
        key = json.dumps(c, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        params = [dict(p, div16=(p["div16"] and p["name"] in KEEP_ALIGNED)) for p in e["params"]]
        if any(consts_of(k) == c and [p["div16"] for p in k["params"]] == [p["div16"] for p in params] for k in ks):
            print("already general:", key[:80]); continue
        ck = compile_variant(params, c, e["num_warps"], e["pdl"])
        (out / "cubins" / f"{ck.hash}.cubin").write_bytes(ck.asm["cubin"])
        new.append(entry(ck, params, c))
        print("compiled", ck.hash[:12], key[:80])
    have = {k["hash"] for k in aot["kernels"]}
    aot["kernels"] += [n for n in new if n["hash"] not in have]
    aot["kernels"].sort(key=lambda x: (x["fn"], x["hash"]))
    (out / "aot.json").write_text(json.dumps(aot, indent=1) + "\n")
    print(f"{len(new)} new variants -> {out}/aot.json ({len(aot['kernels'])} kernels) and {out}/cubins/")


if __name__ == "__main__":
    main()
