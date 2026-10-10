#!/usr/bin/env python3
"""make_topk_sel.py compile|merge: _topk_sel variants the recordings missed, compiled offline and merged into the kit.

compile runs as root in a throwaway container of the deepseek-v41-tp2 engine image (github.com/bertholomus/TensorFold,
branch deepseek-v41-tp2, in nvcr.io/nvidia/pytorch:26.07-py3; no GPU needed), because it replaces the installed package:
  python3 make_topk_sel.py compile --tf <deepseek-v41-zig checkout> --src <deepseek-v41-tp2 checkout>/src/tensorfold \
      --kit <KIT>/aot --out <OUT> --add 256x256
The kit's cubins carry kernels.py's path, _topk_sel's lines and the file's size and mtime in their line tables, and
Triton's hash covers _topk_sel's source and first line. So the package is installed at the kit's path first, with
kernels.py laid out as the kit's line tables give it (blank lines before _topk_sel, a trailing comment to that size,
that mtime; no statement changes). Each variant then goes through Triton's own JIT path as topk_select's launch
compiles it on a GB10 (sm_121, warp 32, num_warps 8 if N > 1024 else 4, PDL) and is packed by the engine branch's
tools/zig/triton_aot_manifest.py and zig/tests/cuda/nemotron/aot_pack.py, as tools/dsv41-zig/build_kit.sh packs a
recording. Every _topk_sel variant the kit already has must come out byte for byte (its aot.json entry and its cubin)
before any --add NxKK is written to OUT/entries.json and OUT/cubins/<hash>.cubin.

merge (standard library only) adds OUT's variants to a kit folder: their cubins, aot.json in aot_pack.py's order and
format, MANIFEST's sums:
  python3 make_topk_sel.py merge --out <OUT> --kit <KIT>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

PACKAGE = Path("/usr/local/lib/python3.12/dist-packages/tensorfold")  # where the kit's image had it installed
KERNELS = PACKAGE / "families/deepseek_v41/cuda/kernels.py"


class OfflineDriver:
    """triton.runtime.driver.active for compiling only: the GB10's target, no CUDA call (nothing is launched)."""

    def __init__(self):
        from triton.backends.compiler import GPUTarget
        self.target = GPUTarget("cuda", 121, 32)

    def get_current_device(self):
        return 0

    def get_current_stream(self, device=None):
        return 0

    def get_current_target(self):
        return self.target


def sections(elf: bytes) -> dict[str, bytes]:
    """An ELF64 file's section contents by name."""
    shoff = struct.unpack_from("<Q", elf, 0x28)[0]
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", elf, 0x3A)
    hdrs = [struct.unpack_from("<IIQQQQIIQQ", elf, shoff + i * shentsize) for i in range(shnum)]
    names = hdrs[shstrndx][4]
    return {elf[names + h[0]:elf.index(b"\0", names + h[0])].decode(): elf[h[4]:h[4] + h[5]] for h in hdrs}


def uleb(b: bytes, i: int) -> tuple[int, int]:
    v = shift = 0
    while True:
        x = b[i]
        i += 1
        v |= (x & 0x7F) << shift
        shift += 7
        if x < 0x80:
            return v, i


def kit_layout(cubin: Path) -> tuple[int, int, int]:
    """(_topk_sel's first line, the file's size, its mtime) as a kit cubin's line tables give kernels.py."""
    line = sections(cubin.read_bytes())[".debug_line"]
    i = line.index(b"kernels.py\0") + len(b"kernels.py\0")
    _, i = uleb(line, i)  # its directory
    mtime, i = uleb(line, i)
    size, _ = uleb(line, i)
    text = subprocess.run(["nvdisasm", "--print-line-info", str(cubin)], capture_output=True, text=True, check=True)
    first = min(int(x) for x in re.findall(r'kernels\.py", line (\d+)', text.stdout))
    return first, size, mtime


def lay_out(src: Path, first: int, size: int, mtime: int) -> None:
    """The public package installed at PACKAGE, kernels.py laid out as the kit's: _topk_sel's def at line `first`,
    `size` bytes, mtime `mtime` (blank lines and a trailing comment: no statement changes)."""
    if PACKAGE.resolve() in (src.resolve(), *src.resolve().parents):
        raise SystemExit(f"--src {src} is the installed package itself: give a checkout's src/tensorfold")
    if PACKAGE.exists():
        shutil.rmtree(PACKAGE)
    shutil.copytree(src, PACKAGE)
    lines = KERNELS.read_text().splitlines(keepends=True)
    at = next(i for i, l in enumerate(lines) if l.startswith("def _topk_sel("))
    if first <= at or not lines[at - 1].startswith("@triton.jit"):
        raise SystemExit(f"_topk_sel is at line {at + 1}: it cannot move to line {first}")
    lines[at - 1:at - 1] = ["\n"] * (first - 1 - at)
    text = "".join(lines)
    pad = size - len(text.encode())
    if pad < 2:
        raise SystemExit(f"kernels.py is {len(text.encode())} bytes: it cannot be laid out to {size}")
    KERNELS.write_text(text + "#" + " " * (pad - 2) + "\n")
    os.utime(KERNELS, (mtime, mtime))


def compile_topk_sel(n: int, k: int):
    """kernels.topk_select's launch for int64 keys [rows, n] (contiguous, 16-byte aligned) and k, compiled only."""
    import torch
    from tensorfold.families.deepseek_v41.cuda import kernels
    fn = kernels._topk_sel
    kernel = fn.warmup(torch.int64, torch.int64, torch.int64, N=n, KK=k, num_warps=8 if n > 1024 else 4,
                       **kernels._pdl(), grid=(1,))
    return fn, kernel


def pack(fn, kernel, cache: Path, work: Path) -> tuple[dict, bytes]:
    """The kernel as an aot.json entry and its cubin, through the kit's manifest and packing tools."""
    import aot_pack
    import triton_aot_manifest as tam
    h = kernel.hash
    launches = work / f"launches-{h}.json"
    launches.write_text(json.dumps({"kernels": {h: tam.Recorder._info(fn, kernel)}, "phases": {}, "grids": {},
                                    "sites": {}}, default=str))
    manifest = tam.build(cache, launches, work / f"manifest-{h}.json", None)
    (k,) = manifest["kernels"]
    jit = {k["function"]: {"do_not_specialize": [p.name for p in fn.params if p.do_not_specialize]}}
    entry = aot_pack.entry(k, jit, cache, work)
    return entry, (work / "cubins" / f"{h}.cubin").read_bytes()


def compile_cmd(a) -> int:
    kit = Path(a.kit)
    have = [e for e in json.loads((kit / "aot.json").read_text())["kernels"] if e["fn"] == "_topk_sel"]
    first, size, mtime = kit_layout(kit / "cubins" / f"{have[0]['hash']}.cubin")
    lay_out(Path(a.src), first, size, mtime)
    print(f"kernels.py laid out as the kit's: _topk_sel at line {first}, {size} bytes, mtime {mtime}")
    sys.path[:0] = [f"{a.tf}/tools/zig", f"{a.tf}/zig/tests/cuda/nemotron"]
    cache = Path(tempfile.mkdtemp(prefix="triton-cache-"))
    os.environ["TRITON_CACHE_DIR"] = str(cache)  # a fresh cache: every variant compiled here, none reused
    from triton.runtime.driver import driver
    driver.set_active(OfflineDriver())
    from tensorfold.families.deepseek_v41.cuda import kernels
    if Path(kernels.__file__) != KERNELS:
        raise SystemExit(f"kernels.py imported from {kernels.__file__}, not {KERNELS}")
    work = Path(tempfile.mkdtemp(prefix="pack-"))
    (work / "cubins").mkdir()
    bad = 0
    for want in sorted(have, key=lambda e: (e["consts"]["N"]["int"], e["consts"]["KK"]["int"])):
        n, k = want["consts"]["N"]["int"], want["consts"]["KK"]["int"]
        entry, cubin = pack(*compile_topk_sel(n, k), cache, work)
        same = entry == want and cubin == (kit / "cubins" / f"{want['hash']}.cubin").read_bytes()
        bad += not same
        print(f"check N={n} KK={k}: {entry['hash'][:12]} {'byte-identical to the kit' if same else 'DIFFERS'}")
    if bad:
        print(f"{bad} of {len(have)} kit variants not reproduced: adding none")
        return 1
    out = Path(a.out)
    (out / "cubins").mkdir(parents=True, exist_ok=True)
    added = []
    for spec in a.add:
        n, k = map(int, spec.lower().split("x"))
        entry, cubin = pack(*compile_topk_sel(n, k), cache, work)
        if entry["global_scratch"] or entry["profile_scratch"]:
            raise SystemExit(f"N={n} KK={k} needs scratch memory (the engine loads no such variant)")
        (out / "cubins" / f"{entry['hash']}.cubin").write_bytes(cubin)
        added.append(entry)
        print(f"added N={n} KK={k}: hash {entry['hash']}, shared {entry['shared']}, num_warps {entry['num_warps']}, "
              f"cubin sha256 {hashlib.sha256(cubin).hexdigest()}")
    (out / "entries.json").write_text(json.dumps(added, indent=1) + "\n")
    return 0


def merge_cmd(a) -> int:
    out, kit = Path(a.out), Path(a.kit)
    path = kit / "aot" / "aot.json"
    text = path.read_text()
    meta = json.loads(text)
    if json.dumps(meta, indent=1) + "\n" != text:
        raise SystemExit(f"{path} is not aot_pack.py's formatting: not rewriting it")
    have = {e["hash"] for e in meta["kernels"]}
    new = [e for e in json.loads((out / "entries.json").read_text()) if e["hash"] not in have]
    for e in new:
        cubin = (out / "cubins" / f"{e['hash']}.cubin").read_bytes()
        (kit / "aot" / "cubins" / f"{e['hash']}.cubin").write_bytes(cubin)
    meta["kernels"] = sorted(meta["kernels"] + new, key=lambda x: (x["fn"], x["hash"]))  # aot_pack.py's order
    path.unlink()  # a new file: a hard-linked copy of the kit keeps its original
    path.write_text(json.dumps(meta, indent=1) + "\n")
    # MANIFEST: sha256sum lines; aot.json's sum updated, the new cubins' added in path order
    lines = [l.split("  ", 1) for l in (kit / "MANIFEST").read_text().splitlines()]
    sums = {p: s for s, p in lines}
    sums["aot/aot.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
    for e in new:
        p = f"aot/cubins/{e['hash']}.cubin"
        sums[p] = hashlib.sha256((kit / p).read_bytes()).hexdigest()
    paths = [p for _, p in lines]
    order = sorted(sums) if paths == sorted(paths) else paths + sorted(set(sums) - set(paths))
    (kit / "MANIFEST").unlink()
    (kit / "MANIFEST").write_text("".join(f"{sums[p]}  {p}\n" for p in order))
    print(f"merged {len(new)} variant(s) into {kit}: {[e['hash'][:12] for e in new]}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compile")
    c.add_argument("--tf", required=True, help="a TensorFold checkout of branch deepseek-v41-zig (the kit tools)")
    c.add_argument("--src", required=True, help="the deepseek-v41-tp2 package to install (a checkout's src/tensorfold)")
    c.add_argument("--kit", required=True, help="the kit's aot folder (aot.json, cubins/)")
    c.add_argument("--out", required=True)
    c.add_argument("--add", action="append", default=[], metavar="NxKK", help="a variant to compile (repeatable)")
    m = sub.add_parser("merge")
    m.add_argument("--out", required=True, help="compile's output folder")
    m.add_argument("--kit", required=True, help="the kit folder to add them to (its aot/, MANIFEST)")
    a = ap.parse_args()
    return compile_cmd(a) if a.cmd == "compile" else merge_cmd(a)


if __name__ == "__main__":
    raise SystemExit(main())
