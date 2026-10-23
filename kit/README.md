# Kernel kit: DeepSeek-V4.1-Flash-1017 on TensorFold 1.0 (Zig), two GB10 nodes

This folder is the kit our served two-node lane loads (engine `bd1b05f`, whose source with the fork's documents is
`ENGINE_COMMIT` `9298495`; the 2.5 bpw 1017 weights). `MANIFEST` lists all 367 files with their sizes and sha256. 366
are byte for byte the served copies on our two nodes; `aot/aot.json` differs only in its packer's note of where its
kernel sets came from, which named private folders (the engine does not read the note, and the kernels it lists are
the same). Git holds 362 of the files (141 MB). The other five are made from public weights instead of committed:

- the four RoPE tables `rope-{plain,compressed}-{cos,sin}.f32` (142,606,336 bytes each), which `tools/make_rope.py`
  computes on the CPU from the checkpoint's `config.json`;
- `vision/gate_bias_vl.safetensors` (67,192 bytes), the MoE gates' routing bias for image tokens: 43 small f32
  tensors of DeepSeek-V4.1-Flash-1017's weights, which `tools/make_bias_vl.py` copies out of its original checkpoint.
  The file of v0.6.1 and earlier came from DeepSeek-V4.1-Flash and does not match.

The `kit/` folder of our Hugging Face repo
[bertholomus/DeepSeek-V4.1-Flash-TensorFold-TP2-2xGB10](https://huggingface.co/bertholomus/DeepSeek-V4.1-Flash-TensorFold-TP2-2xGB10/tree/main/kit)
holds the same files, those five included, for a download instead of a build.

## Make and check it

```sh
kit/verify_kit.sh <EXL3_MODEL_DIR> [<ORIGINAL_CHECKPOINT_DIR>]
```

This makes the RoPE tables (PyTorch on the CPU; without a local `torch` it runs with docker in
`nvcr.io/nvidia/pytorch:26.07-py3`, whose PyTorch made the served tables), and the bias file when you pass DeepSeek's
original checkpoint ([deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)).
Then it checks every file against `MANIFEST` and ends with `verify_kit: 367 of 367 files match MANIFEST (git 362,
rope 4, bias 1)` and exit 0. The bias tensors sit in 43 of the original checkpoint's 48 shards (our EXL3 quant holds
40 of them, not the three MTP layers'), so without a local copy of it take `vision/gate_bias_vl.safetensors` from the
Hugging Face `kit/` instead (`MANIFEST` has its sha256). The EXL3 model folder only has to hold `config.json`.
`sha256sum -c SHA256SUMS` checks the same list.

## Use it

Point both ranks at this folder (the recipe's `KIT_DIR`):

- rank 0 (`tensorfold-native serve`): `TF_DS_KIT=<KIT_DIR>`, `TENSORFOLD_CUDA_KERNELS=<KIT_DIR>/aot` and
  `TENSORFOLD_GRAMMAR_LIB=<KIT_DIR>/grammar/libtfgrammar.so`;
- rank 1 (`tf-dsv41-lanes`): `<KIT_DIR>` as the kit argument.

The kernels are built for the GB10 (sm_121) and the `nvcr.io/nvidia/pytorch:26.07-py3` container, the grammar
library for aarch64 Linux (glibc 2.18 or later; its C++ runtime is linked in). The server of this vision checkpoint
refuses to start without `vision/`. When `TENSORFOLD_GRAMMAR_LIB` is unset, the server looks for `libtfgrammar.so` in
its `../lib`, beside itself, then on the loader's path; with no library it answers structured requests and forced tool
calls with HTTP 400 naming the library, and every other request as before.

## What is in it

`KIT-LICENSES.md` gives each file's source and license; the license texts are in `LICENSES/` and, for the grammar
library, beside it in `grammar/`. In short: our Triton kernels as binaries (`aot/`), the EXL3 kernels (`cubins/`),
Engram's hash constants (`engram.json`), the RoPE tables and their digests (`rope.json`), PyTorch's attention kernel for
the vision tower (`vision/torch_fmha_sm120.cubin`), the image bias, and libtfgrammar (`grammar/`).

It is the kit of our four-node recipe
([bertholomus/deepseek-v4.1-tensorfold-tp4-4xgb10](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp4-4xgb10)
v0.3.1, the engine line this release builds on), file for file, with five more Triton kernel binaries from our
two-node kit v0.6.1: `_gather_tiles` (BN 64, KK 512) and `_topk_sel` for N = KK = 32, 64, 128 and 256. Lucas Peters
([@lucas-peters](https://github.com/lucas-peters)) compiled the N = KK = 256 one, for prompts of exactly 512 or 513
tokens, with `tools/make_topk_sel.py` (recipe pull request #13). 195 of the 367 files are byte for byte the same as in
our two-node kit v0.6.1 with #13 (`MANIFEST` marks them `tp2`). Against that kit, v0.7.0 has the four-node line's
`aot.json` and 162 more Triton kernel binaries, its EXL3 kernels (`cubins/`), the 1017 checkpoint's image bias and the
grammar library; `verify_kit.sh` moves from `tools/` to the kit's top folder.
The model weights are not here: serve
[bertholomus/DeepSeek-V4.1-Flash-1017-EXL3-2.5bpw](https://huggingface.co/bertholomus/DeepSeek-V4.1-Flash-1017-EXL3-2.5bpw).

The engine branch carries the tools that made this folder (`tools/dsv41-zig/kit/`): `pack_kit.py`, which copies a
lane's kit and writes `MANIFEST` and `SHA256SUMS`, the two generators and `verify_kit.sh`. `tools/make_topk_sel.py` is
how the N = KK = 256 `_topk_sel` variant was compiled, from the `deepseek-v41-tp2` branch's source, for the v0.6.1
kit; its `merge` step writes that kit's `MANIFEST` format, not this one's.
