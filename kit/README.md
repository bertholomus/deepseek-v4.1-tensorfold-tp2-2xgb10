# Kernel kit: DeepSeek-V4.1-Flash on TensorFold 1.0 (Zig), v0.6.0

Bertholomus AI. This is the kit our served two-node lane loads, file for file (`MANIFEST` has each file's sha256). It
belongs to the `dsv41` family on [bertholomus/TensorFold, branch `deepseek-v41-zig`](https://github.com/bertholomus/TensorFold/tree/deepseek-v41-zig)
(engine 5e10970a; its code is the served ce20a545). Recipe and run commands: https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10

## Use it

Two copies hold the same bytes (`MANIFEST` lists every file's sha256):

- **git**, folder `kit/` of the recipe repository, at the release tag. It leaves out five generated files: the four
  RoPE tables (136 MB each, over GitHub's file limit) and the image routing bias (DeepSeek weight tensors, which we do
  not commit). Make them with `tools/verify_kit.sh --rope <EXL3_MODEL_DIR> <deepseek-v41-tp2 src> --bias
  <ORIGINAL_CHECKPOINT_DIR>`, which then checks every file.
- **Hugging Face**, folder `kit/` of bertholomus/DeepSeek-V4.1-Flash-TensorFold-TP2-2xGB10: all files. Check it with
  `tools/verify_kit.sh`.

Put the folder on both nodes and point both ranks at it:

- rank 0 (`tensorfold serve`): `TF_DS_KIT=<KIT>` and `TENSORFOLD_CUDA_KERNELS=<KIT>/aot`;
- rank 1 (`tf-dsv41-lanes`): `<KIT>` as its kit argument.

Rank 0 reads `vision/torch_fmha_sm120.cubin`; every rank reads `vision/gate_bias_vl.safetensors`. The kernels are
built for the GB10 (sm_121) and the `nvcr.io/nvidia/pytorch:26.07-py3` container.

## What is in it

See `KIT-LICENSES.md` for each file's source and license. In short: our Triton kernels as binaries (`aot/`), the
EXL3 kernels (`cubins/`), Engram's hash constants (`engram.json`), the RoPE tables (`rope-*.f32`, `rope.json`),
PyTorch's attention kernel for the vision tower (`vision/torch_fmha_sm120.cubin`, BSD-3-Clause), and the gates' image
bias from DeepSeek's weights (`vision/gate_bias_vl.safetensors`, 43 small f32 tensors, MIT). The model weights
themselves are not here: serve [Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw).

## Make parts of it yourself

- `tools/make_bias_vl.py <ORIGINAL_CHECKPOINT_DIR> vision/gate_bias_vl.safetensors` copies the 43 bias tensors from
  your own copy of deepseek-ai/DeepSeek-V4.1-Flash (standard library only). Expected sha256 `0cbdebfb...` (see `MANIFEST`).
- `PYTHONPATH=<deepseek-v41-tp2 checkout>/src python3 tools/make_rope.py <EXL3_MODEL_DIR> .` recomputes the four RoPE
  tables on the CPU and checks them against `rope.json`.
- `tools/make_topk_sel.py compile|merge` compiles `_topk_sel` variants no recording captured (N=KK=256: prompts of
  exactly 512 or 513 tokens) from the deepseek-v41-tp2 source, offline in the engine image, and adds them to the kit.
  It first checks that every `_topk_sel` variant already here comes out byte for byte; see its header.
- The engine branch's `tools/dsv41-zig/build_kit.sh` builds the whole kit from source on two GB10 nodes, and
  `tools/dsv41-zig/build_inputs.sh` makes each node's inputs (rank weight cache, token map); see their headers.
