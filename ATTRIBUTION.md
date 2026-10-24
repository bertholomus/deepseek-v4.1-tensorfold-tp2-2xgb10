# Attribution

This recipe and the engine branch behind it (`deepseek-v41-zig-tp2` of our TensorFold fork) were written
independently. They are clean-room with respect to other people's DeepSeek-V4.1 work: the model math is
re-implemented from DeepSeek's MIT inference code and tech report, and no code from other DeepSeek-V4.1 recipes or
kits was read or copied. The engine branch's own `ATTRIBUTION.md` lists every outside source of the engine; this file
lists what the recipe uses.

| Source | License | What we used | How |
|---|---|---|---|
| [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) and [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | MIT, Copyright (c) 2023 DeepSeek | The model; the 1017 checkpoint's original shards 47 and 48 (Engram tables) and its 43 `ffn.gate.bias_vl` tensors (V4.1-Flash's up to v0.6.1); the rules of its reference code for the Engram constants (`kit/engram.json`) and the RoPE rotation (`kit/tools/make_rope.py`); its reference encoding (`encoding.py`), which our chat template renders | Shards loaded at run time, not redistributed. The bias tensors are copied byte for byte into `kit/vision/gate_bias_vl.safetensors` by `kit/tools/make_bias_vl.py`; git does not hold that file. Rules and the chat format re-implemented (`chat/chat_template.jinja` is ours); no code copied |
| [bertholomus/DeepSeek-V4.1-Flash-1017-EXL3-2.5bpw](https://huggingface.co/bertholomus/DeepSeek-V4.1-Flash-1017-EXL3-2.5bpw) | MIT, inherited from the base model | The weights we serve since v0.7.0 (our EXL3 quant, made with ExLlamaV3's converter and Fabiano Anemone's DeepSeek-V4 support for it) | Loaded at run time. Not redistributed. `tools/serving_1017.py` adds six of its tensors under the engine's names and types to a download |
| [Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw) | MIT, inherited from the base model | The weights we served up to v0.6.1 | Loaded at run time. Not redistributed |
| [TensorFold](https://github.com/ashhart/TensorFold) 1.0.12 (1.0.2 in v0.6.0 and v0.6.1; its Python line before them) | Apache-2.0 from TensorFold 0.6.0, Copyright 2026 TensorFold contributors; code written before 0.6.0 keeps its MIT notice (`kit/LICENSES/TensorFold-MIT.txt`) | The serving engine and its OpenAI-compatible server; compiled in `kit/`: its EXL3 kernels (`kit/cubins/`, with our additions) and its grammar shim (`kit/grammar/libtfgrammar.so`) | Forked: [bertholomus/TensorFold `deepseek-v41-zig-tp2`](https://github.com/bertholomus/TensorFold/tree/deepseek-v41-zig-tp2) |
| Our four-node recipe, [bertholomus/deepseek-v4.1-tensorfold-tp4-4xgb10](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp4-4xgb10) (v0.3.3; its kit v0.3.1) | Apache-2.0, same authors | `scripts/`, `tools/`, `chat/chat_template.jinja`, the layout of `RUNNING.md` and the kit's files | Ours, reused; `scripts/serve.sh` made for two nodes, and `tools/kit_bench.py`'s depth rows take `--seed` (v0.7.1) |
| NVIDIA PyTorch container `nvcr.io/nvidia/pytorch:26.07-py3` (PyTorch BSD-3-Clause, Triton MIT, NCCL, Pillow MIT-CMU) | NVIDIA's container license and the components' own licenses | The run-time environment; its PyTorch on the CPU for the RoPE tables (`kit/verify_kit.sh`) | Used as supplied. Not redistributed, except the kernel below |
| [PyTorch](https://github.com/pytorch/pytorch) and [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass) | BSD-3-Clause | PyTorch's memory-efficient attention kernel for the vision tower (`fmha_cutlassF_f32_aligned_64x64_rf_sm80`, built on CUTLASS) | Extracted unmodified from the container's `libtorch_cuda.so`; redistributed in binary form as `kit/vision/torch_fmha_sm120.cubin` with the license texts in `kit/LICENSES/` |
| [Triton](https://github.com/triton-lang/triton) | MIT | The compiler of our kernels in `kit/aot/` | Compiled with; license text in `kit/LICENSES/` |
| [ExLlamaV3](https://github.com/turboderp-org/exllamav3), with Fabiano Anemone's DeepSeek-V4 support ([pull request 449](https://github.com/turboderp-org/exllamav3/pull/449)) | MIT, Copyright (c) 2025 Turboderp | The EXL3 format the kernels in `kit/cubins/` read (through TensorFold's EXL3 module); the converter that made our quant | Format followed; license text in `kit/LICENSES/` |
| [xgrammar](https://github.com/mlc-ai/xgrammar) v0.2.8, with the [dlpack](https://github.com/dmlc/dlpack) and [picojson](https://github.com/kazuho/picojson) headers it bundles | Apache-2.0, Copyright (c) 2024 by XGrammar Contributors (xgrammar); Apache-2.0 (dlpack); BSD-2-Clause, Copyright 2009-2010 Cybozu Labs, Inc. and 2011-2014 Kazuho Oku (picojson) | The grammar library behind structured output and forced tool calls | Compiled unmodified with TensorFold 1.0.12's shim into `kit/grammar/libtfgrammar.so`; redistributed in binary form, with xgrammar's license and notice and dlpack's license in `kit/grammar/` and picojson's in `kit/LICENSES/` |

## Contributors

The engine is the four-node line's v0.3.1 (v0.7.1 merges its v0.3.2), which carries these pull requests to our fork:
[#4](https://github.com/bertholomus/TensorFold/pull/4) by Lucas Peters ([@lucas-peters](https://github.com/lucas-peters)),
the prompt indexer's aligned row blocks; [#18](https://github.com/bertholomus/TensorFold/pull/18) by
[@lepetitprince716-prog](https://github.com/lepetitprince716-prog), the faster load;
[#26](https://github.com/bertholomus/TensorFold/pull/26) by Manrique ([@hashd1ve](https://github.com/hashd1ve)), which
keeps a conversation's prompt states when another one shares its prefix, and ports of Manrique's
[#28](https://github.com/bertholomus/TensorFold/pull/28) (prompts with images keep their states) and
[#29](https://github.com/bertholomus/TensorFold/pull/29) (`TF_FILL_CHUNKS`, off unless set);
[#30](https://github.com/bertholomus/TensorFold/pull/30) by Scott Woolsey Biggart
([@swbiggart](https://github.com/swbiggart)), the neighbor transport (four nodes, off unless set);
[#31](https://github.com/bertholomus/TensorFold/pull/31) by [@mrsergiopascal](https://github.com/mrsergiopascal), a
test and a reproducer of the replay cut's positions; and [#33](https://github.com/bertholomus/TensorFold/pull/33)
and [#34](https://github.com/bertholomus/TensorFold/pull/34) by [@Blackwellboy](https://github.com/Blackwellboy)
(`TF_DS_SOLO_OVERHEAD` and `TF_DS_NO_COPIES`, each off unless set). Its loop guard's novelty signal follows a report
by [@Capicua25x](https://github.com/Capicua25x) ([issue #18](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10/issues/18)),
and its stream keepalive one by [@den13501](https://github.com/den13501) ([issue #12](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10/issues/12)).

The two-node branch adds nine changes on top in v0.7.0, four of them other people's work: LeetJoe's
([@LeetJoe](https://github.com/LeetJoe)) [#17](https://github.com/bertholomus/TensorFold/pull/17), a message's own
image placeholder escaped instead of refused; [@Blackwellboy](https://github.com/Blackwellboy)'s
[#38](https://github.com/bertholomus/TensorFold/pull/38), a refused Engram row id no longer leaks the AIO batch's
buffers; a port of Manrique's [#25](https://github.com/bertholomus/TensorFold/pull/25) (the native side of
[#20](https://github.com/bertholomus/TensorFold/pull/20) by [@willjriley](https://github.com/willjriley)): DeepSeek-V4.1
calls read from their own tags; and Nathan McBride's ([@nmcbride](https://github.com/nmcbride)) upstream
[TensorFold #973](https://github.com/ashhart/TensorFold/pull/973), the template's append in place (two commits). Its
assistant-turn quoting with `TENSORFOLD_REPLY_IDS` follows a report by [@ahinore](https://github.com/ahinore)
([four-node recipe issue #3](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp4-4xgb10/issues/3)). The
engine branch's `ATTRIBUTION.md` credits each of them, and the picked commits keep their authors.
v0.7.1's two commits that carry the tests of Manrique's [#25](https://github.com/bertholomus/TensorFold/pull/25) into one DSML call reader and the
streamed prose name Manrique as co-author. Its rank-stop watch follows careful testing by
[@plyldavid-maker](https://github.com/plyldavid-maker) ([issue #25](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10/issues/25)), and the kit rebuild's jit
table ([#14](https://github.com/bertholomus/TensorFold/pull/14)) a report by [@den13501](https://github.com/den13501) ([issue #15](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10/issues/15)).
Thank you.

The kit's `_topk_sel` variant for N = KK = 256 (prompts of exactly 512 or 513 tokens) was compiled by Lucas Peters
([@lucas-peters](https://github.com/lucas-peters)) with `kit/tools/make_topk_sel.py`, this recipe's
[pull request #13](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10/pull/13). Thank you.

`scripts/serve.sh` and `RUNNING.md` carry the four-node recipe's two fixes from an outside tester's bring-up, by
Niklas Lenz ([niklaslenz.com](https://niklaslenz.com)): a node that is this host runs locally (no ssh key to itself
needed), and the preflight checks every file rank 0 opens, naming any missing one and starting and stopping nothing.
Thank you.

`RUNNING.md` gives the server's path in the build, as this recipe's issues [#23](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10/issues/23) by
[@willjriley](https://github.com/willjriley) and [#24](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10/issues/24) by [@dylandoesai](https://github.com/dylandoesai)
pointed out, and `scripts/serve.sh` gives both ranks their settings from one file, which [#25](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10/issues/25) by
[@plyldavid-maker](https://github.com/plyldavid-maker) showed is needed (with different values the two ranks wait on
each other). Thank you.

The weights' conversion used Fabiano Anemone's ([@anoane](https://github.com/anoane)) DeepSeek-V4 support for
ExLlamaV3 ([turboderp-org/exllamav3 pull request 449](https://github.com/turboderp-org/exllamav3/pull/449), MIT); the
weights' card lists our four changes to it. Thank you.

Our commits to this recipe, and many of ours to the engine branch, name Claude (Anthropic's AI assistant), which
helped write them, in a `Co-Authored-By` trailer.

## About the files in `results/`

All files in `results/` are our own measurements of our own engine, with private host names, addresses and paths
removed. Nothing else in them was changed, except where `results/README.md` says a file holds only a part of a
check's output.

## Names

- "DeepSeek", "DeepSeek-V4.1-Flash" and "DeepSeek-V4.1-Flash-1017" belong to DeepSeek. "DGX Spark" and "GB10" belong
  to NVIDIA. We use the names only to say what this recipe runs and on what hardware.
- This recipe is not affiliated with or endorsed by DeepSeek, NVIDIA, the TensorFold authors, Mia-AiLab or MiaAI-Lab.
