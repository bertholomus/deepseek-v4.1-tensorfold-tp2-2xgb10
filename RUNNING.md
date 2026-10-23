# Running it

What follows is how we run the served build. `scripts/serve.sh` is our served launch with the site values moved to
`config.env`: the four-node recipe's script (v0.3.3) for two nodes, whose preflight also checks each node's RoCE
address (section 2). v0.7.0 changes the engine, the weights, the kernel kit and the launch (README, v0.7.0).

## 1. The engine

Clone our TensorFold fork at the commit in `ENGINE_COMMIT` (branch `deepseek-v41-zig-tp2`). Its parent, `af07f7b`,
holds the served build's source (the tree of `bd1b05f`, which our lane runs), and `ENGINE_COMMIT` adds the fork's
documents: its README note, NOTICE and ATTRIBUTION.md. The branch is the four-node branch `deepseek-v41-zig-tp4` at
its v0.3.1 with nine changes on top (README, v0.7.0). It is not the engine of v0.6.1 (branch `deepseek-v41-zig`):
run the same build on both nodes. Build the native engine for the GB10's aarch64 + sm_121 with Zig 0.17.0 and the
CUDA toolchain of `nvcr.io/nvidia/pytorch:26.07-py3`:

```sh
zig build install native -Dtarget=aarch64-linux-gnu.2.28 -Doptimize=ReleaseFast -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121
zig build test        # the host-side unit tests (no GPU)
```

Copy `zig-out/native/bin/tensorfold-native` and `zig-out/bin/tf-dsv41-lanes` to `ENGINE_DIR` on both nodes, and for
section 2 `zig-out/bin/tf-dsv41-rank-cache` with `tools/dsv41-zig/inputs/node_inputs.sh` and `make_token_map.py`.
The served launch also passes `rdma_gather.fatbin`, the RDMA kernels of `zig/kernels/cuda/dsv41/rdma_gather.cu` built
as a fatbin for sm_121; without `TF_DS_RDMA_KERNELS` the engine uses its embedded copy.

## 2. Weights and run-time inputs (on both nodes)

- `MODEL_DIR`: [bertholomus/DeepSeek-V4.1-Flash-1017-EXL3-2.5bpw](https://huggingface.co/bertholomus/DeepSeek-V4.1-Flash-1017-EXL3-2.5bpw),
  readied for the engine by this repository's `tools/serving_1017.py`:

  ```sh
  python3 tools/serving_1017.py MODEL_DIR            # standard library only, no GPU; reads about 130 MB
  python3 tools/serving_1017.py MODEL_DIR --check    # exit 0 when both files are there and right
  ```

  It adds two files and changes no downloaded one. `serving-1017.safetensors` holds six tensors under the names and
  types the engine reads: the MTP markov head under `mtp.2.markov_head.embed.weight` and `.head.weight` (byte copies of
  the converter's `markov_w1` and `markov_w2`), and the Engram `q_weight` and `k_weight` of layers 1 and 14 widened from
  BF16 to F32 (exact). The engine maps every `*.safetensors` file of the folder in name order, a later file winning, so
  these four replace the BF16 ones. The six are 16-bit in both of our 1017 quants, so the file is the one the
  four-node recipe makes from the 4.0 bpw quant. `chat_template.jinja` is `chat/chat_template.jinja`: the 1017
  checkpoint ships no chat template, and the server needs one for chat requests. Both files' sha256 are in
  `results/served-v070/inputs.txt`.
- `ENGRAM_DIR`: the original checkpoint's shards 47 and 48 of
  [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (the Engram
  tables).
- `CACHE_DIR`: the node's weight file `rank-cache/rank<g>of2-<hash>.bin` (its half of the checkpoint for the two-node
  split, in the loader's order: about 93 GB) and `dsv41_token_map.json` (Engram's compressed token map). Make both
  from the readied checkpoint, one command a node, no GPU:

  ```sh
  ENGINE_DIR/node_inputs.sh MODEL_DIR CACHE_DIR <g> 2   # g: the node's rank, 0 or 1, in the order of NODES
  scripts/serve.sh inputs                               # or: the same on the two nodes at once
  ```

  `node_inputs.sh` runs `tf-dsv41-rank-cache`, the engine's writer of the weight file, and `make_token_map.py`, which
  needs the `tokenizers` package (without it on the host, it runs in `IMAGE` with docker). `CACHE_DIR/rank-cache` must
  not hold another file of the same rank. Each file's name holds the hash of its index; `results/served-v070/inputs.txt`
  lists the two names our served lane loads and the token map's sha256 (the 1017 tokenizer is V4.1-Flash's, so the
  map is the one the four-node recipe loads since its v0.1.1).
- `KIT_DIR`, the kernel kit: this repository's `kit/` (see `kit/README.md`): `aot/` (the family's Triton kernels,
  compiled and captured by the recorder in the engine's `tools/dsv41-zig/rec/`), `cubins/` (its EXL3 kernels),
  `engram.json`, the RoPE tables and their digests (`rope.json`), for images `vision/`, and `grammar/`
  (`libtfgrammar.so`, the grammar library for structured output and forced tool calls, built for aarch64 Linux, with
  its licenses). Git holds all of it but the four RoPE tables and `vision/gate_bias_vl.safetensors` (43 small tensors
  of DeepSeek's 1017 weights). Make those and check every file against `kit/MANIFEST`:

  ```sh
  kit/verify_kit.sh MODEL_DIR [ORIGINAL_DIR]     # ends with "verify_kit: 367 of 367 files match MANIFEST (git 362, rope 4, bias 1)"
  ```

  It computes the RoPE tables on the CPU (in `nvcr.io/nvidia/pytorch:26.07-py3` with docker when the host has no
  PyTorch) and copies the image bias out of the original 1017 checkpoint when you pass its folder as `ORIGINAL_DIR`
  (its index and the 43 shards that hold the bias tensors; our EXL3 quant has 40 of them, not the three MTP layers').
  Without one, take `vision/gate_bias_vl.safetensors` (sha256 `4745da4b...` in `kit/MANIFEST`) from the `kit/` folder
  of [bertholomus/DeepSeek-V4.1-Flash-TensorFold-TP2-2xGB10](https://huggingface.co/bertholomus/DeepSeek-V4.1-Flash-TensorFold-TP2-2xGB10/tree/main/kit),
  which also holds the whole kit as a download; `verify_kit.sh` refuses any other file. The file of v0.6.1 and earlier
  comes from DeepSeek-V4.1-Flash and does not match. Copy the checked `kit/` to `KIT_DIR` on both nodes. A vision
  checkpoint's server refuses to start without `vision/`.
- What rank 0 opens at start. Rank 0 reads its weights from its weight file, but it also opens these, so its node needs
  them even when the other model shards are not there:
  - in `MODEL_DIR`: `config.json`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` (from
    `tools/serving_1017.py`), `model.safetensors.index.json`, and the vision tower's shard,
    `model-00041-of-00041.safetensors` (266 tensors: `vision.*`, `aligner.*`, `image_start`, `image_newline`,
    `image_end`; the server loads the tower at start);
  - in `CACHE_DIR`: `rank-cache/rank0of2-<hash>.bin` and `dsv41_token_map.json`;
  - in `KIT_DIR`: the kit (`aot/aot.json` and the rest of it); in `ENGINE_DIR`: `tensorfold-native` and
    `rdma_gather.fatbin`; `ENGRAM_DIR`.
- The link, on both nodes. The engine's RDMA ring opens GID index 5 of each device in `RDMA_HCA` (port 1), and the
  launch sets `NCCL_IB_GID_INDEX=5` to match; `show_gids` lists each device's GIDs, and index 5 must be the link
  address's RoCE v2 entry. The engine branch before this one, `deepseek-v41-zig`, reads another index from
  `TF_RDMA_GID_INDEX`; this engine line does not, so a link whose RoCE v2 address sits at another index needs that
  branch.

`scripts/serve.sh start` checks these on rank 0's node, and the GID on both nodes, first. When one is missing, it
names it and starts nothing (it does not stop a running lane either). `scripts/serve.sh preflight` runs the check
alone. It finds the vision shard through the index, so it also works with another shard layout.

## 3. Serve

```sh
cp config.env.example config.env     # fill in every <PLACEHOLDER>
scripts/serve.sh preflight           # exit 0 when every file rank 0 opens at start is there and both GIDs are right
scripts/serve.sh print               # what start will run on each node
scripts/serve.sh start               # preflight, then rank 1, then rank 0; /v1/models, then a 4-token chat
scripts/serve.sh check
scripts/serve.sh stop
```

The server listens on `HOST:PORT` of rank 0 with TensorFold's OpenAI-compatible API. Images go in user turns or tool
results as `image_url` parts (data URLs). Rank 0 prints `{"vision": "Pillow 12.3.0 decodes the formats other than
PNG"}` when it serves images. The chat template's header lists the request fields it reads (`thinking_mode` or
`enable_thinking`, `reasoning_effort`, the tools and the message fields).

Structured output takes `response_format` (`{"type": "json_schema", ...}` or `{"type": "json_object"}`),
`guided_choice` or `guided_regex`; with tools offered, `tool_choice` "required" or a named function answers with a
call. Rank 0 loads `TENSORFOLD_GRAMMAR_LIB` (the kit's `grammar/libtfgrammar.so`, mounted at `/kit`) at the first
such request. A streamed reply that has sent nothing for 15 s sends `: keepalive`, an SSE comment clients skip; rank
0's log gets a stats line (the draft head's accepted chains by depth) at most once a minute, and `/metrics` counts
them.

`config.env.example` holds the served settings: the shape (`CONTEXT`, `WINDOW`, `STREAMS`, `ROUND_ROWS`, which
`scripts/serve.sh` gives both ranks as `TF_DS_*` variables and rank 1 as flags too, and `PARALLEL`), `ENGINE_ENV`
(both ranks' other variables) and `RANK0_FLAGS` (rank 0's own flags after the site's `SERVER_FLAGS`). Leave
`TENSORFOLD_REPLY_IDS`, `TF_FILL_CHUNKS`, `TF_DS_SOLO_OVERHEAD` and `TF_DS_NO_COPIES` unset to serve as we do; the
README says what each does when set. `TF_DS_NEIGHBOR_RDMA` is for a physical four-node ring.

The served window, 2,097,152 positions (`WINDOW`), holds one 1,048,576-token request with room for the other
streams. It left 9.4 GiB free on rank 0's node and 13.1 GiB on rank 1's at the lowest on ours, during a 1M-token
prompt; each 1,048,576 positions less frees 0.87 GiB a node.

## 4. Check

`tools/` holds the four-node recipe's clients (v0.3.3), byte for byte but for the two lines of
`tools/serving_1017.py` that name the 2.5 bpw quant: `kit_bench.py` (the speed rows), `long_check.py` (long prompts:
concurrent equals solo, needles, kept-prompt reuse), `needle.py` (needles in one long haystack; with `--shared` the
follow-up questions resume from the kept prompt), `lensweep.py` (prompt lengths), `gates.py` (the API gates), and
`tools/vision/` (the image gate `vision_http.py`, and the reference tools of the four-node lane, whose
`lane-ref.json` and `lane-probe.json` were recorded on other weights and do not apply here). One client is ours and
new: `long_cold.py` runs `long_check.py`'s cases again, each one alone from a cold start (its own `cache_salt`, so
no kept prompt is reused), against a `long_check.py` run's replies to the same prompts sent together
(`--burst` that run's `--out` file). `results/served-v070/` was made with `kit_bench.py`, `gates.py`, `needle.py`,
`long_check.py` and `long_cold.py`. The exact-chat check (71 cases with DeepSeek's reference
renderings), the 16-reply check, the image checks (solo and four at a time) and the checks of structured output,
forced tool calls, the prompt shield, kept states, the fill and the memory are our lane tools and are not in this
release; their reports are in `results/served-v070/`.
