# DeepSeek-V4.1-Flash-1017 on 2× DGX Spark (GB10) — our TensorFold TP2 engine

**Bertholomus AI recipe: DeepSeek-V4.1-Flash-1017 served tensor-parallel over two NVIDIA DGX Spark / GB10 nodes by the
model family we wrote for [TensorFold](https://github.com/ashhart/TensorFold) (by ashhart).**

> **Weights (v0.7.0 on):** [bertholomus/DeepSeek-V4.1-Flash-1017-EXL3-2.5bpw](https://huggingface.co/bertholomus/DeepSeek-V4.1-Flash-1017-EXL3-2.5bpw)
> (MIT), our EXL3 quant of [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
> (MIT, © 2023 DeepSeek), sized for two nodes. Up to v0.6.1 this recipe served
> [Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw)
> (MIT), an EXL3 quant by Mia-AiLab of [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash).
> This repository holds no weights. The Engram tables are read at run time from DeepSeek's original shards 47 and 48.

**Engine (v0.7.0 on):** [bertholomus/TensorFold, branch `deepseek-v41-zig-tp2`](https://github.com/bertholomus/TensorFold/tree/deepseek-v41-zig-tp2),
Bertholomus AI's `dsv41` family on TensorFold 1.0.12's native Zig engine: our four-node branch at its v0.3.1 with the
two-node recipe's changes on top, merged with that branch's v0.3.2 in v0.7.1. Attribution: [`ATTRIBUTION.md`](https://github.com/bertholomus/TensorFold/blob/deepseek-v41-zig-tp2/ATTRIBUTION.md)
of the branch, and this repository's `ATTRIBUTION.md` and `NOTICE`. v0.6.0 and v0.6.1 ran
[branch `deepseek-v41-zig`](https://github.com/bertholomus/TensorFold/tree/deepseek-v41-zig) (TensorFold 1.0.2). The
Python engine of v0.1.0-v0.5.1 stays on [branch `deepseek-v41-tp2`](https://github.com/bertholomus/TensorFold/tree/deepseek-v41-tp2)
(design, report and attribution in [`tools/dsv41/`](https://github.com/bertholomus/TensorFold/tree/deepseek-v41-tp2/tools/dsv41)).
Clean-room: the model math is re-implemented from DeepSeek's MIT inference code and tech report; no code from other
DeepSeek-V4.1 recipes or kits was read or copied.

**Run it:** [`RUNNING.md`](RUNNING.md): the engine build, the weights and run-time inputs, the kernel kit (`kit/`),
and the launch (`scripts/serve.sh`, `config.env.example`). Our measurements: [`results/`](results/README.md). The
four-node recipe: [bertholomus/deepseek-v4.1-tensorfold-tp4-4xgb10](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp4-4xgb10).

## v0.7.1 (2026-10-23): faster long prompts on the four-node line's v0.3.2, with less scratch memory

- **Engine:** `deepseek-v41-zig-tp2` merges the four-node branch `deepseek-v41-zig-tp4` at its v0.3.2: prompt passes
  of 4,096 rows from position 8,192 on (2,048 below it, `TF_DS_PASS_FROM`), with the gate and indexer GEMMs split at
  2,048 rows and resume points kept on the 2,048-row grid; a stepped prompt pass that runs on while nothing else
  waits; and v0.3.2's `cache_salt` (every word of a salted key goes through the salt), which here also keeps a
  conversation's salt on its compaction notes, as v0.7.0's did. Seven commits of ours come with it:
  `TF_DS_PROMPT_ALIAS=1` (served), which puts the prompt pass's large scratch in five shared regions: 1.9 GiB a node
  for the 4,096-row passes instead of 4.6, less than v0.7.0's 2.5 GiB for its 2,048-row ones; one DSML call reader for
  both of the server's parsers (a call to a tool the request did not offer is a call) and streamed prose that hides a
  DeepSeek-V4.1 block as the parser reads it, both with the tests of Manrique's
  [#25](https://github.com/bertholomus/TensorFold/pull/25); a rank that stops stops the others, with a line that names
  it (a follower whose GPU context failed among them), after careful testing by
  [@plyldavid-maker](https://github.com/plyldavid-maker) ([issue
  #25](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10/issues/25)); the kit rebuild's jit table
  that sees the family's namespace subpackage ([#14](https://github.com/bertholomus/TensorFold/pull/14)), after a
  report by [@den13501](https://github.com/den13501) ([issue
  #15](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10/issues/15)); and documentation addresses in
  two tests.
- **The same weights, kit and launch** as v0.7.0: `config.env.example` adds `TF_DS_PROMPT_ALIAS=1` to `ENGINE_ENV`,
  and the window is 2,097,152 positions.
- **The same replies:** the bench's six greedy replies are v0.7.0's, and after three 131,072-token prompts the alias
  on, the alias off and v0.7.0 write the same text, byte for byte (`results/served-v071/alias/` and `ab-v070/`).
- `TF_FILL_MANY=1` (with `TF_FILL_CHUNKS=1`) admits a request while a prompt pass fills; on our pair it measured level
  with the served settings, so we leave both unset. `tools/kit_bench.py depth` takes `--seed`, so two builds decode
  the same prompts.

Gates, all passed on the served build (47 checks, `results/served-v071/gate/result.txt`), the same checks as v0.7.0's
with the same counts: exact chat 71/71 against DeepSeek's reference encoding; the six API gates; 16 replies as a burst
and staggered == solo; images 6/6, 9/9, and four at a time 18/18 == solo; forced tool calls 10/10 and automatic ones
6/6, each forced row a call of an offered tool within its schema; structured output 11/11, and bursts and staggered
runs of 12 == solo; the prompt shield 14/14; the keepalive; kept states, the fill and the resume probe, each with
v0.7.0's replies; and 1M: 3/3 needles found in a 1,039,836-token prompt, the second and third asks resuming 1,038,336
tokens of it from the kept state. A second launch at the same settings gave the same replies on every check. The
lowest free memory during the 1M prompt: 9.1 GiB on rank 0's node, 13.4 GiB on rank 1's in the gate's samples, one
every 15 s (v0.7.0's gate: 9.4 and 13.1 GiB). Read once a second after the gate's own steps before its 1M prompt
(`results/served-v071/memtrace/gate-steps/`): 9.7 and 13.7 GiB, and v0.7.0 9.7 and 13.1 GiB; on fresh launches
(`memtrace/fresh/`): 10.6 and 14.0 GiB, and v0.7.0 10.2 and 13.6 GiB. After the gate's steps v0.7.1 starts the 1M
prompt with 0.5 GiB more free memory on rank 0's node and dips 0.6 GiB deeper at its start; the difference is gone
within two minutes. Start to ready: 19 s (19 and 19 s for the gate's launches; v0.7.0: 19 s on the same count).

Four long prompts at once, on the served port after the promotion (`tools/long_check.py` and `tools/long_cold.py`,
`results/served-v071/long/` and `long-cold/`): four prompts of 159,925 tokens, each with a needle, sent together gave
the same replies as each one sent alone from a cold start (5 of 5, a seeded-sampled reply among them) and as each one
resumed from its kept state; drafting off gave the same reply as drafting on, and every needle was found. The lowest
free memory meanwhile: 12.6 GiB on rank 0's node, 13.7 GiB on rank 1's (v0.7.0: 10.1 and 11.6 GiB).

Measured on two GB10 nodes (TP2), the served build at the served settings, our own client (`tools/kit_bench.py`),
greedy, DSpark drafting, median of 3 unless noted (`results/served-v071/rows/`). The v0.7.0 column is v0.7.0's own
table below, but for decode after a 128K prompt:

| `--context 1048576 --parallel 4` | v0.7.0 tok/s | v0.7.1 tok/s |
|---|---|---|
| Decode, set a: code / prose / structured (512 tokens) | 106.9 / 61.0 / 119.2 | 106.3 / 60.8 / 119.2 |
| Decode, set b: code / prose / structured (384 tokens) | 113.6 / 68.3 / 161.4 | 113.2 / 68.1 / 160.7 |
| 2 streams, set b, aggregate (median of 9) | 96.2 | 96.0 |
| 4 streams, set b, aggregate (median of 9) | 135.2 | 135.2 |
| 4 streams sustained 90 s, set b, aggregate | 153.1 | 152.8 |
| Prompt 8K / 32K / 64K / 128K | 2,452 / 2,490 / 2,462 / 2,428 | 2,451 / 2,574 / 2,564 / 2,547 |
| Decode after a 128K prompt (median of 3 prompts, seeds 1023-1025) | 99.7 | 100.4 |
| Prompt 93,335 tokens | 2,136 | 2,355 |

The prompt rows repeat each length's words behind a fresh prefix (`--fixed-words`), so the second and third reps read
Engram rows the first one warmed; the first reps alone gave 2,442 / 2,566 / 2,566 / 2,546 tok/s (v0.7.0: 2,453 / 2,488
/ 2,462 / 2,428). Decode after 128K now reads three fixed prompts (`kit_bench.py depth --seed 1023`; v0.7.0 drew new
ones each run, so its README's 101.5 came from other prompts): its v0.7.0 column is v0.7.0 on the same three prompts
(`results/served-v071/ab-v070/`), where both builds wrote the same text. The three prompts decoded at 54.2 / 115.0 /
100.4 tok/s (v0.7.0: 54.0 / 115.0 / 99.7). The 93,335-token prompt is the served port's own bench after the promotion
(`results/served-v071/prefill-93335.json`).

## v0.7.0 (2026-10-23): DeepSeek-V4.1-Flash-1017 at 2.5 bpw, on the four-node engine line

- **New weights:** our EXL3 quant of DeepSeek-V4.1-Flash-1017, 2.52 bpw over the quantized weights, about 185 GB:
  routed experts 2.5 bpw; attention, shared experts, the KV compressor, the indexer's queries and Engram's key/value
  projection 4 bpw; the indexer's keys 8; the head 6; the DSpark drafter 4 and 6; embeddings, router gates, norms and
  the vision tower 16-bit. [Its card](https://huggingface.co/bertholomus/DeepSeek-V4.1-Flash-1017-EXL3-2.5bpw) gives
  the bits by part and how close it stays to DeepSeek's original: KL and top-1 agreement over 122 rows, beside our
  4.0 bpw quant (`results/served-v070/kl-report.md`). Tool-call turns lose the most at 2.5 bpw.
- **New engine line:** `deepseek-v41-zig-tp2` is the four-node branch `deepseek-v41-zig-tp4` at its v0.3.1
  (TensorFold 1.0.12) with nine changes: vLLM's `cache_salt`; three fixes the two-node branch `deepseek-v41-zig` had
  taken (#17, #38, and upstream TensorFold #973's template append, two commits); DeepSeek-V4.1 tool calls read from
  their own tags (a port of #25); an assistant turn's marker text kept as text with `TENSORFOLD_REPLY_IDS`; and a
  parser fix for a `<tool_call>` whose text only starts with a DSML opener. It brings the four-node line's server
  work to two nodes: structured output (`response_format`, `guided_choice`, `guided_regex`) and forced tool calls in
  DeepSeek's DSML, through the kit's grammar library; the prompt shield (`TENSORFOLD_PLAIN_SPECIALS=1`); kept prompt
  states that resume big conversations and image turns (`--prompt-cache-gib 2`); the stream keepalive; `cache_salt`;
  and the draft head's accepted chains on `/metrics`.
- **New kit** (`kit/`, 367 files, `kit/README.md`): the four-node kit v0.3.1 with five Triton kernel variants of our
  v0.6.1 kit (Lucas Peters's `_topk_sel` for N = KK = 256 among them, #13), the 1017 checkpoint's image bias and the
  grammar library.
- **New launch:** `scripts/serve.sh` and `config.env.example`: 1,048,576 tokens a request in a shared window of
  2,097,152 positions, 4 streams, `--parallel 4`. The RDMA ring opens GID index 5 of each device, and the launch's
  preflight checks it on both nodes; this engine line does not read `TF_RDMA_GID_INDEX` (`RUNNING.md`, section 2).

Gates, all passed on the served build (47 checks, `results/served-v070/gate/result.txt`): exact chat 71/71 against
DeepSeek's reference encoding; the six API gates; 16 replies as a burst and staggered == solo; images 6/6, 9/9, and
four at a time 18/18 == solo; forced tool calls 10/10 and automatic ones 6/6, each forced row a call of an offered
tool within its schema; structured output 11/11, and bursts and staggered runs of 12 == solo; the prompt shield 14/14;
the keepalive; kept states, the fill and the resume probe; and 1M: 3/3 needles found in a 1,039,836-token prompt, the
second and third asks resuming 1,038,336 tokens of it from the kept state. A second launch at the same settings gave
the same replies on every check. The lowest free memory: 9.4 GiB on rank 0's node during the 1M prompt, 13.1 GiB on
rank 1's. Start to ready: 18 s (24 to 25 s for the gate's launches).

Four long prompts at once, on the served port after the promotion (`tools/long_check.py` and `tools/long_cold.py`,
`results/served-v070/long/` and `long-cold/`): four prompts of 159,925 tokens, each with a needle, sent together gave
the same replies as each one sent alone from a cold start (5 of 5, a seeded-sampled reply among them) and as each one
resumed from its kept state; drafting off gave the same reply as drafting on, and every needle was found. The lowest
free memory meanwhile: 10.1 GiB on rank 0's node, 11.6 GiB on rank 1's.

Measured on two GB10 nodes (TP2), the served build at the served settings, our own client (`tools/kit_bench.py`),
greedy, DSpark drafting, median of 3 unless noted (`results/served-v070/rows/`). The v0.6.0 column is v0.6.0's own
table below, on the 2.9 bpw weights of DeepSeek-V4.1-Flash:

| `--context 1048576 --parallel 4` | v0.6.0 tok/s | v0.7.0 tok/s |
|---|---|---|
| Decode, set a: code / prose / structured (512 tokens) | 89.8 / 59.0 / 106.6 | 106.9 / 61.0 / 119.2 |
| Decode, set b: code / prose / structured (384 tokens) | 103.1 / 64.1 / 143.2 | 113.6 / 68.3 / 161.4 |
| 2 streams, set b, aggregate (v0.7.0: median of 9) | 83.1 | 96.2 |
| 4 streams, set b, aggregate (median of 9) | 118.9 | 135.2 |
| 4 streams sustained 90 s, set b, aggregate | 142.0 | 153.1 |
| Prompt 8K / 32K / 64K / 128K (v0.6.0: one run each, 8K best of 2) | 2,433 / 2,495 / 2,461 / 2,354 | 2,452 / 2,490 / 2,462 / 2,428 |
| Decode after a 128K prompt (median of 3 seeded prompts) | 100.1 | 101.5 |
| Prompt 93,335 tokens | | 2,136 |

The prompt rows repeat each length's words behind a fresh prefix (`--fixed-words`), so the second and third
reps read Engram rows the first one warmed; the first reps alone gave 2,453 / 2,488 / 2,462 / 2,428 tok/s. The three
prompts after 128K decoded at 39.1 / 101.5 / 123.6 tok/s. The 93,335-token prompt is the served port's own
bench after the promotion (`results/served-v070/prefill-93335.json`).

## v0.6.1 (2026-10-09): the kernel kit in git, no engine change

The kit the Zig engine loads is now in this repository's `kit/`, byte for byte the kit our served lane loads (engine
`ce20a545`, served `65f295ec`), with `MANIFEST`, `KIT-LICENSES.md` and `kit/tools/verify_kit.sh`. The engine branch
gains the builders (`5e10970a`: `tools/dsv41-zig/build_kit.sh`, `build_inputs.sh`); its code is unchanged. Our served
lane started from this kit in a fresh folder gives the same replies as before on every gate (images 9/9 and 18/18,
51 image probes, 14 prompt lengths).

## v0.6.0 (2026-10-09): the engine moves to TensorFold 1.0's native Zig engine, images included

Engine `ce20a545` on branch [`deepseek-v41-zig`](https://github.com/bertholomus/TensorFold/tree/deepseek-v41-zig), on upstream TensorFold v1.0.2. The whole `dsv41` family (prompt
and decode paths, DSpark drafting, Engram, the RDMA ring between the two nodes, the vision tower) now runs in TensorFold
1.0's native server, with no Python in the serving path. It is what we serve now: `--context 1048576`, `--parallel 4`,
images on.

- **Same replies as v0.5.1.** On the same nodes, the prompts' logits of 14 prompt lengths match v0.5.1's byte for byte;
  the release gates (== reference 12/12, drafted == serial, concurrent == solo 12/12 burst and 12/12 staggered,
  images 6/6) pass; image replies match token for token (9/9 alone, 18/18 four at a time), and so do 51 image probes
  (formats, message layouts, refusals and their words).
- **One reply fix: `TF_DS_REPLAY_FLOOR=1`** (on in the served configuration). A last prompt chunk of 16 rows or fewer
  at the bounded replay's cut (prompts of 2048k+112 to 2048k+127 tokens, or a short last chunk) read part of its
  attention window from the previous request's keys, so its reply could depend on what ran before it (v0.5.1 does
  this too). Now it reads from the cut as a larger chunk would: 14 such prompt lengths give the same reply on every
  pass. 2 of the 51 image probes (both in that class) change their reply.
- **Images in full:** the DeepSeek vision tower and aligner at full resolution, PNG decoded natively and every other
  format through the runtime's Pillow, the same refusals as v0.5.1.
- **Memory:** 5 GiB left on the head node while serving at 1M context with 4 streams.

Measured on two GB10 nodes (TP2), engine `ce20a545` (code = served commit 65f295ec) at the served settings below, our own
client (`kit_bench.py` of the `deepseek-v41-tp2` branch), greedy, DSpark drafting, median of 3 unless noted. The
speed rows ran before `TF_DS_REPLAY_FLOOR` was switched on (it changes only prompts of the class above); the gates,
image checks and probes were run again live with it on: all pass.

| v0.6.0, `--context 1048576 --parallel 4` | tok/s |
|---|---|
| Decode, set a: code / prose / structured (512 tokens) | 89.8 / 59.0 / 106.6 |
| Decode, set b: code / prose / structured (384 tokens) | 103.1 / 64.1 / 143.2 |
| 2 streams, set b, aggregate | 83.1 |
| 4 streams, set b, aggregate (median of 9) | 118.9 |
| 4 streams sustained 90 s, set b, aggregate | 142.0 |
| Prompt 8K / 32K / 64K / 128K (one run each; 8K best of 2) | 2,433 / 2,495 / 2,461 / 2,354 |
| Decode after a 128K prompt (median of 3 seeded prompts) | 100.1 |

v0.5.1's engine (`6a6cf68`) measured in the same session, same client and prompts, on the same nodes:

| v0.5.1, `--context 1048576 --parallel 4` | tok/s |
|---|---|
| Decode, set a: code / prose / structured (512 tokens) | 89.4 / 57.7 / 107.2 |
| Decode, set b: code / prose / structured (384 tokens) | 101.5 / 63.4 / 143.4 |
| 2 streams, set b, aggregate | 79.8 |
| 4 streams, set b, aggregate (median of 9) | 113.0 |
| 4 streams sustained 90 s, set b, aggregate | 125.3 |
| Prompt 8K / 32K / 64K / 128K (one run each; 8K best of 2) | 2,387 / 2,421 / 2,363 / 2,242 |
| Decode after a 128K prompt (median of the same 3 seeded prompts) | 49.8 |

The two structured rows are inside v0.5.1's own run-to-run spread in that session (v0.5.1: 106.5-107.4 and
142.2-143.5 tok/s).

## v0.5.1 (2026-10-07): fixes for reasoning loops on long agentic turns (#6), no speed change

Engine `90b68e0` on top of v0.5's `808eb4a`. Replies and speed rows are unchanged on our served configuration.

- **fp32 GEMMs stay fp32 in NVIDIA's PyTorch container.** `nvcr.io/nvidia/pytorch:26.07-py3` sets
  `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`, which ran the prompt chunks' fp32 router, indexer and mHC GEMMs in TF32. That
  gave other bits than our lane and the gates, and the loop in #6 at seed 3001. The engine now sets the switch to 0 at
  start (it prints a line when it was 1) and refuses to start if fp32 GEMMs still run in TF32.
- **Opt-in reasoning loop guard** (default off). When 3 windows of 1,024 reply tokens in a row have fewer than 2% new
  token 8-grams, the server closes the thinking and the reply goes on to its answer. `TF_LOOP_GUARD=1` makes it the
  server default; a request sets it with `"loop_guard": true` / `false`. Replies that do not loop are untouched. The
  signal is from Capicua25x's loop detector (#9).
- Clients on long agentic turns: send the tool schemas the conversation uses. In our runs that alone avoided the loop
  (0 / 60 seeds).

## v0.5 (2026-10-07): 1M context on the shipped configuration, faster long prompts, measured against v0.4

Decode and prefill are what we worked on this time. The same release suite on the served lane for both builds
(v0.4 = engine `bd0024d` at `--context 262144`, v0.5 = `808eb4a` at `--context 1048576`), same client
(`tools/dsv41/kit_bench.py`), same prompts, greedy, DSpark drafting. Median of 3 unless noted.

- **Context window: 262,144 → 1,048,576 tokens** with `--parallel 4`. Needles found to 1M on this configuration;
  four long streams together give the same replies as each one alone.
- **Decode after a 128K prompt:** **89.2 → 128.7 tok/s (+44%)**
- **Prompt speed:** 8K **1,944 → 2,125 (+9%)**, 32K **1,977 → 2,189 (+11%)**, 64K **1,897 → 2,123 (+12%)**, 128K
  **1,732 → 2,004 (+16%)** tok/s
- **Level:** single stream set b code / prose / structured 101.1 / 62.5 / 142.4 → 100.5 / 62.5 / 140.0; 4 streams
  112.5 → 112.4 (9-rep median 112.1 → 112.4); 4 streams sustained 124.5 → 125.1
- **Still exact:** v0.5 replies are bit-identical to v0.4.1's on every gate (== reference 12/12, drafted == serial,
  concurrent == solo 12/12 burst and 12/12 staggered, images 6/6). Same weights and bits, same KV cache.
- **1M qualifier (this build, this config):** needles 3/3 to 1,039,836 tokens; 4 x 160K streams concurrent == solo 5/5; memory floor 6.44 / 5.44 GiB.

Where it came from, all bit-identical: RoPE tables at the context length (cos/sin only) and one shared logits buffer
for the round graphs pay for the 1M window; exact pruned top-k for the long-context indexer, candidate-only reindexing
and tile skipping in deep decode; prompt chunks' grouped expert launches walk a work list built on the device; kept
prompts shrink to a boundary instead of being dropped. The engine is now on upstream TensorFold 0.6.6.
Details in the engine commit message and `tools/dsv41/REPORT.md`.

## v0.4.1 (2026-10-05): fix, no speed change

Engine `d5d7bb3` on top of v0.4's `bd0024d`. The Engram read-ahead now claims its slot by index. Before, `list.remove()`
compared id arrays of other lengths and could stop the server on long, prefix-cached agent sessions (present since
v0.2). One file changed; replies and speed rows are unchanged. If you run v0.4, update to this.

> **Context window, v0.1.0–v0.4.1:** the serve command below and our served lane run `--context 262144`. Earlier text
> here said "window up to 1,048,576": that is the engine's limit, which one earlier build tested with a single needle,
> not the configuration these releases were measured and shipped with.

## v0.4 (2026-10-05): single stream past 100 tok/s, measured against v0.3

The same release suite, run on the served lane for both builds (v0.3 = engine `bbaa6cd`, v0.4 = `bd0024d`), same client
(`tools/dsv41/kit_bench.py`), same prompts, greedy, DSpark drafting. Median of 3 unless noted.

- **Single stream, 384 tokens (set b):** code **86.0 → 101.1 tok/s (+18%)** (reps 101.13 / 101.14 / 101.18), prose
  53.2 → 62.5 (+18%), structured **119.4 → 142.4 (+19%)**
- **Single stream, 512 tokens:** code 72.7 → 88.9 (+22%), prose 45.6 → 57.5 (+26%), structured 87.9 → 106.9 (+22%)
- **4 concurrent streams, 384 tokens, total tokens / wall clock:** **99.4 → 112.5 tok/s (+13%)**; 9-rep median 112.1
- **2 concurrent streams, 384 tokens:** 68.2 → 79.4 tok/s (+16%)
- **4 streams sustained** (4 always in flight for 90 s): **107.5 → 124.5 tok/s (+16%)**, per-stream p50 37.6, first
  token p50 0.20 s
- **Decode after a 128K prompt:** 59.9 → 89.2 tok/s (+49%)
- **Prompt speed:** level (8K / 32K / 64K / 128K 1,944 / 1,977 / 1,897 / 1,732 tok/s)
- **Start to ready:** 36 s (warm restart)
- **Still exact:** v0.4 replies are bit-identical to v0.3's (12/12); drafted == serial; concurrent == solo 12/12 burst
  and 12/12 staggered; images 6/6; needles 12/12 at 8K–250K. Same weights and bits, same 262K window, same KV cache.

Where it came from, all bit-identical and each behind a switch (set it to `0` for the old path):
- Paced L2 prefetch: a side stream pulls the next kernels' weights into L2 (`TF_DS_L2_PREFETCH`).
- Drafter: our own tensor-core vocabulary dot per rank (bit-equal to cuBLAS on all 129,280 rows), one small argmax
  gather, cached bias rows for the 256 most frequent tokens; drafter pass 5.1 → ~3.3 ms.
- Verify forward: dead fp32 scratch dropped from L2, multi-row router as per-chunk sums, split mHC finish on a side
  stream, split q/kv rotations, register-resident attention merge, fused indexer top-k; 6-row forward 39.5 → ~36.8 ms.
- Host path: Engram reads by Linux AIO (O_DIRECT), one pinned copy per round, drafter rows absorbed behind the forward.
- Cross-node gathers: rings and flags in registered memory, one fence per staging block; ~2.9 → ~1.9 ms a round.
Details in the commit message and `tools/dsv41/REPORT.md`.

## v0.3 (2026-10-04): what changed, measured against v0.2

The same release suite, run on the served lane for both builds (v0.2 = engine `08cae28`, v0.3 = `bbaa6cd`), same client
(`tools/dsv41/kit_bench.py`), same prompts, greedy, DSpark drafting. Median of 3 unless noted.

- **Single stream, 384 tokens (set b):** code **81.9 → 86.0 tok/s (+5%)**, prose 49.0 → 53.2 (+9%), structured
  113.1 → 119.4 (+6%)
- **Single stream, 512 tokens:** code 71.2 → 72.7 (+2%), prose 44.4 → 45.6 (+3%), structured 82.6 → 87.9 (+6%)
- **4 concurrent streams, 384 tokens, total tokens / wall clock:** **93.0 → 99.4 tok/s (+7%)**; 9-rep median 93.7 → 99.5
- **2 concurrent streams, 384 tokens:** 64.5 → 68.2 tok/s (+6%)
- **4 streams sustained** (4 always in flight for 90 s): **104.4 → 107.5 tok/s (+3%)**, per-stream p50 31.6 → 32.4
- **Prompt speed:** unchanged (8K–128K 1,739–1,984 tok/s)
- **Still exact:** v0.3 replies are bit-identical to v0.2's (12/12); drafted == serial 8/8; concurrent == solo 12/12
  burst and 12/12 staggered; images 6/6; needles 12/12 at 8K–250K.

Where it came from: less per-row overhead in the 1-row forward, all bit-identical. hc_post is fused into the next
hc_pre, the router and indexer row matmuls are smaller, the rotations and RoPE run in neighbouring kernels' epilogues,
and the indexer and compressor glue is fused. Kernels per 1-row forward: 2,387 → ~990. Every switch defaults on
(`TF_DS_HC_FUSED`, `TF_DS_ROWMM2`, `TF_DS_ROUND_GLUE`, `TF_DS_ROT_*`, `TF_DS_INDEXER_FUSED`, `TF_DS_COMP_FUSED`,
`TF_DS_TRITON_PDL`); set one to `0` to turn it off.

## v0.2 (2026-10-04): what changed, measured against v0.1.0

Same nodes, same public client (`tools/dsv41/kit_bench.py`), same prompts and same method as v0.1.0. Greedy decode,
DSpark drafting, served configuration below. Median of 3 unless noted.

- **Single stream, 384 tokens (prompt set b):** code **69.9 → 80.4 tok/s (+15%)**, prose 42.3 → 49.2 (+16%),
  structured **96.5 → 111.1 (+15%)**
- **Single stream, 512 tokens:** code 60.5 → 70.8 (+17%), prose 38.0 → 44.2 (+16%), structured 74.5 → 84.6 (+14%)
- **4 concurrent streams, 384 tokens, total tokens / wall clock:** **80.5 → 91–93.5 tok/s (+13–16%)** (medians of four
  separate runs of 3–9 reps; this row varies with the server start more than the others); 256 tokens 77.5 → 85.6 (+10%)
- **4 streams sustained (new measure):** 4 requests always in flight for 90 s, distinct prompts: **103 tok/s aggregate**,
  per-stream p50 31 tok/s, first token p50 0.22 s
- **Prompt speed (bounded replay):** 8K **1,357 → 1,935 tok/s (+43%)**, 32K 1,247 → 1,953 (+57%), 64K 1,342 → 1,897 (+41%),
  128K 1,262 → 1,741 (+38%)
- **Start to ready** (restart with the per-rank weight cache, first reply included): **65 s → 35 s (−46%)**
- **Quality (new):** fixed, seeded subsets, greedy: MMLU-200 87.5% (thinking off) / 89.0% (on, 4,096-token cap);
  GSM8K-100 97.0% / 96.0%. Script: `tools/dsv41/quality_eval.py` (subset digest 09c2e6cc43e275af).
- **Still exact:** drafted == serial 8/8 at T=0; every concurrent reply bit-identical to its solo reply (12/12 burst,
  12/12 staggered); solo replies identical to v0.1.0's (12/12: the new kernels are bit-identical); images 6/6;
  needles 12/12 at 8K–250K.

Where it came from: grouped EXL3 decode linears (884 → 350 launches a forward) and a fused grouped-expert decode path,
both bit-identical; Engram reads moved off the critical path; a lighter per-round rank check; NCCL over both RoCE ports
for prompt chunks; a faster weight-cache read and a shorter warm-up. Details in `tools/dsv41/REPORT.md`.

## v0.1.0 (2026-10-04, first release)

Single stream 512 tokens code / prose / structured 60.5 / 38.0 / 74.5 tok/s (set b 69.9 / 42.3 / 96.5); 4 streams
73.2–80.5 tok/s; prefill 1,247–1,365 tok/s at 8K–128K; decode after a 128K prompt 61.8 tok/s; served window 262,144
tokens (the engine accepts up to 1,048,576; an earlier build found one needle at 1,039,833 tokens); start to ready 65 s.

## Credits

- **DeepSeek** — DeepSeek-V4.1-Flash-1017 and DeepSeek-V4.1-Flash, their inference code and tech report (MIT,
  © 2023 DeepSeek).
- **turboderp** — [EXL3 / exllamav3](https://github.com/turboderp-org/exllamav3) (MIT), the weight format and the
  converter that made our quant.
- **Fabiano Anemone** ([@anoane](https://github.com/anoane)) — ExLlamaV3's DeepSeek-V4 support
  ([pull request 449](https://github.com/turboderp-org/exllamav3/pull/449), MIT), which our quant's conversion used.
- **Mia-AiLab** — the EXL3 2.9 bpw quant this recipe served up to v0.6.1 (MIT).
- **ashhart** — [TensorFold](https://github.com/ashhart/TensorFold) (Apache-2.0), the engine this family plugs into.
- **The engine branch's contributors** — Lucas Peters ([@lucas-peters](https://github.com/lucas-peters), #4),
  LeetJoe ([@LeetJoe](https://github.com/LeetJoe), #17), [@lepetitprince716-prog](https://github.com/lepetitprince716-prog)
  (#18), [@willjriley](https://github.com/willjriley) (#20, the base of #25), Manrique
  ([@hashd1ve](https://github.com/hashd1ve), #25, #26, #28 and #29), Scott Woolsey Biggart
  ([@swbiggart](https://github.com/swbiggart), #30), [@mrsergiopascal](https://github.com/mrsergiopascal) (#31),
  [@Blackwellboy](https://github.com/Blackwellboy) (#33, #34 and #38) and Nathan McBride
  ([@nmcbride](https://github.com/nmcbride), upstream TensorFold #973); `ATTRIBUTION.md` says what each one did.
- **Lucas Peters** ([@lucas-peters](https://github.com/lucas-peters)) — the kit's `_topk_sel` variant for
  N = KK = 256 (#13).
- **Capicua25x** — the #6 report and repro, the 8-gram novelty loop signal (#9), and issue #18, which the engine's
  loop guard follows.
- **den13501** — issue #12, which the stream keepalive follows; **ahinore** — the four-node recipe's issue #3, which
  the shield's assistant-turn quoting follows.
- **willjriley**, **dylandoesai** and **plyldavid-maker** — issues #23 and #24 (the server's path in the build) and
  #25 (both ranks need the same settings), which `RUNNING.md` and `scripts/serve.sh` follow.
- **Niklas Lenz** ([niklaslenz.com](https://niklaslenz.com)) — the launch script's local node and preflight, from their
  bring-up of the four-node recipe.
- **Bertholomus AI** (Albert Lee, [bertholomus](https://github.com/bertholomus)) — the `deepseek_v41` TP2 family and
  its Zig port `dsv41`, their kernels, the quant, the deployment and the measurements.

Not affiliated with or endorsed by DeepSeek, NVIDIA, the TensorFold authors, Mia-AiLab or MiaAI-Lab.

## License

Recipe documentation, scripts and tools: Apache-2.0 (`LICENSE`); `NOTICE` and `ATTRIBUTION.md` list what the recipe
uses. The kit's files keep their own licenses (`kit/KIT-LICENSES.md`). The weights stay under their own MIT license.
