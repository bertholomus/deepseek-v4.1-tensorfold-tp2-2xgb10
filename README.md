# DeepSeek-V4.1-Flash on 2× DGX Spark (GB10) — our TensorFold TP2 engine

**BertholomusAI recipe: DeepSeek-V4.1-Flash served tensor-parallel over two NVIDIA DGX Spark / GB10 nodes by a
`deepseek_v41` model family we wrote for [TensorFold](https://github.com/ashhart/TensorFold) (by ashhart).**

> **Weights:** [Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw)
> (MIT), an EXL3 quant by Mia-AiLab of [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
> (MIT, © 2023 DeepSeek). We did not make or modify these weights; this repository holds no weights. The Engram tables
> are read at run time from DeepSeek's original shards 47 and 48.

**Engine:** [bertholomus/TensorFold, branch `deepseek-v41-tp2`](https://github.com/bertholomus/TensorFold/tree/deepseek-v41-tp2).
Design, full report and attribution: [`tools/dsv41/`](https://github.com/bertholomus/TensorFold/tree/deepseek-v41-tp2/tools/dsv41).
Clean-room: the model math is re-implemented from DeepSeek's MIT inference code and tech report; no code from other
DeepSeek-V4.1 recipes or kits was read or copied ([ATTRIBUTION.md](https://github.com/bertholomus/TensorFold/blob/deepseek-v41-tp2/tools/dsv41/ATTRIBUTION.md)).

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

## Run it

Two GB10 nodes with a direct RoCE link. On each node, a throwaway `nvcr.io/nvidia/pytorch:26.07-py3` container with
the `deepseek-v41-tp2` branch installed (`pip install -e`). Start rank 1 (worker) first, then rank 0 (head):

```
TF_DS_REPLAY=1 TF_DS_PREFILL_CHUNK=2048 TF_DS_RANK_CACHE=<CACHE_DIR> TF_DS_RANK_CACHE_READERS=32 \
TF_DS_WARM_LENGTHS=1,17,33,131,514,1024,2113 \
NCCL_IB_HCA=<HCA_PORT_0>,<HCA_PORT_1> NCCL_IB_GID_INDEX=5 NCCL_SOCKET_IFNAME=<IFACE> \
tensorfold serve <MODEL_DIR> --tp 2 --rank R --master <HEAD_IP> --host 127.0.0.1 --port 18891 \
  --context 1048576 --vision --parallel 4 --mtp-drafts 5
```

- `<MODEL_DIR>`: the Mia-AiLab EXL3 checkpoint. Engram tables: `TF_DS_ENGRAM=<ENGRAM_DIR>` (DeepSeek's original shards
  47/48), or a folder next to `<MODEL_DIR>` whose name contains "Engram".
- `NCCL_IB_HCA` lists both RoCE ports when both are cabled (prompt-chunk gathers 153 → 106 ms); one port works too.
- `TF_DS_RANK_CACHE` keeps each rank's weights in one file (~106 GB a rank, written on the first start).
- `--context 1048576` is the measured, served window from v0.5 (v0.1.0–v0.4.1 ran 262144).
- `TF_RDMA_DEVICES=<HCA_PORT_0>,<HCA_PORT_1>` names the RoCE ports of the link between the two nodes when a node has
  more active ports than that link (the RDMA gather needs the same device count on every rank). `TF_API_KEY_FILE=<file>` makes every route but `/health` require a key.
- `--temperature` sets the server-side **default** a client inherits when it omits the field. Greedy (`0`) makes long
  agentic turns loop in `reasoning_content` and return empty `content` (`finish_reason: "length"`); omit it to keep the
  engine default (1.0), and pin `0` only for the deterministic MMLU/GSM8K runs.
- Benchmark: `python3 tools/dsv41/kit_bench.py --base http://127.0.0.1:18891 --model <name> decode|concurrent|sustained|prefill|depth`
- Quality: `python3 tools/dsv41/quality_eval.py --base http://127.0.0.1:18891 --model <name> --mmlu <MMLU_TEST_PARQUET> \
  --gsm8k <GSM8K_TEST_JSONL> --max-tokens-on 4096` (MMLU "all" test from cais/mmlu, GSM8K test from openai/grade-school-math)

Hosts and addresses are placeholders; substitute your own.

## Credits

- **DeepSeek** — DeepSeek-V4.1-Flash, its inference code and tech report (MIT, © 2023 DeepSeek).
- **Mia-AiLab** — the EXL3 2.9 bpw quant this recipe serves (MIT).
- **ashhart** — [TensorFold](https://github.com/ashhart/TensorFold) (Apache-2.0), the engine this family plugs into.
- **turboderp** — [EXL3 / exllamav3](https://github.com/turboderp-org/exllamav3) (MIT), the weight format.
- **BertholomusAI** (Albert Lee, [bertholomus](https://github.com/bertholomus)) — the `deepseek_v41` TP2 family, its
  kernels, the deployment and the measurements.

Not affiliated with or endorsed by DeepSeek, NVIDIA, the TensorFold authors, Mia-AiLab or MiaAI-Lab.

## License

Recipe documentation: Apache-2.0 (`LICENSE`). The weights stay under their own MIT license.
