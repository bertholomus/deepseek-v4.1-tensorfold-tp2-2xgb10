# Results

Our own measurements of our own engine on two GB10 nodes, as the README's v0.7.1 and v0.7.0 sections report them.
Private host names, addresses, devices and paths are not in these files: the nodes are named by rank (`node0` runs
rank 0, `node1` rank 1), the link's addresses and devices are placeholders (`NODE0_LINK_IP`, `RDMA_DEV0`, ...), and a
file the runs name is named without its folder. Nothing else was changed. Releases before v0.7.0 reported their
numbers in the README only.

## `served-v071/`: v0.7.1, the served build `ac6a372` on the 2.5 bpw weights (2026-10-23)

The engine is the source of `ENGINE_COMMIT`'s parent (the same tree but for the fork's three documents); the settings
are `config.env.example`'s. The weights and the run-time inputs are v0.7.0's (`served-v070/weights.sha256`,
`inputs.txt` and `kl-report.md`). The files are laid out as `served-v070/`'s below, with three folders more:

| File | What |
|---|---|
| `gate/` | the gate run before the promotion, as v0.7.0's: launch A, launch B and the served port's checks (47 checks in all, `gate/result.txt`). Where a B check compares replies "as served" or with "the served build's", the served port ran v0.7.0 |
| `rows/` | the speed rows on the served port, as v0.7.0's; `rows-6.json` keeps each prompt's seed (1023 to 1025) and its text |
| `long/`, `long-cold/` | four long prompts at once on the served port, and each one alone from a cold start, as v0.7.0's (our labels `l2` and `l2c`) |
| `prefill-93335.*`, `behind-93335.*` | the served port's 93,335-token prefill (three reps) and the short chat sent 1 s behind it, after the promotion |
| `ab-v070/` | decode after a 128K prompt on the same three prompts (seeds 1023 to 1025): `a` and `a2` are v0.7.1 at the served settings, `b-v070` is v0.7.0's build `bd1b05f` at its own; each line's `rows-6.json` holds the three texts, the same on the three lines; `result.txt` compares the lines |
| `alias/` | the same row on build `ac6a372` with `TF_DS_PROMPT_ALIAS=1` (`b-b14a`) and `TF_DS_PROMPT_ALIAS=0` (`b-b14n`), both at a window of 1,048,576 positions, beside v0.7.0's build (`a` and `a2`); `texts.txt` compares the texts of the four lines (all the same) and `mem.txt` of each line its free memory |
| `memtrace/` | the gate's 1M row (3 needles in a 1,039,836-token prompt) at the served window, with each node's memory read once a second: `gate-steps/` after the gate's own steps before it (the resume probe, the keepalive, the kept and fill checks, the 93,335-token bench), `fresh/` on fresh launches after one 93,335-token prompt; `v071` is v0.7.1 as served, `v070` v0.7.0's build `bd1b05f` as it served, `v071r2` v0.7.1 with v0.7.0's 2,048-row prompt passes (`TF_DS_PASS_ROWS=2048`); `v071na`, v0.7.1 without `TF_DS_PROMPT_ALIAS`, stops at the launch's memory check (10.7 GiB free on rank 0's node after the load, under the 9 GiB floor with the 2 GiB prompt cache), so the alias is what lets v0.7.1's 4,096-row passes run at this window; `*-rank0.txt` and `*-rank1.txt` hold the samples (MiB), `*-phases.txt` when each step began, `summary.txt` each launch's lowest free memory and what moved |

## `served-v070/`: v0.7.0, the served build `bd1b05f` on the 2.5 bpw weights (2026-10-23)

The engine is the source of `ENGINE_COMMIT`'s parent (the same tree); the settings are `config.env.example`'s.

| File | What |
|---|---|
| `inputs.txt` | the served lane's run-time inputs: the two weight files' names (each holds its index's hash) and sizes, the token map's sha256, and what `tools/serving_1017.py` writes into a download |
| `weights.sha256` | the sha256 of every file of our upload of the weights (the 47 files of the repository but its README) |
| `kl-report.md` | the quant's next-token distributions against DeepSeek's original 1017 checkpoint (its own weights, run in 16-bit by ExLlamaV3's reference forward), teacher-forced over the full vocabulary on our 122-row set (294,481 tokens): KL(original ‖ quant) and top-1 agreement with 95 % bootstrap intervals, by kind of text and by depth, for the 2.5 bpw quant and our 4.0 bpw one, and their paired differences |
| `gate/` | the gate run before the promotion. Launch A ran the served settings on a gate port; launch B ran them again (B's window is sized from A's lowest free memory, which kept 2,097,152) and every B check compares its replies with A's; then the served port was checked after the promotion. The other files are the checks' own outputs, `-a`, `-b` or `-served` by launch |
| `gate/result.txt` | the verdicts, a line a check: A's 20 checks and its 1M row, B's 23 checks and its 1M row, the 93,335-token prefill on both, and the served port's 4 checks (47 checks in all), then the promotion |
| `gate/gate.log`, `gate/start-*.log`, `gate/pin.txt` | the run's log with times; each launch's record (its settings, the load, free memory after it); the build's commit |
| `gate/gates-*.json` | `gates.py`: the six API gates |
| `gate/tpl-*.json`, `gate/exact-served.json` | the exact-chat check: 71 chats rendered by the chat template against DeepSeek's reference encoding |
| `gate/pc-*.json`, `gate/done-*.txt` | 16 replies solo, as a burst and staggered (their token ids); rank 0's request lines for the solo ones |
| `gate/vis-*/` | the image gate (`v-vhttp`, 6 checks), 9 image prompts' token ids (`v-vref`), and, in B, the 9 four at a time (`v-vconc`, 18 replies, each equal to A's) |
| `gate/struct-*` | structured output: 12 requests (a JSON schema, a JSON object, a choice and a regex, greedy and seeded, and a forced call), solo, as a burst and staggered |
| `gate/dsml-*` | forced ("required" and named) and automatic tool calls, whole and streamed |
| `gate/sampler-*`, `gate/replyids-*`, `gate/acc-*` | sampling with `top_k` 0, -1, 20, 56, 100 and 1000 (28 requests solo, as a burst and staggered); 16 replies that quote the reasoning end or the DSML marker (`TENSORFOLD_REPLY_IDS` unset); the draft head's accepted chains by depth on `/metrics` |
| `gate/shield-*` | the prompt shield: 14 conversations' token ids and greedy replies, with and without spelled special tokens |
| `gate/keep-*`, `gate/kept-*`, `gate/fill-*`, `gate/resume-*` | the keepalive (two streamed 97,510-token prompts at once: each SSE comment's time); kept prompt states (eight replies, image turns among them); the fill (a 65,010-token prompt arriving beside a running reply: the reply's longest stop, the prompt's first token); the resume probe (four prompts sharing a prefix) |
| `gate/a-1m.*`, `gate/b-1m.*` | the 1M row: three needles in a 1,039,836-token prompt, the second and third asks resumed from the kept state |
| `gate/gate-*-prefill.*` | the 93,335-token prefill on A and B, three reps |
| `gate/mem-*` | each node's free memory (MemAvailable) during the checks, the 1M row and the served port's checks |
| `gate/checkers.sha256`, `gate/*-selftest.log` | the sha256 of the lane tools the gate ran, and their self-tests |
| `rows/` | the speed rows on the served port: `rows-1.json` to `rows-6.json` from `tools/kit_bench.py` (decode set a and set b, 2 and 4 streams, 4 streams sustained 90 s, the 8K to 128K prompts, decode after a 128K prompt), `rows.txt` (each row's arguments), `table.md`, `result.txt`, `mem.txt` (each node's free memory during the rows), `health-before.txt` (the port idle before the rows), `engine*.txt` (each rank's build), `client.sha256`, and both ranks' command lines (`cmd-rank0.json`, `cmd-rank1.json`) |
| `long/` | four long prompts at once on the served port: `long.json` and `long.log` from `tools/long_check.py` (four 160,000-token prompts with a needle each, greedy, and prompt 0 seeded-sampled: all five at once, then one after another, each resuming its prompt from the kept state, then prompt 0 with drafting off; every reply's token ids; 48 tokens a reply with the end token ignored, so a reply runs on past the passphrase), `result.txt`, `mem.txt` (each node's free memory meanwhile), `long-run.log`, `health-before.txt`, `engine*.txt`, `client.sha256` and both ranks' command lines |
| `long-cold/` | the same five cases again, each one alone from a cold start (`tools/long_cold.py`: a fresh `cache_salt` a request, so no kept prompt is reused), and prompt 0 with drafting off, against `long/`'s replies sent together; the same files, and `from.txt` (the run compared with: our labels `l1` and `l1c` are `long/` and `long-cold/`) |
| `prefill-93335.*`, `behind-93335.*` | on the served port after the promotion: the 93,335-token prefill, three reps (median 2,135.8 tok/s), and a short chat sent 1 s behind the cold long prompt (its time to first token) |

The exact-chat, 16-reply, image, structured, tool-call, shield, kept-state, fill, resume and memory checks are our
lane tools, not in this release; `gates.py`, `needle.py` and `kit_bench.py` are in `tools/`.
