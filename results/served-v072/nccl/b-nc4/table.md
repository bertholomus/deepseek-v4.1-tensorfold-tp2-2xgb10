| TP2-25 b-nc4 (engine ac6a372c68e8, window 2097152, streams 4, parallel 4, rows 16), `--context 1048576` | tok/s |
|---|---|
| Decode, set a: code / prose / structured (512 tokens) | 106.0 / 60.7 / 118.5 |
| Decode, set b: code / prose / structured (384 tokens) | 113.1 / 67.9 / 160.4 |
| 2 streams, set b, aggregate (median of 9) | 96.1 |
| 4 streams, set b, aggregate (median of 9) | 135.2 |
| 4 streams sustained 90 s, set b, aggregate | 152.8 |
| Prompt 8K / 32K / 64K / 128K (median of 3) | 2,428 / 2,538 / 2,536 / 2,519 |
| Decode after a 128K prompt (median of 3 seeded prompts) | 100.1 |
