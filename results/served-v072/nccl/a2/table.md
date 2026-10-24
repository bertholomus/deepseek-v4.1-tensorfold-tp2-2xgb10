| TP2-25 a2 (engine ac6a372c68e8, window 2097152, streams 4, parallel 4, rows 16), `--context 1048576` | tok/s |
|---|---|
| Decode, set a: code / prose / structured (512 tokens) | 105.9 / 60.5 / 118.8 |
| Decode, set b: code / prose / structured (384 tokens) | 112.5 / 67.7 / 160.7 |
| 2 streams, set b, aggregate (median of 9) | 95.5 |
| 4 streams, set b, aggregate (median of 9) | 134.7 |
| 4 streams sustained 90 s, set b, aggregate | 152.4 |
| Prompt 8K / 32K / 64K / 128K (median of 3) | 2,442 / 2,575 / 2,563 / 2,548 |
| Decode after a 128K prompt (median of 3 seeded prompts) | 99.9 |
