| TP2-25 (engine 6579b756dcd7, window 2097152, streams 8, parallel 8, rows 16), `--context 1048576` | tok/s |
|---|---|
| Decode, set a: code / prose / structured (512 tokens) | 105.4 / 60.2 / 117.7 |
| Decode, set b: code / prose / structured (384 tokens) | 111.9 / 67.4 / 158.7 |
| 2 streams, set b, aggregate (median of 9) | 95.1 |
| 4 streams, set b, aggregate (median of 9) | 134.1 |
| 4 streams sustained 90 s, set b, aggregate | 152.0 |
| 8 streams sustained 90 s, set b, aggregate | 167.1 |
| Prompt 8K / 32K / 64K / 128K (median of 3) | 2,419 / 2,531 / 2,533 / 2,516 |
| Decode after a 128K prompt (median of 3 seeded prompts) | 99.0 |
