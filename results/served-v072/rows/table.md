| TP2-25 (engine 6579b756dcd7, window 2097152, streams 4, parallel 4, rows 16), `--context 1048576` | tok/s |
|---|---|
| Decode, set a: code / prose / structured (512 tokens) | 105.7 / 60.3 / 119.6 |
| Decode, set b: code / prose / structured (384 tokens) | 112.6 / 67.7 / 160.6 |
| 2 streams, set b, aggregate (median of 9) | 95.6 |
| 4 streams, set b, aggregate (median of 9) | 134.6 |
| 4 streams sustained 90 s, set b, aggregate | 152.5 |
| Prompt 8K / 32K / 64K / 128K (median of 3) | 2,452 / 2,567 / 2,560 / 2,543 |
| Decode after a 128K prompt (median of 3 seeded prompts) | 100.0 |
