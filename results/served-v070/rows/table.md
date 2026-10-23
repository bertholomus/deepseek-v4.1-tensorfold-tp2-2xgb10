| TP2-25 (engine bd1b05fb2a30, window 2097152, streams 4, parallel 4, rows 16), `--context 1048576` | tok/s |
|---|---|
| Decode, set a: code / prose / structured (512 tokens) | 106.9 / 61.0 / 119.2 |
| Decode, set b: code / prose / structured (384 tokens) | 113.6 / 68.3 / 161.4 |
| 2 streams, set b, aggregate (median of 9) | 96.2 |
| 4 streams, set b, aggregate (median of 9) | 135.2 |
| 4 streams sustained 90 s, set b, aggregate | 153.1 |
| Prompt 8K / 32K / 64K / 128K (median of 3) | 2,452 / 2,490 / 2,462 / 2,428 |
| Decode after a 128K prompt (median of 3 seeded prompts) | 101.5 |
