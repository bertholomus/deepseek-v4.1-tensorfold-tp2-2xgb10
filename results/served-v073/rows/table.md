| TP2-25 (engine 6579b756dcd7, window 2097152, streams 8, parallel 8, rows 16), `--context 1048576` | tok/s |
|---|---|
| Decode, set a: code / prose / structured (512 tokens) | 106.0 / 60.5 / 118.3 |
| Decode, set b: code / prose / structured (384 tokens) | 112.7 / 67.5 / 159.5 |
| 2 streams, set b, aggregate (median of 9) | 95.3 |
| 4 streams, set b, aggregate (median of 9) | 134.0 |
| 4 streams sustained 90 s, set b, aggregate | 151.8 |
| 8 streams sustained 90 s, set b, aggregate | 168.3 |
| Prompt 8K / 32K / 64K / 128K (median of 3) | 2,439 / 2,568 / 2,550 / 2,523 |
| Decode after a 128K prompt (median of 3 seeded prompts) | 99.2 |
