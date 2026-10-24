| TP2-25 a (engine ac6a372c68e8, window 2097152, streams 4, parallel 4, rows 16), `--context 1048576` | tok/s |
|---|---|
| Decode, set a: code / prose / structured (512 tokens) | 106.5 / 60.5 / 119.1 |
| Decode, set b: code / prose / structured (384 tokens) | 112.8 / 67.7 / 160.0 |
| 2 streams, set b, aggregate (median of 9) | 95.8 |
| 4 streams, set b, aggregate (median of 9) | 135.0 |
| 4 streams sustained 90 s, set b, aggregate | 152.6 |
| Prompt 8K / 32K / 64K / 128K (median of 3) | 2,445 / 2,574 / 2,565 / 2,547 |
| Decode after a 128K prompt (median of 3 seeded prompts) | 99.8 |
