| TP2-25 (engine 4fcb94c09717, window 2097152, streams 16, parallel 16, rows 32), `--context 1048576` | tok/s |
|---|---|
| Decode, set a: code / prose / structured (512 tokens) | 105.7 / 60.3 / 118.9 |
| Decode, set b: code / prose / structured (384 tokens) | 112.4 / 67.8 / 158.4 |
| 2 streams, set b, aggregate (median of 9) | 95.3 |
| 4 streams, set b, aggregate (median of 9) | 136.4 |
| 4 streams sustained 90 s, set b, aggregate | 152.0 |
| 8 streams sustained 90 s, set b, aggregate | 171.6 |
| Prompt 8K / 32K / 64K / 128K (median of 3) | 2,428 / 2,498 / 2,533 / 2,500 |
| Decode after a 128K prompt (median of 3 seeded prompts) | 98.8 |
