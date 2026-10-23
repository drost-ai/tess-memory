# Embedding candidates: raw results

| model | device | host | R@5 | R@10 | MRR | long-ctx R@5 | multilingual R@5 | query p50/p95 ms | corpus docs/s | load s | peak RSS MB | download MB | dims | max tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-Embedding-0.6B | cpu | Apple M4 Max (macOS) | 0.81 | 0.861 | 0.709 | 0.5 | 0.5 | 96.1/131.1 | 5.5 | 0.26 | 16081.4 | 1207.5 | 1024 | 32768 |
| Qwen3-Embedding-0.6B | cpu | 13th Gen Intel(R) Core(TM) i9-13900KF (Linux) | 0.81 | 0.861 | 0.707 | 0.5 | 0.5 | 203.7/256.8 | 3.5 | 0.35 | 3990.4 | 1207.5 | 1024 | 32768 |
| Qwen3-Embedding-0.6B | mps | Apple M4 Max (macOS) | 0.81 | 0.873 | 0.709 | 0.5 | 0.5 | 19.5/22.1 | 151.7 | 0.4 | 788.7 | 1207.5 | 1024 | 32768 |
| all-MiniLM-L6-v2 | cpu | Apple M4 Max (macOS) | 0.861 | 0.861 | 0.76 | 0.0 | 0.167 | 3.5/3.9 | 1730.3 | 0.06 | 1373.4 | 91.6 | 384 | 256 |
| all-MiniLM-L6-v2 | cpu | 13th Gen Intel(R) Core(TM) i9-13900KF (Linux) | 0.861 | 0.861 | 0.76 | 0.0 | 0.167 | 4.1/4.3 | 870.7 | 0.08 | 1162.6 | 91.6 | 384 | 256 |
| all-MiniLM-L6-v2 | mps | Apple M4 Max (macOS) | 0.861 | 0.861 | 0.76 | 0.0 | 0.167 | 4.7/101.9 | 2395.8 | 0.14 | 611.0 | 91.6 | 384 | 256 |
| bge-small-en-v1.5 | cpu | Apple M4 Max (macOS) | 0.911 | 0.937 | 0.793 | 0.333 | 0.5 | 6.7/7.4 | 598.1 | 0.1 | 2367.7 | 134.4 | 384 | 512 |
| bge-small-en-v1.5 | cpu | 13th Gen Intel(R) Core(TM) i9-13900KF (Linux) | 0.911 | 0.937 | 0.793 | 0.333 | 0.5 | 5.5/5.8 | 314.0 | 0.09 | 1898.1 | 134.4 | 384 | 512 |
| bge-small-en-v1.5 | mps | Apple M4 Max (macOS) | 0.911 | 0.937 | 0.793 | 0.333 | 0.5 | 9.4/9.9 | 636.1 | 0.14 | 616.8 | 134.4 | 384 | 512 |
| multilingual-e5-small | cpu | Apple M4 Max (macOS) | 0.924 | 0.937 | 0.912 | 0.167 | 0.833 | 6.1/7.6 | 598.4 | 0.7 | 2795.1 | 514.9 | 384 | 512 |
| multilingual-e5-small | cpu | 13th Gen Intel(R) Core(TM) i9-13900KF (Linux) | 0.924 | 0.937 | 0.912 | 0.167 | 0.833 | 5.2/5.3 | 303.5 | 1.13 | 2109.7 | 514.9 | 384 | 512 |
| multilingual-e5-small | mps | Apple M4 Max (macOS) | 0.924 | 0.937 | 0.912 | 0.167 | 0.833 | 5.3/6.2 | 1304.3 | 0.83 | 1057.4 | 514.9 | 384 | 512 |
