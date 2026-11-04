| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen2 1.5B Q4_K - Medium       |   1.04 GiB |     1.78 B | CUDA       |  99 |           pp128 |     1407.36 ± 132.67 |
| qwen2 1.5B Q4_K - Medium       |   1.04 GiB |     1.78 B | CUDA       |  99 |           pp256 |      1736.95 ± 56.30 |
| qwen2 1.5B Q4_K - Medium       |   1.04 GiB |     1.78 B | CUDA       |  99 |          pp1024 |      1801.33 ± 36.53 |
| qwen2 1.5B Q4_K - Medium       |   1.04 GiB |     1.78 B | CUDA       |  99 |            tg32 |         44.56 ± 0.23 |
| qwen2 1.5B Q8_0                |   1.76 GiB |     1.78 B | CUDA       |  99 |           pp128 |     1520.63 ± 145.12 |
| qwen2 1.5B Q8_0                |   1.76 GiB |     1.78 B | CUDA       |  99 |           pp256 |      1890.44 ± 62.78 |
| qwen2 1.5B Q8_0                |   1.76 GiB |     1.78 B | CUDA       |  99 |          pp1024 |      1992.94 ± 38.58 |
| qwen2 1.5B Q8_0                |   1.76 GiB |     1.78 B | CUDA       |  99 |            tg32 |         31.77 ± 0.13 |

build: 5aeac37a3 (11719)
