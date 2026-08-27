# Benchmark Data & Formal Comparison (NVIDIA series)

The **structured, machine-readable** dataset lives in [`../src/data/measurements.ts`](../src/data/measurements.ts).
Each entry carries `protocol`, `sourceUrl`, the full scenario (model / gpu / layout /
quant / batch / ISL / OSL) and only the metrics the source actually reported.
This file (and [`measurements.md`](./measurements.md)), the human-readable archive documents provenance and how to interpret the comparison.

## Run the comparison

```
npm run calibrate
```

This renders one row per entry, comparing **measured vs ideal** for
**TTFT / TPOT / throughput / E2E** together. 

A metric the source did not report, or
that does not apply to that protocol, renders as `-`. Columns are `m / i (r)` where
`m` = measured, `i` = ideal, `r` = measured/ideal ratio.


### Results

| id                              | proto | setup                                                     | TTFT m/i (r)            | TPOT m/i (r)       | Thr m/i (r)              | E2E m/i (r)            |
|---------------------------------|-------|-----------------------------------------------------------|-------------------------|--------------------|--------------------------|------------------------|
| splitwise-70b-h100-chat         | LAT   | llama2_70b @ 8xh100_sxm TP8PP1 B2 1020/129 fp16           | 84 / 37 (2.28x)         | 28.0 / 5.3 (5.33x) | - / 361 (-)              | 3,387 / 714 (4.74x)    |
| splitwise-70b-h100-code         | LAT   | llama2_70b @ 8xh100_sxm TP8PP1 B2 1500/13 fp16            | 95 / 55 (1.74x)         | 31.0 / 5.3 (5.89x) | - / 211 (-)              | 493 / 123 (4.01x)      |
| splitwise-70b-a100-chat         | LAT   | llama2_70b @ 8xa100_sxm_80g TP8PP1 B2 1020/129 fp16       | 155 / 117 (1.33x)       | 40.0 / 8.6 (4.64x) | - / 210 (-)              | 4,957 / 1,229 (4.03x)  |
| splitwise-70b-a100-code         | LAT   | llama2_70b @ 8xa100_sxm_80g TP8PP1 B2 1500/13 fp16        | 185 / 173 (1.07x)       | 52.0 / 8.6 (6.02x) | - / 91 (-)               | 856 / 285 (3.00x)      |
| gptj-h100-b1                    | LAT   | gptj_6b @ 1xh100_sxm TP1PP1 B1 128/128 fp8                | - / 1 (-)               | 7.1 / 1.8 (3.90x)  | 185 / 548 (0.34x)        | - / 234 (-)            |
| gptj-h100-b64                   | LAT   | gptj_6b @ 1xh100_sxm TP1PP1 B64 128/128 fp8               | 102 / 50 (2.03x)        | - / 2.6 (-)        | 10,907 / 21,034 (0.52x)  | - / 389 (-)            |
| gptj-a100-b1                    | LAT   | gptj_6b @ 1xa100_sxm_80g TP1PP1 B1 128/128 fp16           | - / 5 (-)               | 12.5 / 6.0 (2.09x) | 111 / 166 (0.67x)        | - / 770 (-)            |
| gptj-a100-b64                   | LAT   | gptj_6b @ 1xa100_sxm_80g TP1PP1 B64 128/128 fp16          | 481 / 319 (1.51x)       | - / 8.7 (-)        | 3,679 / 5,714 (0.64x)    | - / 1,434 (-)          |
| lmsys-qwen3-235b-pp1tp4         | LAT   | qwen3_235b @ 4xh20 TP4PP1 B1 131072/1 fp8                 | 56,000 / 27,218 (2.06x) | - / - (-)          | 2,350 / 4,816 (0.49x)    | - / - (-)              |
| lmsys-qwen3-235b-pp4tp4         | LAT   | qwen3_235b @ 16xh20 TP4PP4 B1 131072/1 fp8                | 17,700 / 6,804 (2.60x)  | - / - (-)          | 8,100 / 19,263 (0.42x)   | - / - (-)              |
| lmsys-qwen3-235b-pp8tp4         | LAT   | qwen3_235b @ 32xh20 TP4PP8 B1 131072/1 fp8                | 10,500 / 3,402 (3.09x)  | - / - (-)          | 14,600 / 38,525 (0.38x)  | - / - (-)              |
| trtllm-8b-h100-1024-2048        | THR   | llama3_1_8b @ 1xh100_sxm TP1PP1 B0 1024/2048 fp8          | - / - (-)               | - / - (-)          | 13,166 / 20,296 (0.65x)  | - / - (-)              |
| trtllm-70b-h100-1024-2048       | THR   | llama3_3_70b @ 2xh100_sxm TP2PP1 B0 1024/2048 fp8         | - / - (-)               | - / - (-)          | 3,785 / 8,576 (0.44x)    | - / - (-)              |
| trtllm-405b-h100-1024-2048      | THR   | llama3_405b @ 8xh100_sxm TP8PP1 B0 1024/2048 fp8          | - / - (-)               | - / - (-)          | 3,237 / 10,827 (0.30x)   | - / - (-)              |
| trtllm-8b-h100-128-128          | THR   | llama3_1_8b @ 1xh100_sxm TP1PP1 B0 128/128 fp8            | - / - (-)               | - / - (-)          | 26,401 / 80,944 (0.33x)  | - / - (-)              |
| trtllm-8b-h200-128-128          | THR   | llama3_1_8b @ 1xh200_sxm TP1PP1 B0 128/128 fp8            | - / - (-)               | - / - (-)          | 27,028 / 91,546 (0.30x)  | - / - (-)              |
| trtllm-8b-h100-128-2048         | THR   | llama3_1_8b @ 1xh100_sxm TP1PP1 B0 128/2048 fp8           | - / - (-)               | - / - (-)          | 21,413 / 38,650 (0.55x)  | - / - (-)              |
| trtllm-8b-h200-128-2048         | THR   | llama3_1_8b @ 1xh200_sxm TP1PP1 B0 128/2048 fp8           | - / - (-)               | - / - (-)          | 23,102 / 57,798 (0.40x)  | - / - (-)              |
| trtllm-8b-h200-1024-2048        | THR   | llama3_1_8b @ 1xh200_sxm TP1PP1 B0 1024/2048 fp8          | - / - (-)               | - / - (-)          | 16,058 / 29,386 (0.55x)  | - / - (-)              |
| trtllm-8b-h100-2048-128         | THR   | llama3_1_8b @ 1xh100_sxm TP1PP1 B0 2048/128 fp8           | - / - (-)               | - / - (-)          | 3,276 / 5,419 (0.60x)    | - / - (-)              |
| trtllm-8b-h200-2048-128         | THR   | llama3_1_8b @ 1xh200_sxm TP1PP1 B0 2048/128 fp8           | - / - (-)               | - / - (-)          | 3,391 / 5,923 (0.57x)    | - / - (-)              |
| trtllm-8b-h100-2048-2048        | THR   | llama3_1_8b @ 1xh100_sxm TP1PP1 B0 2048/2048 fp8          | - / - (-)               | - / - (-)          | 9,462 / 13,112 (0.72x)   | - / - (-)              |
| trtllm-8b-h200-2048-2048        | THR   | llama3_1_8b @ 1xh200_sxm TP1PP1 B0 2048/2048 fp8          | - / - (-)               | - / - (-)          | 11,822 / 18,726 (0.63x)  | - / - (-)              |
| trtllm-70b-h100-128-128         | THR   | llama3_3_70b @ 2xh100_sxm TP2PP1 B0 128/128 fp8           | - / - (-)               | - / - (-)          | 6,092 / 22,230 (0.27x)   | - / - (-)              |
| trtllm-70b-h200-128-128         | THR   | llama3_3_70b @ 2xh200_sxm TP2PP1 B0 128/128 fp8           | - / - (-)               | - / - (-)          | 6,328 / 24,809 (0.26x)   | - / - (-)              |
| trtllm-70b-h100-128-2048        | THR   | llama3_3_70b @ 2xh100_sxm TP2PP1 B0 128/2048 fp8          | - / - (-)               | - / - (-)          | 5,893 / 17,368 (0.34x)   | - / - (-)              |
| trtllm-70b-h200-128-2048        | THR   | llama3_3_70b @ 2xh200_sxm TP2PP1 B0 128/2048 fp8          | - / - (-)               | - / - (-)          | 7,467 / 33,915 (0.22x)   | - / - (-)              |
| trtllm-70b-h200-1024-2048       | THR   | llama3_3_70b @ 2xh200_sxm TP2PP1 B0 1024/2048 fp8         | - / - (-)               | - / - (-)          | 5,480 / 15,008 (0.37x)   | - / - (-)              |
| trtllm-70b-h100-2048-128        | THR   | llama3_3_70b @ 2xh100_sxm TP2PP1 B0 2048/128 fp8          | - / - (-)               | - / - (-)          | 723 / 1,441 (0.50x)      | - / - (-)              |
| trtllm-70b-h200-2048-128        | THR   | llama3_3_70b @ 2xh200_sxm TP2PP1 B0 2048/128 fp8          | - / - (-)               | - / - (-)          | 748 / 1,557 (0.48x)      | - / - (-)              |
| trtllm-70b-h100-2048-2048       | THR   | llama3_3_70b @ 2xh100_sxm TP2PP1 B0 2048/2048 fp8         | - / - (-)               | - / - (-)          | 2,786 / 5,410 (0.51x)    | - / - (-)              |
| trtllm-70b-h200-2048-2048       | THR   | llama3_3_70b @ 2xh200_sxm TP2PP1 B0 2048/2048 fp8         | - / - (-)               | - / - (-)          | 3,776 / 9,113 (0.41x)    | - / - (-)              |
| trtllm-405b-h100-128-128        | THR   | llama3_405b @ 8xh100_sxm TP8PP1 B0 128/128 fp8            | - / - (-)               | - / - (-)          | 3,705 / 17,403 (0.21x)   | - / - (-)              |
| trtllm-405b-h100-128-2048       | THR   | llama3_405b @ 8xh100_sxm TP8PP1 B0 128/2048 fp8           | - / - (-)               | - / - (-)          | 4,517 / 24,642 (0.18x)   | - / - (-)              |
| trtllm-405b-h200-128-2048       | THR   | llama3_405b @ 8xh200_sxm TP8PP1 B0 128/2048 fp8           | - / - (-)               | - / - (-)          | 4,715 / 62,164 (0.08x)   | - / - (-)              |
| trtllm-405b-h200-1024-2048      | THR   | llama3_405b @ 8xh200_sxm TP8PP1 B0 1024/2048 fp8          | - / - (-)               | - / - (-)          | 3,610 / 20,511 (0.18x)   | - / - (-)              |
| trtllm-405b-h100-2048-128       | THR   | llama3_405b @ 8xh100_sxm TP8PP1 B0 2048/128 fp8           | - / - (-)               | - / - (-)          | 433 / 1,106 (0.39x)      | - / - (-)              |
| trtllm-405b-h200-2048-128       | THR   | llama3_405b @ 8xh200_sxm TP8PP1 B0 2048/128 fp8           | - / - (-)               | - / - (-)          | 441 / 1,164 (0.38x)      | - / - (-)              |
| trtllm-405b-h100-2048-2048      | THR   | llama3_405b @ 8xh100_sxm TP8PP1 B0 2048/2048 fp8          | - / - (-)               | - / - (-)          | 2,217 / 6,575 (0.34x)    | - / - (-)              |
| trtllm-405b-h200-2048-2048      | THR   | llama3_405b @ 8xh200_sxm TP8PP1 B0 2048/2048 fp8          | - / - (-)               | - / - (-)          | 2,841 / 11,548 (0.25x)   | - / - (-)              |
| trtllm-maverick-h100-128-4096   | THR   | llama4_maverick @ 8xh100_sxm TP8PP1 B0 128/4096 fp8       | - / - (-)               | - / - (-)          | 11,163 / 38,196 (0.29x)  | - / - (-)              |
| trtllm-maverick-h200-128-4096   | THR   | llama4_maverick @ 8xh200_sxm TP8PP1 B0 128/4096 fp8       | - / - (-)               | - / - (-)          | 18,541 / 107,964 (0.17x) | - / - (-)              |
| trtllm-maverick-h100-1024-2048  | THR   | llama4_maverick @ 8xh100_sxm TP8PP1 B0 1024/2048 fp8      | - / - (-)               | - / - (-)          | 11,584 / 38,942 (0.30x)  | - / - (-)              |
| trtllm-maverick-h200-1024-2048  | THR   | llama4_maverick @ 8xh200_sxm TP8PP1 B0 1024/2048 fp8      | - / - (-)               | - / - (-)          | 16,859 / 102,465 (0.16x) | - / - (-)              |
| trtllm-maverick-h100-2048-128   | THR   | llama4_maverick @ 8xh100_sxm TP8PP1 B0 2048/128 fp8       | - / - (-)               | - / - (-)          | 3,832 / 16,196 (0.24x)   | - / - (-)              |
| trtllm-maverick-h200-2048-128   | THR   | llama4_maverick @ 8xh200_sxm TP8PP1 B0 2048/128 fp8       | - / - (-)               | - / - (-)          | 4,364 / 22,066 (0.20x)   | - / - (-)              |
| trtllm-70b-fp4-b200-128-128     | THR   | llama3_3_70b @ 1xb200_sxm TP1PP1 B0 128/128 fp4           | - / - (-)               | - / - (-)          | 10,614 / 54,616 (0.19x)  | - / - (-)              |
| trtllm-70b-fp4-gb200-128-128    | THR   | llama3_3_70b @ 1xgb200_nvl72_node TP1PP1 B0 128/128 fp4   | - / - (-)               | - / - (-)          | 11,101 / 122,336 (0.09x) | - / - (-)              |
| trtllm-70b-fp4-b200-128-2048    | THR   | llama3_3_70b @ 1xb200_sxm TP1PP1 B0 128/2048 fp4          | - / - (-)               | - / - (-)          | 9,446 / 60,756 (0.16x)   | - / - (-)              |
| trtllm-70b-fp4-gb200-128-2048   | THR   | llama3_3_70b @ 1xgb200_nvl72_node TP1PP1 B0 128/2048 fp4  | - / - (-)               | - / - (-)          | 10,276 / 142,570 (0.07x) | - / - (-)              |
| trtllm-70b-fp4-b200-1024-2048   | THR   | llama3_3_70b @ 1xb200_sxm TP1PP1 B0 1024/2048 fp4         | - / - (-)               | - / - (-)          | 6,547 / 28,167 (0.23x)   | - / - (-)              |
| trtllm-70b-fp4-gb200-1024-2048  | THR   | llama3_3_70b @ 1xgb200_nvl72_node TP1PP1 B0 1024/2048 fp4 | - / - (-)               | - / - (-)          | 7,923 / 65,451 (0.12x)   | - / - (-)              |
| trtllm-70b-fp4-b200-2048-128    | THR   | llama3_3_70b @ 1xb200_sxm TP1PP1 B0 2048/128 fp4          | - / - (-)               | - / - (-)          | 1,330 / 3,461 (0.38x)    | - / - (-)              |
| trtllm-70b-fp4-gb200-2048-128   | THR   | llama3_3_70b @ 1xgb200_nvl72_node TP1PP1 B0 2048/128 fp4  | - / - (-)               | - / - (-)          | 1,418 / 7,734 (0.18x)    | - / - (-)              |
| trtllm-70b-fp4-b200-2048-2048   | THR   | llama3_3_70b @ 1xb200_sxm TP1PP1 B0 2048/2048 fp4         | - / - (-)               | - / - (-)          | 4,528 / 17,376 (0.26x)   | - / - (-)              |
| trtllm-70b-fp4-gb200-2048-2048  | THR   | llama3_3_70b @ 1xgb200_nvl72_node TP1PP1 B0 2048/2048 fp4 | - / - (-)               | - / - (-)          | 5,327 / 40,243 (0.13x)   | - / - (-)              |
| trtllm-405b-fp4-b200-128-128    | THR   | llama3_405b @ 4xb200_sxm TP4PP1 B0 128/128 fp4            | - / - (-)               | - / - (-)          | 6,219 / 42,218 (0.15x)   | - / - (-)              |
| trtllm-405b-fp4-gb200-128-128   | THR   | llama3_405b @ 4xgb200_nvl72_node TP4PP1 B0 128/128 fp4    | - / - (-)               | - / - (-)          | 6,599 / - (-)            | - / - (-)              |
| trtllm-405b-fp4-b200-128-2048   | THR   | llama3_405b @ 4xb200_sxm TP4PP1 B0 128/2048 fp4           | - / - (-)               | - / - (-)          | 7,178 / 119,908 (0.06x)  | - / - (-)              |
| trtllm-405b-fp4-gb200-128-2048  | THR   | llama3_405b @ 4xgb200_nvl72_node TP4PP1 B0 128/2048 fp4   | - / - (-)               | - / - (-)          | 7,497 / - (-)            | - / - (-)              |
| trtllm-405b-fp4-b200-1024-2048  | THR   | llama3_405b @ 4xb200_sxm TP4PP1 B0 1024/2048 fp4          | - / - (-)               | - / - (-)          | 4,833 / 42,211 (0.11x)   | - / - (-)              |
| trtllm-405b-fp4-gb200-1024-2048 | THR   | llama3_405b @ 4xgb200_nvl72_node TP4PP1 B0 1024/2048 fp4  | - / - (-)               | - / - (-)          | 4,686 / - (-)            | - / - (-)              |
| trtllm-405b-fp4-b200-2048-128   | THR   | llama3_405b @ 4xb200_sxm TP4PP1 B0 2048/128 fp4           | - / - (-)               | - / - (-)          | 738 / 2,630 (0.28x)      | - / - (-)              |
| trtllm-405b-fp4-gb200-2048-128  | THR   | llama3_405b @ 4xgb200_nvl72_node TP4PP1 B0 2048/128 fp4   | - / - (-)               | - / - (-)          | 762 / - (-)              | - / - (-)              |
| trtllm-405b-fp4-b200-2048-2048  | THR   | llama3_405b @ 4xb200_sxm TP4PP1 B0 2048/2048 fp4          | - / - (-)               | - / - (-)          | 4,024 / 24,122 (0.17x)   | - / - (-)              |
| trtllm-405b-fp4-gb200-2048-2048 | THR   | llama3_405b @ 4xgb200_nvl72_node TP4PP1 B0 2048/2048 fp4  | - / - (-)               | - / - (-)          | 4,327 / - (-)            | - / - (-)              |
| spheron-70b-h100-fp8            | THR   | llama3_3_70b @ 8xh100_sxm TP8PP1 B0 1024/2048 fp8         | - / - (-)               | - / - (-)          | 24,528 / 2,714 (9.04x)   | - / - (-)              |
| spheron-70b-h200-fp8            | THR   | llama3_3_70b @ 8xh200_sxm TP8PP1 B0 1024/2048 fp8         | - / - (-)               | - / - (-)          | 34,992 / 1,922 (18.20x)  | - / - (-)              |
| spheron-70b-b200-fp8            | THR   | llama3_3_70b @ 8xb200_sxm TP8PP1 B0 1024/2048 fp8         | - / - (-)               | - / - (-)          | 55,776 / 5,698 (9.79x)   | - / - (-)              |
| spheron-70b-b200-fp4            | THR   | llama3_3_70b @ 8xb200_sxm TP8PP1 B0 1024/2048 fp4         | - / - (-)               | - / - (-)          | 102,728 / 18,726 (5.49x) | - / - (-)              |
| koyeb-8b-h200-b1-512-512        | LAT   | llama3_1_8b @ 1xh200_sxm TP1PP1 B1 512/512 fp16           | - / 8 (-)               | - / 3.4 (-)        | 169 / 297 (0.57x)        | 3,160 / 1,726 (1.83x)  |
| koyeb-8b-h100-b1-512-512        | LAT   | llama3_1_8b @ 1xh100_sxm TP1PP1 B1 512/512 fp16           | - / 8 (-)               | - / 4.8 (-)        | 99 / 207 (0.48x)         | 5,390 / 2,469 (2.18x)  |
| koyeb-8b-a100-b1-512-512        | LAT   | llama3_1_8b @ 1xa100_sxm_80g TP1PP1 B1 512/512 fp16       | - / 27 (-)              | - / 7.9 (-)        | 86 / 126 (0.68x)         | 6,160 / 4,070 (1.51x)  |
| koyeb-8b-h200-b8-512-512        | LAT   | llama3_1_8b @ 1xh200_sxm TP1PP1 B8 512/512 fp16           | - / 67 (-)              | - / 3.5 (-)        | 1,309 / 2,202 (0.59x)    | - / 1,860 (-)          |
| koyeb-8b-h100-b8-512-512        | LAT   | llama3_1_8b @ 1xh100_sxm TP1PP1 B8 512/512 fp16           | - / 67 (-)              | - / 5.0 (-)        | 816 / 1,554 (0.53x)      | - / 2,636 (-)          |
| koyeb-8b-a100-b8-512-512        | LAT   | llama3_1_8b @ 1xa100_sxm_80g TP1PP1 B8 512/512 fp16       | - / 214 (-)             | - / 8.2 (-)        | 652 / 924 (0.71x)        | - / 4,434 (-)          |
| koyeb-8b-h200-b32-512-512       | LAT   | llama3_1_8b @ 1xh200_sxm TP1PP1 B32 512/512 fp16          | - / 270 (-)             | - / 4.0 (-)        | 4,621 / 7,062 (0.65x)    | - / 2,320 (-)          |
| koyeb-8b-h100-b32-512-512       | LAT   | llama3_1_8b @ 1xh100_sxm TP1PP1 B32 512/512 fp16          | - / 270 (-)             | - / 5.7 (-)        | 3,008 / 5,108 (0.59x)    | - / 3,208 (-)          |
| koyeb-8b-a100-b32-512-512       | LAT   | llama3_1_8b @ 1xa100_sxm_80g TP1PP1 B32 512/512 fp16      | - / 854 (-)             | - / 9.4 (-)        | 2,083 / 2,884 (0.72x)    | - / 5,681 (-)          |
| koyeb-8b-h200-b1-1024-1024      | LAT   | llama3_1_8b @ 1xh200_sxm TP1PP1 B1 1024/1024 fp16         | - / 17 (-)              | - / 3.4 (-)        | 168 / 295 (0.57x)        | 6,240 / 3,473 (1.80x)  |
| koyeb-8b-h100-b1-1024-1024      | LAT   | llama3_1_8b @ 1xh100_sxm TP1PP1 B1 1024/1024 fp16         | - / 17 (-)              | - / 4.8 (-)        | 99 / 206 (0.48x)         | 10,920 / 4,969 (2.20x) |
| koyeb-8b-a100-b1-1024-1024      | LAT   | llama3_1_8b @ 1xa100_sxm_80g TP1PP1 B1 1024/1024 fp16     | - / 54 (-)              | - / 7.9 (-)        | 86 / 125 (0.69x)         | 12,290 / 8,191 (1.50x) |
| koyeb-8b-h200-b8-1024-1024      | LAT   | llama3_1_8b @ 1xh200_sxm TP1PP1 B8 1024/1024 fp16         | - / 137 (-)             | - / 3.7 (-)        | 1,289 / 2,104 (0.61x)    | - / 3,894 (-)          |
| koyeb-8b-h100-b8-1024-1024      | LAT   | llama3_1_8b @ 1xh100_sxm TP1PP1 B8 1024/1024 fp16         | - / 137 (-)             | - / 5.3 (-)        | 722 / 1,484 (0.49x)      | - / 5,520 (-)          |
| koyeb-8b-a100-b8-1024-1024      | LAT   | llama3_1_8b @ 1xa100_sxm_80g TP1PP1 B8 1024/1024 fp16     | - / 434 (-)             | - / 8.6 (-)        | 632 / 883 (0.72x)        | - / 9,279 (-)          |
| koyeb-8b-h200-b32-1024-1024     | LAT   | llama3_1_8b @ 1xh200_sxm TP1PP1 B32 1024/1024 fp16        | - / 548 (-)             | - / 4.7 (-)        | 4,419 / 6,141 (0.72x)    | - / 5,336 (-)          |
| koyeb-8b-h100-b32-1024-1024     | LAT   | llama3_1_8b @ 1xh100_sxm TP1PP1 B32 1024/1024 fp16        | - / 548 (-)             | - / 6.7 (-)        | 2,401 / 4,423 (0.54x)    | - / 7,409 (-)          |
| koyeb-8b-a100-b32-1024-1024     | LAT   | llama3_1_8b @ 1xa100_sxm_80g TP1PP1 B32 1024/1024 fp16    | - / 1,737 (-)           | - / 11.0 (-)       | 1,888 / 2,519 (0.75x)    | - / 13,009 (-)         |
| koyeb-8b-h200-b1-4096-1024      | LAT   | llama3_1_8b @ 1xh200_sxm TP1PP1 B1 4096/1024 fp16         | - / 75 (-)              | - / 3.5 (-)        | 164 / 283 (0.58x)        | 6,460 / 3,617 (1.79x)  |
| koyeb-8b-h100-b1-4096-1024      | LAT   | llama3_1_8b @ 1xh100_sxm TP1PP1 B1 4096/1024 fp16         | - / 75 (-)              | - / 5.0 (-)        | 99 / 199 (0.50x)         | 10,930 / 5,151 (2.12x) |
| koyeb-8b-a100-b1-4096-1024      | LAT   | llama3_1_8b @ 1xa100_sxm_80g TP1PP1 B1 4096/1024 fp16     | - / 238 (-)             | - / 8.1 (-)        | 83 / 119 (0.70x)         | 12,690 / 8,577 (1.48x) |
| koyeb-8b-h200-b8-4096-1024      | LAT   | llama3_1_8b @ 1xh200_sxm TP1PP1 B8 4096/1024 fp16         | - / 601 (-)             | - / 4.3 (-)        | 1,162 / 1,624 (0.72x)    | - / 5,046 (-)          |
| koyeb-8b-h100-b8-4096-1024      | LAT   | llama3_1_8b @ 1xh100_sxm TP1PP1 B8 4096/1024 fp16         | - / 601 (-)             | - / 6.2 (-)        | 616 / 1,175 (0.52x)      | - / 6,969 (-)          |
| koyeb-8b-a100-b8-4096-1024      | LAT   | llama3_1_8b @ 1xa100_sxm_80g TP1PP1 B8 4096/1024 fp16     | - / 1,906 (-)           | - / 10.2 (-)       | 544 / 662 (0.82x)        | - / 12,368 (-)         |
| koyeb-8b-h200-b32-4096-1024     | LAT   | llama3_1_8b @ 1xh200_sxm TP1PP1 B32 4096/1024 fp16        | - / 2,405 (-)           | - / 7.4 (-)        | 3,209 / 3,296 (0.97x)    | - / 9,942 (-)          |
| koyeb-8b-h100-b32-4096-1024     | LAT   | llama3_1_8b @ 1xh100_sxm TP1PP1 B32 4096/1024 fp16        | - / 2,405 (-)           | - / 10.5 (-)       | 1,591 / 2,482 (0.64x)    | - / 13,204 (-)         |
| koyeb-8b-a100-b32-4096-1024     | LAT   | llama3_1_8b @ 1xa100_sxm_80g TP1PP1 B32 4096/1024 fp16    | - / 7,624 (-)           | - / 17.3 (-)       | 1,202 / 1,292 (0.93x)    | - / 25,367 (-)         |
| koyeb-q25-7b-h200-b1-512-512    | LAT   | qwen25_7b @ 1xh200_sxm TP1PP1 B1 512/512 fp16             | - / 8 (-)               | - / 3.2 (-)        | 182 / 313 (0.58x)        | 2,810 / 1,634 (1.72x)  |
| koyeb-q25-7b-h100-b1-512-512    | LAT   | qwen25_7b @ 1xh100_sxm TP1PP1 B1 512/512 fp16             | - / 8 (-)               | - / 4.6 (-)        | 105 / 219 (0.48x)        | 5,100 / 2,338 (2.18x)  |
| koyeb-q25-7b-a100-b1-512-512    | LAT   | qwen25_7b @ 1xa100_sxm_80g TP1PP1 B1 512/512 fp16         | - / 25 (-)              | - / 7.5 (-)        | 93 / 133 (0.70x)         | 5,740 / 3,853 (1.49x)  |
| koyeb-q25-7b-h200-b8-512-512    | LAT   | qwen25_7b @ 1xh200_sxm TP1PP1 B8 512/512 fp16             | - / 64 (-)              | - / 3.2 (-)        | 1,371 / 2,378 (0.58x)    | - / 1,723 (-)          |
| koyeb-q25-7b-h100-b8-512-512    | LAT   | qwen25_7b @ 1xh100_sxm TP1PP1 B8 512/512 fp16             | - / 64 (-)              | - / 4.6 (-)        | 808 / 1,678 (0.48x)      | - / 2,441 (-)          |
| koyeb-q25-7b-a100-b8-512-512    | LAT   | qwen25_7b @ 1xa100_sxm_80g TP1PP1 B8 512/512 fp16         | - / 202 (-)             | - / 7.6 (-)        | 699 / 997 (0.70x)        | - / 4,108 (-)          |
| koyeb-q25-7b-h200-b32-512-512   | LAT   | qwen25_7b @ 1xh200_sxm TP1PP1 B32 512/512 fp16            | - / 255 (-)             | - / 3.5 (-)        | 4,523 / 8,083 (0.56x)    | - / 2,027 (-)          |
| koyeb-q25-7b-h100-b32-512-512   | LAT   | qwen25_7b @ 1xh100_sxm TP1PP1 B32 512/512 fp16            | - / 255 (-)             | - / 5.0 (-)        | 2,937 / 5,864 (0.50x)    | - / 2,794 (-)          |
| koyeb-q25-7b-a100-b32-512-512   | LAT   | qwen25_7b @ 1xa100_sxm_80g TP1PP1 B32 512/512 fp16        | - / 809 (-)             | - / 8.1 (-)        | 2,486 / 3,290 (0.76x)    | - / 4,980 (-)          |
| koyeb-q25-7b-h200-b1-1024-1024  | LAT   | qwen25_7b @ 1xh200_sxm TP1PP1 B1 1024/1024 fp16           | - / 16 (-)              | - / 3.2 (-)        | 182 / 312 (0.58x)        | 5,620 / 3,278 (1.71x)  |
| koyeb-q25-7b-h100-b1-1024-1024  | LAT   | qwen25_7b @ 1xh100_sxm TP1PP1 B1 1024/1024 fp16           | - / 16 (-)              | - / 4.6 (-)        | 106 / 218 (0.49x)        | 10,040 / 4,689 (2.14x) |
| koyeb-q25-7b-a100-b1-1024-1024  | LAT   | qwen25_7b @ 1xa100_sxm_80g TP1PP1 B1 1024/1024 fp16       | - / 51 (-)              | - / 7.5 (-)        | 92 / 132 (0.69x)         | 11,410 / 7,729 (1.48x) |
| koyeb-q25-7b-h200-b8-1024-1024  | LAT   | qwen25_7b @ 1xh200_sxm TP1PP1 B8 1024/1024 fp16           | - / 129 (-)             | - / 3.3 (-)        | 1,368 / 2,326 (0.59x)    | - / 3,522 (-)          |
| koyeb-q25-7b-h100-b8-1024-1024  | LAT   | qwen25_7b @ 1xh100_sxm TP1PP1 B8 1024/1024 fp16           | - / 129 (-)             | - / 4.7 (-)        | 800 / 1,641 (0.49x)      | - / 4,991 (-)          |
| koyeb-q25-7b-a100-b8-1024-1024  | LAT   | qwen25_7b @ 1xa100_sxm_80g TP1PP1 B8 1024/1024 fp16       | - / 410 (-)             | - / 7.8 (-)        | 682 / 976 (0.70x)        | - / 8,397 (-)          |
| koyeb-q25-7b-h200-b32-1024-1024 | LAT   | qwen25_7b @ 1xh200_sxm TP1PP1 B32 1024/1024 fp16          | - / 517 (-)             | - / 3.8 (-)        | 4,719 / 7,513 (0.63x)    | - / 4,361 (-)          |
| koyeb-q25-7b-h100-b32-1024-1024 | LAT   | qwen25_7b @ 1xh100_sxm TP1PP1 B32 1024/1024 fp16          | - / 517 (-)             | - / 5.4 (-)        | 2,802 / 5,438 (0.52x)    | - / 6,025 (-)          |
| koyeb-q25-7b-a100-b32-1024-1024 | LAT   | qwen25_7b @ 1xa100_sxm_80g TP1PP1 B32 1024/1024 fp16      | - / 1,640 (-)           | - / 8.8 (-)        | 2,363 / 3,066 (0.77x)    | - / 10,689 (-)         |
| koyeb-q25-7b-h200-b1-4096-1024  | LAT   | qwen25_7b @ 1xh200_sxm TP1PP1 B1 4096/1024 fp16           | - / 70 (-)              | - / 3.2 (-)        | 180 / 304 (0.59x)        | 5,690 / 3,369 (1.69x)  |
| koyeb-q25-7b-h100-b1-4096-1024  | LAT   | qwen25_7b @ 1xh100_sxm TP1PP1 B1 4096/1024 fp16           | - / 70 (-)              | - / 4.6 (-)        | 104 / 213 (0.49x)        | 10,200 / 4,797 (2.13x) |
| koyeb-q25-7b-a100-b1-4096-1024  | LAT   | qwen25_7b @ 1xa100_sxm_80g TP1PP1 B1 4096/1024 fp16       | - / 221 (-)             | - / 7.6 (-)        | 90 / 128 (0.70x)         | 11,860 / 7,987 (1.48x) |
| koyeb-q25-7b-h200-b8-4096-1024  | LAT   | qwen25_7b @ 1xh200_sxm TP1PP1 B8 4096/1024 fp16           | - / 558 (-)             | - / 3.6 (-)        | 1,143 / 1,927 (0.59x)    | - / 4,252 (-)          |
| koyeb-q25-7b-h100-b8-4096-1024  | LAT   | qwen25_7b @ 1xh100_sxm TP1PP1 B8 4096/1024 fp16           | - / 558 (-)             | - / 5.2 (-)        | 750 / 1,400 (0.54x)      | - / 5,851 (-)          |
| koyeb-q25-7b-a100-b8-4096-1024  | LAT   | qwen25_7b @ 1xa100_sxm_80g TP1PP1 B8 4096/1024 fp16       | - / 1,769 (-)           | - / 8.5 (-)        | 636 / 783 (0.81x)        | - / 10,464 (-)         |
| koyeb-q25-7b-h200-b32-4096-1024 | LAT   | qwen25_7b @ 1xh200_sxm TP1PP1 B32 4096/1024 fp16          | - / 2,232 (-)           | - / 4.9 (-)        | 1,253 / 4,502 (0.28x)    | - / 7,279 (-)          |
| koyeb-q25-7b-h100-b32-4096-1024 | LAT   | qwen25_7b @ 1xh100_sxm TP1PP1 B32 4096/1024 fp16          | - / 2,232 (-)           | - / 7.1 (-)        | 2,156 / 3,463 (0.62x)    | - / 9,464 (-)          |
| koyeb-q25-7b-a100-b32-4096-1024 | LAT   | qwen25_7b @ 1xa100_sxm_80g TP1PP1 B32 4096/1024 fp16      | - / 7,076 (-)           | - / 11.6 (-)       | 1,913 / 1,729 (1.11x)    | - / 18,957 (-)         |

## Protocol classification (read before interpreting)

| Protocol | Meaning | How it is compared |
|---|---|---|
| `LAT` | Latency protocol, low-concurrency / single-request | Directly comparable to the single-batch model; ideal from `evaluate()` |
| `THR` | Throughput protocol, continuous-batching saturated steady state | Compared against the **decode roofline ceiling** at the VRAM-limited max batch; only an upper bound |
| `CFG` | Deployment config only, no perf numbers | Not compared (kept in the archive for layout sanity) |

## Interpreting the ratios

- **LAT rows**: ratio ~2–5× is the expected calibration region — at batch=1 the
  small-message collectives are fully exposed (α) and prefill MFU is well below 1.
  The TTFT ratio ≈ prefill-MFU gap; the TPOT ratio ≈ decode efficiency + exposed α.
- **LAT prefill-only rows** (LMSYS, `outputLen=1`): throughput = input tok/s; ratio <1
  reflects prefill MFU × PP strong-scaling efficiency.
- **THR rows**: uses `evaluate()` with IDEAL calibration (flash attention enabled,
  VRAM-limited max batch). Ratio ≈ 1 means the engine achieves the ideal steady-state
  throughput. Ratio >1 indicates the engine outperforms the single-batch model
  (e.g. continuous batching hides prefill overhead better than the model assumes).

## Calibration anchors (v0, order-of-magnitude only)

| Constant | Anchor | Evidence |
|---|---|---|
| `bwEffDecode` | ~0.55 | throughput cross-checks 0.48–0.57 |
| `mfuPrefill` | ~0.5 (TRT-LLM batch=64, SGLang H20 PP=1) | prefill comparisons ×3 |
| `ppEfficiency` | ~0.8–0.9 (chunked prefill, PP≤8) | LMSYS strong scaling 0.77–0.91 |
| α (small-msg collective latency) | default intra 0.01 ms / inter 0.03 ms; measure on target (below) | Splitwise batch=1 ratios 4–6 indirectly confirm its presence |

## Measuring calibration constants on the target machine

Later the UI will expose a `Calibration` input box; per-constant measurement:

### α (`alphaIntraMs` / `alphaInterMs`): per-call small-message collective latency

**Measure directly; do not back it out of inference data** — batch=1 decode TBT also
contains small-GEMM kernel latency and would contaminate α.

1. **Intra-node** (= TP-group α): nccl-tests small messages,
   `./build/all_reduce_perf -b 8 -e 8K -g <tp>`; read the `avg time` in the 8B–4KB
   range (bandwidth term negligible there, so the time ≈ α_intra).
   Without nccl-tests: a torch.distributed micro-benchmark (`torchrun --nproc_per_node=8`,
   loop `dist.all_reduce` on a small tensor, warm up, average ~1000 iters with cuda sync).
2. **Inter-node**: same command across 2 nodes with `-g 1` → α_inter. For EP all-to-all,
   use `alltoall_perf` at small sizes (the model currently reuses α_intra; may split later).
3. **Why this is correct**: NCCL auto-selects LL/LL128/tree at small sizes, so the
   measured per-call total latency already includes the real algorithm choice — exactly
   the α semantics of `ringAllreduceMs = bandwidth term + α` (per-call, not per-hop).
4. **Sanity ranges**: NVLink intra-node 8-GPU ≈ 5–30 μs; PCIe ≈ 50–200 μs; IB inter-node
   ≈ 20–100 μs. Off by an order of magnitude ⇒ check the environment (PCIe gen,
   NCCL_P2P_LEVEL, shared NVSwitch). Fill in ms (divide a μs reading by 1000).
5. **Defaults when unmeasured**: `DEFAULT_ALPHA_INTRA_MS = 0.01`, `DEFAULT_ALPHA_INTER_MS = 0.03`,
   same order as the tps old heuristic (8/25 μs per hop) and measured NCCL small-message
   totals (6–11 μs). Note α only takes effect when the matching `*CommOverlap < 1`
   (decode small-batch collectives are not overlapped). Do not derive α from batch=1 TBT.

### Other constants

| Constant | How to measure |
|---|---|
| `mfuPrefill` | measured large-batch prefill tok/s ÷ ideal tok/s |
| `bwEffDecode` | single-card decode tok/s ÷ (BW / weight bytes) |
| `commEffIntra/Inter` | nccl-tests **large** message (≥128MB) measured BW ÷ peak BW |
| `tpCommOverlap` / `epCommOverlap` / `ppCommOverlap` | fit from an exposed-comm scenario (e.g. batch=1 decode measured minus bandwidth term); ideal = 1, conservative start ≈ 0.5 |
| `ppEfficiency` (not yet a Calibration field) | LMSYS anchor 0.8–0.9 |
