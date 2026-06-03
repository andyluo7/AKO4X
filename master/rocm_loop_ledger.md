# ROCm closed-loop ledger — fp8_gemm_dsr1 on MI355X

Append-only round summary. One line per round.

| round | variant slug | result | takeaway |
|------:|---|---|---|
| 1 | v6_lds_swizzle | WIN 137.7 µs / 873 TFLOP/s | 8-byte LDS row pad eliminates 8-way bank conflicts; 1.76× over v2 — conflicts were the dominant bottleneck |
| 2 | v7_8wave | WIN 102.3 µs / 1175 TFLOP/s | 8 waves/block (24/CU) vs 4 (12/CU) hides 64-cycle MFMA latency; 1.35× over v6 |
| 3 | v8_wider_n | DEAD-END 109.8 µs | BN=256 cuts DRAM 25% but 16 waves/CU (2 blocks×8 waves) starves MFMA pipeline — occupancy loss dominates |
| 4 | v9_256n_16w | DEAD-END 113.5 µs | BN=256+32 waves/CU still 12% slower; 2 blocks/CU → 128 concurrent blocks (vs 192) reduces chip-level ILP — BN=256 ruled out |
| 5 | v10_global_lds | FAILED (correctness) | global_load_lds writes to M0+lane_id*4 (contiguous); our padded tile (ROW_STRIDE=72, LDS_PAD=8) mismatches that stride — removing pad would restore v6's bank conflicts |
