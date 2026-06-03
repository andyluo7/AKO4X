# ROCm closed-loop ledger — fp8_gemm_dsr1 on MI355X

Append-only round summary. One line per round.

| round | variant slug | result | takeaway |
|------:|---|---|---|
| 1 | v6_lds_swizzle | WIN 137.7 µs / 873 TFLOP/s | 8-byte LDS row pad eliminates 8-way bank conflicts; 1.76× over v2 — conflicts were the dominant bottleneck |
