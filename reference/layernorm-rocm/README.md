# layernorm-rocm — second operator archive

Phase 1.5 — proves the AKO4X-rocm scaffolding generalizes beyond RMSNorm.

LayerNorm BF16: `y = (x - mean(x)) / sqrt(var(x) + eps) * w + b`.

Tested at 12 shapes: 4 batch sizes × 3 hidden dims (128 MLA-per-head, 4096 Llama, 7168 DSR1).

See [`RESULTS.md`](RESULTS.md) for the full table and analysis.

**Headline:** v1 vector_loads beats AITER on 11/12 shapes; 1.81× geomean speedup.
