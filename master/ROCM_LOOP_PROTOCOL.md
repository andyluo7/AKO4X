# ROCm closed-loop round protocol — FP8 GEMM, MI355X gfx950

You are the **master + engineer** for an AKO4X-style closed-loop kernel-optimization campaign on AMD MI355X (gfx950, CDNA4). You play both roles in one session: you decide what to try, you implement it, you measure it, you archive the result.

## Hardware/software

- **Box:** `smci355-ccs-aus-m15-17` (you are running on it). 8× AMD Instinct MI355X.
- **Container:** `rocm/pytorch-private:mxfp8-gfx950-v26.6` (torch 2.12+rocm7.1, AITER 25.9, FP8 e4m3fn native on gfx950).
- **Docker invocation:** see `reference/fp8-gemm-dsr1-rocm/README.md`.

## Operator

`fp8_gemm_dsr1` — DSR1 FFN-like FP8 GEMM. Single shape (M=4096, N=7168, K=2048), e4m3fn inputs, BF16 output, per-token A scale + per-channel B scale. 120 GFLOP per call. See `reference/fp8-gemm-dsr1-rocm/RESULTS.md` for the full baseline + variant table.

## State you inherit (read this first, in order)

1. `reference/fp8-gemm-dsr1-rocm/RESULTS.md` — measured table, baseline bar, what's been tried.
2. `reference/fp8-gemm-dsr1-rocm/variants/v2_wider_tile/kernel.hip` — current best (the anchor).
3. `reference/fp8-gemm-dsr1-rocm/variants/{v3_double_buffer, v4_bigger_mfma, v5_block_256x128}/kernel.hip` — three documented dead-ends. **READ THE `DEAD-END` HEADERS** before proposing a direction; do not re-attempt levers that were already rejected with reasoning.
4. `reference/fp8-gemm-dsr1-rocm/baseline_mi355x.json` — most recent MI355X measurements.

Current state on MI355X (8% faster than MI350X across the board):
- **Bar to beat:** AITER `gemm_a8w8_bpreshuffle` = 72.2 µs / 1665 TFLOP/s.
- **Our anchor (v2):** 243 µs / 495 TFLOP/s (≈ 3.4× behind bar).
- **Three rejected hypotheses:** load latency (v3), MFMA dispatch overhead (v4), DRAM amplification (v5). Remaining unexplored: LDS bank-conflict-free swizzle, `__builtin_amdgcn_global_load_lds`, multi-stage software pipelining.

## Round protocol (10 steps, repeat per round)

1. **Pick direction.** Choose one untried lever from the v5 header's "remaining gap" list, or invent a new one. Write a one-line justification + a one-line falsification criterion ("if this hypothesis is right, X should happen") before coding.

2. **Pick variant slug.** Use `vN_<short_name>` where N continues the sequence (v6, v7, ...). Slug must be `<= 24 chars`, kebab- or snake-case.

3. **Create `reference/fp8-gemm-dsr1-rocm/variants/<slug>/`** with `kernel.hip` and `binding.cpp`. Copy structure from v2 (the anchor) and apply your delta.

4. **Header.** `kernel.hip` MUST carry the 5-section header (Identity / Delta from parent / Lessons / Dead-ends / Open directions) per `templates/agent/lessons-convention.md`.

5. **Build + correctness.** Run the benchmark; verify the variant passes the correctness gate. If correctness fails: debug, iterate. Do NOT mark the round complete until correctness passes.

6. **Time.** With correctness passing, capture the full latency table from `python benchmark_fp8_gemm.py --iters 100 --warmup 5 --out baseline_mi355x.json`.

7. **Classify the result:**
    - **Win** (beats v2's current best, < 243 µs): becomes the new anchor. Update RESULTS.md anchor row.
    - **Tie / loss** (within 5% of v2, or slower): document as dead-end. **Append a `## DEAD-END` block to the variant's `kernel.hip` header** explaining what was tried, what was measured, and which hypothesis was falsified.
    - **Crash / never-correct**: move the variant dir to `_failed/round-<N>/<slug>/` and write a one-paragraph postmortem.

8. **Archive.** Whether win / dead-end / failed, the artifact stays in the repo. Future rounds (and future humans) read it.

9. **Ledger.** Append a one-line entry to `master/rocm_loop_ledger.md`:
    ```
    round-N | <slug> | <result: WIN µs / DEAD-END / FAILED> | <one-line takeaway>
    ```

10. **Commit + push** the round's changes to `rocm-port` with a message of the form `round-N <slug>: <one-line result>`. Then move to the next round.

## Constraints

- **Do not modify** the benchmark driver, baseline torch_naive reference, AITER probes, or hip_stock. They are part of the comparison harness.
- **Do not retry** any of the dead-end hypotheses (load latency overlap via SW double-buffer, MFMA-shape tuning at fixed tile, naïve bigger block) without a new mechanism distinguishing your attempt from the rejected one.
- **5 rounds total.** Stop after round 5 and write a summary at `reference/fp8-gemm-dsr1-rocm/ROUND_LOG.md`.

## What "good iteration" looks like

The point is not that every round produces a winner. It is that every round produces a **measurement** and a **structured artifact** (variant + header) so the archive's signal-to-noise improves monotonically. A round that documents a clean dead-end is a successful round.
