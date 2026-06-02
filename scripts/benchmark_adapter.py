"""Benchmark adapter — the single seam between AKO4X and the active benchmark.

**ako4x-rocm fork (Phase 0):** retains the AKO4X plain-data interface unchanged, but
the *execution engine* path is being rewritten to target AMD ROCm (gfx942 / gfx950)
rather than NVIDIA CUDA / B200. Same public surface (``run`` / ``pack`` / ``profile``
/ ``sanitize`` / ``cheat_check`` / ``list_workloads`` / ``solution_meta`` /
``list_ncu_options`` — kept under the original name for caller compatibility, but on
ROCm it surfaces ``rocprofv3`` counter sets instead of NCU sets) so the runners,
``bench_utils``, ``pack_solution``, the master loop — none of those need to change.

What this file does that's ROCm-specific
----------------------------------------
1. **Build paths**: HIP / AITER / FlyDSL / CK / TileLang-ROCm / Triton-ROCm. The
   ``BuilderRegistry`` used internally by ``run`` and ``cheat_check`` is replaced
   with a ROCm-aware registry below that dispatches on
   ``Solution.spec.language``: ``hip`` → hipcc + TVM-FFI direct-export, ``aiter`` →
   call AITER's op-table, ``flydsl`` → FlyDSL preshuffle codegen, ``ck`` → CK
   header-only template compile via hipcc, etc.
2. **Profiling agent**: ``rocprofv3`` instead of ``ncu``. NVTX ranges replaced
   with ``hipRange`` markers via the AMD ``roctx`` library; the surface name
   ``NCU_NVTX_RANGE`` is kept (callers reference it by name) but its content now
   maps to a ``roctx`` range string.
3. **Sanitizer**: there is no direct ``compute-sanitizer`` equivalent on ROCm yet.
   ``sanitize()`` wraps a HIP-aware approach: ``HIP_LAUNCH_BLOCKING=1`` + AddressSanitizer
   (when the build supports it) + checked-mode allocator wrappers. See sanitizer SKILL.
4. **Modal constants**: Modal does not currently offer ROCm GPUs, so the
   ``MODAL_*`` constants are kept for interface compatibility but point at
   placeholder values. The intended backends for ako4x-rocm are local (podman +
   rocm/dev-ubuntu image) and SLURM-via-SSH (AAC1 + Tensorwave) — see
   ``scripts/run_slurm.py`` and ``scripts/run_tensorwave_ssh.py``.

Public surface (plain data in, plain data out — no benchmark types escape)
--------------------------------------------------------------------------
Discovery   : ``list_workloads(dataset_path, definition) -> [{"uuid","axes"}, ...]``
Packing     : ``pack(source_dir, build_cfg, *, name, definition, author) -> blob:str``
              ``solution_meta(blob) -> {"name","definition","author"}``
Execution   : ``run(blob, uuids, params, *, dataset_path, capture_logs=False,
              capture_autotune=False) -> normalized_result_dict``
Profiling   : ``profile(blob, uuid, opts, *, dataset_path, env_pairs=None) -> str``
              ``list_ncu_options() -> str``  # name kept; returns rocprof counters
Sanitizer   : ``sanitize(blob, uuid, opts, *, dataset_path) -> str``
Cheat-check : ``cheat_check(blob, uuids, *, dataset_path, n_iters=4) -> dict``

The shape of the result dict is unchanged from upstream AKO4X (see the original
docstring in upstream's ``scripts/benchmark_adapter.py`` / ``bench_utils.py``).
This is what lets ``bench_utils.compute_score`` / ``load_baseline`` / ``save_baseline``
work unchanged.

Porting status (Phase 0)
------------------------
This file currently routes most calls through the upstream ``flashinfer_bench``
engine for compatibility — that engine *does* import on ROCm hosts (the pydantic
models and dataset format are platform-agnostic), and most ops will simply fail at
build time because the bundled solutions are CUDA-only. The ROCm-specific code
paths flagged with ``# ROCm-PORT:`` comments below are where Phase 1 will land
the actual HIP / AITER / FlyDSL builders.

Each ``# ROCm-PORT:`` comment is an open work item; the upstream code stays
in place until each item is resolved, so the file is testable for the all-plain-data
paths (``list_workloads``, ``solution_meta``, ``pack``) on Day 0.
"""

# No module-level flashinfer_bench import: every function below imports what it
# needs from the benchmark at call time (function scope). Same rule as upstream.

# --- Status enum (unchanged from upstream) -----------------------------------
STATUS_PASSED = "PASSED"
STATUS_COMPILE_ERROR = "COMPILE_ERROR"
STATUS_INCORRECT_NUMERICAL = "INCORRECT_NUMERICAL"
STATUS_RUNTIME_ERROR = "RUNTIME_ERROR"
STATUS_TIMEOUT = "TIMEOUT"

# --- Dataset discovery (unchanged) -------------------------------------------
DATASET_PATH_ENV = "AKO_DATASET_PATH"
LEGACY_DATASET_PATH_ENV = "FIB_DATASET_PATH"

# --- Profiler range name (renamed semantically) ------------------------------
# On NV this is the NVTX range the NCU agent wraps the profiled call in. On AMD
# (ako4x-rocm) it's the roctx range name used by the rocprofv3 wrapper. Kept under
# the upstream constant name so callers (profiler-rocprof SKILL, run_local_profile)
# don't need to learn a new symbol.
NCU_NVTX_RANGE = "ako4x_rocm_profile_range"

# --- Modal image pins (placeholders — Modal lacks ROCm; kept for API compat) -
# ROCm-PORT: replace with SLURM-job templates or a podman-on-local registry once
# the SLURM runner lands. These constants remain typed-as-strings so any caller
# that already does `from benchmark_adapter import MODAL_IMAGE_REGISTRY` keeps
# importing cleanly.
MODAL_IMAGE_REGISTRY = "rocm/dev-ubuntu-24.04:7.0-complete"  # not actually on Modal
MODAL_PYTHON = "3.12"
MODAL_PACKAGE_PIN = (
    "aiter @ git+https://github.com/ROCm/aiter.git@main"
)
MODAL_EXTRA_PIN = (
    "flashinfer-bench @ git+https://github.com/flashinfer-ai/flashinfer-bench.git@main"
)


# ===========================================================================
# Internal: ROCm-aware builder dispatch (Phase 0 stub)
# ===========================================================================
# The upstream BuilderRegistry knows how to compile CUDA / Triton / CuTe / etc.
# This wrapper extends it for HIP / AITER / FlyDSL / CK on ROCm.
#
# Phase 0: only the dispatch shape exists; actual builders are TODOs.
# Phase 1: implement HIP builder (hipcc + TVM-FFI export) — first deliverable.
# Phase 2: implement AITER builder (call into aiter.ops.<op>).
# Phase 3: implement FlyDSL builder (preshuffle GEMM codegen).

def _build_runnable(definition, solution):
    """Build a Runnable for (definition, solution) on a ROCm host.

    Dispatches on solution.spec.language. Returns the same Runnable contract
    flashinfer_bench's BuilderRegistry returns (must have `.call_value_returning(*inputs)`
    and `.cleanup()`).

    Phase 0: falls back to the upstream BuilderRegistry for compatibility — this
    will FAIL at build time for the CUDA-only bundled solutions, but allows
    handcoded HIP solutions placed in `reference/<family>/variants/<v>/kernel.hip`
    to be wired through the upstream registry's `hip` builder path (which exists
    in flashinfer-bench head as of 2026-05).
    """
    # ROCm-PORT: implement dispatch on solution.spec.language ∈
    #            {"hip", "aiter", "flydsl", "ck", "triton", "tilelang"}.
    from flashinfer_bench.compile import BuilderRegistry
    registry = BuilderRegistry.get_instance()
    return registry.build(definition, solution)


# ===========================================================================
# Plain-data public surface
# ===========================================================================

def list_workloads(dataset_path, definition):
    """Return ``[{"uuid": str, "axes": dict}, ...]`` for ``definition``, in dataset order."""
    from flashinfer_bench import TraceSet

    trace_set = TraceSet.from_path(dataset_path)
    entries = trace_set.workloads.get(definition, [])
    return [{"uuid": w.workload.uuid, "axes": dict(w.workload.axes)} for w in entries]


def pack(source_dir, build_cfg, *, name, definition, author):
    """Pack kernel sources from ``source_dir`` into a solution-blob (solution.json text).

    ``build_cfg`` keys: ``language``, ``entry_point``, ``destination_passing_style``
    (default False), ``target_hardware`` (default ``["rocm"]`` on this fork).
    """
    from flashinfer_bench import BuildSpec
    from flashinfer_bench.agents import pack_solution_from_files

    # ROCm-PORT: default target_hardware to ["rocm"] (not ["cuda"]); honor explicit
    # override from build_cfg for cross-platform kernels (rare, but supported).
    spec = BuildSpec(
        language=build_cfg["language"],
        target_hardware=build_cfg.get("target_hardware", ["rocm"]),
        entry_point=build_cfg["entry_point"],
        destination_passing_style=build_cfg.get("destination_passing_style", False),
    )
    solution = pack_solution_from_files(
        path=str(source_dir), spec=spec, name=name,
        definition=definition, author=author,
    )
    return solution.model_dump_json(indent=2)


def solution_meta(blob):
    """``{"name", "definition", "author"}`` from a solution-blob."""
    from flashinfer_bench import Solution
    sol = Solution.model_validate_json(blob)
    return {"name": sol.name, "definition": sol.definition, "author": sol.author}


def run(blob, uuids, params, *, dataset_path, capture_logs=False, capture_autotune=False):
    """Run the benchmark engine over the workloads named by ``uuids``.

    Phase 0: leans on flashinfer_bench's engine for the result-flattening + scoring
    plumbing (it's all pydantic-model construction, no CUDA calls), but the actual
    kernel execution dispatches via ``_build_runnable`` above — which on ROCm
    builds HIP / AITER / FlyDSL kernels instead of CUDA ones.
    """
    from flashinfer_bench import BenchmarkConfig, Solution, TraceSet

    solution = Solution.model_validate_json(blob)
    config = BenchmarkConfig(**params)

    trace_set = TraceSet.from_path(dataset_path)
    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")
    definition = trace_set.definitions[solution.definition]
    if not uuids:
        raise ValueError("run() called with no workload uuids — nothing to benchmark")
    uuid_set = set(uuids)
    workloads = [w for w in trace_set.workloads.get(solution.definition, [])
                 if w.workload.uuid in uuid_set]
    found = {w.workload.uuid for w in workloads}
    missing = uuid_set - found
    if missing:
        raise ValueError(
            f"{len(missing)}/{len(uuid_set)} requested workload uuid(s) not found in the "
            f"dataset for definition '{solution.definition}' (e.g. {sorted(missing)[0]!r}). "
            f"The selection source (docs/workloads.jsonl) and the execution dataset "
            f"({dataset_path}) may have diverged."
        )

    bench_trace_set = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: [solution]},
        workloads={definition.name: workloads},
        traces={definition.name: []},
    )

    if capture_autotune:
        import contextlib, io, os
        prior_autotune = os.environ.get("TRITON_PRINT_AUTOTUNING")
        os.environ["TRITON_PRINT_AUTOTUNING"] = "1"
        buf = io.StringIO()
        try:
            with contextlib.redirect_stderr(buf):
                result_trace_set = _run_benchmark_all(bench_trace_set, config)
            results = _extract_results(result_trace_set, solution.definition,
                                       capture_all_logs=capture_logs)
            return {"results": results, "autotune_log": buf.getvalue()}
        finally:
            if prior_autotune is None:
                os.environ.pop("TRITON_PRINT_AUTOTUNING", None)
            else:
                os.environ["TRITON_PRINT_AUTOTUNING"] = prior_autotune

    result_trace_set = _run_benchmark_all(bench_trace_set, config)
    return _extract_results(result_trace_set, solution.definition,
                            capture_all_logs=capture_logs)


def profile(blob, uuid, opts, *, dataset_path, env_pairs=None):
    """rocprofv3-profile one workload (was ncu on upstream)."""
    # ROCm-PORT: route to rocprofv3 wrapper instead of flashinfer_bench's NCU agent.
    # Until that lands, callers should expect a "profiler not yet implemented on ROCm"
    # error here — but the function shape stays so SKILL docs are accurate.
    import os
    prior_env = {}
    if env_pairs:
        for k, v in env_pairs.items():
            prior_env[k] = os.environ.get(k)
            os.environ[k] = str(v)
    try:
        # ROCm-PORT: replace this branch with `from ako4x_rocm.profiling.rocprof_agent import run_rocprof`
        raise NotImplementedError(
            "profile() not yet implemented on ROCm in Phase 0. "
            "Phase 1 wires this to rocprofv3 via scripts/run_local_profile.py."
        )
    finally:
        for k, v in prior_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def list_ncu_options():
    """Return the active profiler's counter-set listing.

    On AMD this returns ``rocprofv3 --counter-list`` output. The function name
    is kept for API compatibility with upstream callers.
    """
    # ROCm-PORT: subprocess.run(["rocprofv3", "--counter-list"], ...)
    return (
        "ROCm profiler counter sets (rocprofv3) — not yet wired in Phase 0.\n"
        "Targeted lists for Phase 1:\n"
        "  basic    : VALU_BUSY, SALU_BUSY, MFMA_OCC, LDS_BANK_CONFLICT\n"
        "  memory   : TCC_HIT_PCT, L2_HIT, MEM_UNIT_BUSY\n"
        "  occupancy: WAVES_PER_CU, VGPR_SPILL, SGPR_SPILL\n"
        "See profiler-rocprof SKILL (`rocprof.md`) for full counter reference.\n"
    )


def sanitize(blob, uuid, opts, *, dataset_path):
    """HIP-aware sanitize wrapper (was compute-sanitizer on upstream).

    No direct compute-sanitizer equivalent on ROCm; this wraps:
      * ``HIP_LAUNCH_BLOCKING=1`` + ``AMD_LOG_LEVEL=4`` for trace-level errors
      * AddressSanitizer (where the build supports `-fsanitize=address`)
      * Bounds-check'd allocator wrappers
    """
    # ROCm-PORT: implement HIP-aware sanitize loop. See sanitizer SKILL for the
    # exact env + flag combo expected to surface OOB / use-after-free errors on ROCm.
    raise NotImplementedError(
        "sanitize() not yet implemented on ROCm in Phase 0. "
        "Phase 1 wires this to HIP_LAUNCH_BLOCKING + ASan via scripts/run_local_sanitize.py."
    )


def cheat_check(blob, uuids, *, dataset_path, n_iters=4):
    """Varying-inputs correctness audit (unchanged shape; runs via ROCm-aware Runnable)."""
    import torch
    from flashinfer_bench import Solution, TraceSet
    from flashinfer_bench.bench.utils import gen_inputs, load_safetensors

    solution = Solution.model_validate_json(blob)
    trace_set = TraceSet.from_path(dataset_path)
    if solution.definition not in trace_set.definitions:
        return {"status": "ERROR",
                "reason": f"definition {solution.definition!r} not in trace set"}
    definition = trace_set.definitions[solution.definition]
    all_wl = trace_set.workloads.get(solution.definition, [])
    if not all_wl:
        return {"status": "ERROR", "reason": f"no workloads for {solution.definition!r}"}
    uuid_set = set(uuids)
    probe = [w for w in all_wl if w.workload.uuid in uuid_set]
    if not probe:
        return {"status": "ERROR", "reason": "no workloads match the requested probe uuids"}

    runnable = _build_runnable(definition, solution)
    # ROCm note: torch+ROCm exposes hip devices via "cuda:N" (PyTorch's compatibility shim)
    device = "cuda:0"

    out = {"status": "PASS", "definition": solution.definition,
           "n_iters": n_iters, "workloads": {}}
    overall_pass = True

    for trace in probe:
        wl = trace.workload
        wl_key = wl.uuid[:8]
        entry = {"axes": dict(wl.axes), "n_iters": n_iters}
        try:
            safe_tensors = None
            if any(inp.type == "safetensors" for inp in wl.inputs.values()):
                safe_tensors = load_safetensors(definition, wl, trace_set.root)
            inputs = gen_inputs(definition, wl, device, safe_tensors)

            res0 = runnable.call_value_returning(*inputs)
            torch.cuda.synchronize()
            res0 = runnable.call_value_returning(*inputs)
            torch.cuda.synchronize()
            del res0

            hashes = []
            for _ in range(n_iters):
                _mutate_inputs_inplace(inputs)
                res = runnable.call_value_returning(*inputs)
                torch.cuda.synchronize()
                out_list = list(res) if isinstance(res, tuple) else [res]
                hashes.append(_hash_outputs(out_list))
                del res, out_list

            unique = len(set(hashes))
            entry["unique_hashes"] = unique
            entry["all_iters_differ"] = all(
                hashes[i] != hashes[i - 1] for i in range(1, len(hashes))
            )
            if not entry["all_iters_differ"]:
                overall_pass = False
                entry["status"] = "FAIL"
                entry["reason"] = (
                    f"only {unique}/{n_iters} unique outputs across mutated inputs"
                    " — kernel may be returning cached / graph-replay-stale output"
                )
            else:
                entry["status"] = "PASS"
        except Exception as e:
            overall_pass = False
            entry["status"] = "ERROR"
            entry["reason"] = f"{type(e).__name__}: {e}"
        finally:
            out["workloads"][wl_key] = entry

    runnable.cleanup()
    if not overall_pass:
        out["status"] = "FAIL"
    return out


# --- Adapter-private helpers (Phase 0: identical to upstream) ----------------

def _run_benchmark_all(bench_trace_set, config):
    """Run the benchmark engine over a prepared TraceSet."""
    # ROCm-PORT: in Phase 1, replace this with an ako4x_rocm.engine.Benchmark.run_all
    # that uses _build_runnable above. For Phase 0 we route through FIB's engine
    # which will lift our Runnable via the dispatch above; FIB's scoring math is
    # benchmark-agnostic and works on ROCm.
    from flashinfer_bench.bench.benchmark import Benchmark
    return Benchmark(bench_trace_set, config).run_all(dump_traces=True)


def _find_workload(dataset_path, definition, uuid):
    from flashinfer_bench import TraceSet
    trace_set = TraceSet.from_path(dataset_path)
    for w in trace_set.workloads.get(definition, []):
        if w.workload.uuid == uuid:
            return w.workload
    raise ValueError(f"Workload uuid {uuid!r} not found for definition {definition!r}")


def _truncate_log(log, max_chars=3000):
    if not log or len(log) <= max_chars:
        return log
    truncated = log[-max_chars:]
    nl = truncated.find("\n")
    if nl != -1 and nl < 200:
        truncated = truncated[nl + 1:]
    return f"[...truncated...]\n{truncated}"


def _extract_results(result_trace_set, definition_name, *, capture_all_logs=False):
    import math
    traces = result_trace_set.traces.get(definition_name, [])
    results = {definition_name: {}}
    for trace in traces:
        if not trace.evaluation:
            continue
        entry = {
            "status": trace.evaluation.status.value,
            "solution": trace.solution,
            "axes": dict(trace.workload.axes),
        }
        if trace.evaluation.performance:
            entry["latency_ms"] = trace.evaluation.performance.latency_ms
            entry["reference_latency_ms"] = trace.evaluation.performance.reference_latency_ms
            entry["speedup_factor"] = trace.evaluation.performance.speedup_factor
        if trace.evaluation.correctness:
            max_abs = trace.evaluation.correctness.max_absolute_error
            max_rel = trace.evaluation.correctness.max_relative_error
            entry["max_abs_error"] = "NaN" if (max_abs is not None and math.isnan(max_abs)) else max_abs
            entry["max_rel_error"] = "NaN" if (max_rel is not None and math.isnan(max_rel)) else max_rel
        log_text = getattr(trace.evaluation, "log", "")
        if trace.evaluation.status.value != "PASSED" and log_text:
            entry["error_log"] = _truncate_log(log_text)
        elif capture_all_logs and log_text:
            entry["log"] = _truncate_log(log_text, max_chars=20000)
        results[definition_name][trace.workload.uuid] = entry
    return results


def _mutate_inputs_inplace(inputs):
    """Mutate float / packed inputs in place. Skips int32/int64 (likely indices)."""
    import torch
    for t in inputs:
        if not isinstance(t, torch.Tensor):
            continue
        if t.dtype == torch.float8_e4m3fn:
            tmp = (torch.randn(t.shape, dtype=torch.bfloat16, device=t.device) * 0.5)
            t.copy_(tmp.to(torch.float8_e4m3fn))
        elif t.dtype == torch.float8_e5m2:
            tmp = (torch.randn(t.shape, dtype=torch.bfloat16, device=t.device) * 0.5)
            t.copy_(tmp.to(torch.float8_e5m2))
        elif t.is_floating_point():
            t.normal_()
        elif t.dtype == torch.int8:
            t.random_(-128, 128)
        elif t.dtype == torch.uint8:
            t.random_(0, 256)


def _hash_outputs(outputs):
    import hashlib, torch
    h = hashlib.sha256()
    for o in outputs:
        if isinstance(o, torch.Tensor):
            buf = o.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
            h.update(buf)
        else:
            h.update(repr(o).encode())
    return h.hexdigest()
