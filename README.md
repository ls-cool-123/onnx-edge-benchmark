# Edge inference benchmark: PyTorch vs ONNX Runtime vs INT8

A small, reproducible harness for the kind of measurement an edge-AI evaluation
role does every day: **run one model across several backends under identical
conditions, then report accuracy, latency, throughput and memory** instead of
a single feel-good number.

It answers four questions an edge deployment actually lives or dies on:

1. How much accuracy do I lose when I leave PyTorch? (`max |Δ|`, `cosine`)
2. How much latency do I gain or lose? (`p50 / p95 / p99`)
3. How much throughput do I get? (`sentences / second`)
4. How much memory does it cost, and how much does INT8 save? (`peak RSS`, model size)

## Why the methodology matters

A benchmark is only meaningful if everything except the variable is pinned.
This harness pins:

| Variable | How it is pinned |
|---|---|
| Input data | fixed sentence set, fixed `seq_len`, same batch for every backend |
| Thread count | `torch.set_num_threads(n)` and `intra_op_num_threads=n` |
| Warmup | 10 untimed runs before measuring (first call pays lazy init / kernel selection) |
| Measurement | `perf_counter` per call, percentiles reported — not just the mean |
| Comparison | output tensors diffed element-wise against the PyTorch reference |

Without the warmup and the thread pinning, the numbers swing by 2-3x on the
same machine. That is the single most common way edge benchmarks mislead.

## Quick start

```bash
pip install -r requirements.txt
python bench.py --model BAAI/bge-small-zh-v1.5 --seq-len 32 --runs 200 --threads 4
```

Windows one-shot:

```powershell
.\run_all.ps1
```

## Results

Machine: Intel Core i7-14650HX (16C / 24T), 15.8 GB RAM, Windows 11, CPU only,
`--threads 4`, `--seq-len 32`, batch 8, 200 measured runs after 10 warm-up runs.

| Backend | Precision | p50 (ms) | p95 (ms) | p99 (ms) | Throughput (sent/s) | Peak RSS (MB) | max \|Δ\| vs FP32 | cosine vs FP32 |
|---|---|---|---|---|---|---|---|---|
| PyTorch FP32 | FP32 | 76.6 | 82.9 | 86.5 | 103.6 | 468.7 | 0 | 1.0000 |
| ONNX Runtime FP32 | FP32 | 69.1 | 72.0 | 73.2 | 115.6 | 596.3 | 3.0e-06 | 1.0000 |
| ONNX Runtime INT8 | INT8 (dynamic) | 38.4 | 41.1 | 43.4 | 208.6 | 562.2 | 0.7964 | 0.9618 |

Model size: 90.5 MB → 22.8 MB (25%, 4.0x smaller).

Derived:

| Comparison | p50 | p99 | Throughput | Verdict |
|---|---|---|---|---|
| ONNX FP32 vs PyTorch | -9.8% | -15.3% | +11.5% | safe, bit-identical within float noise |
| INT8 vs ONNX FP32 | -44.5% (1.80x) | -40.7% | +81% (1.81x) | **rejected: cosine 0.9618** |

## What this run actually showed

- **Export is faithful.** ONNX FP32 differs from PyTorch by `3e-6` — float noise,
  not a conversion error. Operator fusion gave ~10% on p50 and ~15% on p99.
- **INT8 is fast but not accurate enough here.** 1.8x throughput and a 4x smaller
  model, but `max |Δ| = 0.80` and `cosine = 0.962`. For an embedding model that
  feeds cosine retrieval, a 4-point cosine drift is enough to reorder results —
  so the honest conclusion is *rejected pending a better recipe*, not "2x faster".
- **ONNX Runtime is not automatically cheaper in memory.** Peak RSS went **up**
  27% versus PyTorch (469 → 596 MB): session state and the optimised graph cost
  memory even though the file on disk got smaller. Model size is not peak memory.
- **The p99 gap matters more than the p50 gap** on a device: INT8 p50 improves
  44% but p99 still sits at 43 ms, so the worst-case frame budget is set by the
  tail, not by the average.

Next step for INT8 would be **static quantization with a calibration set**
(and per-channel weights) rather than dynamic — dynamic only rewrites MatMul
weights and leaves the numerically sensitive LayerNorm/Softmax path in fp32,
which is where most of this drift comes from.

## Failure modes this repo actually hit

- **Non-ASCII path breaks quantization.** `onnx` round-trips the model through a
  temp file before quantizing; with `C:\Users\<中文名>\...` as the working path
  the save/load pair dies with `FileNotFoundError: ...model-inferred.onnx`.
  Fixed by running the whole quantization step inside an ASCII scratch directory
  (`_ascii_scratch()` in `bench.py`) and moving the artefact back.
- `torch.onnx.export` on an `AutoModel` returns a `ModelOutput` dataclass; the
  export path is cleaner if you wrap it in an `nn.Module` that returns a single
  tensor.
- Dynamic axes must be declared explicitly, otherwise the exported graph is
  frozen to the batch size used at export time.
- Dynamic INT8 quantization only touches MatMul weights, so the accuracy cost is
  not spread evenly across the graph — LayerNorm and Softmax stay in fp32 and
  dominate the drift you see.
- Peak RSS sampled inside the loop is not free — sample every N iterations and
  note it in the report.

## Structure

```
bench.py           # the harness: export, quantize, measure, compare, report
run_all.ps1        # convenience wrapper for a standard three-way run
REPORT.md          # one-page report template (method / results / limits)
results.csv        # appended on every run, one row per backend
artifacts/         # exported ONNX models (fp32 + int8), git-ignored
```

## Scope and limits

- CPU execution only; no Jetson / Ascend hardware. The harness is written so a
  device backend is a swap of the execution provider, not a rewrite.
- Dynamic quantization is the cheap baseline. Static quantization with a
  calibration set usually gives better latency at the cost of a calibration
  pipeline — that is the next step, not a claim being made here.
- Single model, single input distribution. Numbers do not transfer to a
  different sequence length without re-measuring.