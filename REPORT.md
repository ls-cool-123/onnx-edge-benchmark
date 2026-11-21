# Benchmark report — `bge-small-zh-v1.5` on CPU: PyTorch vs ONNX Runtime vs INT8

## 1. Objective

Determine whether ONNX Runtime and dynamic INT8 quantization are viable for
serving `BAAI/bge-small-zh-v1.5` (a Chinese text-embedding model) as a local
retrieval encoder, measuring the trade between latency, memory footprint and
numerical agreement with the PyTorch reference.

## 2. Test plan

| Item | Setting |
|---|---|
| Model | `BAAI/bge-small-zh-v1.5`, 24M params, 512-dim output, FP32 baseline |
| Input | fixed set of 8 Chinese sentences, `max_length = 32`, batch = 8 |
| Backends | PyTorch FP32 / ONNX Runtime FP32 / ONNX Runtime INT8 (dynamic) |
| Threads | 4 for every backend (`set_num_threads` / `intra_op_num_threads`) |
| Warm-up | 10 untimed iterations |
| Measured runs | 200 per backend |
| Hardware | Intel Core i7-14650HX (16C/24T), 15.8 GB RAM, Windows 11, CPU only |
| Metrics | p50/p95/p99 latency, throughput, peak RSS, model size, `max |Δ|`, cosine |

Conditions were pinned deliberately: identical inputs, identical thread count and
a warm-up phase for every backend. Without those three, repeated runs on the same
machine varied by more than 2x.

## 3. Results

| Backend | p50 (ms) | p95 (ms) | p99 (ms) | Throughput (sent/s) | Peak RSS (MB) | max \|Δ\| | cosine |
|---|---|---|---|---|---|---|---|
| PyTorch FP32 | 76.6 | 82.9 | 86.5 | 103.6 | 468.7 | 0 | 1.0000 |
| ONNX FP32 | 69.1 | 72.0 | 73.2 | 115.6 | 596.3 | 3.0e-06 | 1.0000 |
| ONNX INT8 | 38.4 | 41.1 | 43.4 | 208.6 | 562.2 | 0.7964 | 0.9618 |

Model size: 90.5 MB → 22.8 MB (25%, 4.0x smaller).

## 4. Observations

1. **The export itself is lossless.** ONNX FP32 agrees with PyTorch to `3e-6`,
   which is float noise. The ~10% p50 / ~15% p99 gain comes from operator fusion
   and a leaner runtime, not from a change in computation.
2. **INT8 fails the accuracy gate for this use case.** Throughput improves 1.8x
   and the artefact shrinks 4x, but `cosine = 0.9618` and `max |Δ| = 0.80`. For
   an encoder whose output feeds cosine retrieval, a 0.04 cosine drift is large
   enough to reorder neighbours — the change is **rejected**, not accepted with
   a caveat.
3. **Smaller file did not mean smaller memory.** Peak RSS rose 27% from PyTorch
   to ONNX Runtime (469 → 596 MB) because of session state and graph
   optimization buffers. Model size and peak memory are different numbers and
   should be reported separately.
4. **The tail sets the budget.** INT8 improves p50 by 44% but p99 remains 43 ms,
   so on a device with a frame deadline the p99 is what decides feasibility.

## 5. Limits and next steps

- CPU only; no Jetson or Ascend device. Absolute numbers do not transfer — the
  ranking and the method do.
- Single sequence length (32) and a single input distribution; long-input
  behaviour is unmeasured and encoder latency grows super-linearly with length.
- Dynamic quantization only rewrites MatMul weights; LayerNorm and Softmax stay
  in fp32 and dominate the drift. The next experiment is **static quantization
  with a calibration set plus per-channel weights**, acceptance criterion:
  `cosine ≥ 0.999` on all 8 samples and p50 no worse than the current INT8.
- Repeated-run variance was not characterised: each backend was measured once.
  A second sweep would quantify run-to-run noise before trusting a 10% delta.