#!/usr/bin/env python3
"""
Edge-inference benchmark harness
===============================

Purpose
-------
Run **one model** across several execution backends under **identical input and
measurement conditions**, then report accuracy, latency, throughput and memory.
This is the measurement loop an edge-AI evaluation role runs every day:

    PyTorch (reference)  ->  ONNX Runtime FP32  ->  ONNX Runtime INT8

Everything that can make a benchmark lie is pinned on purpose:
  * same fixed inputs and sequence length for every backend
  * same thread count for every backend
  * warmup runs before timing (first call pays lazy init / kernel compile)
  * percentile latency, not just mean (p99 is what users feel)
  * output tensors compared element-wise against the PyTorch reference

Usage
-----
    pip install -r requirements.txt
    python bench.py                                  # all backends, default model
    python bench.py --model BAAI/bge-small-zh-v1.5 --seq-len 32 --runs 200
    python bench.py --backends onnx,onnx-int8        # skip the torch reference

Output
------
    results.csv        one row per backend, appends on every run
    markdown table     printed to stdout, ready to paste into a README
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, List, Sequence

import numpy as np

DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"

SENTENCES = [
    "这个岗位需要在 Jetson 和 Atlas 上跑同一个模型并比较结果。",
    "固定测试条件是基准测试有意义的前提。",
    "量化之后模型体积更小，但要确认精度掉在可接受范围内。",
    "推理延迟的 p99 比平均值更能反映真实体验。",
    "模型转换会做算子融合、量化和内存规划。",
    "同一批输入分别喂给两个后端，输出才具有可比性。",
    "边缘设备的内存带宽往往比算力更早成为瓶颈。",
    "把线程数对齐之后，两个后端的数字才谈得上比较。",
]

# --------------------------------------------------------------------------- #
# measurement helpers
# --------------------------------------------------------------------------- #


def pct(values: Sequence[float], q: float) -> float:
    """q-th percentile (0-100) with linear interpolation."""
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def peak_rss_mb() -> float:
    """Peak resident set size of this process, in MB.

    Sampled rather than atomic: cheap enough to call inside a loop, and good
    enough to compare backends against each other.
    """
    try:
        import psutil

        info = psutil.Process(os.getpid()).memory_info()
        return info.rss / (1024 * 1024)
    except Exception:
        return float("nan")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(np.dot(a, b) / denom)


def compare(reference: np.ndarray, candidate: np.ndarray) -> Dict[str, float]:
    """Element-wise agreement between two output tensors."""
    a = reference.astype(np.float64)
    b = candidate.astype(np.float64)
    diff = np.abs(a - b)
    return {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "cosine": cosine(a, b),
    }


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #


def build_inputs(model_id: str, seq_len: int):
    """Tokenize a fixed sentence list into a (batch, seq_len) int64 pair."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    enc = tok(
        SENTENCES,
        padding="max_length",
        truncation=True,
        max_length=seq_len,
        return_tensors="np",
    )
    input_ids = enc["input_ids"].astype(np.int64)
    attention_mask = enc["attention_mask"].astype(np.int64)
    return input_ids, attention_mask


# --------------------------------------------------------------------------- #
# backends
# --------------------------------------------------------------------------- #


def load_torch(model_id: str):
    import torch
    from transformers import AutoModel

    model = AutoModel.from_pretrained(model_id).eval()
    return model, torch


def export_onnx(model, input_ids, attention_mask, path: str) -> None:
    import torch

    class Encoder(torch.nn.Module):
        """Export only last_hidden_state: one output, no dataclass wrapping."""

        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, attention_mask):
            return self.inner(input_ids=input_ids, attention_mask=attention_mask)[
                "last_hidden_state"
            ]

    wrapper = Encoder(model).eval()
    kwargs = dict(
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch"},
            "attention_mask": {0: "batch"},
            "last_hidden_state": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    with torch.no_grad():
        try:
            # torch >= 2.6 defaults to the dynamo exporter; keep the legacy path
            torch.onnx.export(
                wrapper,
                (torch.from_numpy(input_ids), torch.from_numpy(attention_mask)),
                path,
                dynamo=False,
                **kwargs,
            )
        except TypeError:
            # torch < 2.6 has no `dynamo` keyword
            torch.onnx.export(
                wrapper,
                (torch.from_numpy(input_ids), torch.from_numpy(attention_mask)),
                path,
                **kwargs,
            )


def _ascii_scratch():
    """Return a writable, ASCII-only directory, or None.

    onnxruntime / onnx round-trip the model through a temporary file before
    quantizing. If any directory in that path contains non-ASCII characters
    (very common on Chinese Windows profiles) the save/load pair breaks with
    ``FileNotFoundError`` — so the quantization step is run entirely inside an
    ASCII scratch directory and the result is moved back afterwards.
    """
    import tempfile

    for cand in (os.environ.get("ORT_SCRATCH"), r"C:\ort_scratch", "/tmp"):
        if not cand or not str(cand).isascii():
            continue
        try:
            os.makedirs(cand, exist_ok=True)
            probe = os.path.join(cand, ".probe")
            with open(probe, "w", encoding="utf-8") as fh:
                fh.write("ok")
            os.remove(probe)
            return cand
        except Exception:
            continue
    return None


def quantize_int8(src: str, dst: str) -> None:
    """Dynamic INT8 quantization, executed inside an ASCII-safe workspace."""
    import shutil
    import tempfile

    from onnxruntime.quantization import QuantType, quantize_dynamic

    scratch = _ascii_scratch()
    if scratch is None:
        # No ASCII path available: try in place and let the error surface.
        quantize_dynamic(model_input=src, model_output=dst, weight_type=QuantType.QInt8)
        return

    previous = tempfile.tempdir
    with tempfile.TemporaryDirectory(prefix="ortq_", dir=scratch) as wd:
        tempfile.tempdir = wd
        try:
            src_local = os.path.join(wd, "model.onnx")
            dst_local = os.path.join(wd, "model_int8.onnx")
            shutil.copyfile(src, src_local)
            quantize_dynamic(
                model_input=src_local,
                model_output=dst_local,
                weight_type=QuantType.QInt8,
            )
        finally:
            tempfile.tempdir = previous
        shutil.move(dst_local, dst)


def run_torch(model, input_ids, attention_mask, runs: int) -> Dict:
    import torch

    ids = torch.from_numpy(input_ids)
    mask = torch.from_numpy(attention_mask)

    ref = None
    lat: List[float] = []
    peak = peak_rss_mb()

    with torch.no_grad():
        for _ in range(10):  # warmup
            model(input_ids=ids, attention_mask=mask)
        for i in range(runs):
            t0 = now_ms()
            out = model(input_ids=ids, attention_mask=mask)["last_hidden_state"]
            lat.append(now_ms() - t0)
            if i == 0:
                ref = out.numpy()
            if i % 20 == 0:
                peak = max(peak, peak_rss_mb())

    return {"lat": lat, "out": ref, "peak": peak}


def run_onnx(path: str, input_ids, attention_mask, runs: int, threads: int) -> Dict:
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])

    feeds = {"input_ids": input_ids, "attention_mask": attention_mask}
    lat: List[float] = []
    peak = peak_rss_mb()

    for _ in range(10):
        sess.run(None, feeds)
    ref = None
    for i in range(runs):
        t0 = now_ms()
        out = sess.run(None, feeds)[0]
        lat.append(now_ms() - t0)
        if i == 0:
            ref = out
        if i % 20 == 0:
            peak = max(peak, peak_rss_mb())

    return {"lat": lat, "out": ref, "peak": peak}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser(description="Edge inference benchmark harness")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--backends", default="pytorch,onnx,onnx-int8")
    ap.add_argument("--onnx-dir", default="artifacts")
    ap.add_argument("--csv", default="results.csv")
    args = ap.parse_args()

    os.makedirs(args.onnx_dir, exist_ok=True)
    wanted = [b.strip() for b in args.backends.split(",") if b.strip()]

    print(f"model     : {args.model}")
    print(f"seq_len   : {args.seq_len}   runs: {args.runs}   threads: {args.threads}")
    print(f"backends  : {', '.join(wanted)}\n")

    try:
        input_ids, attention_mask = build_inputs(args.model, args.seq_len)
    except ImportError as exc:
        print(f"! missing dependency: {exc}")
        print("  install with:  pip install -r requirements.txt")
        return 3
    batch = input_ids.shape[0]
    fp32_onnx = os.path.join(args.onnx_dir, "model_fp32.onnx")
    int8_onnx = os.path.join(args.onnx_dir, "model_int8.onnx")

    rows: List[Dict] = []
    reference = None

    if "pytorch" in wanted:
        try:
            import torch

            torch.set_num_threads(args.threads)
            model, _ = load_torch(args.model)
            export_onnx(model, input_ids, attention_mask, fp32_onnx)
            r = run_torch(model, input_ids, attention_mask, args.runs)
            reference = r["out"]
            rows.append(
                {
                    "backend": "PyTorch FP32",
                    "precision": "FP32",
                    "p50_ms": pct(r["lat"], 50),
                    "p95_ms": pct(r["lat"], 95),
                    "p99_ms": pct(r["lat"], 99),
                    "throughput": batch / (sum(r["lat"]) / 1000.0 / len(r["lat"])),
                    "peak_rss_mb": r["peak"],
                    "max_abs_diff": 0.0,
                    "mean_abs_diff": 0.0,
                    "cosine": 1.0,
                }
            )
        except ImportError:
            print("! torch not installed - skipping the PyTorch reference backend\n")
        except Exception as exc:  # export can fail on exotic ops
            print(f"! PyTorch backend failed: {exc}\n")

    if ("onnx" in wanted or "onnx-int8" in wanted) and not os.path.exists(fp32_onnx):
        print("! no exported ONNX model found; need the pytorch backend first")
        return 2

    if "onnx" in wanted:
        r = run_onnx(fp32_onnx, input_ids, attention_mask, args.runs, args.threads)
        cmp = compare(reference, r["out"]) if reference is not None else {}
        rows.append(
            {
                "backend": "ONNX Runtime FP32",
                "precision": "FP32",
                "p50_ms": pct(r["lat"], 50),
                "p95_ms": pct(r["lat"], 95),
                "p99_ms": pct(r["lat"], 99),
                "throughput": batch / (sum(r["lat"]) / 1000.0 / len(r["lat"])),
                "peak_rss_mb": r["peak"],
                **cmp,
            }
        )

    if "onnx-int8" in wanted:
        quantize_int8(fp32_onnx, int8_onnx)
        r = run_onnx(int8_onnx, input_ids, attention_mask, args.runs, args.threads)
        cmp = compare(reference, r["out"]) if reference is not None else {}
        rows.append(
            {
                "backend": "ONNX Runtime INT8",
                "precision": "INT8 (dynamic)",
                "p50_ms": pct(r["lat"], 50),
                "p95_ms": pct(r["lat"], 95),
                "p99_ms": pct(r["lat"], 99),
                "throughput": batch / (sum(r["lat"]) / 1000.0 / len(r["lat"])),
                "peak_rss_mb": r["peak"],
                **cmp,
            }
        )

    # ---------------- report
    cols = [
        ("backend", "Backend"),
        ("precision", "Precision"),
        ("p50_ms", "p50 (ms)"),
        ("p95_ms", "p95 (ms)"),
        ("p99_ms", "p99 (ms)"),
        ("throughput", "Throughput (sent/s)"),
        ("peak_rss_mb", "Peak RSS (MB)"),
        ("max_abs_diff", "max |Δ| vs FP32"),
        ("cosine", "cosine vs FP32"),
    ]

    def fmt(v):
        if isinstance(v, float):
            if np.isnan(v):
                return "-"
            return f"{v:.4f}" if abs(v) < 1 else f"{v:.1f}"
        return str(v if v is not None else "-")

    print("| " + " | ".join(t for _, t in cols) + " |")
    print("|" + "|".join(["---"] * len(cols)) + "|")
    for row in rows:
        print("| " + " | ".join(fmt(row.get(k)) for k, _ in cols) + " |")

    # ---------------- persist
    total_mb = os.path.getsize(fp32_onnx) / (1024 * 1024) if os.path.exists(fp32_onnx) else 0
    int8_mb = os.path.getsize(int8_onnx) / (1024 * 1024) if os.path.exists(int8_onnx) else 0
    if total_mb:
        print(f"\nmodel size : FP32 {total_mb:.1f} MB -> INT8 {int8_mb:.1f} MB "
              f"({int8_mb / total_mb * 100:.0f}%)")

    write_header = not os.path.exists(args.csv)
    with open(args.csv, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[k for k, _ in cols])
        if write_header:
            w.writeheader()
        for row in rows:
            w.writerow({k: (round(row.get(k, float("nan")), 6)
                           if isinstance(row.get(k), float) else row.get(k, ""))
                        for k, _ in cols})
    print(f"\nappended {len(rows)} rows -> {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())