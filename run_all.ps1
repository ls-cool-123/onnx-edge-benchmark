# One-shot three-way run on Windows.
# Requires: pip install -r requirements.txt
# Usage:   .\run_all.ps1

$ErrorActionPreference = "Stop"

Write-Host "== check deps ==" -ForegroundColor Cyan
python -c "import torch, onnxruntime, transformers, onnx, psutil; print('deps ok')"

Write-Host "`n== benchmark: 3 backends, 200 runs, 4 threads ==" -ForegroundColor Cyan
python bench.py --model BAAI/bge-small-zh-v1.5 --seq-len 32 --runs 200 --threads 4 --backends pytorch,onnx,onnx-int8

Write-Host "`n== done. paste the markdown table into README.md 'Results' ==" -ForegroundColor Green