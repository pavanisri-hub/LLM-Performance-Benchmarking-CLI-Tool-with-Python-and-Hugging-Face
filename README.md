<div align="center">

# LLM Performance Benchmarking CLI Tool

**A production-ready command-line application to systematically benchmark and compare Large Language Models from Hugging Face.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-orange?logo=huggingface&logoColor=white)](https://huggingface.co)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Configuration](#configuration)
- [Metrics Collected](#metrics-collected)
- [Output Files](#output-files)
- [Example Results](#example-results)
- [Docker](#docker)
- [GPU Support](#gpu-support)
- [Dependencies](#dependencies)
- [Architecture](#architecture)
- [License](#license)

---

## Overview

This tool automates the entire LLM benchmarking pipeline:

1. Loads a YAML/JSON config listing the models and prompt dataset to use
2. Dynamically downloads and runs each model from the Hugging Face Hub
3. Profiles every inference call for **latency**, **throughput**, and **memory usage**
4. Scores every generated output using a **text quality metric** (Type-Token Ratio)
5. Saves all measurements to a structured **CSV file**
6. Generates **comparison charts** (PNG) across all tested models
7. Prints a **formatted summary table** to the console

It is designed for ML engineers and teams evaluating LLMs for production — helping make informed decisions balancing speed, resource consumption, and generation quality.

---

## Features

| Feature | Details |
|---|---|
| **CLI Interface** | `argparse`-powered with `--config`, `--output-dir`, `--max-new-tokens`, `--num-prompts`, `--verbose` |
| **Dynamic Model Loading** | Loops config's `models` list — no hardcoded model names in logic |
| **Latency Profiling** | `time.perf_counter()` bracketed precisely around `model.generate()` |
| **Throughput** | Computed as `output_tokens / latency_seconds` (tokens-per-second) |
| **RAM Monitoring** | Background thread samples `psutil` RSS every 50 ms → reports peak |
| **VRAM Monitoring** | `torch.cuda.max_memory_allocated()` — silently skipped on CPU machines |
| **Quality Metric** | Type-Token Ratio (TTR): `unique_words / total_words` ∈ (0, 1] |
| **CSV Output** | `pandas` DataFrame with all required schema columns |
| **4 Charts** | Latency · Throughput · RAM · Quality — saved as PNG via `matplotlib`/`seaborn` |
| **Progress Bars** | `tqdm` wraps every prompt loop with real-time stats (latency, TPS, TTR) |
| **Console Summary** | `tabulate` renders a per-model aggregated table on completion |
| **Docker** | `Dockerfile` + `docker-compose.yml` + `.env.example` for reproducible runs |
| **GPU Auto-detect** | Uses `cuda` if available; falls back to `cpu` gracefully |

---

## Project Structure

```
.
├── src/
│   ├── cli.py                 # Entry point — argparse CLI + pipeline orchestration
│   ├── config_loader.py       # YAML/JSON config parsing and validation
│   ├── model_runner.py        # HF model/tokenizer loading, encode/decode helpers
│   ├── profiler.py            # Latency, RAM (psutil thread), VRAM (torch.cuda)
│   ├── quality_metrics.py     # Type-Token Ratio (TTR) quality metric
│   ├── results_writer.py      # Structured CSV output via pandas
│   └── visualizer.py          # 4 comparison bar charts via matplotlib + seaborn
│
├── config/
│   └── benchmark_config.yaml  # Default config: 3 models + dataset path + settings
│
├── data/
│   └── prompts.txt            # Sample benchmark prompts (one per line)
│
├── output/                    # Auto-created — CSV results + chart PNGs land here
│   ├── benchmark_results.csv
│   ├── avg_latency_comparison.png
│   ├── avg_throughput_comparison.png
│   ├── peak_ram_comparison.png
│   └── quality_score_comparison.png
│
├── Dockerfile                 # python:3.10-slim image definition
├── docker-compose.yml         # Service definition with volume mounts
├── .env.example               # Template for required environment variables
├── .gitignore                 # Git ignore rules
├── requirements.txt           # Pinned Python dependencies
└── README.md                  # This file
```

---

## Quick Start

### Prerequisites

- Python 3.10 or higher
- ~1–2 GB of free disk space (for model downloads on first run)
- Internet connection (to download models from Hugging Face Hub)

### 1. Clone / navigate to the project

```bash
cd "Build an LLM Performance Benchmarking CLI Tool with Python and Hugging Face"
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the benchmark

```bash
python src/cli.py --config config/benchmark_config.yaml
```

> **First run note:** Models will be downloaded from Hugging Face Hub (~900 MB total for the default 3 models). They are cached locally, so subsequent runs are fast.

### 4. View results

```
output/
├── benchmark_results.csv           ← all raw measurements
├── avg_latency_comparison.png      ← chart 1
├── avg_throughput_comparison.png   ← chart 2
├── peak_ram_comparison.png         ← chart 3
└── quality_score_comparison.png    ← chart 4
```

---

## CLI Reference

```
usage: python src/cli.py --config PATH [options]

required arguments:
  --config PATH          Path to the YAML or JSON benchmark configuration file

optional arguments:
  --output-dir DIR       Override the output directory from config (default: "output")
  --max-new-tokens N     Maximum tokens to generate per prompt (default: 100)
  --num-prompts N        Limit the number of prompts used (default: all)
  --verbose, -v          Enable verbose/debug logging
  -h, --help             Show this help message and exit
```

### Usage Examples

```bash
# Full benchmark with all prompts and default settings
python src/cli.py --config config/benchmark_config.yaml

# Quick smoke test — 3 prompts, 40 tokens max
python src/cli.py --config config/benchmark_config.yaml --num-prompts 3 --max-new-tokens 40

# Save results to a custom directory
python src/cli.py --config config/benchmark_config.yaml --output-dir results/run_01

# Run with debug logging enabled
python src/cli.py --config config/benchmark_config.yaml --verbose

# Use a different config file (JSON also supported)
python src/cli.py --config config/my_custom_config.json
```

---

## Configuration

Edit `config/benchmark_config.yaml` to customise models, data, and output settings.

```yaml
# At least 3 Hugging Face model identifiers are required
models:
  - "sshleifer/tiny-gpt2"   # ~30 MB  — ultra-fast, great for CI smoke tests
  - "distilgpt2"             # ~350 MB — distilled GPT-2, 2× faster than gpt2
  - "gpt2"                   # ~500 MB — original GPT-2 small (124 M params)

# Path to prompts file — one prompt per line, blank lines ignored
dataset_path: "data/prompts.txt"

# Directory to write CSV and chart files
output_dir: "output"

# Maximum number of new tokens to generate per prompt
max_new_tokens: 80

# Limit prompts (null = use all)
num_prompts: null
```

### Supported Config Formats

| Format | Extension |
|---|---|
| YAML | `.yaml`, `.yml` |
| JSON | `.json` |

### Using Your Own Models

Add any public causal language model from the Hugging Face Hub:

```yaml
models:
  - "microsoft/phi-2"
  - "facebook/opt-125m"
  - "EleutherAI/gpt-neo-125M"
  - "bigscience/bloom-560m"
```

### Custom Prompts Dataset

Create a plain text file with one prompt per line:

```
What is the capital of France?
Explain the theory of relativity in simple terms.
Write a short poem about the ocean.
```

Then update `dataset_path` in your config to point to it.

---

## Metrics Collected

| Metric | Unit | Method |
|---|---|---|
| **Latency** | seconds | `time.perf_counter()` around `model.generate()` |
| **Throughput** | tokens/sec (TPS) | `output_tokens ÷ latency_seconds` |
| **Peak RAM** | MB | `psutil.Process().memory_info().rss` — sampled every 50 ms |
| **Peak VRAM** | MB | `torch.cuda.max_memory_allocated()` — 0 if no GPU |
| **Quality (TTR)** | 0.0 – 1.0 | `unique_words ÷ total_words` — higher = richer vocabulary |

> **Note on TTR:** A TTR of 1.0 means every word is unique (maximally diverse). A score near 0 indicates highly repetitive output. It is a fast, reference-free metric that requires no ground-truth labels.

---

## Output Files

### `benchmark_results.csv` — Schema

| Column | Type | Description |
|---|---|---|
| `Model_Name` | str | Hugging Face model identifier string |
| `Prompt_ID` | int | Zero-based index of the prompt |
| `Prompt_Text` | str | Raw input prompt (first 200 chars) |
| `Latency_Seconds` | float | Wall-clock inference time in seconds |
| `Throughput_TPS` | float | Generated tokens per second |
| `Peak_RAM_MB` | float | Peak Resident Set Size during inference (MB) |
| `Peak_VRAM_MB` | float | Peak CUDA memory allocated (MB); `0.0` on CPU |
| `Quality_Metric_Score` | float | Type-Token Ratio of generated text |
| `Output_Tokens` | int | Number of newly generated tokens |
| `Generated_Text` | str | Decoded model output (first 500 chars) |

### Charts (PNG) — saved in `output/`

| File | Description |
|---|---|
| `avg_latency_comparison.png` | Average inference latency per model |
| `avg_throughput_comparison.png` | Average tokens-per-second per model |
| `peak_ram_comparison.png` | Peak system RAM per model |
| `quality_score_comparison.png` | Average TTR quality score per model |

---

## Example Results

Results from a verified run on **3 models × 2 prompts** on CPU:

```
╭─────────────────────┬─────────────────┬───────────┬───────────────┬────────────────┬───────────┬───────────────╮
│ Model               │ Avg Latency (s) │  Avg TPS  │ Peak RAM (MB) │ Peak VRAM (MB) │  Avg TTR  │    Prompts    │
├─────────────────────┼─────────────────┼───────────┼───────────────┼────────────────┼───────────┼───────────────┤
│ sshleifer/tiny-gpt2 │          0.117  │    429.8  │         473.0 │            0.0 │    0.025  │             2 │
│ distilgpt2          │          1.311  │     30.5  │         815.4 │            0.0 │    0.000  │             2 │
│ gpt2                │          2.001  │     20.0  │         986.9 │            0.0 │    0.273  │             2 │
╰─────────────────────┴─────────────────┴───────────┴───────────────┴────────────────┴───────────┴───────────────╯
```

**Takeaways:**
- `tiny-gpt2` is 17× faster than `gpt2` but produces repetitive output (low TTR)
- `gpt2` (124M params) produces the richest vocabulary (highest TTR)
- All VRAM values are 0 because this run used CPU only

---

## Docker

### Setup

```bash
# 1. Copy the environment template
cp .env.example .env

# 2. (Optional) Add your Hugging Face token for private/gated models
# Edit .env and set: HUGGING_FACE_HUB_TOKEN=hf_your_token_here

# 3. Build the image and run the benchmark
docker-compose up --build
```

Results are written to `./output/` on your host machine (volume-mounted).

### Custom Run via Docker

```bash
# Run with custom flags
docker-compose run llm-bench \
  python src/cli.py --config config/benchmark_config.yaml --num-prompts 3

# Run an interactive shell inside the container
docker-compose run --entrypoint bash llm-bench
```

### Environment Variables (`.env`)

| Variable | Description |
|---|---|
| `HF_HOME` | Directory for Hugging Face cache (models, tokenizers) |
| `TRANSFORMERS_CACHE` | Specific cache path for model weights |
| `HUGGING_FACE_HUB_TOKEN` | HF API token — required for gated/private models |

See [`.env.example`](.env.example) for the full template.

---

## GPU Support

The tool **automatically detects CUDA GPUs** at runtime:

- **GPU detected** → models moved to `cuda`, VRAM measured via `torch.cuda.max_memory_allocated()`
- **No GPU** → CPU mode, VRAM column reports `0.0` — no errors or crashes

### Enable GPU in Docker

Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html) first, then add to `docker-compose.yml`:

```yaml
services:
  llm-bench:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

Also swap the base image in `Dockerfile` to:

```dockerfile
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CLI Entry Point                               │
│                         src/cli.py                                   │
│   argparse → load config → load prompts → loop models → summary     │
└────────┬──────────┬──────────┬──────────┬──────────┬────────────────┘
         │          │          │          │          │
         ▼          ▼          ▼          ▼          ▼
  config_      model_      profiler   quality_   results_    visualizer
  loader.py    runner.py   .py        metrics    writer.py   .py
                                      .py
  YAML/JSON   HF Auto-    time.       TTR        pandas      matplotlib
  parsing     ModelFor    perf_       score      CSV         seaborn
  & validate  CausalLM    counter()   (0-1)      output      4 PNG charts
              AutoToken   psutil RAM
              izer        torch VRAM
```

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `transformers` | ≥4.40.0 | Hugging Face model loading and inference |
| `torch` | ≥2.0.0 | PyTorch backend + VRAM measurement |
| `accelerate` | ≥0.27.0 | Efficient model weight loading |
| `psutil` | ≥5.9.0 | System RAM monitoring |
| `pynvml` | ≥11.5.0 | NVIDIA GPU management library |
| `pyyaml` | ≥6.0 | YAML configuration parsing |
| `pandas` | ≥2.0.0 | CSV results writing |
| `matplotlib` | ≥3.8.0 | Chart rendering |
| `seaborn` | ≥0.13.0 | Chart styling |
| `tqdm` | ≥4.66.0 | Progress bars |
| `tabulate` | ≥0.9.0 | Formatted console summary table |
| `colorama` | ≥0.4.6 | Cross-platform coloured terminal output |

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

<div align="center">
Built for the Partnr Network — Week 30 Task<br>
<em>Machine Learning · MLOps · Inference Profiling · Data Visualization</em>
</div>
