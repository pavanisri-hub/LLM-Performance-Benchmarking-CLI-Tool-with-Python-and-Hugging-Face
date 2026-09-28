from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import colorama
from colorama import Fore, Style
from tabulate import tabulate

import io
import sys as _sys
if hasattr(_sys.stdout, 'buffer'):
    _sys.stdout = io.TextIOWrapper(_sys.stdout.buffer, encoding='utf-8', errors='replace')
if hasattr(_sys.stderr, 'buffer'):
    _sys.stderr = io.TextIOWrapper(_sys.stderr.buffer, encoding='utf-8', errors='replace')

import pandas as pd
from tqdm import tqdm

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from config_loader import load_config
from model_runner import decode_output, encode_prompt, iter_models
from profiler import profile_inference
from quality_metrics import compute_quality
from results_writer import ResultsWriter
from visualizer import generate_all_charts


logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("llm_bench")

colorama.init(autoreset=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm_bench",
        description=(
            "LLM Performance Benchmarking CLI Tool\n"
            "--------------------------------------\n"
            "Benchmarks multiple Hugging Face language models on latency,\n"
            "throughput, memory usage, and text quality (TTR)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python src/cli.py --config config/benchmark_config.yaml\n"
            "  python src/cli.py --config config/benchmark_config.yaml "
            "--num-prompts 3 --max-new-tokens 50\n"
        ),
    )
    parser.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="Path to the YAML or JSON benchmark configuration file.",
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        default=None,
        help=(
            "Directory to save the CSV results and chart images. "
            "Overrides the value in the config file."
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Maximum number of tokens to generate per prompt. "
            "Overrides the value in the config file (default: 100)."
        ),
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=None,
        metavar="N",
        help="Limit the number of prompts used. Uses all prompts if omitted.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose / debug logging.",
    )
    return parser


def load_prompts(dataset_path: str, num_prompts: int | None) -> list[str]:
    path = Path(dataset_path)
    if not path.exists():
        sys.exit(
            f"{Fore.RED}ERROR: Dataset file not found: {dataset_path}{Style.RESET_ALL}"
        )
    prompts = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not prompts:
        sys.exit(
            f"{Fore.RED}ERROR: No prompts found in '{dataset_path}'.{Style.RESET_ALL}"
        )
    if num_prompts is not None:
        prompts = prompts[:num_prompts]
    return prompts


def print_summary(df: pd.DataFrame) -> None:
    print()
    print(Fore.CYAN + Style.BRIGHT + "=" * 70)
    print(Fore.CYAN + Style.BRIGHT + "  BENCHMARK SUMMARY")
    print(Fore.CYAN + Style.BRIGHT + "=" * 70)

    agg = (
        df.groupby("Model_Name")
        .agg(
            Avg_Latency_s=("Latency_Seconds", "mean"),
            Avg_TPS=("Throughput_TPS", "mean"),
            Peak_RAM_MB=("Peak_RAM_MB", "max"),
            Peak_VRAM_MB=("Peak_VRAM_MB", "max"),
            Avg_TTR=("Quality_Metric_Score", "mean"),
            Prompts_Run=("Prompt_ID", "count"),
        )
        .reset_index()
    )

    agg["Avg_Latency_s"] = agg["Avg_Latency_s"].round(4)
    agg["Avg_TPS"] = agg["Avg_TPS"].round(2)
    agg["Peak_RAM_MB"] = agg["Peak_RAM_MB"].round(1)
    agg["Peak_VRAM_MB"] = agg["Peak_VRAM_MB"].round(1)
    agg["Avg_TTR"] = agg["Avg_TTR"].round(4)

    print(
        tabulate(
            agg,
            headers={
                "Model_Name": "Model",
                "Avg_Latency_s": "Avg Latency (s)",
                "Avg_TPS": "Avg TPS",
                "Peak_RAM_MB": "Peak RAM (MB)",
                "Peak_VRAM_MB": "Peak VRAM (MB)",
                "Avg_TTR": "Avg TTR",
                "Prompts_Run": "Prompts",
            },
            tablefmt="rounded_outline",
            showindex=False,
            floatfmt=".4f",
        )
    )
    print(Fore.CYAN + Style.BRIGHT + "=" * 70)
    print()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    print(Fore.YELLOW + Style.BRIGHT + "\n[*] Loading configuration ...")
    try:
        cfg = load_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(f"{Fore.RED}ERROR: {exc}{Style.RESET_ALL}")

    if args.output_dir is not None:
        cfg["output_dir"] = str(Path(args.output_dir).resolve())
    if args.max_new_tokens is not None:
        cfg["max_new_tokens"] = args.max_new_tokens
    if args.num_prompts is not None:
        cfg["num_prompts"] = args.num_prompts

    output_dir: str = cfg["output_dir"]
    max_new_tokens: int = cfg["max_new_tokens"]
    model_ids: list[str] = cfg["models"]

    print(f"  {Fore.GREEN}[ok]{Style.RESET_ALL}  Models     : {', '.join(model_ids)}")
    print(f"  {Fore.GREEN}[ok]{Style.RESET_ALL}  Dataset    : {cfg['dataset_path']}")
    print(f"  {Fore.GREEN}[ok]{Style.RESET_ALL}  Output dir : {output_dir}")
    print(f"  {Fore.GREEN}[ok]{Style.RESET_ALL}  Max tokens : {max_new_tokens}")

    prompts = load_prompts(cfg["dataset_path"], cfg.get("num_prompts"))
    print(f"  {Fore.GREEN}[ok]{Style.RESET_ALL}  Prompts    : {len(prompts)} loaded\n")

    writer = ResultsWriter(output_dir=output_dir)
    total_start = time.perf_counter()

    for model_id, model, tokenizer in iter_models(model_ids):
        print(
            Fore.MAGENTA + Style.BRIGHT
            + f"\n[MODEL] Benchmarking: {model_id}"
            + Style.RESET_ALL
        )

        prompt_iter = tqdm(
            enumerate(prompts),
            total=len(prompts),
            desc=f"  {model_id[:30]}",
            unit="prompt",
            ncols=80,
            colour="green",
        )

        for prompt_idx, prompt in prompt_iter:
            input_ids = encode_prompt(prompt, tokenizer)

            output_ids, perf = profile_inference(
                model=model,
                tokenizer=tokenizer,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
            )

            generated_text = decode_output(output_ids, input_ids, tokenizer)

            quality = compute_quality(generated_text)

            writer.add_row(
                model_name=model_id,
                prompt_id=prompt_idx,
                prompt_text=prompt,
                latency_seconds=perf.latency_seconds,
                throughput_tps=perf.throughput_tps,
                peak_ram_mb=perf.peak_ram_mb,
                peak_vram_mb=perf.peak_vram_mb,
                quality_metric_score=quality.score,
                output_tokens=perf.output_tokens,
                generated_text=generated_text,
            )

            prompt_iter.set_postfix(
                latency=f"{perf.latency_seconds:.2f}s",
                tps=f"{perf.throughput_tps:.1f}",
                ttr=f"{quality.score:.3f}",
            )

    total_elapsed = time.perf_counter() - total_start

    print(Fore.YELLOW + Style.BRIGHT + "\n[*] Saving results ...")
    csv_path = writer.save()
    print(f"  {Fore.GREEN}[ok]{Style.RESET_ALL}  CSV saved  -> {csv_path}")

    print(Fore.YELLOW + Style.BRIGHT + "\n[*] Generating charts ...")
    df = writer.dataframe
    chart_paths = generate_all_charts(df, output_dir)
    for p in chart_paths:
        print(f"  {Fore.GREEN}[ok]{Style.RESET_ALL}  Chart      -> {p}")

    print_summary(df)
    print(
        Fore.GREEN + Style.BRIGHT
        + f"[DONE] Benchmark complete in {total_elapsed:.1f}s  "
        + f"({len(writer)} total measurements)\n"
        + Style.RESET_ALL
    )


if __name__ == "__main__":
    main()
