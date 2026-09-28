from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import psutil
import torch


@dataclass
class ProfileResult:
    latency_seconds: float = 0.0
    throughput_tps: float = 0.0
    peak_ram_mb: float = 0.0
    peak_vram_mb: float = 0.0
    output_tokens: int = 0


def profile_inference(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int = 100,
) -> tuple[torch.Tensor, ProfileResult]:
    result = ProfileResult()

    process = psutil.Process()
    _ram_samples: list[float] = []
    _stop_event = threading.Event()

    def _ram_sampler():
        while not _stop_event.is_set():
            try:
                rss_mb = process.memory_info().rss / (1024 ** 2)
                _ram_samples.append(rss_mb)
            except psutil.NoSuchProcess:
                break
            time.sleep(0.05)

    ram_thread = threading.Thread(target=_ram_sampler, daemon=True)

    has_gpu = torch.cuda.is_available()
    if has_gpu:
        torch.cuda.reset_peak_memory_stats()

    ram_thread.start()
    t_start = time.perf_counter()

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    t_end = time.perf_counter()
    _stop_event.set()
    ram_thread.join(timeout=0.5)

    latency = t_end - t_start
    result.latency_seconds = round(latency, 6)

    n_input = input_ids.shape[-1]
    n_output = output_ids.shape[-1] - n_input
    result.output_tokens = max(n_output, 0)

    if latency > 0 and result.output_tokens > 0:
        result.throughput_tps = round(result.output_tokens / latency, 4)

    if _ram_samples:
        result.peak_ram_mb = round(max(_ram_samples), 2)

    if has_gpu:
        vram_bytes = torch.cuda.max_memory_allocated()
        result.peak_vram_mb = round(vram_bytes / (1024 ** 2), 2)

    return output_ids, result
