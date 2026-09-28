from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


COLUMNS = [
    "Model_Name",
    "Prompt_ID",
    "Prompt_Text",
    "Latency_Seconds",
    "Throughput_TPS",
    "Peak_RAM_MB",
    "Peak_VRAM_MB",
    "Quality_Metric_Score",
    "Output_Tokens",
    "Generated_Text",
]


class ResultsWriter:
    def __init__(self, output_dir: str) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._rows: list[dict[str, Any]] = []

    def add_row(
        self,
        *,
        model_name: str,
        prompt_id: int,
        prompt_text: str,
        latency_seconds: float,
        throughput_tps: float,
        peak_ram_mb: float,
        peak_vram_mb: float,
        quality_metric_score: float,
        output_tokens: int,
        generated_text: str,
    ) -> None:
        self._rows.append(
            {
                "Model_Name": model_name,
                "Prompt_ID": prompt_id,
                "Prompt_Text": prompt_text[:200],
                "Latency_Seconds": latency_seconds,
                "Throughput_TPS": throughput_tps,
                "Peak_RAM_MB": peak_ram_mb,
                "Peak_VRAM_MB": peak_vram_mb,
                "Quality_Metric_Score": quality_metric_score,
                "Output_Tokens": output_tokens,
                "Generated_Text": generated_text[:500],
            }
        )

    def save(self, filename: str = "benchmark_results.csv") -> str:
        csv_path = self._output_dir / filename
        df = pd.DataFrame(self._rows, columns=COLUMNS)
        df.to_csv(csv_path, index=False, encoding="utf-8")
        return str(csv_path)

    @property
    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows, columns=COLUMNS)

    def __len__(self) -> int:
        return len(self._rows)


def load_results(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path, encoding="utf-8")
