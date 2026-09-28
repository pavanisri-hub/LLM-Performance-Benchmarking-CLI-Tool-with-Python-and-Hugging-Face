from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    suffix = path.suffix.lower()
    with open(path, "r", encoding="utf-8") as fh:
        if suffix in {".yaml", ".yml"}:
            cfg = yaml.safe_load(fh)
        elif suffix == ".json":
            cfg = json.load(fh)
        else:
            raise ValueError(
                f"Unsupported config format '{suffix}'. Use .yaml, .yml, or .json."
            )

    _validate(cfg, config_path)

    cfg.setdefault("max_new_tokens", 100)
    cfg.setdefault("num_prompts", None)
    cfg.setdefault("output_dir", "output")

    cfg["output_dir"] = str(Path(cfg["output_dir"]).resolve())

    return cfg


def _validate(cfg: dict[str, Any], path: str) -> None:
    errors: list[str] = []

    models = cfg.get("models")
    if not models:
        errors.append("'models' key is missing.")
    elif not isinstance(models, list):
        errors.append("'models' must be a list of model identifier strings.")
    elif len(models) < 3:
        errors.append(
            f"'models' must contain at least 3 model identifiers, "
            f"but only {len(models)} were provided."
        )
    else:
        for i, m in enumerate(models):
            if not isinstance(m, str) or not m.strip():
                errors.append(
                    f"'models[{i}]' must be a non-empty string (model identifier)."
                )

    dataset_path = cfg.get("dataset_path")
    if not dataset_path:
        errors.append("'dataset_path' key is missing.")
    elif not isinstance(dataset_path, str):
        errors.append("'dataset_path' must be a string path.")

    if errors:
        msg = "\n  ".join(errors)
        raise ValueError(
            f"Invalid configuration file '{path}':\n  {msg}"
        )
