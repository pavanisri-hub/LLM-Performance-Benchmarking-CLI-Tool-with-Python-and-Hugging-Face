from __future__ import annotations

import logging
from typing import Generator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def iter_models(
    model_ids: list[str],
) -> Generator[tuple[str, AutoModelForCausalLM, AutoTokenizer], None, None]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    for model_id in model_ids:
        logger.info("Loading model '%s'", model_id)
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                use_fast=True,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )
            model.to(device)
            model.eval()

            yield model_id, model, tokenizer

        except Exception as exc:
            logger.error(
                "Failed to load model '%s': %s — skipping.", model_id, exc
            )
            continue

        finally:
            try:
                del model  # type: ignore[possibly-undefined]
            except NameError:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def encode_prompt(
    prompt: str,
    tokenizer: AutoTokenizer,
    device: str | None = None,
) -> torch.Tensor:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    return inputs["input_ids"].to(device)


def decode_output(
    output_ids: torch.Tensor,
    input_ids: torch.Tensor,
    tokenizer: AutoTokenizer,
) -> str:
    n_prompt = input_ids.shape[-1]
    new_token_ids = output_ids[0, n_prompt:]
    return tokenizer.decode(new_token_ids, skip_special_tokens=True)
