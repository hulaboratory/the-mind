"""Labels, data-layout constants, and small helpers shared by the analysis and figure scripts."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

from PIL import ImageFont

# One participant excluded by the attention check (Appendix C), identified by the unique
# "<partner>_<timestamp>" suffix of the record key so that no participant ID appears in the code.
HUMAN_EXCLUSIONS = {"_rule_1784305906287"}

# Partner conditions, in plotting order.  "counting" is the paper's "Linear" partner.
CONDITION_ORDER = ["random", "counting_off", "counting_on", "bayesian_off", "bayesian_on"]

# LLMs: (label, results directory prefix, separator before the prompt type).
LLM_MODELS = [
    ("Gemini 3.7", "gemini3.7", "_"),
    ("GPT-6", "gpt-6", "_"),
    ("GPT-5.6 Luna", "gpt-5.6-luna", "_"),
    ("GPT-OSS 120B", "gpt-oss-120b", "_"),
    ("GPT-OSS 20B", "gpt-oss-20b", "_"),
    ("Llama 70B", "llama70b", "_"),
    ("Llama 8B", "llama8b", "_"),
    ("Llama 1B", "llama1b", "_"),
    ("Qwen3 32B", "Qwen3-32B", ""),
    ("Qwen3 8B", "Qwen3-8B", "_"),
]

# LLM prompt conditions (Section 3.3.3): (results directory suffix, display label).
PROMPTS = [
    ("time_based_q_second", "0.25 s"),
    ("time_based_second", "1 s"),
    ("time_based_half_second", "0.5 s"),
    ("time_based2", "Real time"),
    ("wait_based", "Wait"),
    ("wait_count_based", "Wait + count"),
]

# Class-name prefixes identifying the rule-based partner in an LLM game file.
STRATEGY_PREFIXES = ("RandomPlayer", "RuleBasedAgent", "BayesianAgent")


def condition_from_model(model_id: str) -> str:
    """Partner condition from a rule-based partner's model id, e.g. 'RuleBasedAgent0.5_True'."""
    if model_id.startswith("RandomPlayer"):
        return "random"
    calibrated = "_True" in model_id
    if model_id.startswith("RuleBasedAgent"):
        return "counting_on" if calibrated else "counting_off"
    if model_id.startswith("BayesianAgent"):
        return "bayesian_on" if calibrated else "bayesian_off"
    raise ValueError(f"Unrecognized strategy model: {model_id}")


def hand_from_result_filename(path: Path) -> int:
    """Hand (game seed) number from 'game_<model>_<partner>_<hand>_<hand>_<run>.json'."""
    parts = path.stem.rsplit("_", 2)
    if len(parts) != 3 or not parts[-2].isdigit():
        raise ValueError(f"Cannot identify hand from result filename: {path.name}")
    return int(parts[-2])


def akaike_weights(scores: Iterable[float]) -> tuple[list[float], list[float]]:
    """Deltas from the best score and the corresponding Akaike weights exp(-delta/2)/sum."""
    values = list(scores)
    minimum = min(values)
    deltas = [value - minimum for value in values]
    raw = [math.exp(-0.5 * delta) for delta in deltas]
    total = sum(raw)
    return deltas, [value / total for value in raw]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to CSV; dict/list cells (fitted parameters) are stored as JSON strings."""
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                    for key, value in row.items()
                }
            )


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """Helvetica or Arial on macOS, the standard sans-serif fonts on Linux/Windows, else Pillow's default."""
    candidates = [
        ("/System/Library/Fonts/Helvetica.ttc", 1 if bold else 0),
        (f"/System/Library/Fonts/Supplemental/Arial{' Bold' if bold else ''}.ttf", 0),
        (f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf", 0),
        (f"/usr/share/fonts/truetype/liberation/LiberationSans-{'Bold' if bold else 'Regular'}.ttf", 0),
        (f"C:/Windows/Fonts/arial{'bd' if bold else ''}.ttf", 0),
    ]
    for candidate, index in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size, index=index)
    return ImageFont.load_default(size=size)
