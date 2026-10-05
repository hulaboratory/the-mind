#!/usr/bin/env python3
"""Draw the paper's best-fit-strategy figures (AIC: Figure 3, BIC: Figure 10).

Reads aic_winners.csv / bic_winners.csv written by generative_aic_analysis_compact.py and draws
one 14-panel figure per criterion: Human, three computer-use agents, ten LLMs; each panel shows
the share of game records won by each strategy model for the five partner conditions.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

import analysis_common as common

HERE = Path(__file__).resolve().parent  # default input/output paths are relative to this file


FAMILY_ORDER = (
    "random",
    "constant_wait",
    "counting_off",
    "counting_on",
    "confidence_bayesian_off",
    "confidence_bayesian_on",
)

FAMILY_LABELS = {
    "random": "Random",
    "constant_wait": "Constant Wait",
    "counting_off": "Linear, -Calibration",
    "counting_on": "Linear, +Calibration",
    "confidence_bayesian_off": "Bayesian, -Calibration",
    "confidence_bayesian_on": "Bayesian, +Calibration",
}

FAMILY_COLORS = {
    "random": "#4D4D4D",
    "constant_wait": "#7A5195",
    "counting_off": "#A6CEE3",
    "counting_on": "#1F78B4",
    "confidence_bayesian_off": "#FDBF6F",
    "confidence_bayesian_on": "#FF7F00",
}

CONDITION_ORDER = (
    "random",
    "counting_off",
    "counting_on",
    "bayesian_off",
    "bayesian_on",
)

CONDITION_LABELS = (
    "Random",
    "Linear-",
    "Linear+",
    "Bayesian-",
    "Bayesian+",
)

LLM_ORDER = (
    "Gemini 3.7",
    "GPT-6",
    "GPT-5.6 Luna",
    "GPT-OSS 120B",
    "GPT-OSS 20B",
    "Llama 70B",
    "Llama 8B",
    "Llama 1B",
    "Qwen3 32B",
    "Qwen3 8B",
)

CUA_ORDER = ("Gemini", "GPT-5.6 Luna", "GPT-6")

MODEL_DISPLAY_NAMES = {
    "Gemini 3.7": "Gemini-3.7-flash",
    "GPT-6": "GPT-6-astra",
    "GPT-5.6 Luna": "GPT-5.6-luna",
    "GPT-OSS 120B": "GPT-oss-120b",
    "GPT-OSS 20B": "GPT-oss-20b",
    "Llama 70B": "Llama-3.1-70B",
    "Llama 8B": "Llama-3.1-8B",
    "Llama 1B": "Llama-3.2-1B",
    "Qwen3 32B": "Qwen3-32B",
    "Qwen3 8B": "Qwen3-8B",
}

CUA_DISPLAY_NAMES = {
    "Gemini": "Gemini-3.7-flash-CUA",
    "GPT-5.6 Luna": "GPT-5.6-luna-CUA",
    "GPT-6": "GPT-6-astra-CUA",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read winner rows produced by the compact analysis."""
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def draw_grid(
    draw: ImageDraw.ImageDraw,
    left: int,
    top: int,
    right: int,
    bottom: int,
    tick_font: Any,
    show_tick_labels: bool,
) -> None:
    tick_labels = ((0.0, "0"), (0.25, "0.25"), (0.5, "0.5"), (0.75, "0.75"), (1.0, "1"))
    for fraction, label in tick_labels:
        y = bottom - int(round(fraction * (bottom - top)))
        draw.line((left, y, right, y), fill="#E3E3E3", width=1)
        if show_tick_labels:
            width = draw.textlength(label, font=tick_font)
            draw.text((left - width - 11, y - 23), label, fill="#333333", font=tick_font)
    draw.line((left, top, left, bottom), fill="#777777", width=2)
    draw.line((left, bottom, right, bottom), fill="#777777", width=2)


def draw_bar(
    draw: ImageDraw.ImageDraw,
    rows: list[dict[str, str]],
    x0: int,
    x1: int,
    top: int,
    bottom: int,
) -> None:
    if not rows:
        return
    counts = Counter(row["strategy_family"] for row in rows)
    total = len(rows)
    cumulative = 0
    prior_y = bottom
    for family in FAMILY_ORDER:
        cumulative += counts[family]
        next_y = bottom - int(round(cumulative / total * (bottom - top)))
        if counts[family]:
            draw.rectangle((x0, next_y, x1, prior_y), fill=FAMILY_COLORS[family])
        prior_y = next_y


def draw_rotated_label(
    image: Image.Image,
    text: str,
    center_x: int,
    top_y: int,
    font: Any,
) -> None:
    scratch = Image.new("RGBA", (300, 100), (255, 255, 255, 0))
    scratch_draw = ImageDraw.Draw(scratch)
    scratch_draw.multiline_text((4, 2), text, fill="#222222", font=font, spacing=0, align="right")
    bounds = scratch.getbbox()
    if bounds is None:
        return
    cropped = scratch.crop(bounds)
    rotated = cropped.rotate(45, expand=True, resample=Image.Resampling.BICUBIC)
    # Match Matplotlib's rotation=45, ha="right": the label's right edge is
    # anchored at its tick and the text extends down and to the left.
    image.paste(rotated, (center_x - rotated.width, top_y), rotated)


def vertical_label(image: Image.Image, text: str, x: int, center_y: int, font: Any) -> None:
    scratch = Image.new("RGBA", (1200, 100), (255, 255, 255, 0))
    scratch_draw = ImageDraw.Draw(scratch)
    scratch_draw.text((0, 0), text, fill="#222222", font=font)
    bounds = scratch.getbbox()
    if bounds is None:
        return
    cropped = scratch.crop(bounds)
    rotated = cropped.rotate(90, expand=True)
    image.paste(rotated, (x, center_y - rotated.height // 2), rotated)


def draw_panel(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    bounds: tuple[int, int, int, int],
    title: str,
    rows: list[dict[str, str]],
    header_font: Any,
    tick_font: Any,
    condition_font: Any,
    show_tick_labels: bool,
    show_condition_labels: bool,
) -> None:
    x0, y0, x1, y1 = bounds
    draw.text((x0 + 16, y0 + 5), title, fill="#111111", font=header_font)

    # Use identical plotting widths in every panel. 
    # The global left margin has enough room for the first column's y-axis tick labels.
    chart_left = x0 + 16
    chart_right = x1 - 10
    chart_top = y0 + 74
    chart_bottom = y1 - 110
    draw_grid(draw, chart_left, chart_top, chart_right, chart_bottom, tick_font, show_tick_labels)
    category_width = (chart_right - chart_left) / len(CONDITION_ORDER)
    bar_half_width = min(29, int(category_width * 0.34))

    for index, (condition, label) in enumerate(zip(CONDITION_ORDER, CONDITION_LABELS)):
        center = int(round(chart_left + (index + 0.5) * category_width))
        condition_rows = [row for row in rows if row["opponent_condition"] == condition]
        draw_bar(draw, condition_rows, center - bar_half_width, center + bar_half_width, chart_top, chart_bottom)
        if show_condition_labels:
            draw_rotated_label(image, label, center, chart_bottom + 9, condition_font)


def draw_horizontal_legend(
    draw: ImageDraw.ImageDraw,
    width: int,
    y: int,
    legend_font: Any,
) -> None:
    swatch_width, swatch_height = 50, 38
    text_gap = 15
    title = "Best-fit player strategy:"
    title_width = draw.textlength(title, font=legend_font)
    item_widths = [
        swatch_width + text_gap + draw.textlength(FAMILY_LABELS[family], font=legend_font)
        for family in FAMILY_ORDER
    ]
    side_margin = 20
    item_gap = max(
        10,
        (width - 2 * side_margin - title_width - sum(item_widths)) / len(item_widths),
    )
    cursor_x = side_margin
    draw.text((cursor_x, y), title, fill="#222222", font=legend_font)
    cursor_x += title_width + item_gap
    for index, family in enumerate(FAMILY_ORDER):
        draw.rectangle((cursor_x, y + 5, cursor_x + swatch_width, y + 5 + swatch_height), fill=FAMILY_COLORS[family])
        draw.text((cursor_x + swatch_width + text_gap, y), FAMILY_LABELS[family], fill="#222222", font=legend_font)
        cursor_x += item_widths[index] + item_gap



def create_figure(
    path: Path,
    human_cua_rows: list[dict[str, str]],
    llm_rows: list[dict[str, str]],
) -> None:
    width, height = 3750, 1540
    margin_left, margin_right = 200, 30
    margin_top = 18
    panel_height = 670
    row_gap = -90
    plot_bottom = margin_top + 2 * panel_height + row_gap
    column_gap = 22
    columns, rows_count = 7, 2
    panel_width = int((width - margin_left - margin_right - (columns - 1) * column_gap) / columns)

    image = Image.new("RGB", (width, height), "#FFFFFF")
    draw = ImageDraw.Draw(image)
    header_font = common.load_font(41, bold=True)
    tick_font = common.load_font(48)
    condition_font = common.load_font(39)
    legend_font = common.load_font(42)

    panels: list[tuple[str, list[dict[str, str]]]] = [
        ("Human", [row for row in human_cua_rows if row["cohort"] == "human"]),
    ]
    panels.extend(
        (
            CUA_DISPLAY_NAMES[agent],
            [
                row
                for row in human_cua_rows
                if row["cohort"] == "computer_use" and row["llm_model"] == agent
            ],
        )
        for agent in CUA_ORDER
    )
    panels.extend(
        (
            MODEL_DISPLAY_NAMES[model],
            [
                row
                for row in llm_rows
                if row["cohort"] == "llm" and row["llm_model"] == model
            ],
        )
        for model in LLM_ORDER
    )

    for index, (title, panel_rows) in enumerate(panels):
        row_index, column_index = divmod(index, columns)
        x0 = margin_left + column_index * (panel_width + column_gap)
        y0 = margin_top + row_index * (panel_height + row_gap)
        bounds = (x0, y0, x0 + panel_width, y0 + panel_height)
        draw_panel(
            image,
            draw,
            bounds,
            title,
            panel_rows,
            header_font,
            tick_font,
            condition_font,
            show_tick_labels=column_index == 0,
            show_condition_labels=row_index == 1,
        )

    vertical_label(
        image,
        "Proportion of game records",
        3,
        (margin_top + plot_bottom) // 2,
        common.load_font(64),
    )
    x_axis_label = "Partner strategy"
    x_axis_font = common.load_font(60)
    x_axis_width = draw.textlength(x_axis_label, font=x_axis_font)
    draw.text(
        ((width - x_axis_width) / 2, 1360),
        x_axis_label,
        fill="#222222",
        font=x_axis_font,
    )
    draw_horizontal_legend(
        draw,
        width,
        1450,
        legend_font,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    image.save(path.with_suffix(".pdf"), "PDF", resolution=300.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--human-cua-dir",
        type=Path,
        default=HERE / "aic_generative_results_compact",
    )
    parser.add_argument(
        "--llm-dir",
        type=Path,
        default=HERE / "aic_generative_results_compact",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "combined_strategy_figures",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for criterion in ("aic", "bic"):
        human_cua_rows = read_csv(args.human_cua_dir / f"{criterion}_winners.csv")
        llm_rows = read_csv(args.llm_dir / f"{criterion}_winners.csv")
        create_figure(
            args.output_dir / f"combined_{criterion}.png",
            human_cua_rows,
            llm_rows,
        )


if __name__ == "__main__":
    main()
