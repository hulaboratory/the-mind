import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from scipy import stats
plt.rcParams['hatch.linewidth'] = 1.0
plt.rcParams['font.size'] = 20
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


json_path = "analysis_results/overall_time_llm.json"

with open(json_path, "r") as f:
    raw = json.load(f)

rows = []
for model_name, partners in raw.items():
    for partner_name, prompts in partners.items():
        for prompt_name, hands in prompts.items():
            for hand_id, t in hands.items():
                if t is None:
                    continue
                rows.append({"model": model_name, "partner": partner_name,
                             "prompt_type": prompt_name, "hand": hand_id, "time": t})
long_df = pd.DataFrame(rows)


partner_labels = {
    "random": "Random",
    "count_false": "Linear (-Calibration)",
    "count_true": "Linear (+Calibration)",
    "bayesian_false": "Bayesian (-Calibration)",
    "bayesian_true": "Bayesian (+Calibration)",
    "llm": "LLM",
}
partner_base_colors = {
    "Random": "#4B4C4E",
    "Linear (-Calibration)": "#A6CEE3",
    "Linear (+Calibration)": "#1F78B4",
    "Bayesian (-Calibration)": "#FDBF6F",
    "Bayesian (+Calibration)": "#FF7F00",
    "LLM": "#33A02C",
}
long_df["partner"] = long_df["partner"].map(partner_labels).fillna(long_df["partner"])
condition_order = list(partner_labels.values())


summary_df = (
    long_df.groupby(["model", "partner", "prompt_type"], as_index=False)
    .agg(mean_time=("time", "mean"), sd_time=("time", "std"), n=("time", "count"))
)
sem = summary_df["sd_time"] / np.sqrt(summary_df["n"])
t_crit = stats.t.ppf(0.975, df=summary_df["n"] - 1)
summary_df["err"] = t_crit * sem
summary_df = summary_df[summary_df["partner"].isin(condition_order)].copy()


prompt_display_names = {
    "time_based_q_second": "1/4 sec",
    "time_based_half_second": "1/2 sec",
    "time_based_second": "1 sec",
    "time_based2": "time est.",
    "wait_based": "wait action",
    "wait_count_based": "wait count",
}
prompt_order = list(prompt_display_names.keys())

prompt_hatch = {
    "time_based_second": "",
    "time_based_half_second": "",
    "time_based_q_second": "",
    "time_based2": "",
    "wait_based": "\\\\",
    "wait_count_based": "//",
}

def make_shades(base_hex, n, lighten=0.5, darken=0.4):

    base = np.array(to_rgb(base_hex))
    shades = []
    for t in np.linspace(-lighten, darken, n):
        if t < 0:
            c = base + (1 - base) * (-t)
        else:
            c = base * (1 - t)
        shades.append(tuple(np.clip(c, 0, 1)))
    return shades


partner_prompt_colors = {
    partner: dict(zip(prompt_order, make_shades(base, len(prompt_order))))
    for partner, base in partner_base_colors.items()
}

summary_model = summary_df[
    (summary_df["model"] != "Human") & (summary_df["prompt_type"].isin(prompt_order))
].copy()


def add_strip_title(ax, text):
    """Partner name above the panel, left-aligned, no background"""
    ax.text(
        0.0, 1.02, text,
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=20, fontweight="normal"
    )


def apply_theme_bw(ax):
    ax.set_facecolor("white")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#333333")
        spine.set_linewidth(0.8)
    ax.tick_params(length=0)


def plot_partner_panel(ax, model, partner, show_xticks=True):
    sub = summary_model[(summary_model["model"] == model) &
                        (summary_model["partner"] == partner)]
    colors = partner_prompt_colors[partner]
    for p_idx, prompt in enumerate(prompt_order):
        row = sub[sub["prompt_type"] == prompt]
        if row.empty:
            continue
        r = row.iloc[0]
        hatch = prompt_hatch[prompt]
        ax.bar(p_idx, r["mean_time"], width=0.75,
               color=colors[prompt],
               hatch=hatch,
               edgecolor="white" if hatch else "none",
               linewidth=0, zorder=3)
        ax.errorbar(p_idx, r["mean_time"], yerr=r["err"], fmt="none",
                    ecolor="black", elinewidth=1, capsize=2, zorder=4)
    apply_theme_bw(ax)
    add_strip_title(ax, partner)
    ax.tick_params(axis="both", labelsize=20)
    ax.set_xticks(range(len(prompt_order)))
    if show_xticks:
        ax.set_xticklabels([prompt_display_names[p] for p in prompt_order],
                           rotation=45, ha="right", fontsize=20)
    else:
        ax.set_xticklabels([])


output_names = {
    "gpt-5.6-luna": "gpt-5_time_by_partner.pdf",
    "Llama-3.2-1B-Instruct": "llama-1b_time_by_partner.pdf",
}

focus_model = "gpt-5.6-luna"
# focus_model = "Llama-3.2-1B-Instruct"

available_partners = summary_model.loc[summary_model["model"] == focus_model, "partner"].unique()
partners_to_plot = [p for p in condition_order if p in available_partners]
missing = [p for p in condition_order if p not in available_partners]
if missing:
    print(f"Note: no data for {focus_model} with partner(s): {missing}")

n_panels = len(partners_to_plot)
n_cols = 3 if n_panels <= 6 else 4
n_rows = math.ceil(n_panels / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 4 * n_rows), sharey=True,
                         gridspec_kw=dict(wspace=0.12, hspace=0.18), squeeze=False)
axes = axes.flatten()

for i, ax in enumerate(axes):
    if i >= n_panels:
        ax.axis("off")
        continue
    show_x = (i + n_cols >= n_panels)
    plot_partner_panel(ax, focus_model, partners_to_plot[i], show_xticks=show_x)
    if i % n_cols == 0:
        ax.set_ylabel("Mean Time (95% CI)", fontsize=20)

plt.subplots_adjust(left=0.07, right=0.98, top=0.95, bottom=0.14)
out_path = "gpt5_time_by_partner.pdf"
plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
print(f"Saved to {out_path}")
plt.show()