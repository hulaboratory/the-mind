import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams['font.size'] = 18
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

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
    "count_false": "Linear\n(- Calibration)",
    "count_true": "Linear\n(+ Calibration)",
    "bayesian_false": "Bayesian\n(- Calibration)",
    "bayesian_true": "Bayesian\n(+ Calibration)",
    "llm": "LLM",
}
long_df["partner"] = long_df["partner"].map(partner_labels).fillna(long_df["partner"])
condition_order = list(dict.fromkeys(partner_labels.values()))


prompt_order = [
    "time_based_second", "time_based_half_second", "time_based_q_second",
    "time_based2",
]
long_df = long_df[(long_df["model"] != "Human") & (long_df["prompt_type"].isin(prompt_order))]


per_prompt_mean = (
    long_df.groupby(["model", "partner", "prompt_type"], as_index=False)
    .agg(mean_time=("time", "mean"))
)

cv_df = (
    per_prompt_mean.groupby(["model", "partner"], as_index=False)
    .agg(
        cv=("mean_time", lambda x: x.std(ddof=1) / x.mean() if x.mean() != 0 else np.nan),
        n_prompts=("mean_time", "count"),
    )
)

cv_df.loc[cv_df["n_prompts"] < 2, "cv"] = np.nan

cv_df.to_csv("analysis_results/prompt_time_cv_by_model_partner.csv", index=False)

model_order = [
    "gpt-5.6-luna", "gemini-3.7-flash", "gpt-oss-120b", "gpt-oss-20b",
    "Llama-3.1-70B-Instruct", "Llama-3.1-8B-Instruct", "Llama-3.2-1B-Instruct",
    "Qwen3-32B", "Qwen3-8B",
]
model_order = [m for m in model_order if m in cv_df["model"].unique()]
partner_order = [p for p in condition_order if p in cv_df["partner"].unique()]

cv_matrix = cv_df.pivot(index="model", columns="partner", values="cv")
cv_matrix = cv_matrix.reindex(index=model_order, columns=partner_order)

fig, ax = plt.subplots(figsize=(9.33, 6.66))

purple_blue_hex = [
    "#6B0077", "#6C1D7D", "#6D2D83", "#6E3A89", "#704590",
    "#724F96", "#74599C", "#7663A2", "#786CA8", "#7B75AE",
    "#7D7EB3", "#8086B9", "#848FBE", "#8797C3", "#8B9FC7",
    "#90A7CC", "#94AED0", "#9AB5D4", "#9FBDD8", "#A5C3DB",
    "#ABCADF", "#B2D0E2", "#B9D6E5", "#C0DCE8", "#C7E1EA",
    "#CFE5ED", "#D6EAEF", "#DEEDF0", "#E7F0F1", "#F1F1F1",
]
purple_blue_cmap = LinearSegmentedColormap.from_list("purple_blue", purple_blue_hex, N=256)

im = ax.imshow(cv_matrix.values, cmap=purple_blue_cmap, aspect="auto")

ax.set_xticks(range(len(partner_order)))
ax.set_xticklabels(partner_order, rotation=45, ha="right")
ax.set_yticks(range(len(model_order)))
ax.set_yticklabels(model_order)

vmax = np.nanmax(cv_matrix.values)
for i in range(len(model_order)):
    for j in range(len(partner_order)):
        val = cv_matrix.values[i, j]
        if np.isnan(val):
            ax.text(j, i, "–", ha="center", va="center", color="grey", fontsize=13)
        else:
            text_color = "black" if val > vmax * 0.55 else "white"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                     color=text_color, fontsize=13)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("CV (SD / mean, across prompt types)")

plt.tight_layout()
plt.savefig("cv.pdf", dpi=300, bbox_inches="tight", pad_inches=0.15)
plt.show()