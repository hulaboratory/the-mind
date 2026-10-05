# The Mind: Rational Use of Time for Social Coordination

This repository contains the code and analysis materials for the paper: Rational Use of Time for Social Coordination in Humans and AI by Xiulin Yang, Sirisha Gudavalli, and Jennifer Hu. 

## How the game The Mind works

- Each game has 10 levels.
- At level `N`, each player receives `N` cards.
- Cards are played asynchronously in ascending order.
- Players cannot see their partner's cards.
- Playing before a lower partner card produces an error; lower cards are then
  automatically removed.
- Results record actions, waiting time, errors, model prompts, and accuracy.

If you are interested in the game demo, please feel free to contact Xiulin for the source code!
## Repository structure

```text
experiments/
├── CUA/       Browser/computer-use agent experiments
├── Human/     Prolific study and participant-management scripts (not publicly available due to privacy issues)
└── LLM/       Game engine, prompts, model agents, and experiment runners

analysis_results/   Precomputed JSON summaries used by the analysis scripts
results/            Saved game-level JSON results (If you want to analyse the results, please contact Xiulin)
simulation/         Parameter-sweep simulation outputs (If you want to access the simulation results, you can either contact Xiulin or simply run LLM/agent_play_simulation.py to generate the results)
vis_results/        Visualizations generated from the results
r_code/             R scripts for statistical analysis and figures
strategy_fit/       Strategy-model fitting and AIC/BIC model comparison (Figures 3 and 10)
submit/             Submission or batch-job helpers
requirements.txt    Python dependencies
```

## Installation

Create or activate a Python environment, then install the dependencies:

```bash
conda create -n themind python=3.12
conda activate themind
python -m pip install -r requirements.txt
```

Several experiments require substantial GPU memory and model checkpoints. The
LLM runners use `vllm` for local models and API clients for hosted models.
Download or configure the required model separately before starting an
experiment.

## Running LLM experiments

The main LLM runner accepts a model name, prompt type, and GPU count:

```bash
python experiments/LLM/agent_play.py <model_name> <prompt_type> <gpu_count>
# e.g.,
python experiments/LLM/agent_play.py qwen3-32b time_based_second 1
python experiments/LLM/agent_play.py llama8b time_based_half_second 1
```

Model names currently handled by the runner include:

```text
gpt-oss-120b  gpt-oss-20b qwen3-32b llama8b       llama1b      llama70b         qwen3-8b
gemini        gpt-5.6-luna gpt-6
```

Prompt/timing modes include `time_based_second`,
`time_based_half_second`, `time_based_q_second`,
`time_based_millisecond`, `wait_based`, and `wait_count_based`.

Game outputs are saved under a model- and prompt-specific directory in
`results/`.

## Human and computer-use experiments

- `experiments/Human/prolific_API.py` contains helpers for creating and
  managing Prolific studies. Review the study configuration and credentials
  before use. Not publicly available due to privacy issues.
- `experiments/CUA/` contains computer-use experiments for GPT and
  Gemini-based agents.

## Analysis

The R scripts in `r_code/` read JSON summaries from `analysis_results/`.
Some analysis scripts currently contain absolute local paths. Update their
`ANALYSIS_DIR` or `results_dir` variables before running them on another
machine.

The strategy-fit analysis (which cognitive strategy best explains each player's timing;
Figure 3 for AIC and Figure 10 for BIC) is in `strategy_fit/`. It reads the game-level
records from `results/`; see `strategy_fit/README.md` for how to run it.

## Citation
To be updated.

If you have questions about the code or experiments, please open an issue or contact Xiulin Yang (xiulin.yang.compling@gmail.com)
