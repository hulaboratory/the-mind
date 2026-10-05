# Rational Use of Time for Social Coordination in Humans and AI — strategy-fit analysis

Code to reproduce the best-fit-strategy figures of the paper (Figure 3, AIC; Figure 10, BIC):
for every game record, six timing-strategy models are fitted to the player's timing decisions
and the model with the lowest AIC/BIC is selected.

## Data

The scripts read the game data from the repository's `results/` directory:

- `results/<model>_<prompt>/game_*.json` — text-prompted LLM game records
- `results/cua_raw_data/the-mind-human-agent.json` — human game records
- `results/cua_raw_data/the-mind-{gemini,gpt-luna,gpt6}_updated.json` — computer-use-agent game records

The scripts read `results/` from the repository root (a `results/` directory placed next to the
scripts also works). The game-level data are not included in the repository; see the main README
for how to obtain them.

## Setup

Python 3.10 or newer.

```bash
pip install -r requirements.txt
```

## Reproduce the figures

```bash
./run_all.sh
```

Equivalent manual steps (both scripts can be run from any directory):

```bash
python3 generative_aic_analysis_compact.py   # fits all models, writes CSVs to aic_generative_results_compact/
python3 combined_strategy_figure.py          # draws the figures from those CSVs into combined_strategy_figures/
```

## Files

| File | Role |
|---|---|
| `generative_aic_analysis_compact.py` | Fits the six models (Random, Constant Wait, Linear ±Calibration, Bayesian ±Calibration; Appendix B) to each record and writes `raw_fits.csv`, `aic_winners.csv`, `bic_winners.csv`, winner proportions, `exclusions.csv`, `run_metadata.json`. |
| `combined_strategy_figure.py` | Draws the 14-panel AIC and BIC figures (`combined_aic.png|pdf`, `combined_bic.png|pdf`) from the winner CSVs. |
| `analysis_common.py` | Shared constants (model and partner labels, data layout) and helpers. |

## Notes

- In the code the paper's "Linear" strategy is called `counting`; the figure labels use the paper's names.
- The paper's figures use the 1-second time-based LLM prompt (`--llm-prompt-type time_based_second`, the default).
- Fonts: the figures use Helvetica/Arial on macOS and the standard sans-serif system fonts on Linux
  and Windows. If no system font is found, Pillow's built-in font is used; the bars are unaffected.
