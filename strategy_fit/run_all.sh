#!/usr/bin/env bash
# Reproduce the paper's best-fit-strategy figures (AIC: Figure 3, BIC: Figure 10).
# Usage: ./run_all.sh        (from any directory; about 8 minutes on 4 cores)
set -euo pipefail
cd "$(dirname "$0")"
python3 generative_aic_analysis_compact.py   # fit all models -> aic_generative_results_compact/
python3 combined_strategy_figure.py          # draw figures  -> combined_strategy_figures/combined_{aic,bic}.{png,pdf}
echo "Figures written to $(pwd)/combined_strategy_figures/"
