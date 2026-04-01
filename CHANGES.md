# Development Changes

## Session: Refactoring for Hyperparameter Optimisation

### New files

- **`src/config.py`** — `ModelConfig` dataclass consolidating all hyperparameters (architecture, optimiser, LR scheduler, early stopping, verbosity) with documented defaults
- **`src/model.py`** — `build_mlp()` factory for configurable MLP architectures; `NMSELoss` moved here from the training script
- **`src/training.py`** — reusable training infrastructure:
  - `EarlyStopping` class with configurable patience and min-delta
  - `load_and_scale_data()` — loads splits, fits scaler on training set only
  - `fit()` — full training loop with best-checkpoint restoration, early stopping, GPU support, and an `epoch_callback` hook for Optuna pruning
  - `save_artifacts()` — saves model, scaler, and loss history
- **`scripts/hparam_search.py`** — Optuna hyperparameter search with TPE sampler, MedianPruner, per-trial progress callbacks, results summary, and final model re-training

### Modified files

- **`scripts/train_model.py`** — reduced to configuration and glue; uses the new `src/` helpers throughout

### Key design decisions

- `training.py` has no dependency on Optuna — the `epoch_callback(val_loss, epoch) -> bool` pattern decouples the two
- Data is loaded once outside the Optuna objective to avoid repeated disk I/O
- `fit()` always restores the best-checkpoint weights before returning, regardless of exit path
- `device` is a runtime flag, not a hyperparameter — lives in the script, not in `ModelConfig`
- Verbosity is controlled via `ModelConfig.verbose` (0 = silent, 1 = summary, 2 = per-epoch)
