import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # suppress duplicate OpenMP warning on Windows

import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch

from src.util import check_folder_structure
from src.config import ModelConfig
from src.model import build_mlp
from src.plotting import loss_plot
from src.training import load_and_scale_data, fit, save_artifacts, load_artifacts, display_model_info

check_folder_structure()

# Suppress Optuna's per-trial INFO logs; fit() already prints epoch-level output
optuna.logging.set_verbosity(optuna.logging.WARNING)

# -- settings ------------------------------------------------------------------

DATA_ID           = '2026-02-18_14-04-47'  # name of the data set
N_TRIALS          = 15
SAVE              = True
TRIAL_N_EPOCHS    = 800
TRIAL_ES_PATIENCE = 100
device            = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -- data — loaded once and reused across all trials ---------------------------

X_train, y_train, X_val, y_val, *_, scaler = load_and_scale_data(DATA_ID)


# -- search space --------------------------------------------------------------

def config_from_trial(trial: optuna.Trial) -> ModelConfig:
    """Map an Optuna trial to a ModelConfig by sampling the search space."""
    n_layers    = trial.suggest_int('n_layers', 1, 6)
    hidden_size = trial.suggest_categorical('hidden_size', [4, 8, 16, 32, 64])
    return ModelConfig(
        # All layers share the same width; keeps the search space manageable.
        hidden_sizes            = [hidden_size] * n_layers,
        activation              = trial.suggest_categorical('activation',
                                      ['relu', 'gelu', 'tanh', 'sigmoid', 'elu']),
        dropout                 = trial.suggest_float('dropout', 0.0, 0.2),
        batch_size              = trial.suggest_categorical('batch_size',
                                      [64, 128, 256, 512]),
        learning_rate           = trial.suggest_float('learning_rate',
                                      1e-4, 1e-2, log=True),
        n_epochs                = TRIAL_N_EPOCHS,
        early_stopping_patience = TRIAL_ES_PATIENCE,
        verbose                 = 1,
    )


# -- objective — called once per trial -----------------------------------------

def objective(trial: optuna.Trial) -> float:
    config = config_from_trial(trial)
    model  = build_mlp(X_train.shape[1], y_train.shape[1],
                       config.hidden_sizes, config.activation, config.dropout)

    # Wrap Optuna's pruning API in the epoch_callback interface so that
    # training.py stays independent of Optuna (see training.py for details).
    def prune_callback(val_loss: float, epoch: int) -> bool:
        trial.report(val_loss, epoch)
        return trial.should_prune()

    history = fit(model, X_train, y_train, X_val, y_val, config,
                  epoch_callback=prune_callback, device=device)

    if history['pruned']:
        raise optuna.exceptions.TrialPruned()
    return history['best_val_loss']


# -- run the study -------------------------------------------------------------

# TPESampler: Tree-structured Parzen Estimator — models the search space and
# focuses samples on promising regions; much more efficient than grid search.
# MedianPruner: prunes a trial when its val loss is worse than the median of
# completed trials at the same epoch. n_startup_trials and n_warmup_steps
# prevent the pruner from firing before it has enough data to judge.
sampler = optuna.samplers.TPESampler(seed=42)
pruner  = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=50)
study   = optuna.create_study(direction='minimize', sampler=sampler, pruner=pruner)

print(f'\nStarting hyperparameter search:'
      f'\n  Trials       : {N_TRIALS}'
      f'\n  Device       : {device}'
      f'\n  Sampler      : {type(study.sampler).__name__}'
      f'\n  Pruner       : {type(study.pruner).__name__}'
      f'\n  Epochs/trial : {TRIAL_N_EPOCHS}  (early stop patience: {TRIAL_ES_PATIENCE})\n')


def trial_summary_callback(study: optuna.Study,
                            trial: optuna.trial.FrozenTrial) -> None:
    """Print a one-line progress update after each completed or pruned trial."""
    status = 'pruned' if trial.state == optuna.trial.TrialState.PRUNED else 'done  '
    value  = f'{trial.value:.4e}' if trial.value is not None else '      n/a'
    print(f'  Trial {trial.number + 1:3d}/{N_TRIALS} [{status}] — '
          f'val loss: {value} | best: {study.best_value:.4e}')


study.optimize(objective, n_trials=N_TRIALS, callbacks=[trial_summary_callback])

save_date = np.datetime64('now').astype('str').replace(':', '-').replace('T', '_')

# -- results -------------------------------------------------------------------

states     = study.trials_dataframe()['state'].value_counts()
n_complete = states.get('COMPLETE', 0)
n_pruned   = states.get('PRUNED', 0)

completed_values = np.array([t.value for t in study.trials
                              if t.state == optuna.trial.TrialState.COMPLETE])

print(f'\nStudy complete — {N_TRIALS} trials ({n_complete} complete, {n_pruned} pruned).'
      f'\nBest trial    : #{study.best_trial.number}'
      f'\nBest val loss : {study.best_value:.4e}'
      f'\n\nCompleted trial val losses:'
      f'\n  min    : {completed_values.min():.4e}'
      f'\n  median : {np.median(completed_values):.4e}'
      f'\n  mean   : {completed_values.mean():.4e}'
      f'\n  max    : {completed_values.max():.4e}'
      f'\n  std    : {completed_values.std():.4e}')

# Optimization history: each trial's val loss + running best (Optuna built-in)
optuna.visualization.matplotlib.plot_optimization_history(study)
plt.tight_layout()
plt.yscale('log')
if SAVE:
    plt.savefig(f'figures/optuna_history_{save_date}.svg', bbox_inches='tight')

print('\nBest hyperparameters:')
for k, v in study.best_params.items():
    print(f'  {k}: {v}')

# Top 5 trials ranked by val loss (study.trials_dataframe() is Optuna built-in)
df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
df = df[df['state'] == 'COMPLETE'].sort_values('value').head(5)
print(f'\nTop 5 trials:\n{df.to_string(index=False)}')

if SAVE:
    with open(f'models/hparamsearch_top5_{save_date}.txt', 'w') as f:
        f.write(f'Top 5 trials — hyperparameter search {save_date}\n')
        f.write('=' * 55 + '\n\n')
        f.write(df.to_string(index=False))
        f.write('\n')

# -- re-train with best hyperparameters and a full training budget -------------

# Build the config directly from best_params rather than re-running the trial,
# since FrozenTrial does not support suggest_* methods.
p = study.best_params
final_config = ModelConfig(
    hidden_sizes            = [p['hidden_size']] * p['n_layers'],
    activation              = p['activation'],
    dropout                 = p['dropout'],
    batch_size              = p['batch_size'],
    learning_rate           = p['learning_rate'],
    n_epochs                = 2000,
    early_stopping_patience = 200,
    verbose                 = 1,
)
final_model = build_mlp(X_train.shape[1], y_train.shape[1],
                        final_config.hidden_sizes, final_config.activation,
                        final_config.dropout)

print('\nRe-training final model with best hyperparameters...')
history = fit(final_model, X_train, y_train, X_val, y_val, final_config, device=device)
print(f'Final best val loss : {history["best_val_loss"]:.4e}')

# -- save ----------------------------------------------------------------------

if SAVE:
    save_artifacts(final_model, scaler, history, save_date)

# -- plot loss curves of final model -------------------------------------------

loss_plot(history['train_losses'], history['val_losses'],
          figure_name=f'loss_{save_date}', save_figure=SAVE)
plt.show()

# -- display the best model's configuration ------------------------------------------------

if SAVE:
    loaded_model, loaded_scaler, loaded_history = load_artifacts(save_date)
    display_model_info(loaded_model, loaded_scaler, loaded_history, model_id=save_date)
