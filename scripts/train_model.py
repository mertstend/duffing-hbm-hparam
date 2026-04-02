import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # suppress duplicate OpenMP warning on Windows

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.util import check_folder_structure
from src.config import ModelConfig
from src.model import build_mlp
from src.plotting import loss_plot
from src.training import load_and_scale_data, fit, save_artifacts

check_folder_structure()

###############################################################################
# IDs and flags
###############################################################################
SAVE   = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not SAVE:
    RED = "\033[91m"
    RESET = "\033[0m"
    input(f'{RED}WARNING: SAVE is set to {SAVE}. To save the model, '
          f'set SAVE = True. Press Enter to continue...{RESET}')

data_id = '2026-02-18_14-04-47'

###############################################################################
# Configuration — edit here to change hyperparameters
###############################################################################
config = ModelConfig(
    hidden_sizes=[32, 32, 32, 32, 32],
    activation='relu',
    dropout=0.0,
    n_epochs=1000,
    batch_size=64,
    early_stopping_patience=50,
)

###############################################################################
# Data
###############################################################################
X_train, y_train, X_val, y_val, *_, scaler = load_and_scale_data(data_id)

###############################################################################
# Model
###############################################################################
model = build_mlp(
    input_size=X_train.shape[1],
    output_size=y_train.shape[1],
    hidden_sizes=config.hidden_sizes,
    activation=config.activation,
    dropout=config.dropout,
)
print('\nModel architecture:')
print(model)

###############################################################################
# Training
###############################################################################
history = fit(model, X_train, y_train, X_val, y_val, config, device=device)

print(f'\nFinal train loss : {history["train_losses"][-1]:.3e}')
print(f'Final val   loss : {history["val_losses"][-1]:.3e}')
print(f'Best  val   loss : {history["best_val_loss"]:.3e}')

###############################################################################
# Save
###############################################################################
save_date = (np.datetime64('now').astype('str')
             .replace(':', '-').replace('T', '_'))
if SAVE:
    save_artifacts(model, scaler, history, save_date)

###############################################################################
# Plot loss curves
###############################################################################
loss_plot(history['train_losses'], history['val_losses'],
          figure_name=f'loss_{save_date}', save_figure=SAVE)
plt.show()

