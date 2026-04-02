from typing import Callable

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.config import ModelConfig


class EarlyStopping:
    """Tracks validation loss and signals when training should stop.

    Call ``step(val_loss)`` once per epoch; it returns True when no
    improvement larger than *min_delta* has been seen for *patience* epochs.
    """

    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        self.patience  = patience
        self.min_delta = min_delta
        self._best    = float('inf')
        self._counter = 0

    def step(self, val_loss: float) -> bool:
        if val_loss < self._best - self.min_delta:
            self._best    = val_loss
            self._counter = 0
        else:
            self._counter += 1
        return self._counter >= self.patience


def load_and_scale_data(data_id: str):
    """
    Load train/val/test splits, fit a per-feature standardisation scaler on
    the training set only, and return scaled float32 tensors plus the scaler.

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test : torch.Tensor
    scaler : dict  — keys: X_mean, X_std, y_mean, y_std  (numpy arrays)
    """
    train = np.load(f'data/duffing_train_data_H3_N64_{data_id}.npz')
    val   = np.load(f'data/duffing_val_data_H3_N64_{data_id}.npz')
    test  = np.load(f'data/duffing_test_data_H3_N64_{data_id}.npz')

    X_train_raw, y_train_raw = train['q_coeffs'], train['fnl_coeffs']
    X_val_raw,   y_val_raw   = val['q_coeffs'],   val['fnl_coeffs']
    X_test_raw,  y_test_raw  = test['q_coeffs'],  test['fnl_coeffs']

    # Scaler fitted on training data only
    X_mean = X_train_raw.mean(axis=0)
    X_std  = X_train_raw.std(axis=0);  X_std[X_std == 0] = 1.0
    y_mean = y_train_raw.mean(axis=0)
    y_std  = y_train_raw.std(axis=0);  y_std[y_std == 0] = 1.0

    scaler = {'X_mean': X_mean, 'X_std': X_std, 'y_mean': y_mean, 'y_std': y_std}

    def _scale(X, y):
        X = torch.tensor((X - X_mean) / X_std, dtype=torch.float32)
        y = torch.tensor((y - y_mean) / y_std, dtype=torch.float32)
        return X, y

    X_tr, y_tr = _scale(X_train_raw, y_train_raw)
    X_v,  y_v  = _scale(X_val_raw,   y_val_raw)
    X_te, y_te = _scale(X_test_raw,  y_test_raw)

    print('Data shapes:')
    print(f'  X_train={tuple(X_tr.shape)}, y_train={tuple(y_tr.shape)}')
    print(f'  X_val  ={tuple(X_v.shape)},  y_val  ={tuple(y_v.shape)}')
    print(f'  X_test ={tuple(X_te.shape)}, y_test ={tuple(y_te.shape)}')

    return X_tr, y_tr, X_v, y_v, X_te, y_te, scaler


def fit(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    config: ModelConfig,
    epoch_callback: Callable[[float, int], bool] | None = None,
    device: torch.device | None = None,
) -> dict:
    """
    Train *model* in-place and return loss history.

    Parameters
    ----------
    epoch_callback : callable(val_loss: float, epoch: int) -> bool, optional
        Called at the end of every epoch.  Return True to stop training early
        (e.g. for Optuna pruning).  Raising ``optuna.exceptions.TrialPruned``
        is the caller's responsibility after checking ``history['pruned']``.
    device : torch.device, optional
        Compute device.  Auto-detects CUDA if not provided.

    Returns
    -------
    dict with keys:
        'train_losses'  : list[float]
        'val_losses'    : list[float]
        'best_val_loss' : float
        'pruned'        : bool
        'stopped_early' : bool
    """
    # ------------------------------------------------------------------ setup --
    # Flow per epoch:
    #   1. mini-batch SGD pass over the training set
    #   2. full-batch validation pass (no gradients)
    #   3. LR scheduler step based on validation loss
    #   4. log, early-stopping check, optional Optuna callback
    #   5. shuffle training data for the next epoch
    # At exit (any path) the model weights are restored to the best checkpoint.

    # Auto-detect device if not specified by the caller
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if config.verbose >= 2:
        print(f'Using device: {device}')

    # Move model and all data tensors to the target device
    model.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_val   = X_val.to(device)
    y_val   = y_val.to(device)

    # MSE on standardised data is equivalent to a per-feature normalised error
    loss_fn   = nn.MSELoss()

    # Adam as standard. Weight decay is L2 regularisation on the weights, 
    # which can help with overfitting by penalising large weights.
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    # ReduceLROnPlateau halves the LR whenever the validation loss stops
    # improving for `lr_patience` epochs, down to a floor of `lr_min`.
    # This lets the optimiser take large steps early and fine-tune later.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config.lr_factor,
        patience=config.lr_patience,
        threshold=1e-4,
        threshold_mode='rel',
        min_lr=config.lr_min,
    )

    # Work on clones so the caller's tensors are not shuffled in-place
    X_tr = X_train.clone()
    y_tr = y_train.clone()

    train_losses: list[float] = []
    val_losses:   list[float] = []
    best_val_loss = float('inf')

    # Best-checkpoint: clone the state dict whenever val loss improves so we
    # can restore the best weights at the end, regardless of when we stop.
    best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # instantiate early stopper if enabled in config; else None to disable
    early_stopper = (
        EarlyStopping(config.early_stopping_patience, config.early_stopping_min_delta)
        if config.early_stopping_patience is not None else None
    )

    if config.verbose == 1:
        print(f'Training: hidden={config.hidden_sizes}, act={config.activation}, '
              f'lr={config.learning_rate:.2e}, bs={config.batch_size}')
    elif config.verbose >= 2:
        print('\nStarting training:')
    model.train()
    for epoch in range(config.n_epochs):

        # --- training pass ---
        epoch_loss, n_batches = 0.0, 0
        for i in range(0, len(X_tr), config.batch_size):
            Xb = X_tr[i:i + config.batch_size]
            yb = y_tr[i:i + config.batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(Xb), yb)
            loss.backward()
            optimizer.step()
            # .detach() drops the computation graph; addition stays on the GPU
            # so there is no CPU-GPU transfer inside the batch loop
            epoch_loss += loss.detach()
            n_batches  += 1
        # Single .item() call per epoch — the only CPU sync in the training pass
        train_loss = (epoch_loss / n_batches).item()
        train_losses.append(train_loss)

        # --- validation pass ---
        # model.eval() disables dropout and switches batchnorm to running stats,
        # giving a deterministic output that reflects inference-time behaviour.
        # torch.no_grad() skips building the computation graph — no gradients
        # are needed here, so this saves memory and speeds up the forward pass.
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val), y_val).item()
        model.train()  # restore training mode for the next epoch
        val_losses.append(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Snapshot the weights — clone keeps them independent of the live model
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        scheduler.step(val_loss)

        if config.verbose >= 2:
            current_lr = optimizer.param_groups[0]['lr']
            print(
                f'Epoch {epoch:4d} | '
                f'train loss = {train_loss:.3e} | '
                f'val loss = {val_loss:.3e} | '
                f'lr = {current_lr:.3e}'
            )

        # --- early stopping ---
        if early_stopper is not None and early_stopper.step(val_loss):
            if config.verbose >= 1:
                print(f'  Early stopping triggered after {epoch + 1} epochs '
                      f'({config.early_stopping_patience} epochs without improvement).')
            model.load_state_dict(best_state)  # restore best weights before returning
            return {
                'train_losses':  train_losses,
                'val_losses':    val_losses,
                'best_val_loss': best_val_loss,
                'pruned':        False,
                'stopped_early': True,
            }

        # --- optional external hook (e.g. Optuna pruning) ---
        # epoch_callback decouples this module from Optuna: the caller wraps
        # trial.report() and trial.should_prune() in a plain callable so that
        # training.py has no dependency on the optimisation framework.
        # If the callback signals that this trial is unpromising (returns True),
        # we exit immediately and let the caller raise TrialPruned.
        if epoch_callback is not None and epoch_callback(val_loss, epoch):
            if config.verbose >= 2:
                print(f'  Pruned at epoch {epoch}.')
            model.load_state_dict(best_state)  # restore best weights before returning
            return {
                'train_losses':  train_losses,
                'val_losses':    val_losses,
                'best_val_loss': best_val_loss,
                'pruned':        True,
                'stopped_early': False,
            }

        # --- shuffle for next epoch ---
        # Re-ordering the training data prevents the model from memorising
        # the fixed batch sequence, which can bias gradient estimates.
        perm = torch.randperm(len(X_tr))
        X_tr, y_tr = X_tr[perm], y_tr[perm]

    # Restore best weights — the final epoch is rarely the best one
    model.load_state_dict(best_state)
    if config.verbose >= 1:
        print(f'Done. Best val loss: {best_val_loss:.3e} (epoch {val_losses.index(best_val_loss)})')
    return {
        'train_losses':  train_losses,
        'val_losses':    val_losses,
        'best_val_loss': best_val_loss,
        'pruned':        False,
        'stopped_early': False,
    }


def save_artifacts(
    model: nn.Module,
    scaler: dict,
    history: dict,
    model_id: str,
) -> None:
    """Save model, scaler, and loss history to models/."""
    torch.save(model, f'models/duffing_mlp_h3_{model_id}.pt')
    joblib.dump(history, f'models/duffing_losses_h3_{model_id}.joblib')
    joblib.dump(scaler,  f'models/duffing_scaler_h3_{model_id}.joblib')
    print(f'Artifacts saved with id {model_id}')


def load_artifacts(model_id: str) -> tuple[nn.Module, dict, dict]:
    """Load model, scaler, and loss history saved by ``save_artifacts``.

    Parameters
    ----------
    model_id : str
        The timestamp string used when the artifacts were saved, e.g.
        ``'2026-03-25_16-59-05'``.

    Returns
    -------
    model   : nn.Module   — restored PyTorch model (eval mode)
    scaler  : dict        — keys: X_mean, X_std, y_mean, y_std
    history : dict        — keys: train_losses, val_losses, best_val_loss,
                            pruned, stopped_early
    """
    model   = torch.load(f'models/duffing_mlp_h3_{model_id}.pt', weights_only=False)
    scaler  = joblib.load(f'models/duffing_scaler_h3_{model_id}.joblib')
    history = joblib.load(f'models/duffing_losses_h3_{model_id}.joblib')
    model.eval()
    return model, scaler, history


def display_model_info(model: nn.Module, scaler: dict, history: dict, model_id: str = '') -> None:
    """Print a summary of a loaded model to the console.

    Covers three aspects:
    - Architecture: layer structure and trainable parameter count.
    - Scaler: per-feature mean and standard deviation for inputs and outputs.
    - Training history: epochs run, best validation loss, early-stopping status.

    Parameters
    ----------
    model    : nn.Module  — the loaded model
    scaler   : dict       — as returned by ``load_artifacts``
    history  : dict       — as returned by ``load_artifacts``
    model_id : str        — optional identifier shown in the header
    """
    sep   = '=' * 55
    thin  = '-' * 55

    header = f'  Model: duffing_mlp_h3_{model_id}' if model_id else '  Model'
    print(f'\n{sep}\n{header}\n{sep}')

    # ---------------------------------------------------------------- architecture --
    print('\nArchitecture')
    print(thin)
    print(model)

    # Count trainable parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Extract input/output sizes from first and last Linear layers
    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    in_features  = linear_layers[0].in_features  if linear_layers else '?'
    out_features = linear_layers[-1].out_features if linear_layers else '?'

    print(f'Inputs : {in_features}   Outputs : {out_features}')
    print(f'Trainable parameters: {n_params:,}')

    # -------------------------------------------------------------------- scaler --
    print('\nScaler')
    print(thin)
    np.set_printoptions(precision=4, suppress=True)
    print(f'  X  mean : {scaler["X_mean"]}')
    print(f'     std  : {scaler["X_std"]}')
    print(f'  y  mean : {scaler["y_mean"]}')
    print(f'     std  : {scaler["y_std"]}')

    # ----------------------------------------------------------- training history --
    print('\nTraining History')
    print(thin)
    train_losses = history['train_losses']
    val_losses   = history['val_losses']
    best_epoch   = int(np.argmin(val_losses))
    print(f'  Epochs trained   : {len(train_losses)}')
    print(f'  Best val loss    : {history["best_val_loss"]:.4e}  (epoch {best_epoch})')
    print(f'  Final train loss : {train_losses[-1]:.4e}')
    print(f'  Final val loss   : {val_losses[-1]:.4e}')
    print(f'  Early stopped    : {"Yes" if history["stopped_early"] else "No"}')
    print(f'  Pruned           : {"Yes" if history["pruned"] else "No"}')
    print()
