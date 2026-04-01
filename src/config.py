from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """Hyperparameters for model architecture and training.

    Defaults reproduce the original hand-tuned configuration.  Fields are
    grouped by concern so an Optuna objective can override only what it tunes.
    """

    # ---------------------------------------------------------------- architecture --

    # One entry per hidden layer; controls both depth and width.
    # Recommended: 3–6 layers, 32–128 units each.
    hidden_sizes: List[int] = field(default_factory=lambda: [32, 32, 32, 32, 32])

    # Nonlinearity after every hidden layer.
    # "relu" and "gelu" are the most reliable defaults for regression.
    # Options: "relu" | "gelu" | "tanh" | "elu" | "sigmoid"
    activation: str = "relu"

    # Dropout probability after each hidden activation (0 = disabled).
    # Start at 0; increase to 0.1–0.3 if overfitting is observed.
    dropout: float = 0.0

    # ---------------------------------------------------------------- optimisation --

    # Upper bound on training epochs; early stopping can terminate before this.
    n_epochs: int = 1000

    # Samples per gradient update. Powers of 2 use GPU memory most efficiently.
    # Recommended range: 16–256.
    batch_size: int = 64

    # Initial Adam learning rate. Recommended range: 1e-4 – 1e-2 (log scale).
    learning_rate: float = 1e-3

    # L2 weight regularisation. Start at 0; try 1e-5 – 1e-3 if overfitting.
    weight_decay: float = 0.0

    # -------------------------------------------- LR scheduler (ReduceLROnPlateau) --

    # LR is multiplied by this factor on each reduction. Default: halve the LR.
    lr_factor: float = 0.5

    # Epochs without val-loss improvement before reducing LR. Range: 10–50.
    lr_patience: int = 20

    # Hard lower bound on the learning rate.
    lr_min: float = 1e-8

    # --------------------------------------------------------------- early stopping --

    # Epochs without improvement before stopping. None = disabled.
    # Should be larger than lr_patience so the scheduler can act before we stop.
    # Recommended range: 50–200.
    early_stopping_patience: Optional[int] = None

    # Minimum improvement that counts as progress. 0 means any decrease counts.
    early_stopping_min_delta: float = 0.0

    # ----------------------------------------------------------------- verbosity --
    # Controls how much is printed during training.
    #   0 — silent
    #   1 — one summary line at start + final result (recommended for Optuna)
    #   2 — full per-epoch output (default)
    verbose: int = 2
