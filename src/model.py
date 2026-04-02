import torch
import torch.nn as nn
from typing import List


# Registry mapping string names to PyTorch activation classes.
# Using strings keeps the interface serialisation-friendly (e.g. for Optuna
# trial suggestions or saving config to disk).
_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "elu":  nn.ELU,
    "sigmoid": nn.Sigmoid,
}


class NMSELoss(nn.Module):
    """Normalized Mean Squared Error: MSE / (mean(targets²) + eps).

    Normalising by the target energy makes the loss scale-invariant, which
    can help when output features differ in magnitude.
    """
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mse = torch.mean((predictions - targets) ** 2)
        target_norm = torch.mean(targets ** 2)
        return mse / (target_norm + 1e-8)


def build_mlp(
    input_size: int,
    output_size: int,
    hidden_sizes: List[int],
    activation: str = "relu",
    dropout: float = 0.0,
) -> nn.Sequential:
    """
    Build a fully-connected MLP.

    Parameters
    ----------
    input_size, output_size : int
    hidden_sizes : list of int — one entry per hidden layer
    activation   : one of "relu", "gelu", "tanh", "elu", "sigmoid"
    dropout      : dropout probability after each hidden activation (0 = disabled)
    """
    if activation not in _ACTIVATIONS:
        raise ValueError(
            f"Unknown activation '{activation}'. Choose from {list(_ACTIVATIONS)}."
        )

    activation_cls = _ACTIVATIONS[activation]

    # Build the layer list dynamically so that depth and width are controlled
    # entirely by hidden_sizes — e.g. [32, 64, 32] gives three hidden layers.
    layers = []
    in_features = input_size
    for h in hidden_sizes:
        layers.append(nn.Linear(in_features, h))
        layers.append(activation_cls())
        if dropout > 0.0:
            # Dropout is placed after the activation, which is the standard
            # convention for feed-forward networks.
            layers.append(nn.Dropout(dropout))
        in_features = h  # next layer reads what this layer wrote

    # Final linear layer — no activation, so the output is unbounded
    layers.append(nn.Linear(in_features, output_size))
    return nn.Sequential(*layers)

