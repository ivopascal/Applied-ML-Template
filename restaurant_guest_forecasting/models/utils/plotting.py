import torch
import matplotlib.pyplot as plt
from typing import Union

def plot_avg_loss_over_epochs(train_losses: Union[list, torch.Tensor],
                              val_losses: Union[list, torch.Tensor],
                              title: str = "Average Training and Validation Loss Across Epochs"):
    """
    Plots average training and validation loss per task across epochs.

    Args:
        train_losses: (epochs,) or (epochs, 1) — single-task losses.
        val_losses:   (epochs,) or (epochs, 1)
    """
    train_losses = torch.tensor(train_losses).squeeze()
    val_losses = torch.tensor(val_losses).squeeze()

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses.tolist(), label="Train Loss", marker='o')
    plt.plot(epochs, val_losses.tolist(), label="Validation Loss", marker='s')

    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()