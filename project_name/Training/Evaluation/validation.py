import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict


def validate_model(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    device: torch.device,
    loss_fn: torch.nn.Module = torch.nn.MSELoss(),
    min_depth: float = 1e-3,
) -> Dict[str, float]:
    """
    Evaluate depth estimation model on dataset.

    Returns metrics: mse, rmse, mae, absrel, delta thresholds.
    """
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    total_mse = 0.0
    total_mae = 0.0
    total_absrel = 0.0
    total_delta1 = 0.0
    total_delta2 = 0.0
    total_delta3 = 0.0
    n_batches = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)

            # Basic losses
            mse = loss_fn(pred, y).item()
            total_mse += mse

            pred_np = pred.cpu().numpy()
            y_np = y.cpu().numpy()

            abs_err = np.abs(pred_np - y_np)
            total_mae += abs_err.mean()

            # Mask invalid / zero depths
            valid = y_np > min_depth

            # Absolute relative error on valid pixels
            absrel_map = abs_err / (y_np + 1e-6)
            total_absrel += absrel_map[valid].mean()

            # Threshold accuracies
            ratio = np.maximum(
                pred_np / (y_np + 1e-6),
                y_np / (pred_np + 1e-6),
            )
            total_delta1 += np.mean(ratio[valid] < 1.25)
            total_delta2 += np.mean(ratio[valid] < 1.25 ** 2)
            total_delta3 += np.mean(ratio[valid] < 1.25 ** 3)

            n_batches += 1

    # Compute averages
    avg_mse = total_mse / n_batches
    rmse = np.sqrt(avg_mse)
    mae = total_mae / n_batches
    absrel = total_absrel / n_batches
    delta1 = total_delta1 / n_batches
    delta2 = total_delta2 / n_batches
    delta3 = total_delta3 / n_batches

    return {
        "mse": avg_mse,
        "rmse": rmse,
        "mae": mae,
        "absrel": absrel,
        "delta1": delta1,
        "delta2": delta2,
        "delta3": delta3,
    }
