import time
import numpy as np
import torch  # type: ignore
import torch.nn.functional as F  # type: ignore
import os
from PIL import Image
import cv2  # type: ignore

from typing import type_check_only

if type_check_only:
    from torch.utils.data import Dataset
    import torch.nn as nn


def save_prediction_images(img_tensor: torch.Tensor,
                           pred_tensor: torch.Tensor,
                           index: int,
                           save_dir: str = "pred") -> None:
    """Save the prediction"""

    os.makedirs(save_dir, exist_ok=True)

    img_norm = normalize(img_tensor)
    pred_norm = normalize(pred_tensor.squeeze().cpu())

    input_np = (img_norm.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    pred_np = (pred_norm.cpu().numpy() * 255).astype(np.uint8)

    h, w = input_np.shape[:2]
    pred_np_resized = cv2.resize(pred_np, (w, h), cv2.INTER_LINEAR)

    if input_np.shape[2] == 1:
        input_np = np.repeat(input_np, 3, axis=2)

    pred_img_3ch = np.stack([pred_np_resized] * 3, axis=2)

    combined_img = np.concatenate((input_np, pred_img_3ch), axis=1)

    combined_img_pil = Image.fromarray(combined_img)
    combined_img_pil.save(os.path.join(save_dir, f"combined_{index}.png"))


def normalize(img: torch.Tensor) -> torch.Tensor:
    """Normalize array for visualization."""
    if isinstance(img, torch.Tensor):
        img_min = img.min()
        img_max = img.max()
        return (img - img_min) / (img_max - img_min + 1e-8)
    else:
        img_min = np.min(img)
        img_max = np.max(img)
        return (img - img_min) / (img_max - img_min + 1e-8)


def evaluate_zoedepth_model(model: nn.module,
                            dataloader: Dataset,
                            device: torch.device) -> dict:
    """Evaluate model

    Args:
        model (model): the model to be evaluated
        dataloader (dataloader with data):data loader with the data to use
        device (torch.device): device to run the model on

    Returns:
        dict: the eval results
    """
    model.eval()
    model.to(device)

    maes, rmses, mses, absrels = [], [], [], []
    delta1s, delta2s, delta3s = [], [], []
    times = []

    with torch.no_grad():
        for i, (images, depths_gt) in enumerate(dataloader):
            print(f"\n--- Batch {i} ---")

            images = images.to(device)
            depths_gt = depths_gt.to(device)

            start = time.time()
            preds = model(images)
            times.append(time.time() - start)

            if isinstance(preds, dict):
                preds = preds["metric_depth"]
            elif isinstance(preds, (list, tuple)):
                preds = preds[0]

            if preds.shape[-2:] != depths_gt.shape[-2:]:
                i, s = F.interpolate, depths_gt.shape[-2:]
                preds = i(preds, s, "bilinear", False)

            preds_np = preds.cpu().numpy().reshape(-1)
            gt_np = depths_gt.cpu().numpy().reshape(-1)

            valid_mask = (gt_np > 0) & (preds_np > 0)
            gt_np = gt_np[valid_mask]
            preds_np = preds_np[valid_mask]

            mae = np.mean(np.abs(gt_np - preds_np))
            mse = np.mean((gt_np - preds_np) ** 2)
            rmse = np.sqrt(mse)
            absrel = np.mean(np.abs(gt_np - preds_np) / gt_np)

            thresh = np.maximum(gt_np / preds_np, preds_np / gt_np)
            delta1 = np.mean(thresh < 1.25)
            delta2 = np.mean(thresh < 1.25 ** 2)
            delta3 = np.mean(thresh < 1.25 ** 3)

            maes.append(mae)
            mses.append(mse)
            rmses.append(rmse)
            absrels.append(absrel)
            delta1s.append(delta1)
            delta2s.append(delta2)
            delta3s.append(delta3)

            img, pred = images[0].cpu(), preds[0].cpu()
            save_prediction_images(img, pred, index=i, save_dir='save_dir')

    return {
        "MAE": np.mean(maes),
        "MSE": np.mean(mses),
        "RMSE": np.mean(rmses),
        "AbsRel": np.mean(absrels),
        "Delta1": np.mean(delta1s),
        "Delta2": np.mean(delta2s),
        "Delta3": np.mean(delta3s),
        "Inference Time (s)": np.mean(times),
    }
