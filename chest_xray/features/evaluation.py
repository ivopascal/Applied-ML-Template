import time
from typing import Dict, List, Tuple, Any

import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count the total and trainable parameters of a PyTorch model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total_params, trainable_params

def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    disease_labels: List[str],
    threshold: float = 0.5,
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, Dict[str, Any]]]:
    """Evaluate a multi-label classification model."""
    model.eval()

    all_probs = []
    all_labels = []

    start_time = time.time()

    # Disable gradient calculation during evaluation.
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device).float()

            # Model outputs raw logits.
            logits = model(images)

            # Convert logits to probabilities for multi-label classification.
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    end_time = time.time()
    inference_time = end_time - start_time

    # Combine all batches into one array.
    y_prob = np.vstack(all_probs)
    y_true = np.vstack(all_labels)

    # Convert probabilities to binary predictions.
    y_pred = (y_prob >= threshold).astype(int)

    results = []
    confusion_matrices = {}

    # Evaluate each disease as a separate binary classification task.
    for i, disease in enumerate(disease_labels):
        true_label = y_true[:, i]
        pred_label = y_pred[:, i]
        prob_label = y_prob[:, i]

        # AUROC can fail if only one class is present in y_true.
        try:
            auroc = roc_auc_score(true_label, prob_label)
        except ValueError:
            auroc = np.nan

        precision = precision_score(true_label, pred_label, zero_division=0)
        recall = recall_score(true_label, pred_label, zero_division=0)
        f1 = f1_score(true_label, pred_label, zero_division=0)

        # Matrix order with labels=[0, 1]:
        # [[TN, FP],
        #  [FN, TP]]
        cm = confusion_matrix(true_label, pred_label, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        confusion_matrices[disease] = {
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
            "matrix": cm,
        }

        results.append(
            {
                "Disease": disease,
                "AUROC": auroc,
                "Precision": precision,
                "Recall": recall,
                "F1-score": f1,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "TP": tp,
            }
        )

    results_df = pd.DataFrame(results)

    # Macro metrics are the average across all disease labels.
    macro_auroc = np.nanmean(results_df["AUROC"])
    macro_precision = results_df["Precision"].mean()
    macro_recall = results_df["Recall"].mean()
    macro_f1 = results_df["F1-score"].mean()

    total_params, trainable_params = count_parameters(model)

    summary = {
        "Macro AUROC": macro_auroc,
        "Macro Precision": macro_precision,
        "Macro Recall": macro_recall,
        "Macro F1-score": macro_f1,
        "Inference Time Seconds": inference_time,
        "Inference Time Per Image Seconds": inference_time / len(dataloader.dataset),
        "Total Parameters": total_params,
        "Trainable Parameters": trainable_params,
    }

    return results_df, summary, confusion_matrices