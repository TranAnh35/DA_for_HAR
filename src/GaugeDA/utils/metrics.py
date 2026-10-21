from typing import Dict, Any, Union
import numpy as np
import torch

def compute_metrics(
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred: Union[torch.Tensor, np.ndarray]
) -> Dict[str, Any]:
    """
    Computes standard evaluation metrics for HAR:
    - Overall Accuracy (%)
    - Macro-F1 (unweighted mean across classes)
    - Per-class Precision, Recall, and F1
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = y_true.flatten().astype(int)
    y_pred = y_pred.flatten().astype(int)

    classes = np.unique(np.concatenate([y_true, y_pred]))
    num_classes = len(classes)

    total_samples = len(y_true)
    accuracy = 100.0 * np.sum(y_true == y_pred) / max(total_samples, 1)

    f1_list = []
    precision_list = []
    recall_list = []

    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        f1_list.append(f1)
        precision_list.append(prec)
        recall_list.append(rec)

    macro_f1 = 100.0 * np.mean(f1_list) if len(f1_list) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class_f1": [100.0 * f for f in f1_list],
        "classes": classes.tolist()
    }
