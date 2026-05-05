import torch
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

def threshold_predictions(logits, threshold=0.5):
    """convert raw logits to binary predictions."""
    probs = torch.sigmoid(logits)
    return (probs > threshold).float()

def compute_iou(preds, targets):
    """compute intersection over union for binary masks."""
    preds   = preds.view(-1).cpu().numpy()
    targets = targets.view(-1).cpu().numpy()

    intersection = ((preds == 1) & (targets == 1)).sum()
    union        = ((preds == 1) | (targets == 1)).sum()

    if union == 0:
        return 1.0
    return intersection / union

def compute_f1(preds, targets):
    """compute f1 score for binary change detection."""
    preds   = preds.view(-1).cpu().numpy().astype(int)
    targets = targets.view(-1).cpu().numpy().astype(int)
    return f1_score(targets, preds, zero_division=0)

def compute_precision(preds, targets):
    preds   = preds.view(-1).cpu().numpy().astype(int)
    targets = targets.view(-1).cpu().numpy().astype(int)
    return precision_score(targets, preds, zero_division=0)

def compute_recall(preds, targets):
    preds   = preds.view(-1).cpu().numpy().astype(int)
    targets = targets.view(-1).cpu().numpy().astype(int)
    return recall_score(targets, preds, zero_division=0)

def compute_confusion_matrix(preds, targets):
    preds   = preds.view(-1).cpu().numpy().astype(int)
    targets = targets.view(-1).cpu().numpy().astype(int)
    return confusion_matrix(targets, preds, labels=[0, 1])

def evaluate_batch(logits, targets, threshold=0.5):
    """compute all metrics for a batch."""
    preds = threshold_predictions(logits, threshold)
    return {
        "f1":        compute_f1(preds, targets),
        "iou":       compute_iou(preds, targets),
        "precision": compute_precision(preds, targets),
        "recall":    compute_recall(preds, targets)
    }

def aggregate_metrics(metrics_list):
    """average a list of per-batch metric dicts."""
    keys = metrics_list[0].keys()
    return {k: np.mean([m[k] for m in metrics_list]) for k in keys}
