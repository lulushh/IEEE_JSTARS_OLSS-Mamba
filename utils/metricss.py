"""Torch metrics for HSI classification.

Labels are expected to start from 1 when computing final OA/AA/Kappa, matching
common HSI reporting conventions. Training internally uses 0-based labels.
"""

from __future__ import annotations

import torch


def th_confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes=None) -> torch.Tensor:
    y_true = y_true.long().view(-1)
    y_pred = y_pred.long().view(-1)
    if num_classes is None:
        num_classes = int(max(y_true.max(), y_pred.max()).item())
    valid = (y_true >= 1) & (y_true <= num_classes) & (y_pred >= 1) & (y_pred <= num_classes)
    y_true = y_true[valid] - 1
    y_pred = y_pred[valid] - 1
    idx = y_true * num_classes + y_pred
    cm = torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return cm.float()


def th_overall_accuracy_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    y_true = y_true.view(-1).long()
    y_pred = y_pred.view(-1).long()
    return (y_true == y_pred).float().mean()


def th_average_accuracy_score(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    num_classes=None,
    return_accuracys: bool = False,
):
    cm = th_confusion_matrix(y_true, y_pred, num_classes)
    per_class = torch.diag(cm) / (cm.sum(dim=1) + 1e-6)
    aa = per_class.mean()
    if return_accuracys:
        return aa, per_class
    return aa


def th_cohen_kappa_score(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes=None) -> torch.Tensor:
    cm = th_confusion_matrix(y_true, y_pred, num_classes)
    total = cm.sum()
    if total <= 0:
        return torch.tensor(0.0, device=cm.device)
    po = torch.diag(cm).sum() / total
    pe = (cm.sum(dim=0) * cm.sum(dim=1)).sum() / (total * total + 1e-6)
    return (po - pe) / (1.0 - pe + 1e-6)


def th_intersection_over_union_per_class(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes=None) -> torch.Tensor:
    cm = th_confusion_matrix(y_true, y_pred, num_classes)
    diag = torch.diag(cm)
    denominator = cm.sum(dim=0) + cm.sum(dim=1) - diag
    return diag / (denominator + 1e-6)


def th_mean_intersection_over_union(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes=None) -> torch.Tensor:
    return th_intersection_over_union_per_class(y_true, y_pred, num_classes).mean()
