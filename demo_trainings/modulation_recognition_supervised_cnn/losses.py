from __future__ import annotations

import torch


def classification_loss(
    class_logits: torch.Tensor,
    class_ids: torch.Tensor,
    *,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(
        class_logits.float(),
        class_ids.long(),
        label_smoothing=float(label_smoothing),
    )


def classification_metrics(
    class_logits: torch.Tensor,
    class_ids: torch.Tensor,
    class_count: int = 3,
) -> dict[str, float]:
    predictions = torch.argmax(class_logits, dim=1)
    truth = class_ids.long()
    confusion = torch.zeros((class_count, class_count), dtype=torch.float64)
    for actual, predicted in zip(truth.detach().cpu(), predictions.detach().cpu()):
        confusion[int(actual), int(predicted)] += 1.0
    support = confusion.sum(dim=1)
    predicted_count = confusion.sum(dim=0)
    diagonal = torch.diag(confusion)
    recall = torch.where(support > 0, diagonal / support, torch.zeros_like(support))
    precision = torch.where(
        predicted_count > 0,
        diagonal / predicted_count,
        torch.zeros_like(predicted_count),
    )
    f1 = torch.where(
        precision + recall > 0,
        2.0 * precision * recall / (precision + recall),
        torch.zeros_like(precision),
    )
    return {
        "accuracy": float(torch.mean((predictions == truth).float()).cpu()),
        "balanced_accuracy": float(torch.mean(recall).cpu()),
        "macro_f1": float(torch.mean(f1).cpu()),
    }
