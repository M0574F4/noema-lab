from __future__ import annotations

import torch


def image_mse(reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((reconstruction.float() - target.float()) ** 2)


def build_loss(name: str):
    normalized = str(name or "image.mse").strip().lower()
    if normalized != "image.mse":
        raise ValueError("Unsupported DeepJSCC loss: %s" % name)
    return image_mse
