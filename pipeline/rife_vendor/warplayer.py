# Derived from Practical-RIFE (MIT): model/warplayer.py
# Grid sampling uses the same tensor device as the activations.
from __future__ import annotations

import torch
import torch.nn.functional as F

_backwarp_grid: dict[tuple[str, str], torch.Tensor] = {}


def warp(tenInput: torch.Tensor, tenFlow: torch.Tensor) -> torch.Tensor:
    dev = tenInput.device
    k = (str(dev), str(tenFlow.size()))
    if k not in _backwarp_grid:
        h, w = tenFlow.shape[2], tenFlow.shape[3]
        ten_horizontal = torch.linspace(-1.0, 1.0, w, device=dev).view(
            1, 1, 1, w
        ).expand(tenFlow.shape[0], -1, h, -1)
        ten_vertical = torch.linspace(-1.0, 1.0, h, device=dev).view(
            1, 1, h, 1
        ).expand(tenFlow.shape[0], -1, -1, w)
        _backwarp_grid[k] = torch.cat([ten_horizontal, ten_vertical], 1)

    scaled = torch.cat(
        [
            tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
            tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0),
        ],
        1,
    )
    g = (_backwarp_grid[k] + scaled).permute(0, 2, 3, 1)
    return F.grid_sample(
        input=tenInput,
        grid=g,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
