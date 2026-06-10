import math

import torch
import torch.nn as nn


class LoRAQKV(nn.Module):
    """Fused qkv projection with LoRA on a chosen subset of q, k, v.

    timm packs q, k, v into a single Linear (dim -> 3*dim). We keep that
    projection frozen and learn a low-rank delta on the chosen slices.
    The default ("q", "v") follows the placement study in the original
    LoRA paper, which found q+v the best use of a fixed budget.
    """

    def __init__(self, qkv: nn.Linear, r: int, alpha: int, dropout: float = 0.0,
                 targets=("q", "v")):
        super().__init__()
        self.qkv = qkv
        self.dim = qkv.in_features
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)
        self.targets = tuple(targets)

        for t in self.targets:
            a = nn.Linear(self.dim, r, bias=False)
            b = nn.Linear(r, self.dim, bias=False)
            nn.init.kaiming_uniform_(a.weight, a=math.sqrt(5))
            nn.init.zeros_(b.weight)  # start as a no-op so training begins from the pretrained model
            setattr(self, f"lora_a_{t}", a)
            setattr(self, f"lora_b_{t}", b)

    def forward(self, x):
        """Add the low-rank deltas for the chosen slices on top of the frozen qkv output."""
        qkv = self.qkv(x)
        xd = self.dropout(x)
        deltas = {}
        for t in self.targets:
            a = getattr(self, f"lora_a_{t}")
            b = getattr(self, f"lora_b_{t}")
            deltas[t] = b(a(xd)) * self.scaling
        zero = torch.zeros_like(next(iter(deltas.values())))
        return qkv + torch.cat([deltas.get(t, zero) for t in ("q", "k", "v")], dim=-1)


def inject_lora(model, r: int, alpha: int, dropout: float = 0.0, targets=("q", "v")):
    """Wrap every fused qkv projection in the model with a LoRAQKV."""
    found = [m for m in model.modules() if isinstance(getattr(m, "qkv", None), nn.Linear)]
    for m in found:
        m.qkv = LoRAQKV(m.qkv, r, alpha, dropout, targets)
    return len(found)
