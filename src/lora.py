import math

import torch
import torch.nn as nn


class LoRAQKV(nn.Module):
    """Fused qkv projection with low-rank adapters on the q and v blocks.

    timm packs q, k, v into a single Linear (dim -> 3*dim). We keep that
    projection frozen and learn a small delta on the q and v slices only,
    following the original LoRA setup (k is left untouched).
    """

    def __init__(self, qkv: nn.Linear, r: int, alpha: int, dropout: float = 0.0):
        super().__init__()
        self.qkv = qkv
        self.dim = qkv.in_features
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)

        self.lora_a_q = nn.Linear(self.dim, r, bias=False)
        self.lora_b_q = nn.Linear(r, self.dim, bias=False)
        self.lora_a_v = nn.Linear(self.dim, r, bias=False)
        self.lora_b_v = nn.Linear(r, self.dim, bias=False)

        for a in (self.lora_a_q, self.lora_a_v):
            nn.init.kaiming_uniform_(a.weight, a=math.sqrt(5))
        for b in (self.lora_b_q, self.lora_b_v):
            nn.init.zeros_(b.weight)  # start as a no-op so training begins from the pretrained model

    def forward(self, x):
        qkv = self.qkv(x)
        xd = self.dropout(x)
        dq = self.lora_b_q(self.lora_a_q(xd)) * self.scaling
        dv = self.lora_b_v(self.lora_a_v(xd)) * self.scaling
        dk = torch.zeros_like(dq)
        return qkv + torch.cat([dq, dk, dv], dim=-1)


def inject_lora(model, r: int, alpha: int, dropout: float = 0.0):
    """Wrap every fused qkv projection in the model with a LoRAQKV."""
    targets = [m for m in model.modules() if isinstance(getattr(m, "qkv", None), nn.Linear)]
    for m in targets:
        m.qkv = LoRAQKV(m.qkv, r, alpha, dropout)
    return len(targets)
