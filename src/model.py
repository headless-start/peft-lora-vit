import timm

from .lora import LoRAQKV, inject_lora


def build_model(cfg, num_classes):
    model = timm.create_model(
        cfg.model.backbone,
        pretrained=cfg.model.pretrained,
        num_classes=num_classes,
    )
    n_lora = inject_lora(model, cfg.model.lora.r, cfg.model.lora.alpha, cfg.model.lora.dropout)
    freeze_backbone(model)
    return model, n_lora


def freeze_backbone(model):
    """Freeze everything, then re-enable the LoRA adapters and the classifier head."""
    for p in model.parameters():
        p.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, LoRAQKV):
            for adapter in (m.lora_a_q, m.lora_b_q, m.lora_a_v, m.lora_b_v):
                for p in adapter.parameters():
                    p.requires_grad_(True)
    for p in model.get_classifier().parameters():
        p.requires_grad_(True)
