import timm

from .lora import LoRAQKV, inject_lora


def build_model(cfg, num_classes):
    """Build the ViT, attach LoRA to q/v, swap the head, and freeze the rest."""
    model = timm.create_model(
        cfg.model.backbone,
        pretrained=cfg.model.pretrained,
        num_classes=num_classes,
        drop_path_rate=cfg.model.get("drop_path_rate", 0.0),
    )
    r = cfg.model.lora.r
    alpha = cfg.model.lora.alpha_factor * r
    n_lora = inject_lora(model, r, alpha, cfg.model.lora.dropout)
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


def trainable_state_dict(model):
    """State dict restricted to trainable tensors (LoRA adapters + head) — a few MB, not 344."""
    keep = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v for k, v in model.state_dict().items() if k in keep}
