import argparse

import torch
from omegaconf import OmegaConf
from PIL import Image

from src.data import CLASSES, NUM_CLASSES, build_transforms
from src.model import build_model


def load_model(ckpt_path, backbone, r, alpha_factor, device):
    """Rebuild the model on pretrained weights, then load the trained LoRA weights and head."""
    cfg = OmegaConf.create({
        "model": {"backbone": backbone, "pretrained": True, "drop_path_rate": 0.0,
                  "lora": {"r": r, "alpha_factor": alpha_factor, "dropout": 0.0}},
    })
    model, _ = build_model(cfg, NUM_CLASSES)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    assert not unexpected, f"checkpoint keys not in model: {unexpected[:3]}"
    model.to(device).eval()
    return model


@torch.no_grad()
def predict(model, image_path, device, topk=3):
    """Classify one image and return the top-k (breed, probability) pairs."""
    _, eval_tf = build_transforms(224)
    img = Image.open(image_path).convert("RGB")
    x = eval_tf(img).unsqueeze(0).to(device)
    probs = model(x).softmax(-1).squeeze(0)
    top = probs.topk(topk)
    return [(CLASSES[i], p.item()) for p, i in zip(top.values, top.indices)]


def main():
    parser = argparse.ArgumentParser(description="Classify pet images with the fine-tuned ViT")
    parser.add_argument("images", nargs="+", help="image file(s) to classify")
    parser.add_argument("--ckpt", default="outputs/best.pt")
    parser.add_argument("--backbone", default="vit_base_patch16_224")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--alpha-factor", type=int, default=2)
    parser.add_argument("--topk", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, args.backbone, args.lora_r, args.alpha_factor, device)

    for path in args.images:
        preds = predict(model, path, device, args.topk)
        labels = ", ".join(f"{name} ({p:.1%})" for name, p in preds)
        print(f"{path}: {labels}")


if __name__ == "__main__":
    main()
