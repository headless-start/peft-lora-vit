import json
import os

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

from src.data import build_loaders
from src.engine import run_epoch
from src.model import build_model
from src.plot import save_curve
from src.utils import count_parameters, set_seed


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    use_cuda = cfg.device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    use_amp = cfg.train.amp and use_cuda

    train_loader, val_loader, num_classes = build_loaders(cfg)
    model, n_lora = build_model(cfg, num_classes)
    model.to(device)

    trainable, total = count_parameters(model)
    print(f"backbone={cfg.model.backbone}  lora_layers={n_lora}  "
          f"trainable={trainable:,}/{total:,} ({100 * trainable / total:.2f}%)")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    wandb.init(project=cfg.wandb.project, mode=cfg.wandb.mode,
               config=OmegaConf.to_container(cfg, resolve=True))

    history, best_acc = [], 0.0
    for epoch in range(cfg.train.epochs):
        tr_loss, tr_acc = run_epoch(model, train_loader, device, optimizer, scaler,
                                    use_amp, cfg.train.limit_train_batches)
        va_loss, va_acc = run_epoch(model, val_loader, device, use_amp=use_amp,
                                    limit_batches=cfg.train.limit_val_batches)
        scheduler.step()

        wandb.log({"epoch": epoch, "lr": scheduler.get_last_lr()[0],
                   "train/loss": tr_loss, "train/acc": tr_acc,
                   "val/loss": va_loss, "val/acc": va_acc})
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va_loss, "val_acc": va_acc})
        print(f"epoch {epoch:3d}  train {tr_loss:.3f}/{tr_acc:.3f}  val {va_loss:.3f}/{va_acc:.3f}")

        if va_acc > best_acc:
            best_acc = va_acc
            os.makedirs("outputs", exist_ok=True)
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_acc": best_acc},
                       "outputs/best.pt")

    os.makedirs("results", exist_ok=True)
    save_curve(history, "results/training_curve.png")
    metrics = {
        "backbone": cfg.model.backbone,
        "dataset": cfg.data.name,
        "epochs": cfg.train.epochs,
        "top1_acc": round(best_acc, 4),
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(100 * trainable / total, 3),
        "lora": OmegaConf.to_container(cfg.model.lora, resolve=True),
    }
    with open("results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    wandb.finish()
    print(f"best val acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()
