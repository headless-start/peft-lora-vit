import argparse
import gc
import json
import os

import torch
from hydra import compose, initialize

from src.plot import save_ablation
from train import run_training


def main():
    parser = argparse.ArgumentParser(description="LoRA rank ablation")
    parser.add_argument("--ranks", default="4,8,16,32",
                        help="comma-separated lora ranks to sweep")
    parser.add_argument("overrides", nargs="*",
                        help="extra hydra overrides, e.g. data=flowers")
    args = parser.parse_args()
    args.ranks = [int(r) for r in args.ranks.split(",")]

    rows = []
    with initialize(version_base=None, config_path="configs"):
        for r in args.ranks:
            overrides = [f"model.lora.r={r}", f"wandb.name=lora-r{r}", *args.overrides]
            cfg = compose(config_name="config", overrides=overrides)
            print(f"\n=== lora r={r} ===")
            res = run_training(cfg, ckpt_path=f"outputs/best_r{r}.pt")
            rows.append({"r": r, "top1_acc": round(res["best_acc"], 4),
                         "trainable_params": res["trainable"],
                         "trainable_pct": round(100 * res["trainable"] / res["total"], 3)})
            # write after every run so a crash doesn't lose the finished ones
            os.makedirs("results", exist_ok=True)
            with open("results/ablation.json", "w") as f:
                json.dump(rows, f, indent=2)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    save_ablation(rows, "results/ablation.png")

    print("\n| rank | top-1 acc | trainable params | % of total |")
    print("|------|-----------|------------------|------------|")
    for row in rows:
        print(f"| {row['r']} | {row['top1_acc']:.1%} | "
              f"{row['trainable_params'] / 1e3:.0f}K | {row['trainable_pct']:.2f}% |")


if __name__ == "__main__":
    main()
