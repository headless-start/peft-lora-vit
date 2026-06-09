import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def save_curve(history, path):
    """Plot train/val loss and accuracy across epochs and save the figure to path."""
    epochs = [h["epoch"] for h in history]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))

    ax_loss.plot(epochs, [h["train_loss"] for h in history], label="train")
    ax_loss.plot(epochs, [h["val_loss"] for h in history], label="val")
    ax_loss.set(xlabel="epoch", ylabel="loss", title="loss")
    ax_loss.legend()

    ax_acc.plot(epochs, [h["train_acc"] for h in history], label="train")
    ax_acc.plot(epochs, [h["val_acc"] for h in history], label="val")
    ax_acc.set(xlabel="epoch", ylabel="accuracy", title="accuracy")
    ax_acc.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_ablation(rows, path):
    """Plot accuracy and trainable-parameter count against LoRA rank."""
    ranks = [r["r"] for r in rows]
    fig, (ax_acc, ax_par) = plt.subplots(1, 2, figsize=(10, 4))

    ax_acc.plot(ranks, [r["top1_acc"] for r in rows], marker="o")
    ax_acc.set(xlabel="lora rank", ylabel="top-1 accuracy", title="accuracy vs rank",
               xscale="log", xticks=ranks, xticklabels=[str(r) for r in ranks])

    ax_par.plot(ranks, [r["trainable_params"] / 1e3 for r in rows], marker="o", color="tab:orange")
    ax_par.set(xlabel="lora rank", ylabel="trainable params (K)", title="cost vs rank",
               xscale="log", xticks=ranks, xticklabels=[str(r) for r in ranks])

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
