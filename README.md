# Parameter-Efficient Fine-Tuning of a Vision Transformer (LoRA)

## 📌 Project Overview
This project demonstrates **parameter-efficient fine-tuning (LoRA)** of a **Vision Transformer (ViT-B/16)** for image classification. An ImageNet-pretrained backbone is adapted to a new dataset by training small low-rank adapters on the attention **query/value** projections, while the backbone itself stays frozen. This reaches near full fine-tuning accuracy while updating only a tiny fraction of the weights.

**Dataset**: Oxford-IIIT Pets (37 cat and dog breeds).  
**Backbone**: `vit_base_patch16_224`, pretrained on ImageNet via `timm`.  
**Goal**: Strong top-1 accuracy while training well under 5% of the model's parameters.

---

## 🚀 Key Features
1. **Hand-Written LoRA**:
   - Low-rank adapters injected into the fused q/v attention projections (`B · A · x · α/r`, with `B` zero-initialised so training starts from the pretrained model).
   - Only the adapters and the classifier head are trainable; the backbone is fully frozen.
2. **Configurable with Hydra**:
   - Data, model, and training settings live in `configs/` and can be overridden straight from the command line.
3. **Experiment Tracking**:
   - Metrics are logged to Weights & Biases in **offline** mode by default, so it runs without an account.
4. **CPU Smoke Test**:
   - A tiny end-to-end run on random data with no downloads, for quick sanity checks.

---

## 🔍 Findings
- **Top-1 Accuracy**: **94.4%** on the validation set.
- **Trainable Parameters**: 323K out of 86.1M — just **0.38%** of the model.
- **Setup**: LoRA rank 8 on q/v, 10 epochs, AdamW with a cosine schedule, mixed precision.
- **Takeaway**: LoRA recovers near full fine-tuning accuracy on Pets while training under half a percent of the weights.

![Training Curves](results/training_curve.png)

---

## ⚙️ How to Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# full run (downloads Oxford-IIIT Pets on first use)
python train.py

# override anything from the command line
python train.py train.epochs=20 data.batch_size=64 model.lora.r=16
```

Quick smoke test (CPU, no downloads):

```bash
python train.py +experiment=smoke
```

Sync tracking to the cloud instead of running offline:

```bash
wandb login
python train.py wandb.mode=online
```

Results are written to `results/` (training curve + `metrics.json`); the best checkpoint goes to `outputs/`.

Classify your own images with the trained checkpoint:

```bash
python predict.py path/to/cat.jpg path/to/dog.jpg
# path/to/cat.jpg: Abyssinian (100.0%), Russian Blue (0.0%), Shiba Inu (0.0%)
```

---

## 🛠 System Requirements
### Dependencies
- Python 3.10+
- Libraries: `torch`, `torchvision`, `timm`, `hydra-core`, `wandb`, `matplotlib`
- Hardware: CUDA GPU recommended (a CPU smoke run is supported)

---

## 📄 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
