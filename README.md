# Parameter-Efficient Fine-Tuning of a Vision Transformer (LoRA)

## 📌 Project Overview
This project demonstrates **parameter-efficient fine-tuning** of a **Vision Transformer (ViT-B/16)** for image classification using **LoRA** — strictly low-rank updates, no other PEFT method. An ImageNet-pretrained backbone is adapted to a new dataset by learning small low-rank deltas on the attention **query/value** projections, while the backbone itself stays frozen. This reaches near full fine-tuning accuracy while updating only a tiny fraction of the weights.

**Datasets**: Oxford-IIIT Pets (37 cat and dog breeds) and Oxford Flowers-102.  
**Backbone**: `vit_base_patch16_224`, pretrained on ImageNet via `timm`.  
**Goal**: Strong top-1 accuracy while training well under 5% of the model's parameters.

I built this as hands-on preparation for the PEFT/LoRA side of my thesis; everything here is a standalone prototype on public data and public weights.

![Dataset Samples](results/pet_samples.png)

---

## 🚀 Key Features
1. **Hand-Written LoRA**:
   - Low-rank matrices injected into the fused q/v attention projections (`B · A · x · α/r`, with `α = 2r` and `B` zero-initialised so training starts exactly from the pretrained model).
   - Placement follows the original LoRA paper (Hu et al., 2022), whose placement study found adapting **q and v** the best use of a fixed parameter budget — k contributes least.
   - Only the LoRA matrices and the classifier head are trainable; the backbone is fully frozen.
2. **Rank Ablation**:
   - One command sweeps the LoRA rank over {4, 8, 16, 32} and plots accuracy and cost against rank.
3. **Tiny Checkpoints**:
   - Only the LoRA weights and head are saved — a few MB instead of the full 344 MB backbone. Inference rebuilds the model from public pretrained weights and loads the LoRA weights on top.
4. **Solid Training Recipe**:
   - AdamW with a 2-epoch linear warmup into cosine decay, drop-path 0.1, mixed precision.
5. **Configurable with Hydra**:
   - Data, model, and training settings live in `configs/` and can be overridden straight from the command line.
6. **Experiment Tracking**:
   - Metrics are logged to Weights & Biases in **offline** mode by default, so it runs without an account.

---

## 🔍 Findings
- **Top-1 Accuracy**: **95.2%** on the Pets validation set (weighted average recall, WAR).
- **Trainable Parameters**: 323K out of 86.1M — just **0.38%** of the model.
- **Setup**: LoRA rank 8 on q/v, 25 epochs, AdamW with warmup + cosine decay, mixed precision.
- **Takeaway**: LoRA recovers near full fine-tuning accuracy while training under half a percent of the weights.

![Training Curves](results/training_curve.png)

### Rank Ablation
Sweeping the LoRA rank shows accuracy saturates almost immediately — rank 4 is already within 0.4 points of the best, and pushing to rank 32 buys nothing for 7× the parameters:

| rank | top-1 acc (WAR) | trainable params | % of total |
|------|-----------------|------------------|------------|
| 4    | 94.9%           | 176K             | 0.20%      |
| 8    | **95.2%**       | 323K             | 0.38%      |
| 16   | 94.7%           | 618K             | 0.72%      |
| 32   | 94.8%           | 1.21M            | 1.39%      |

![Rank Ablation](results/ablation.png)

---

## ⚙️ How to Run
Works on Linux, macOS and Windows.

```bash
git clone https://github.com/headless-start/peft-lora-vit.git
cd peft-lora-vit

python -m venv .venv
source .venv/bin/activate          # linux / macos
# .venv\Scripts\activate           # windows

pip install -r requirements.txt
```

For GPU training install the CUDA build of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/) first; the plain `pip install` gives you a CPU build on some platforms.

```bash
# full run on Oxford-IIIT Pets (downloads on first use)
python train.py

# or train on Flowers-102 instead
python train.py data=flowers

# override anything from the command line
python train.py train.epochs=40 data.batch_size=32 model.lora.r=16
```

Sweep the LoRA rank (writes `results/ablation.json` and `results/ablation.png`):

```bash
python ablate.py                    # ranks 4, 8, 16, 32
python ablate.py --ranks 4,8 data=flowers
```

Classify your own images with a trained checkpoint:

```bash
python predict.py path/to/cat.jpg path/to/dog.jpg
# path/to/cat.jpg: Abyssinian (100.0%), Russian Blue (0.0%), Shiba Inu (0.0%)
```

Quick smoke test (CPU, small backbone, no downloads):

```bash
python train.py +experiment=smoke
```

Runs are logged to Weights & Biases offline by default; to sync to the cloud:

```bash
wandb login
python train.py wandb.mode=online
```

Training curves and `metrics.json` are written to `results/`; checkpoints go to `outputs/`.

---

## 🛠 System Requirements
### Dependencies
- Python 3.10+
- Libraries: `torch`, `torchvision`, `timm`, `hydra-core`, `wandb`, `matplotlib`
- Hardware: CUDA GPU recommended (a CPU smoke run is supported)

---

## 📄 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
