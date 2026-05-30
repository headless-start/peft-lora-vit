# peft-vit-demo

Parameter-efficient fine-tuning (LoRA) of a Vision Transformer for image
classification. A ViT-B/16 pretrained on ImageNet is adapted to the
Oxford-IIIT Pets dataset by training small LoRA adapters on the attention
projections while keeping the backbone frozen — under ~5% of parameters
trainable.

Work in progress.
