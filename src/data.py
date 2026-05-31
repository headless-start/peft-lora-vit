from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FakeData, OxfordIIITPet

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NUM_CLASSES = 37


def build_transforms(image_size):
    """ImageNet-normalised transforms: light augmentation for train, center crop for eval."""
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(round(image_size * 256 / 224)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


def build_loaders(cfg):
    """Build the train/val dataloaders — real Pets, or random data for the smoke test."""
    train_tf, eval_tf = build_transforms(cfg.data.image_size)
    size = cfg.data.image_size

    if cfg.data.fake:
        # random images, no download — used for the quick smoke run
        train_ds = FakeData(256, (3, size, size), NUM_CLASSES, train_tf)
        val_ds = FakeData(128, (3, size, size), NUM_CLASSES, eval_tf)
    else:
        train_ds = OxfordIIITPet(cfg.data.root, split="trainval", download=True, transform=train_tf)
        val_ds = OxfordIIITPet(cfg.data.root, split="test", download=True, transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=cfg.data.batch_size, shuffle=True,
                              num_workers=cfg.data.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.data.batch_size, shuffle=False,
                            num_workers=cfg.data.num_workers, pin_memory=True)
    return train_loader, val_loader, NUM_CLASSES
