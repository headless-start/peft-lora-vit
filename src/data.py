from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FakeData, Flowers102, OxfordIIITPet

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Oxford-IIIT Pets breed names, in torchvision's label order
CLASSES = [
    "Abyssinian", "American Bulldog", "American Pit Bull Terrier", "Basset Hound",
    "Beagle", "Bengal", "Birman", "Bombay", "Boxer", "British Shorthair",
    "Chihuahua", "Egyptian Mau", "English Cocker Spaniel", "English Setter",
    "German Shorthaired", "Great Pyrenees", "Havanese", "Japanese Chin",
    "Keeshond", "Leonberger", "Maine Coon", "Miniature Pinscher", "Newfoundland",
    "Persian", "Pomeranian", "Pug", "Ragdoll", "Russian Blue", "Saint Bernard",
    "Samoyed", "Scottish Terrier", "Shiba Inu", "Siamese", "Sphynx",
    "Staffordshire Bull Terrier", "Wheaten Terrier", "Yorkshire Terrier",
]
NUM_CLASSES = len(CLASSES)


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


def build_datasets(cfg, train_tf, eval_tf):
    """Return (train_ds, val_ds, num_classes) for the configured dataset."""
    size = cfg.data.image_size
    if cfg.data.fake:
        # random images, no download — used for the quick smoke run
        return (FakeData(256, (3, size, size), NUM_CLASSES, train_tf),
                FakeData(128, (3, size, size), NUM_CLASSES, eval_tf), NUM_CLASSES)
    if cfg.data.name == "oxford_pets":
        return (OxfordIIITPet(cfg.data.root, split="trainval", download=True, transform=train_tf),
                OxfordIIITPet(cfg.data.root, split="test", download=True, transform=eval_tf), 37)
    if cfg.data.name == "flowers102":
        return (Flowers102(cfg.data.root, split="train", download=True, transform=train_tf),
                Flowers102(cfg.data.root, split="val", download=True, transform=eval_tf), 102)
    raise ValueError(f"unknown dataset: {cfg.data.name}")


def build_loaders(cfg):
    """Build the train/val dataloaders for the configured dataset."""
    train_tf, eval_tf = build_transforms(cfg.data.image_size)
    train_ds, val_ds, num_classes = build_datasets(cfg, train_tf, eval_tf)

    train_loader = DataLoader(train_ds, batch_size=cfg.data.batch_size, shuffle=True,
                              num_workers=cfg.data.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.data.batch_size, shuffle=False,
                            num_workers=cfg.data.num_workers, pin_memory=True)
    return train_loader, val_loader, num_classes
