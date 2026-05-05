import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class LEVIRDataset(Dataset):
    def __init__(self, split, config, augment=False):
        """
        split   : 'train_patches' or 'test_patches'
        config  : dict loaded from config.yaml
        augment : whether to apply augmentations
        """
        self.a_dir     = os.path.join(config["paths"]["data_processed"], split, "A")
        self.b_dir     = os.path.join(config["paths"]["data_processed"], split, "B")
        self.label_dir = os.path.join(config["paths"]["data_processed"], split, "label")
        self.augment   = augment

        self.files = sorted(os.listdir(self.a_dir))

        norm = config["normalization"]
        self.transform_a = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=norm["mean_t1"], std=norm["std_t1"])
        ])
        self.transform_b = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=norm["mean_t2"], std=norm["std_t2"])
        ])

        self.aug_config = config.get("augmentation", {})

    def __len__(self):
        return len(self.files)

    def _apply_augmentation(self, img_a, img_b, mask):
        """apply identical spatial augmentations to both images and mask."""
        aug = self.aug_config

        # random horizontal flip
        if aug.get("horizontal_flip") and torch.rand(1) > 0.5:
            img_a = transforms.functional.hflip(img_a)
            img_b = transforms.functional.hflip(img_b)
            mask  = transforms.functional.hflip(mask)

        # random vertical flip
        if aug.get("vertical_flip") and torch.rand(1) > 0.5:
            img_a = transforms.functional.vflip(img_a)
            img_b = transforms.functional.vflip(img_b)
            mask  = transforms.functional.vflip(mask)

        # color jitter applied independently to each time point
        if aug.get("color_jitter"):
            jitter = transforms.ColorJitter(
                brightness=aug.get("color_jitter_brightness", 0.2),
                contrast=aug.get("color_jitter_contrast", 0.2)
            )
            img_a = jitter(img_a)
            img_b = jitter(img_b)

        return img_a, img_b, mask

    def __getitem__(self, idx):
        fname = self.files[idx]

        img_a = Image.open(os.path.join(self.a_dir,     fname)).convert("RGB")
        img_b = Image.open(os.path.join(self.b_dir,     fname)).convert("RGB")
        mask  = Image.open(os.path.join(self.label_dir, fname)).convert("L")

        if self.augment:
            img_a, img_b, mask = self._apply_augmentation(img_a, img_b, mask)

        img_a = self.transform_a(img_a)
        img_b = self.transform_b(img_b)

        # mask to binary tensor (0 or 1)
        mask = torch.tensor(np.array(mask), dtype=torch.float32)
        mask = (mask > 0).float().unsqueeze(0)

        return img_a, img_b, mask

class SyntheticDataset(Dataset):
    def __init__(self, config):
        self.a_dir     = os.path.join(config["paths"]["synthetic_pairs"], "A")
        self.b_dir     = os.path.join(config["paths"]["synthetic_pairs"], "B")
        self.label_dir = os.path.join(config["paths"]["synthetic_pairs"], "label")
        self.files     = sorted(os.listdir(self.a_dir))

        norm = config["normalization"]
        self.transform_a = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=norm["mean_t1"], std=norm["std_t1"])
        ])
        self.transform_b = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=norm["mean_t2"], std=norm["std_t2"])
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img_a = Image.open(os.path.join(self.a_dir,     fname)).convert("RGB")
        img_b = Image.open(os.path.join(self.b_dir,     fname)).convert("RGB")
        mask  = Image.open(os.path.join(self.label_dir, fname)).convert("L")
        img_a = self.transform_a(img_a)
        img_b = self.transform_b(img_b)
        mask  = torch.tensor(np.array(mask), dtype=torch.float32)
        mask  = (mask > 0).float().unsqueeze(0)
        return img_a, img_b, mask
