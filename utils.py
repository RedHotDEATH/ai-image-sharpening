# utils.py
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
import random
import numpy as np
from PIL import Image
import os

class NoisyImageDataset(Dataset):
    def __init__(self, image_dir, patch_size=64, noise_std=25, transform=None):
        self.image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)]
        self.patch_size = patch_size
        self.noise_std = noise_std
        self.transform = transform or transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("L")
        image = image.resize((256, 256))
        clean = self.transform(image)
        noise = torch.randn_like(clean) * (self.noise_std / 255.0)
        noisy = clean + noise
        noisy = torch.clamp(noisy, 0., 1.)

        # Random pixel masking for self-supervision
        mask = torch.ones_like(noisy)
        num_mask = int(mask.numel() * 0.05)  # 5% masking
        mask.view(-1)[torch.randperm(mask.numel())[:num_mask]] = 0
        masked_input = noisy * mask

        return masked_input, noisy, clean
