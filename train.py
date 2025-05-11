# train.py
import sys
sys.path.append(".")
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from models.blindspot_unet import BlindSpotUNet
from utils import NoisyImageDataset
import os
from tqdm import tqdm

# Config
image_dir = 'data/BSD68'
epochs = 20
batch_size = 8
lr = 1e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
save_path = 'checkpoints'
os.makedirs(save_path, exist_ok=True)

# DataLoader
dataset = NoisyImageDataset(image_dir=image_dir)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model, loss, optimizer
model = BlindSpotUNet().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Training loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    loop = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")

    for i, (masked_input, noisy, _) in enumerate(loop):
        masked_input = masked_input.to(device)
        target = noisy.to(device)

        optimizer.zero_grad()
        output = model(masked_input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    # Save a checkpoint every 5 epochs
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), f"{save_path}/model_epoch{epoch+1}.pth")
        print(f"Checkpoint saved at epoch {epoch+1}")

print("✅ Training complete.")
