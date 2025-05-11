# evaluate.py
import os
import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from models.blindspot_unet import BlindSpotUNet
from utils import NoisyImageDataset
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import numpy as np

# Configuration
image_dir = 'data/BSD68'
checkpoint_path = 'checkpoints/model_epoch20.pth'
output_dir = 'results'
os.makedirs(output_dir, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataset
test_dataset = NoisyImageDataset(image_dir=image_dir)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Load model
model = BlindSpotUNet().to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# Evaluate
psnr_scores, ssim_scores = [], []

print("🔍 Running evaluation...")
with torch.no_grad():
    for i, (masked_input, noisy_input, clean) in enumerate(test_loader):
        masked_input = masked_input.to(device)
        clean = clean.squeeze().numpy()
        noisy_input = noisy_input.to(device)

        output = model(masked_input).cpu().squeeze().numpy()
        noisy = noisy_input.cpu().squeeze().numpy()

        # Clamp output to [0, 1]
        output = np.clip(output, 0, 1)

        # Save side-by-side comparison
        combined = np.stack([noisy, output, clean], axis=0)  # [noisy, denoised, clean]
        save_image(torch.tensor(combined), f'{output_dir}/sample_{i}.png', nrow=3)

        # Calculate PSNR/SSIM
        psnr_scores.append(psnr(clean, output, data_range=1.0))
        ssim_scores.append(ssim(clean, output, data_range=1.0))

print(f"\n📊 Average PSNR: {np.mean(psnr_scores):.2f}")
print(f"📊 Average SSIM: {np.mean(ssim_scores):.4f}")
print("✅ Results saved in the 'results/' folder.")
