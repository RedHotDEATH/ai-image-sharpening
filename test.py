import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from models.blindspot_unet import BlindSpotUNet
import torchvision.transforms as transforms
import sys
import os

# Set device to CPU
device = torch.device("cpu")

# Load model
model = BlindSpotUNet().to(device)
model.load_state_dict(torch.load("checkpoints/model_epoch10.pth", map_location=device))
model.eval()

# Check input path from command line
if len(sys.argv) != 2:
    print("Usage: python test.py <path_to_noisy_image>")
    sys.exit(1)

img_path = sys.argv[1]
if not os.path.exists(img_path):
    print(f"File does not exist: {img_path}")
    sys.exit(1)

# Image transform
transform = transforms.Compose([transforms.ToTensor()])
img = Image.open(img_path).convert('L')
input_tensor = transform(img).unsqueeze(0).to(device)

# Denoise
with torch.no_grad():
    output = model(input_tensor)
    output = torch.clamp(output, 0., 1.)

# Convert to numpy
output_np = output.squeeze().cpu().numpy()
input_np = input_tensor.squeeze().cpu().numpy()

# Plot
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Noisy Input")
plt.imshow(input_np, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Denoised Output")
plt.imshow(output_np, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

#output
from torchvision.utils import save_image
import os

# Create output folder if not exists
os.makedirs("outputs", exist_ok=True)

# Save denoised image
output_path = os.path.join("outputs", os.path.basename(image_path))
save_image(denoised.clamp(0.0, 1.0), output_path)

print(f"Denoised image saved at: {output_path}")
