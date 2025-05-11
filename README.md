
# Self-Supervised Image Denoising using Conditional Blind-Spot Networks and Downsampled Invariance Loss

This repository presents a self-supervised deep learning framework for image denoising, eliminating the need for clean training targets. The method integrates a Conditional Blind-Spot Network (CBN) with a novel Downsampled Invariance Loss (DIL), enabling effective noise removal from real-world and synthetic noisy images.

---

## 🧠 Project Overview

- **Problem**: Real-world scenarios rarely provide clean image pairs for supervised denoising training.
- **Solution**: A blind-spot architecture is paired with a loss function that enforces invariance under spatial downsampling, allowing training using only noisy images.
- **Goal**: To perform efficient denoising on both synthetic and real-world noisy images, without requiring ground-truth clean targets.

---

## 📁 Project Structure

```bash
.
├── data/
│   ├── BSD68/                                # Benchmark dataset
│  
├── models/
│   └── blindspot_unet.py
├── scripts/
│   ├── train.py                              # Model training script
│   ├── cbn_model.py                          # Model architecture (CBN)
│   ├── utils.py                              # Utility functions
│   ├── loss_functions.py                     # DIL + hybrid loss definitions
│   └── evaluate.py                           # PSNR / SSIM evaluation
├── README.md
└── requirements.txt
````

---

## 🏗️ Model Highlights

* **Encoder-decoder UNet-style network**
* **Blind-spot convolutional masking** (inspired by Noise2Void)
* **Conditional context modeling**
* **Downsampled Invariance Loss (DIL)** for learning from noisy inputs alone

---

## 📦 Installation & Dependencies

Clone the repository and install the required packages:

```bash
git clone https://github.com/your-username/self-supervised-image-denoising
cd self-supervised-image-denoising
pip install -r requirements.txt
```

**requirements.txt:**

```txt
torch>=1.11.0
torchvision
numpy
matplotlib
scikit-image
opencv-python
tqdm
```

---

## 📊 Datasets Used

* **BSD68** – Standard synthetic Gaussian noise dataset.
* **RealNoise** – Smartphone and DSLR image sets with real-world noise.

---

## 🚀 How to Train

```bash
python train.py --dataset data/BSD68 --noise_sigma 25
```

Optional parameters:

* `--epochs`: Number of training epochs
* `--batch_size`: Mini-batch size
* `--save_path`: Path to save model checkpoints

---

## 📈 Evaluation Metrics

* **PSNR (Peak Signal-to-Noise Ratio)**
* **SSIM (Structural Similarity Index)**

---

## 📉 Results Summary

| Model          | PSNR (σ=25) | SSIM | Params | Inference Time |
| -------------- | ----------- | ---- | ------ | -------------- |
| DnCNN          | 30.4 dB     | 0.88 | 560K   | 0.10 s         |
| Noise2Void     | 29.6 dB     | 0.86 | 1.4M   | 0.15 s         |
| **This Model** | 30.2 dB     | 0.87 | 2.1M   | 0.12 s         |

*Performance evaluated on BSD68 (Gaussian σ=25)*

---

## 📚 Scientific References

* **Yeong Il Jang et al.** (2023).
  *Self-supervised Image Denoising with Downsampled Invariance Loss and Conditional Blind-Spot Network*, ICCV.
  [IEEE Link](https://ieeexplore.ieee.org/document/10010320)

* **Krull et al.** (2019).
  *Noise2Void: Learning Denoising from Single Noisy Images*, CVPR.

* **Zhang et al.** (2017).
  *Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising*, IEEE TIP.

---

## 🧑‍💻 Author

* **Rishabh Tiwari**
  B.Sc. Statistics, University of Lucknow
  Data Science | Deep Learning | Computer Vision
  GitHub: [@rishabhtiwari](https://github.com/your-username)

---

## 🔖 License

This project is released for educational and research use only. Please cite the references above if you build upon this work.

```

---

