
---

## 🧠 Project Overview

- **Problem**: Traditional denoising models require clean images which are often unavailable in practice.
- **Solution**: We use blind-spot networks and a novel DIL strategy to train only on noisy data.
- **Goal**: Achieve competitive denoising results without relying on ground-truth clean targets.

---

## 🏗️ Model Architecture

- **Encoder-decoder backbone (UNet-based)**
- **Blind-spot convolutional masking**
- **Conditional context blocks**
- **Downsampled Invariance Loss function**

No architecture diagram is used. Please refer to the `paper/` folder for detailed textual explanation.

---

## 📁 Project Structure

```bash
.
├── paper/
│   ├── Image Denoising Case Study.docx       # Full research paper (8 pages)
│   └── Cover_Title_Page.docx                 # Title page for submission
├── presentation/
│   └── SelfSupervised_ImageDenoising_Presentation.pptx
├── scripts/
│   ├── train.py                              # Model training script
│   ├── cbn_model.py                          # Model architecture (CBN)
│   ├── utils.py                              # Utility functions
│   ├── loss_functions.py                     # DIL + hybrid losses
│   └── evaluate.py                           # PSNR / SSIM evaluation
├── data/
│   ├── BSD68/                                # Benchmark dataset
│   └── RealNoise/                            # Real-world noisy images
├── results/
│   └── visual_comparisons/                   # Placeholders for output samples
├── video/
│   └── video_script.txt                      # Script for 15-min walkthrough
├── README.md
└── requirements.txt                          # List of dependencies
````

---

## 🧪 Datasets Used

* **BSD68** – Standard benchmark for synthetic noise
* **Set12** – Widely used in low-level vision tasks
* **RealNoise** – Smartphone-captured noisy images (no clean ground truths)

---

## 📊 Evaluation Metrics

* **PSNR**: Peak Signal-to-Noise Ratio
* **SSIM**: Structural Similarity Index

---

## 📦 Installation & Dependencies

Clone the repo and install the dependencies:

```bash
git clone https://github.com/your-username/self-supervised-image-denoising
cd self-supervised-image-denoising
pip install -r requirements.txt
```

**requirements.txt**:

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

## 🚀 How to Train

```bash
python train.py --dataset data/BSD68 --noise_sigma 25
```

Optional flags:

* `--epochs`: number of epochs
* `--batch_size`: batch size
* `--save_path`: model save directory

---

## 📈 Results Summary

| Model           | PSNR (σ=25) | SSIM | Params | Inference Time |
| --------------- | ----------- | ---- | ------ | -------------- |
| DnCNN           | 30.4 dB     | 0.88 | 560K   | 0.10 s         |
| Noise2Void      | 29.6 dB     | 0.86 | 1.4M   | 0.15 s         |
| **SelfDIL-CBN** | 30.2 dB     | 0.87 | 2.1M   | 0.12 s         |

---

## 📽️ Video Presentation

A 15-minute walkthrough covering:

* Novelty of our DIL-based self-supervision
* Architecture and coding details
* Comparative evaluation
* Visual demos

Video script is in `video/video_script.txt`.

---

## ✍️ Authors & Credits

* **Rishabh Tiwari** – B.Sc. Statistics, University of Lucknow
* Inspired by:
  *"Self-supervised Image Denoising with Downsampled Invariance Loss and Conditional Blind-Spot Network"* – Yeong Il Jang et al., ICCV 2023

---

## 📚 Suggested Journals (for publication)

1. Signal, Image and Video Processing – Springer (Q2)
2. Journal of Electronic Imaging – SPIE (Q2)
3. IET Image Processing – Wiley (Q2)
4. Journal of Real-Time Image Processing – Springer (Q3)
5. EURASIP Journal on Image and Video Processing – SpringerOpen (Q3, open access)

---

## 🔖 License

This project is for academic and non-commercial research use only.

---

