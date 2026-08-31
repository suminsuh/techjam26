# RIFT — Robust Image Forgery Tracer

**TikTok TechJam 2026 · Track 5 · *Robust Detection of AI-Generated Images Under Real-World Transformations***

RIFT is a high-performance AIGC detector designed around the core requirement of the brief: **maintaining stable, accurate detection under real-world social media degradations** (JPEG re-encoding, Gaussian blur, thumbnail resizing, additive noise, color jitter, and cropping) — not just reporting laboratory accuracy on pristine images.

---

## 🏗️ Architecture: Dual-Stream Gated Fusion

Research shows that **forensic frequency cues are sensitive on pristine images but degrade under heavy compression**, while **semantic/spatial backbones degrade smoothly**. RIFT combines both paradigms via learned trust-gated routing:

```text
                                  ┌──> ConvNeXt-Tiny (Spatial Stream) ────────┐
                                  │    [Texture regularity & spatial cues]    │
                                  │                                           ├──> [ Gated Fusion ] ──> [ Classifier ] ──> P(AIGC)
Input Image (RGB) ────────────────┤                                           │    (Trust Gate)
                                  │                                           │
                                  └──> SRM + FFT Magnitude + Phase ───────────┘
                                       [9-Channel Forensic Stream]
```

1. **Spatial Stream:** `convnext_tiny` (~28M parameters, ImageNet pretrained) captures high-level structural patterns and semantic textures that survive aggressive compression.
2. **Forensic Stream:** A 9-channel frontend extracting 3 Steganalysis Rich Model (SRM) spatial residuals, 3 FFT log-magnitude channels, and 3 FFT phase channels (phase preserves edge geometry across JPEG quantization).
3. **Learned Trust Gate:** Dynamically balances spatial vs. forensic evidence per image. When severe compression quantizes high frequencies, the gate shifts confidence toward the spatial stream.
4. **Consistency Regularization:** Trained with two-view perturbation matching to prevent decision flips upon image reposting.
5. **Parameter Budget:** **~28.67M parameters**, well below the competition's **< 2B** ceiling (utilizing only **1.4%** of the allowed budget).

---

## 📊 Benchmark Results

### 1. Robustness Evaluation on Validation Set (`data/sid_set/val`, $n=1000$, TTA)
*Operating threshold frozen on clean validation at 5% target FPR:* `0.0852`

| Condition | Accuracy | AUROC | AP | FPR | FNR | Spatial Gate | Forensic Gate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`clean`** | **97.6%** | **99.8%** | **99.9%** | 4.2% | **0.6%** | 63.0% | 37.0% |
| **`jpeg_90`** | 97.4% | 99.9% | 99.9% | 4.2% | 1.0% | 63.0% | 37.0% |
| **`jpeg_70`** | 97.5% | 99.9% | 99.9% | 3.8% | 1.2% | 63.0% | 37.0% |
| **`jpeg_50`** | 97.6% | 99.8% | 99.8% | 3.6% | 1.2% | 62.9% | 37.1% |
| **`jpeg_30`** | 97.8% | 99.9% | 99.9% | 3.4% | 1.0% | 62.9% | 37.1% |
| **`blur_0.5`** | 97.8% | 99.9% | 99.9% | 3.4% | 1.0% | 63.1% | 36.9% |
| **`blur_1.0`** | 97.9% | 99.8% | 99.9% | 2.6% | 1.6% | 63.3% | 36.7% |
| **`blur_2.0`** | 98.1% | 99.8% | 99.8% | 2.0% | 1.8% | 63.1% | 36.9% |
| **`resize_0.5`** | 98.1% | 99.8% | 99.9% | 2.0% | 1.8% | 63.3% | 36.7% |
| **`resize_0.25`** | 97.9% | 99.8% | 99.8% | 1.8% | 2.4% | 62.5% | 37.5% |
| **`noise_0.02`** | 97.6% | 99.8% | 99.9% | 4.0% | 0.8% | 63.0% | 37.0% |
| **`noise_0.05`** | 98.0% | 99.9% | 99.9% | 2.6% | 1.4% | 62.9% | 37.1% |
| **`noise_0.10`** | 98.3% | 99.8% | 99.9% | 1.6% | 1.8% | 62.5% | 37.5% |
| **`color_jitter`** | 97.6% | 99.9% | 99.9% | 3.8% | 1.0% | 62.7% | 37.3% |
| **`center_crop_0.8`** | 97.0% | 99.5% | 99.5% | 2.8% | 3.2% | 58.6% | 41.4% |

---

### 2. Generalization on Unseen Holdout Set (`data/wildfake_holdout`, $n=1000$, TTA)
*Zero-shot transfer to unseen generators (COCO Real vs. DALL·E 3 AIGC):*

| Condition | Accuracy | AUROC | AP | FPR | FNR |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`clean`** | **64.9%** | **78.4%** | **79.0%** | 5.0% | **65.2%** |
| **`jpeg_70`** | 65.0% | 79.8% | 80.2% | 4.0% | 66.0% |
| **`jpeg_30`** | 63.5% | 78.6% | 79.4% | 3.8% | 69.2% |
| **`blur_0.5`** | 65.1% | 77.7% | 79.1% | 4.6% | 65.2% |
| **`noise_0.05`** | 64.6% | 90.2% | 90.9% | 1.2% | 69.6% |
| **`noise_0.10`** | 63.2% | 94.7% | 94.8% | 0.4% | 73.2% |

---

## ⚡ Quickstart

### 1. Environment Setup
Requires Python 3.10–3.13 (CUDA or CPU).

```powershell
# Clone and enter directory
cd techjam26

# Install dependencies
pip install -r requirements.txt

# For NVIDIA GPU acceleration (e.g. RTX 4060 / CUDA 12.6):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Run unit tests
python -m pytest -v
```

---

## 🚀 Execution & Deliverables

### A. Predict (Required JSON Output)
Scores any directory of images and writes the standard competition format:

```powershell
python scripts/predict.py --config configs/convnext_tiny.yaml --checkpoint checkpoints/best.pt --input_dir path/to/images --output predictions.json --aux
```

```json
[
  {
    "image_path": "data/samples/FAKE/fake_001.png",
    "pred": 0.941,
    "gate_spatial": 0.628,
    "gate_forensic": 0.372
  }
]
```

### B. Run Robustness Benchmark
Computes the 15-condition robustness table under frozen 5% FPR:

```powershell
# Validation set evaluation
python scripts/evaluate.py --config configs/convnext_tiny.yaml --checkpoint checkpoints/best.pt --output_dir outputs/convnext_tiny

# Holdout evaluation
python scripts/evaluate.py --config configs/convnext_tiny.yaml --checkpoint checkpoints/best.pt --data_dir data/wildfake_holdout --output_dir outputs/convnext_holdout
```

### C. Launch Interactive Gradio Demo
Interactive web interface for single-image upload, live gate breakdown, spatial Grad-CAM activation overlay, and instant 8-transform perturbation stress-testing:

```powershell
python scripts/demo.py --config configs/convnext_tiny.yaml --checkpoint checkpoints/best.pt
```
Open **`http://127.0.0.1:7860`** in your browser.

### D. Train Model
Train the ConvNeXt-Tiny + Forensic dual-stream model on SID_Set:

```powershell
python scripts/train.py --config configs/convnext_tiny.yaml
```

---

## 📂 Repository Layout

```text
techjam26/
├── configs/
│   ├── convnext_tiny.yaml             # Primary submission config
│   └── default.yaml                   # Starter base config
├── checkpoints/
│   ├── best.pt                        # Best model checkpoint (ConvNeXt-Tiny)
│   └── last.pt                        # Last epoch weights
├── outputs/
│   ├── convnext_tiny/                 # Validation robustness benchmark outputs
│   └── convnext_holdout/              # WildFake holdout benchmark outputs
├── scripts/
│   ├── predict.py                     # Official folder scorer -> predictions.json
│   ├── evaluate.py                    # 15-condition frozen-threshold evaluator
│   ├── demo.py                        # Gradio UI & Grad-CAM visualizer
│   ├── train.py                       # Training engine with consistency loss
│   ├── error_analysis.py              # FP/FN report generator
│   └── prepare_sid_set.py             # Training dataset setup
├── src/rift/                          # Core RIFT package
│   ├── models/dual_stream.py          # ConvNeXt-Tiny + SRM/FFT Dual-Stream
│   ├── features.py                    # 9-channel SRM + FFT Magnitude & Phase
│   ├── transforms.py                  # 15 official Track 5 transformations
│   ├── metrics.py                     # Fixed-threshold AUROC/FPR/FNR metrics
│   └── engine/                        # Predict, train, evaluate, explain engines
└── predictions.json                   # Sample scored output deliverable
```

---

## 📜 License
MIT License. Datasets remain subject to their respective licenses (SID_Set, WildFake, COCO).
