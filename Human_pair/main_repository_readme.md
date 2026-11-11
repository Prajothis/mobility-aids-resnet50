# Mobility Aid Recognition — ResNet-18 Evaluation & Diagnostic Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-Freiburg_Mobility_Aids-green.svg)](http://mobility-aids.informatik.uni-freiburg.de/)

## 📌 Project Overview

This repository contains the fine-tuning, robustness evaluation, and diagnostic error analysis for a **ResNet-18** binary image classifier trained to identify mobility aids using the **Freiburg Mobility Aids Dataset**.

### Classification Goal:
* **Class 0 (`No mobility aid`):** No mobility aid present in the frame.
* **Class 1 (`Mobility aid`):** At least one mobility aid present (wheelchair, walking frame, crutches, push wheelchair).

---

## 📊 Summary of Key Results

| Condition / Experiment | Overall Accuracy | Class 0 Recall | Class 1 Recall | Primary Observation |
| :--- | :---: | :---: | :---: | :--- |
| **Baseline Test Evaluation** | **95.39%** | **0.51** | **0.98** | High accuracy driven by majority class (Class 1); severe Class 0 recall deficit. |
| **AI-Aided Perturbations** | $\ge 92.88\%$ | — | — | Robust against minor noise/blur (Gaussian blur caused highest drop). |
| **Human-Pair Color Inversion** | **45.75%** | **1.00** | **0.42** | Severe drop (-49.64 percentage points); model relies heavily on chromatic shortcuts. |

---

## 🛠️ Dataset Setup

### Dataset Source
This project uses the **Freiburg Mobility Aids Dataset** (17,079 RGB images and 15,278 YAML annotations).
* **Download link:** [Freiburg Mobility Aids Dataset](http://mobility-aids.informatik.uni-freiburg.de/)

### Local Directory Structure
Place `Images_RGB` and `Annotations_RGB` directly in the root directory:

```text
Mobility-Aid-Classification/
├── Images_RGB/           # Raw RGB images (excluded via .gitignore)
├── Annotations_RGB/      # Annotation files (excluded via .gitignore)
├── data/                 # Cleaned dataset CSV split files (train, val, test)
├── models/               # Model checkpoints (best_resnet18.pth)
├── notebooks/            # Evaluation & analysis Jupyter notebooks
├── src/                  # Dataset loader & evaluation transforms
├── README.md             # Repository documentation
└── requirements.txt      # Project dependencies
```

---

## 🚀 Quick Start: Model Evaluation

### 1. Load Trained Checkpoint

```python
import torch
import torch.nn as nn
import pandas as pd
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Recreate architecture & load checkpoint
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

checkpoint = torch.load("models/best_resnet18.pth", map_location=device)
model.load_state_dict(checkpoint)
model = model.to(device)
model.eval()
```

### 2. Run Inference on Test Dataset

```python
from src import MobilityAidDataset, eval_transform
from torch.utils.data import DataLoader

test_df = pd.read_csv("data/test.csv")
test_dataset = MobilityAidDataset(test_df, transform=eval_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
```

---

## 🔬 Analysis Breakdown

This evaluation was split into two analytical tracks:

### 1. AI-Aided Robustness Analysis
* Focuses on global environmental perturbations: **Brightness increase**, **Gaussian Noise**, and **Gaussian Blur**.
* Documented in `ai-interactions/03_phase1_robustness_analysis_k12545103.md`.

### 2. Human-Pair Diagnostic Analysis
* Deep-dive into structural failure modes:
  * **Temporal Tracking:** Sequential frame prediction path analysis revealing error clustering in video sequences (`[14–58]` & `[90–115]`).
  * **Spatial Sensitivity:** ROI Cropping (restored Class 0 accuracy to 99.9%) and Dual-Mode Masking (Foreground vs. Background).
  * **Full Dataset Color Inversion:** Scaled $N=1,952$ chromatic perturbation showing 1,091 prediction shifts and confirming shortcut learning.

---

## 👥 Project Team

* **Project Group:** The Deadliners
* **Architecture:** ResNet-18 (Pre-trained on ImageNet, Fine-Tuned)