# Technical Report: Model Error Analysis & Robustness Evaluation

**Project Team:** Human-Aided Pair  
**Authors:** Prajothis Prajothis Student ID K12538726 | Aldo Vincent Student ID K12517527  
**Model Architecture:** Fine-Tuned ResNet-18  
**Dataset:** Mobility Aid Identification Dataset (`Images_RGB`)  

---

## Abstract & Executive Summary

This report presents a thorough classification error analysis and out-of-distribution (OOD) robustness evaluation for a fine-tuned ResNet-18 model trained on the Mobility Aid Identification dataset (`Images_RGB`).

While the baseline model achieves a high top-line accuracy of $95.39\%$ across $1,952$ test images, in-depth error diagnosis reveals a critical vulnerability due to severe class imbalance. Specifically, **Class 0 (No Mobility Aid)** exhibits an unacceptable baseline recall of $0.51$, compared to $0.98$ for **Class 1 (Mobility Aid)**.

To investigate whether the network relies on invariant geometric features or chromatic/background shortcuts, a systematic series of spatial and chromatic perturbation experiments were conducted—including bounding box cropping, dual-mode masking, a 12-sample pilot color inversion test, and a full 1,952-sample color inversion evaluation.

Under full-dataset color inversion, overall accuracy collapsed by $49.64$ percentage points to $45.75\%$, proving that the baseline model suffers heavily from shortcut learning via chromatic priors.

---

## Experimental Methodology & Pipeline

To evaluate feature reliance, spatial sensitivity, and color channel dependency, we executed a 4-stage analytical workflow:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        1,952 Test Image Dataset                        │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
       ┌───────────────────────────┼───────────────────────────┐
       ▼                           ▼                           ▼
[ 1. ROI Cropping ]       [ 2. Dual-Mode Masking ]   [ 3. Color Inversion ]
   Isolates object           ├── Type 1: Foreground      ├── Pilot Test (12 samples)
   geometry from background  └── Type 2: Background      └── Full Scale (1,952 samples)
```

1. **Bounding Box Cropping:** Extracted key Regions of Interest (ROI) to isolate object geometry and test model behavior when background context is stripped.
2. **Dual-Mode Masking:**
   - **Type 1 (Foreground Masking):** Occluded the primary subject to evaluate the network's reliance on peripheral and background context.
   - **Type 2 (Background Masking):** Zeroed out background pixels to test whether foreground boundaries alone are sufficient for accurate classification.
3. **12-Sample Pilot Inversion Test:** Evaluated color inversion ($I_{\text{inverted}} = 255 - I_{\text{original}}$) on a controlled batch of 12 representative images to observe qualitative prediction shifts prior to large-scale execution.
4. **Full-Dataset Inversion Experiment:** Scaled color inversion across all 1,952 test images to quantify statistical robustness, recall shifts, and prediction stability.

---

## Baseline Model Performance

Evaluated on the standard 1,952-image test split:

- **Class 0 (No Mobility Aid):** Precision: $0.89$ | Recall: $0.51$ | F1-Score: $0.65$ | Support: $124$ *(Underperforming - Minority Class)*
- **Class 1 (Mobility Aid):** Precision: $0.96$ | Recall: $0.98$ | F1-Score: $0.97$ | Support: $1,828$ *(Optimal Performance - Majority Class)*
- **Overall Model Accuracy:** $95.39\%$ (Total Support: $1,952$)

### Detailed Classification Report

| Class | Precision | Recall | F1-Score | Support | Status / Note |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Class 0 (No Aid)** | 0.89 | 0.51 | 0.65 | 124 | Minority Class (Underperforming) |
| **Class 1 (Mobility Aid)** | 0.96 | 0.98 | 0.97 | 1,828 | Majority Class |
| **Accuracy** | | | **0.95** | **1,952** | Top-Line Benchmark |
| **Macro Avg** | 0.92 | 0.75 | 0.81 | 1,952 | Unweighted Average |
| **Weighted Avg** | 0.95 | 0.95 | 0.95 | 1,952 | Class-Weighted Average |

---

## Diagnostic Error & Confusion Matrix Analysis

### 1. Bounding Box Cropping & Masking Insights

- **Spatial Cropping:** Removing environmental background clutter improved feature focus on Class 0 objects, confirming that non-target clutter in original frames contributed to false negatives.
- **Foreground vs. Background Masking:**
  - **Type 2 (Background Masking):** Severely degraded Class 0 detection, revealing that the model heavily relies on background context as a proxy for identifying the absence of a mobility aid.
  - **Type 1 (Foreground Masking):** Demonstrated that Class 1 predictions remain falsely confident even when the target object is occluded, confirming reliance on environmental shortcuts.

---

### 2. Full Dataset Color Inversion Matrix Breakdown

Below is the exact empirical breakdown extracted from the comparative confusion matrices (*ResNet-18: Original vs Colour Inversion*):

#### Matrix Comparison Table ($N = 1,952$)

| Ground Truth Class | Original Pred: No Aid | Original Pred: Mobility Aid | Inverted Pred: No Aid | Inverted Pred: Mobility Aid |
| :--- | :---: | :---: | :---: | :---: |
| **True: No Aid (Class 0)** | 63 (TN) | 61 (FP) | 124 (TN) | 0 (FP) |
| **True: Mobility Aid (Class 1)** | 29 (FN) | 1,799 (TP) | 1,059 (FN) | 769 (TP) |

---

### 3. Detailed Confusion Matrix Analysis & Key Metric Shifts

1. **Class 0 (No Aid) Transformation:**
   - **Baseline:** Out of 124 true "No Aid" images, the baseline model correctly identified 63 (TN) and misclassified 61 as "Mobility Aid" (FP), yielding a $49.2\%$ error rate within Class 0.
   - **Color Inverted:** All 124 "No Aid" samples were predicted as "No Aid" ($\text{TN}=124, \text{FP}=0$), driving Class 0 Recall to $100\%$ ($1.00$).

2. **Class 1 (Mobility Aid) Collapse:**
   - **Baseline:** Out of 1,828 true "Mobility Aid" samples, the model correctly identified 1,799 (TP) and misclassified only 29 (FN), achieving $98.4\%$ recall.
   - **Color Inverted:** False negatives exploded from 29 to 1,059, dropping true positives down to 769. Class 1 Recall collapsed from $0.98$ to $0.42$.

3. **Overall Systematic Decision Bias:**
   - **Inversion Bias Toward Class 0:** Under color inversion, total predictions for "No Aid" increased dramatically from 92 ($63+29$) to 1,183 ($124+1,059$).
   - **Prediction Transition Summary:**
     - 1,030 previously correct predictions (61 Class 0 + 969 Class 1) were flipped into errors.
     - 61 previously incorrect Class 0 predictions shifted to correct predictions ($\text{FP} \rightarrow \text{TN}$).
     - 861 total predictions remained unchanged ($63 + 798$).

4. **Conclusion on Feature Shortcuts:** The massive shift of 1,059 Class 1 images being misclassified as Class 0 under inverted color space confirms that the ResNet-18 baseline relies on chromatic distribution and brightness levels as strong classification priors rather than robust visual geometry.

---

## AI-Aided Image Perturbation Benchmarking

*Note: As done by the Ai-Aided pair.*

- **Original Accuracy:** $95.39\%$
- **Brightness Increased:** $93.49\%$
- **Gaussian Noise Added:** $94.72\%$
- **Gaussian Blur Applied:** $92.88\%$ *(Largest overall performance reduction among standard image modifications)*

While general noise and blur slightly decrease accuracy, the model maintains $>92\%$ accuracy under standard perturbations. However, this high surface robustness obscures the severe structural shortcut dependencies identified in Sections 3 and 4.

---

## Phase 2 Improvement Roadmap

To eliminate chromatic bias, resolve the Class 0 recall deficit, and enforce true geometric feature extraction, Phase 2 training will execute the following mitigations:

- **Data Augmentation Enhancements:**
  - Introduce heavy `ColorJitter` (brightness, contrast, saturation, hue) to decouple classification from color distributions.
  - Implement **Random Grayscale** conversion ($p = 0.3$) to force reliance on structural geometry and edges.
  - Apply **Random Cutout / Random Erasing** to emulate Type 1 & Type 2 masking during training.

- **Class Imbalance & Loss Function Optimization:**
  - Replace standard Cross-Entropy Loss with **Weighted Cross-Entropy** or **Focal Loss** ($\gamma = 2.0$) to penalize Class 0 misclassifications more aggressively.

