## Dataset Setup

### Dataset Source

This project uses the **Freiburg Mobility Aids Dataset**.

The dataset consists of two main components:

- **RGB images:** Images of people and mobility-aid-related objects.
- **Annotations:** YAML files containing annotated object categories and bounding-box coordinates.

The annotations include categories such as:

- `person`
- `wheelchair`
- `walking_frame`
- `crutches`
- `push_wheelchair`

The original dataset contains 17,079 RGB images and 15,278 annotation files.

**Dataset download:** [Mobility Aids](http://mobility-aids.informatik.uni-freiburg.de/)

### Downloading and Setting Up the Dataset Locally

The original images and annotations are not included in this GitHub repository. They must be downloaded separately.

1. Download the Freiburg Mobility Aids Dataset from its original source.
2. Extract the downloaded dataset.
3. Locate the `Images_RGB` and `Annotations_RGB` folders.
4. Place both folders directly in the root directory of this repository.

The project directory should then look like this:

```text
Mobility-Aid-Classification/
│
├── Images_RGB/
│   ├── seq_....png
│   └── ...
│
├── Annotations_RGB/
│   ├── seq_....yml
│   └── ...
│
├── data/
├── models/
├── notebooks/
└── README.md
```

The folder names must match exactly because the data preparation notebook expects these directory names.

**Note:** `Images_RGB/` and `Annotations_RGB/` are excluded from Git through `.gitignore`. They remain available locally but are not uploaded to GitHub.


# Model Fine-Tuning

## Starting Phase I: Loading the Trained Model

The AI-assisted work for data preparation and model fine-tuning is documented in:

`ai-interactions/01_data_preparation_and_training_k51822097.ipynb`

After training, the best ResNet-18 checkpoint is saved to:

```text
models/best_resnet18.pth
```

The data preparation notebook also exports the dataset splits:

```text
data/
    cleaned_dataset.csv
    train.csv
    validation.csv
    test.csv
```

These files allow the same dataset splits to be reused without repeating the preprocessing and splitting process.

### Loading the Best Model Checkpoint

```python
import torch
import torch.nn as nn
import pandas as pd
from torchvision import models

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

Load the test dataset:

```python
test_df = pd.read_csv("data/test.csv")
```

Recreate the ResNet-18 architecture and load the saved checkpoint:

```python
model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    2
)

checkpoint = torch.load(
    "models/best_resnet18.pth",
    map_location=device
)

model.load_state_dict(checkpoint)
model = model.to(device)
model.eval()
```

The model is now ready for inference.

**Training Approach:** The goal was to establish a reasonable baseline model rather than achieve optimal classification performance. Training therefore focused on fine-tuning a pretrained ResNet-18 with limited optimisation. The resulting model serves as the basis for the Phase I investigation of its predictions, errors and behaviour.

## Loading the Test Dataset and Applying ImageNet Preprocessing
```python
from src import MobilityAidDataset
from src import eval_transform

import pandas as pd
from torch.utils.data import DataLoader

test_df = pd.read_csv("data/test.csv")
test_dataset = MobilityAidDataset(test_df, transform=eval_transform)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

```

# AI-assisted analysis

### Error Analysis

The AI-assisted work for error analysis is documented in:

`ai-interactions/02_phase1_error_analysis_k51822097.ipynb`

The error analysis focused on understanding where the model makes mistakes and identifying possible patterns or factors that could explain them.


### Robustness Analysis

The AI-assisted work for model evaluation and robustness testing is
documented in `ai-interactions/03_phase1_robustness_analysis_k12545103.md`.

The robustness of the model was tested by using three different perturbations:

- Increased brightness
- Gaussian noise
- Gaussian blur

For each test image, the original version and also three perturbed versions were evaluated. The original images are the reference condition. 

To reproduce the evaluation and robustness test, ensure that the
following files and folders are available:

- `data/test.csv`
- `models/best_resnet18.pth`
- `Images_RGB/`
- `Annotations_RGB/`
- `src/`

Run the notebook cells from top to bottom. 

# Mobility Aid Identification: Model Error Diagnosis & Robustness Evaluation

## Team Information (Human Pair)
- **Team Name:** The Deadliners


## Human Pair Error Analysis & Diagnostics

The human pair contribution focuses on detailed error diagnosis, spatial sensitivity testing, and chromatic channel dependence testing. All corresponding code, reports, and figures are located in the `Human_pair/` folder:

- **`Human_pair/model_error_analysis_report.md`**: Comprehensive technical report detailing baseline error clustering, class imbalance deficit (Class 0 recall = 0.51), spatial masking analysis, and full-dataset color inversion results.
- **`Human_pair/human_inference.ipynb`**: Experimental notebook implementing ROI bounding box cropping, dual-mode masking (Type 1 foreground & Type 2 background), and RGB color inversion ($I_{\text{inverted}} = 255 - I$).
- **`Human_pair/Figures/`**: Generated confusion matrices, color inversion evaluation plots, and qualitative failure sample figures.
- **`Phase1_midterm.pptx`**: 4-slide presentation summarizing error diagnosis, spatial/chromatic key findings, and Phase 2 data augmentation recommendations.

### Summary of Human Pair Findings
1. **Class Imbalance Deficit:** High top-line baseline accuracy ($95.39\%$) masks a critical minority class failure, where **Class 0 (No Aid)** recall is only $0.51$.
2. **Spatial Context Dependency:** Background masking (Type 2) severely degraded performance, while foreground masking (Type 1) showed the model remained falsely confident in predicting mobility aids based on corridor background clutter alone.
3. **Chromatic Prior Shortcut:** Inverting RGB color channels across all 1,952 test images caused top-line accuracy to collapse by $49.64\%$ down to $45.75\%$, proving heavy model reliance on color/brightness priors over structural geometry.

---

## Dataset Setup

### Dataset Source

This project uses the **Freiburg Mobility Aids Dataset**.

The dataset consists of two main components:

- **RGB images:** Images of people and mobility-aid-related objects.
- **Annotations:** YAML files containing annotated object categories and bounding-box coordinates.

The annotations include categories such as:

- `person`
- `wheelchair`
- `walking_frame`
- `crutches`
- `push_wheelchair`

The original dataset contains 17,079 RGB images and 15,278 annotation files.

**Dataset download:** [Mobility Aids](http://mobility-aids.informatik.uni-freiburg.de/)

### Downloading and Setting Up the Dataset Locally

The original images and annotations are not included in this GitHub repository. They must be downloaded separately.

1. Download the Freiburg Mobility Aids Dataset from its original source.
2. Extract the downloaded dataset.
3. Locate the `Images_RGB` and `Annotations_RGB` folders.
4. Place both folders directly in the root directory of this repository.

The project directory should then look like this:

```text
Mobility-Aid-Classification/
│
├── Human_pair/
│   ├── Figures/
│   ├── human_inference.ipynb
│   ├── main_repository_readme.md
│   └── model_error_analysis_report.md
│
├── ai-interactions/
├── data/
├── models/
├── src/
├── Images_RGB/
├── Annotations_RGB/
├── Phase1_midterm.pptx
├── README.md
├── team-info.json
└── workload.json
```

The folder names must match exactly because the data preparation notebook expects these directory names.

**Note:** `Images_RGB/` and `Annotations_RGB/` are excluded from Git through `.gitignore`. They remain available locally but are not uploaded to GitHub.

## Starting Phase I: Loading the Trained Model

The model is trained in `01_data_preparation_and_training.ipynb`.

After training, the best ResNet-18 checkpoint is saved to:

```text
models/best_resnet18.pth
```

The data preparation notebook also exports the dataset splits:

```text
data/
    cleaned_dataset.csv
    train.csv
    validation.csv
    test.csv
```

These files allow the same dataset splits to be reused without repeating the preprocessing and splitting process.

### Loading the Best Model Checkpoint

```python
import torch
import torch.nn as nn
import pandas as pd
from torchvision import models

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

Load the test dataset:

```python
test_df = pd.read_csv("data/test.csv")
```

Recreate the ResNet-18 architecture and load the saved checkpoint:

```python
model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    2
)

checkpoint = torch.load(
    "models/best_resnet18.pth",
    map_location=device
)

model.load_state_dict(checkpoint)
model = model.to(device)
model.eval()
```

The model is now ready for inference.

**Training Approach:** The goal was to establish a reasonable baseline model rather than achieve optimal classification performance. Training therefore focused on fine-tuning a pretrained ResNet-18 with limited optimisation. The resulting model serves as the basis for the Phase I investigation of its predictions, errors and behaviour.

## Loading the Test Dataset and Applying ImageNet Preprocessing
```python
from src import MobilityAidDataset
from src import eval_transform

import pandas as pd
from torch.utils.data import DataLoader

test_df = pd.read_csv("data/test.csv")
test_dataset = MobilityAidDataset(test_df, transform=eval_transform)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)



