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
