# Interaction 03 – Robustness Analysis

## 1. Context

Task: Evaluate the performance of fine-tuned ResNet-18 binary
classifier and investigate its robustness under image perturbations.

The evaluation focused on overall classification performance,
confusion-matrix analysis, and the effect of brightness, Gaussian noise,
and Gaussian blur on model predictions. The goal was to understand not
only how accurate the model is, but also how stable its predictions are
when image conditions changes.

ChatGBT was used to support the analytical process, suggest investigation methods, and implement the corresponding analyses.

## 2. Important Prompts

Prompt 1 – Evaluation structure

I asked how to structure an evaluation notebook for a fine-tuned
ResNet-18 binary classifier, including accuracy, precision, recall,
F1-score, and a confusion matrix.

Prompt 2 – Model inference

I asked for a Python workflow to run inference on the test images,
compare predictions with ground-truth labels, and calculate standard
classification metrics.

Prompt 3 – Confusion matrix

I asked how to create and interpret a confusion matrix for the binary
classification task.

Prompt 4 – Robustness test design

I asked how to test the model under image perturbations, specifically
increased brightness, Gaussian noise, and Gaussian blur.

Prompt 5 – Robustness implementation

I asked for code that creates perturbed versions of each test image,
runs model inference on the original and perturbed images, and compares
the accuracy across all conditions.

Prompt 6 – Results interpretation

I asked for help interpreting the robustness results and explaining
which perturbation had the largest and smallest effect on accuracy.

## 3. AI Contributions

Contribution 1: ChatGBT suggested a structured evaluation
workflow using the metrics accuracy, precision, recall, F1-score, and a confusion
matrix.

Contribution 2: It helped structure the model inference and evaluation
process, including prediction generation, comparison with ground-truth
labels, and calculation of classification metrics.

Contribution 3: It provided an approach for creating and interpreting
the confusion matrix.

Contribution 4: It suggested testing robustness by comparing original
images with perturbed versions while keeping the evaluation
preprocessing consistent.

Contribution 5: It helped generate code for brightness increase,
Gaussian noise, and Gaussian blur perturbations.

Contribution 6: It helped summarise the robustness results in a table
and interpret the relative effect of each perturbation.

## 4. Implementation and Decisions

The suggested analyses were implemented in the evaluation ´02_image_perturbation´ notebook.

The main steps included:

- Loading the predefined test split and verifying the image files.
- Running model inference on 1,952 test images.
- Calculating accuracy, precision, recall, and F1-score.
- Creating and interpreting the confusion matrix.
- Creating brightness, noise, and blur versions of each test image.
- Comparing model accuracy across the original and perturbed conditions.
- Visualising the robustness results in a bar chart.

The original images were used as a reference condition. All perturbed
images were processed using the same evaluation preprocessing as the
original test images.

## 5. Critical Evaluation

What worked well?

ChatGBT was useful for structuring the evaluation workflow and suggesting a
clear comparison between the original images and the perturbed images.
It also helped produce code for the image perturbations and supported
the interpretation of the results.

What needed correction or refinement?

The generated code needed to be adapted to the existing notebook and
dataset structure. The perturbation settings also had to be selected and
checked carefully, because different values for brightness, noise, and
blur would produce different results.

What were the limitations of AI assistance?

ChatGBT could suggest suitable methods, but it could not decide whether the
selected perturbation settings were meaningful for the specific dataset.
The interpretation also had to be checked carefully, because the results
show performance under the tested conditions only and do not prove
robustness to stronger or combined perturbations.

## 6. Lessons Learned

This project showed me that AI is useful for structuring an evaluation,
generating code, and improving documentation. However, the choice of
evaluation methods and perturbation settings still requires an
understanding of the dataset and the research question.

I also learned that a robustness test should compare a reference
condition with controlled changes. In this project, the original images
served as the reference, while brightness, noise, and blur were tested
separately.

Finally, I learned that results should be interpreted carefully. The
model remained above 92% accuracy in all tested conditions, but this
does not mean that it would remain stable under stronger perturbations
or combinations of perturbations.
