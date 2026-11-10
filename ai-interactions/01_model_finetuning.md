# Interaction 01 – Model Fine-Tuning and Pipeline Setup

## 1. Context

**Task:** Prepare and fine-tune a ResNet-18 model for binary mobility-aid image classification.

The objective was to develop a training pipeline using the Freiburg Mobility Aids Dataset.

The classification task was defined as:

- **Class 0:** No mobility aid present.
- **Class 1:** At least one mobility aid present.

ChatGPT was used to support the implementation and organization of the training pipeline, from dataset preparation to model fine-tuning.

## 2. Important Prompts

**Prompt 1 – Building the training pipeline**

> Develop a complete PyTorch pipeline in a Jupyter Notebook to fine-tune a pretrained ResNet-18 model on a custom dataset. The input data contains images and separate annotation files.

**Prompt 2 – Choosing a dataset splitting strategy**

> The images are extracted from video sequences. What is the best approach to split the dataset into training, validation, and test sets while avoiding data leakage between sequences?

**Prompt 3 – Identifying video sequences**

> The following lines contain the names of my annotation files. Can you provide a script that identifies which annotation files belong to the same video sequence?

**Prompt 4 – Improving notebook structure and documentation**

> I have modified the Jupyter Notebook. Could you restructure it and add clear, well-organized Markdown cells with understandable explanations and annotations?

## 3. AI Contributions

- **Contribution 1:** ChatGPT provided a training pipeline, including loading images and annotations, matching them, and combining the information into a dataset. It also implemented the pretrained ResNet-18 model and the fine-tuning workflow.
- **Contribution 2:** ChatGPT suggested avoiding a purely random split and using a group-based split to separate training, validation, and test data by video sequence.
- **Contribution 3:** ChatGPT provided a script to identify sequence groups based on annotation filenames and timestamps. These groups were then used for splitting the dataset.
- **Contribution 4:** ChatGPT restructured the notebook while retaining the existing functional code cells. It added Markdown explanations to make the pipeline easier to understand and follow.

## 4. Implementation and Decisions

The AI generated most of the initial pipeline, which we then adapted to our actual dataset and project structure.

The suggested group-based splitting strategy was adopted instead of random splitting because images from the same video sequence should not be distributed across different dataset splits.

The generated code required some adjustments, particularly regarding imports, file paths, and the structure of the annotation files. The notebook documentation was also generated with AI assistance.

## 5. Critical Evaluation

**What worked well?**

- ChatGPT helped establish a structured training pipeline in a relatively short time.
- It provided useful implementations for dataset preparation, model fine-tuning, and dataset splitting.
- The Markdown explanations made the notebook easier to navigate and understand.
- The suggestion to use group-based splitting helped address a potential data leakage issue.

**What needed correction or refinement?**

- Generated code needed to be checked against the actual dataset structure.
- File paths and imports needed to be adapted to the project organization.
- The sequence identification and grouping logic needed to be checked to ensure that images from the same recording were assigned to the same split.

**What were the limitations of AI assistance?**

- AI-generated code did not automatically guarantee a reproducible and functional pipeline.
- The quality of the suggestions depended on how much context about the dataset and project structure was provided.
- Sequence grouping based on filenames and timestamps required verification rather than being accepted without checking.
- Class imbalance wasn#t recognized or asked about. I had to come up with this. 

## 6. Lessons Learned

AI assistance helped establish a more structured model training pipeline in less time.

An important lesson was that AI is useful for designing and organizing a technical workflow, but its suggestions must always be checked against the actual project and dataset.

Providing additional context, such as example images, annotation files, and details about the dataset structure, could have helped make the generated code more accurate from the beginning.