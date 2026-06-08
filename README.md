# Distorted Sequence OCR: Advanced AI & Deep Learning

This repository features a complete, end-to-end **Artificial Intelligence (AI)** and **Machine Learning (ML)** pipeline designed to solve complex Optical Character Recognition (OCR) tasks. Specifically, it decodes highly distorted, noisy, and overlapping text sequences from images using state-of-the-art **Neural Networks (NN)**.

## Core Technologies: AI & ML

This project leverages cutting-edge Deep Learning architectures to process unstructured image data and map it to sequential text labels:

### 1. Convolutional Neural Networks (CNNs) for Computer Vision
The first half of the model utilizes a 5-layer deep **CNN** to perform visual feature extraction. Rather than relying on traditional algorithmic image processing, the AI learns to identify high-level visual concepts (edges, curves, and character structures) while inherently ignoring noise, blurring, and heavy occlusion.

### 2. Bidirectional Gated Recurrent Units (BiGRU) for Sequence Modeling
Text recognition is a sequential problem. The visual features extracted by the CNN are passed into a **BiGRU** (a specialized form of Recurrent Neural Network). This allows the AI to learn *context*—analyzing the sequence both left-to-right and right-to-left. By understanding surrounding characters, the network can predict letters that are heavily blurred or overlapping based on sequential probability.

### 3. Connectionist Temporal Classification (CTC)
Traditional ML OCR requires strict "bounding boxes" around every single character during training. This project uses **CTC Loss**, an advanced loss function that allows the AI to calculate the error of an unaligned sequence. The Neural Network learns to dynamically align its predictions with the target text, making it highly robust against irregular spacing.

## Evaluation Metric

The model's performance is strictly evaluated using the **Character Error Rate (CER)**. This is calculated using the Levenshtein Distance algorithm, which measures the minimum number of single-character edits (insertions, deletions, substitutions) required to fix the AI's prediction.

## Repository Contents

- `submission_notebook.ipynb`: The fully documented Jupyter Notebook containing Exploratory Data Analysis, preprocessing, the CRNN Neural Network definition, training loop, evaluation, and inference.
- `best_model.pth`: The saved PyTorch weights of the fully-trained AI model.
- `submission.csv`: The finalized model predictions generated on the test dataset.
- `train-labels.csv`: The true labels for the training sequences.

## Getting Started

Ensure you have the required ML dependencies installed:
```bash
pip install torch torchvision numpy pandas pillow jupyter
```

1. Place the `train_images/` and `test_images/` directories in the root folder (ignored by git due to size).
2. Run the `submission_notebook.ipynb` to reproduce the AI training pipeline, or load `best_model.pth` directly for rapid inference.
