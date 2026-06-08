# Distorted Visual Sequence Pattern Recognition

This repository contains a complete deep learning workflow to recognize and reconstruct text sequences from visually distorted grayscale images.

## Overview

Real-world text recognition systems often struggle with overlapping characters, background noise, blur, shape deformation, and irregular spacing. This project builds a robust model to decode sequences from such noisy images using a Convolutional Recurrent Neural Network (CRNN).

## Methodology & Architecture

The model is built using PyTorch and follows a standard sequence recognition architecture:

1. **Visual Feature Extraction (CNN):** 
   - 5 Convolutional layers paired with Max Pooling.
   - Robustly extracts features like curves and edges while handling noise and occlusion.
   - Compresses the vertical dimension to 1 so the sequence can be read left-to-right.

2. **Sequential Learning (BiGRU):** 
   - A Bidirectional Gated Recurrent Unit (BiGRU).
   - Reads the visual features contextually (left-to-right and right-to-left) to resolve ambiguous or heavily overlapping characters.

3. **Loss & Alignment (CTC Loss):** 
   - Uses Connectionist Temporal Classification (CTC) loss to calculate the error between raw model predictions and the target sequence without requiring strict character bounding boxes.

4. **Evaluation Metric:** 
   - Model performance is measured using **Character Error Rate (CER)** based on the Levenshtein Distance (minimum insertions, deletions, and substitutions).

## Repository Contents

- `submission_notebook.ipynb`: The fully documented Jupyter Notebook containing Exploratory Data Analysis, preprocessing, model definition, training loop, evaluation, and inference.
- `best_model.pth`: The saved weights of the best-performing model checkpoint during training.
- `submission.csv`: The finalized model predictions generated on the test dataset.
- `train-labels.csv`: The true labels for the training sequences.

## How to Run

Ensure you have the required dependencies installed:
```bash
pip install torch torchvision numpy pandas pillow jupyter
```

1. Place the `train_images/` and `test_images/` directories in the root folder (ignored by git due to size).
2. Run the `submission_notebook.ipynb` top-to-bottom to reproduce the preprocessing, or use the pre-trained `best_model.pth` to jump straight to testing.
