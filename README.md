## Malaria Detection: Hybrid Deep Learning Approach

## 📌 Project Overview

This project implements a Multi-Modal Hybrid Neural Network to identify Malaria parasites (Trophozoites) and White Blood Cells (WBCs) in blood smear images. Unlike standard models that only look at pixels, this architecture combines Computer Vision with Geometric Spatial Analysis to achieve higher diagnostic accuracy.

## Key Achievements:

Validation Accuracy: ~93-96%

Architecture: Dual-input (CNN + Dense Spatial Branch)

Efficiency: Utilizes custom Sequence Data Generators for low-RAM environments.

## 🧬 The Architecture

The model processes two distinct types of data simultaneously:

Image Branch: A Convolutional Neural Network (CNN) that extracts visual features from $224 \times 224$ RGB blood smear patches.

Spatial Branch: A Multi-Layer Perceptron (MLP) that processes geometric metadata including:

x_center, y_center (Location)

width, height (Bounding box dimensions)

area (Size of the detected object)

These branches are merged via a Concatenate layer before passing through the final classification head.

## 🛠️ Tech StackLanguage: 

Python 3.10+

Deep Learning: TensorFlow / Keras

Data Handling: Pandas, NumPy

Image Processing: OpenCV (cv2)

Visualization: Matplotlib, Seaborn

## 🚀 How to Run

1. Installation

Bash

pip install tensorflow pandas opencv-python matplotlib seaborn scikit-learn

2. Training the Model

Run the main script or notebook. The project uses a custom MalariaHybridGenerator to handle data streaming from your local directories:

Python

### Initialize Generators
train_gen = MalariaHybridGenerator(train_df, image_dir, batch_size=32)

### Train with Callbacks
history = model.fit(train_gen, epochs=25, callbacks=[early_stopping, checkpoint])

3. Inference / Prediction

To predict a single image using the hybrid model:

Python

from model_utils import predict_malaria
result, conf = predict_malaria("path/to/cell_image.jpg", x, y, w, h, area)
print(f"Result: {result} ({conf}%)")

## 📊 Performance Metrics

The model was evaluated using:

Confusion Matrix: Used to ensure high recall for Trophozoites (minimizing False Negatives).

ROC-AUC Curve: Measuring the separation capability between Parasites, WBCs, and Negatives.

Early Stopping: Training was regulated to prevent overfitting once validation loss stabilized.

## 📝 Author

Muhammad Abdulkareem

Malaria Hybrid Classification 

ProjectStatus: Trained & Validated
