# Exploring-Fashion-MNIST-Building-a-Basic-Neural-Network
This repository contains a comprehensive, end-to-end pipeline for classifying fashion items using the Fashion MNIST dataset. The project demonstrates the fundamental steps of a Computer Vision (CV) workflow, from data ingestion to model evaluation using TensorFlow and Keras.

# Case Study: Fashion MNIST Image Classification using Artificial Neural Networks (ANN)

## Project Overview
The goal of this project is to build a basic **Artificial Neural Network (ANN)** that can accurately classify $28 \times 28$ grayscale images into one of 10 fashion categories. This serves as a foundational exercise for understanding how image data is structured and processed in deep learning.

---

##  Dataset: Fashion MNIST
The dataset consists of **70,000 images** (60,000 for training and 10,000 for testing). Each image is a $28 \times 28$ grayscale image associated with a label from 10 distinct classes:

| Label | Description | Label | Description |
| :--- | :--- | :--- | :--- |
| 0 | T-shirt/top | 5 | Sandal |
| 1 | Trouser | 6 | Shirt |
| 2 | Pullover | 7 | Sneaker |
| 3 | Dress | 8 | Bag |
| 4 | Coat | 9 | Ankle boot |

---

##  Project Pipeline

### 1.  Loading and Understanding the Data
We use the `tensorflow_datasets` (tfds) library to load the data efficiently.
* **Split:** The data is partitioned into 'train' and 'test' sets.
* **Supervised Learning:** Data is loaded as (image, label) pairs.
* **Metadata:** Information such as class names and image dimensions are extracted using `with_info=True`.

### 2.  Data Visualization
Visualizing the data is crucial for intuition. Using `matplotlib`, we plot sample images with their corresponding labels to understand what the model will "see."

### 3.  Preprocessing (Normalization)
Neural networks converge faster when input values are on a similar scale. We normalize the pixel values from the range $[0, 255]$ to $[0, 1]$ using the following transformation:

$$x_{normalized} = \frac{x}{255.0}$$

### 4.  Model Architecture
We constructed a **Sequential** model with the following layers:

* **Flatten Layer:** Converts the $28 \times 28$ 2D image into a 1D vector of 784 pixels.
* **Hidden Layer (Dense):** 128 neurons with **ReLU** (Rectified Linear Unit) activation to learn non-linear patterns.
* **Output Layer (Dense):** 10 neurons (one for each class) with **Softmax** activation to output probability distributions.

!

### 5. Compiling and Training
The model is compiled with the following configurations:
* **Optimizer:** Adam (adaptive moment estimation).
* **Loss Function:** `sparse_categorical_crossentropy` (ideal for multi-class classification).
* **Metric:** Accuracy.

**Training Details:**
* **Batch Size:** 32 (data is fed in small groups for efficiency).
* **Epochs:** 10 (the model iterates over the entire dataset 10 times).

### 6. Evaluation
The final performance is measured on the unseen test set to check for generalization. In this basic setup, the model achieves approximately **82.64% accuracy**.

---

##  Requirements
To replicate this project, the following dependencies are required:
* Python 3.x
* TensorFlow
* TensorFlow Datasets
* Matplotlib
* NumPy

---

##  Future Improvements
While this basic model provides a solid foundation, further improvements can be made by:
* Adding more Dense layers or increasing neuron counts.
* Implementing **Convolutional Neural Networks (CNNs)** for better spatial feature extraction.
* Using **Dropout** layers to prevent overfitting.

---


