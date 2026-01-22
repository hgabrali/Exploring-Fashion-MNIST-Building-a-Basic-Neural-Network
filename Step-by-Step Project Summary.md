## Step-by-Step Project Summary 

This repository  follows these specific steps using the **Fashion MNIST** dataset to build an end-to-end image classification pipeline:

1.  **Data Loading :** We fetch the ready-to-use dataset using the `tfds.load` function from the **TensorFlow Datasets** library. This allows us to easily access the raw image data and labels.

2.  **Data Visualization :** We create plots (grafikler) and figures to understand the data's content and structure. Seeing the images helps us gain intuition about the features the model will learn.

3.  **Normalization:** We scale the pixel values (piksel değerleri) to a range between 0 and 1. This step is crucial for **Optimization** (Eğitimi Optimize Etme) as it helps the neural network converge faster and more effectively.

4.  **Model Construction:** We build the architecture by connecting various layers:
    * **Flattening :** Converting 2D images into 1D vectors.
    * **Activation Functions :** Using **ReLU** for hidden layers and **Softmax** for the output layer to handle multi-class classification.

5.  **Model Training :** We train the model using the **Adam Optimizer** (Adam Optimizasyonu) for a total of 10 **Epochs** (Dönem). During this phase, the model learns to associate visual patterns with clothing categories.

6.  **Model Evaluation:** We measure the **Accuracy** (Başarı Oranı) of the model on the **Test Set** (Test Verisi) to ensure it can generalize well to new, unseen images.
