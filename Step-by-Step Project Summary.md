## Step-by-Step Project Summary 

This repository  follows these specific steps using the **Fashion MNIST** dataset to build an end-to-end image classification pipeline:

##  Dataset: Fashion MNIST
The dataset consists of **70,000 images** (60,000 for training and 10,000 for testing). Each image is a $28 \times 28$ grayscale image associated with a label from 10 distinct classes:

| Label | Description | Label | Description |
| :--- | :--- | :--- | :--- |
| 0 | T-shirt/top | 5 | Sandal |
| 1 | Trouser | 6 | Shirt |
| 2 | Pullover | 7 | Sneaker |
| 3 | Dress | 8 | Bag |
| 4 | Coat | 9 | Ankle boot 


<img width="744" height="402" alt="image" src="https://github.com/user-attachments/assets/75977bbe-0e4e-4db9-b045-775b466e5dd6" />

* Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total.
* Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker.
* This pixel-value is an integer between 0 and 255. Let’s now have a quick glance at the actual images from the dataset:

<img width="766" height="700" alt="image" src="https://github.com/user-attachments/assets/c0ba0718-30f5-43d9-b8c2-661e73f64507" />



# 1.  **Data Loading :** We fetch the ready-to-use dataset using the `tfds.load` function from the **TensorFlow Datasets** library. This allows us to easily access the raw image data and labels.

### Data Initialization and Dataset Partitioning

The following section provides a detailed breakdown of the output generated during the initial data handling phase of the project. This stage is critical for ensuring data integrity and optimizing the training workflow.

---

###  Data Retrieval and Storage
The initial execution confirms that the **Fashion MNIST** dataset has been successfully retrieved and stored within a specific local directory: 
`path: /root/tensorflow_datasets/...`.

* **Persistence:** By storing the dataset locally, the system ensures that all **subsequent calls** will load the data instantaneously from the disk.
* **Efficiency:** This eliminates the need for redundant downloads, significantly reducing latency and bandwidth consumption during iterative development cycles.

---

### Dataset Partitioning
The output validates the automatic partitioning of the data as defined in the source code. Proper segmentation is vital for unbiased model evaluation.

| Segment | Sample Count | Purpose |
| :--- | :--- | :--- |
| **Training Samples** | 60,000 | Used to optimize model weights and learn feature patterns. |
| **Test Samples** | 10,000 | Reserved as a "hold-out" set to evaluate generalization on unseen data. |

!

---

### Significance of this Phase
This stage serves as the structural foundation of the entire machine learning pipeline. It establishes a rigorous **Supervised Learning** framework where:

1.  **Feature-Label Pairing:** Every image is precisely mapped to its corresponding ground-truth label (e.g., an image of a boot is correctly associated with the integer label **'9'**).
2.  **Pipeline Readiness:** Ensuring the data is correctly partitioned and accessible is a prerequisite before proceeding to **Normalization**, **Feature Engineering**, and **Model Architecture** definition.
3.  **Experimental Integrity:** By separating the test set early, we ensure that the model evaluation remains objective and free from data leakage.

!






# 2.  **Data Visualization :** We create plots (grafikler) and figures to understand the data's content and structure. Seeing the images helps us gain intuition about the features the model will learn.

# 3.  **Normalization:** We scale the pixel values (piksel değerleri) to a range between 0 and 1. This step is crucial for **Optimization** (Eğitimi Optimize Etme) as it helps the neural network converge faster and more effectively.

# 4.  **Model Construction:** We build the architecture by connecting various layers:
    * **Flattening :** Converting 2D images into 1D vectors.
    * **Activation Functions :** Using **ReLU** for hidden layers and **Softmax** for the output layer to handle multi-class classification.

# 5.  **Model Training :** We train the model using the **Adam Optimizer** (Adam Optimizasyonu) for a total of 10 **Epochs** (Dönem). During this phase, the model learns to associate visual patterns with clothing categories.

# 6.  **Model Evaluation:** We measure the **Accuracy** (Başarı Oranı) of the model on the **Test Set** (Test Verisi) to ensure it can generalize well to new, unseen images.
