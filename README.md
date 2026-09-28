# Intel Image Classification using CNN

A Convolutional Neural Network (CNN) built from scratch using TensorFlow/Keras to classify natural scene images into six different categories from the **Intel Image Classification dataset**.

## 📌 Project Overview

This project demonstrates an end-to-end image classification workflow using a CNN.

The model learns visual features from images and classifies them into one of six scene categories:

* 🏢 Buildings
* 🌲 Forest
* 🏔️ Glacier
* ⛰️ Mountain
* 🌊 Sea
* 🛣️ Street

The project covers dataset exploration, image preprocessing, data augmentation, CNN architecture design, model training, evaluation, and prediction on external images.

---

## 📂 Dataset

**Dataset:** Intel Image Classification

The dataset contains images belonging to six different natural and urban scene categories.

### Classes

```text
buildings
forest
glacier
mountain
sea
street
```

### Dataset Split

The original training dataset was divided into:

* **80% Training**
* **20% Validation**

A separate test dataset was used for final evaluation.

For this implementation:

| Dataset    | Images |
| ---------- | -----: |
| Training   | 11,228 |
| Validation |  2,806 |
| Test       |  3,000 |

Images are RGB with a resolution of **150 × 150 pixels**.

---

## 🛠️ Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Matplotlib
* Scikit-learn
* Google Colab / Kaggle

---

## 🔄 Project Workflow

```text
Dataset
   ↓
Dataset Exploration
   ↓
Train / Validation Split
   ↓
Image Normalization
   ↓
Data Augmentation
   ↓
CNN Model
   ↓
Model Training
   ↓
Validation
   ↓
Test Evaluation
   ↓
Classification Report
   ↓
Confusion Matrix
   ↓
External Image Prediction
```

---

## 🧹 Image Preprocessing

The images are already available at **150 × 150 RGB resolution**, so they are loaded using:

```python
image_size=(150, 150)
```

### Normalization

Pixel values originally range from:

```text
0 - 255
```

They are scaled to:

```text
0 - 1
```

using:

```python
tf.keras.layers.Rescaling(1./255)
```

### Data Augmentation

To improve the model's ability to generalize, augmentation is applied during training:

```python
tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomTranslation(0.1, 0.1)
])
```

The augmentation is applied only during training and is not applied to validation or test images during inference.

---

## 🧠 CNN Architecture

The model is built from scratch using multiple convolutional blocks.

```text
Input Image
150 × 150 × 3
      ↓
Conv2D - 32 filters
      ↓
MaxPooling
      ↓
Conv2D - 64 filters
      ↓
MaxPooling
      ↓
Conv2D - 128 filters
      ↓
MaxPooling
      ↓
Conv2D - 256 filters
      ↓
MaxPooling
      ↓
Global Average Pooling
      ↓
Dense - 128
      ↓
Dropout
      ↓
Dense - 6
      ↓
Softmax
```

### Main Components

**Convolutional layers**

Extract visual features such as edges, textures, shapes, and higher-level patterns.

**MaxPooling**

Reduces the spatial dimensions of feature maps while retaining important features.

**Global Average Pooling**

Converts the feature maps into a compact representation before classification and reduces the number of parameters compared with using a large `Flatten()` layer.

**Dropout**

Helps reduce overfitting during training.

**Softmax**

Produces probabilities for the six possible classes.

---

## ⚙️ Model Compilation

The model is compiled using the Adam optimizer and sparse categorical cross-entropy loss.

```python
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

Sparse categorical cross-entropy is appropriate because the class labels are represented as integer class IDs.

---

## ⏹️ Early Stopping

Early stopping is used to prevent unnecessary training and reduce overfitting.

```python
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)
```

The training process monitors validation loss and restores the model weights from the epoch with the best validation performance.

---

## 📊 Model Performance

The trained CNN achieved approximately:

| Metric        |      Score |
| ------------- | ---------: |
| Test Accuracy | **84.60%** |
| Test Loss     |  **0.455** |

The model was evaluated on the separate test dataset that was not used during training.

> Performance can vary depending on the exact preprocessing, augmentation, architecture, training configuration, and random initialization.

---

## 📈 Evaluation

The model is evaluated using:

### Accuracy

Measures the overall percentage of correctly classified images.

### Precision

Measures how many images predicted as a particular class actually belong to that class.

### Recall

Measures how many images belonging to a class were correctly identified.

### F1-Score

Provides a combined measure of precision and recall.

A classification report is generated using Scikit-learn.

```python
from sklearn.metrics import classification_report

print(classification_report(
    y_true,
    y_pred,
    target_names=class_names
))
```

---

## 🔲 Confusion Matrix

A confusion matrix is used to understand which classes the model predicts correctly and which classes are commonly confused with each other.

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(xticks_rotation=45)
```

This is particularly useful for identifying visually similar categories such as **mountain, glacier, forest, and sea**.

---

## 🖼️ Test Image Visualization

The project also visualizes test images together with their:

```text
Actual Label
Predicted Label
```

Example:

```text
Actual: mountain
Predicted: mountain
```

This provides a visual way to inspect the model's predictions.

---

## 🌐 External Image Prediction

The trained model can also be used to classify an image outside the original dataset.

The external image is:

1. Loaded
2. Resized to `150 × 150`
3. Converted into an array
4. Given a batch dimension
5. Passed to the trained CNN

Example:

```python
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np

img = load_img(
    "my_image.jpg",
    target_size=(150, 150)
)

img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)

predicted_class = np.argmax(prediction[0])

print("Predicted class:", test.class_names[predicted_class])
```

Since normalization is already included inside the model, the external image does not need to be manually divided by `255`.

---

## 📁 Suggested Repository Structure

```text
Intel-Image-Classification-CNN/
│
├── README.md
├── CNN_Image_Classification.ipynb
│
├── results/
│   ├── accuracy_plot.png
│   ├── loss_plot.png
│   └── confusion_matrix.png
│
└── requirements.txt
```

The dataset itself does not need to be uploaded to the repository.

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Intel-Image-Classification-CNN.git
```

### 2. Install dependencies

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

### 3. Download the Intel Image Classification dataset

Download the dataset from Kaggle and place it in the appropriate directory.

### 4. Open the notebook

Run:

```text
CNN_Image_Classification.ipynb
```

The notebook contains the complete workflow from dataset loading to model evaluation and external image prediction.

---

## 🎯 Learning Outcomes

Through this project, I worked with:

* Image dataset exploration
* Train/validation/test splitting
* Image normalization
* Data augmentation
* Convolutional Neural Networks
* Convolution and pooling layers
* Feature extraction
* Dropout and regularization
* Softmax classification
* Model training and validation
* Early stopping
* Classification metrics
* Confusion matrix analysis
* External image prediction

---

## 🔮 Possible Improvements

Future improvements could include:

* Batch Normalization
* Learning-rate scheduling
* Hyperparameter tuning
* Deeper CNN architectures
* Transfer learning using pretrained models
* Deployment as a web application

---

## 👨‍💻 Author

**Harigovind P**

B.Tech Computer Science and Engineering

GitHub: [Haariiiii](https://github.com/Haariiiii)

LinkedIn: [Harigovind P](https://www.linkedin.com/in/harigovindp2004/)
