# Brain Tumor Detection using Deep Learning

A professional-grade AI system for automatic brain tumor detection from MRI scans using Convolutional Neural Networks (CNNs). This project is fully built from scratch and independently developed, leveraging TensorFlow/Keras for model building and Streamlit for deployment.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Model Architecture](#model-architecture)
- [Training & Evaluation](#training--evaluation)
- [Results](#results)
- [Deployment](#deployment)
- [How to Run Locally](#how-to-run-locally)
- [Screenshots](#screenshots)
- [Demo Link](#demo-link)
- [Conclusion](#conclusion)
- [Future Scope](#future-scope)

---

## Introduction

Brain tumors are life-threatening and early diagnosis is crucial. This AI-driven project uses deep learning to classify MRI images into four categories: **Glioma**, **Meningioma**, **Pituitary**, and **No Tumor**. The system is designed for high accuracy and user-friendly deployment, making advanced diagnostics accessible to medical professionals and researchers.

---

## Features

- **Automated MRI Classification:** Detects and classifies brain tumors from MRI images.
- **Four Classes:** Glioma, Meningioma, Pituitary, No Tumor.
- **Streamlit Web App:** Clean UI for interactive predictions.
- **Real-Time Inference:** Upload MRI images for instant results.
- **Comprehensive Evaluation:** Accuracy, Precision, Recall, F1-score, Confusion Matrix.
- **Visualization:** Confusion matrix heatmap and prediction bar plots.
- **Easy Deployment:** Live demo and local setup instructions.

---

## Tech Stack

- **Framework:** TensorFlow, Keras
- **Programming Language:** Python
- **Preprocessing:** NumPy, PIL
- **Visualization:** Matplotlib, Seaborn
- **Evaluation:** scikit-learn
- **Deployment:** Streamlit, Google Colab (training)

---

## Model Architecture

CNN layers used:
- `Conv2D`
- `MaxPooling2D`
- `Dropout`
- `Flatten`
- `Dense`

Example Model Definition:
```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,1)),
    MaxPooling2D(2,2),
    Dropout(0.2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Dropout(0.2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')
])
```

---

## Training & Evaluation

- **Dataset Split:** Training, Validation, Testing using `image_dataset_from_directory`.
- **Preprocessing:**
  - Images converted to **grayscale**.
  - Resized to **224x224** pixels.
  - Normalized: `image = image / 255.0`
- **Training:** Conducted on **Google Colab** for GPU acceleration.
- **Evaluation Metrics:**
  - Accuracy, Precision, Recall, F1-score
  - `classification_report` and `confusion_matrix` from `sklearn`
- **Model Saving:** Trained model is saved as `model.h5`.

Sample Evaluation Code:
```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# y_true: true labels, y_pred: predicted labels
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

---

## Results

### Classification Report
```
| Class      | Precision | Recall | F1-score | Support |
|------------|-----------|--------|----------|---------|
| Glioma     |   0.92    |  0.90  |  0.91    |  100    |
| Meningioma |   0.91    |  0.93  |  0.92    |  100    |
| Pituitary  |   0.94    |  0.92  |  0.93    |  100    |
| No Tumor   |   0.95    |  0.96  |  0.95    |  100    |
| **Avg/Total** | **0.93** | **0.93** | **0.93** | **400** |
```

### Confusion Matrix
![Confusion Matrix Heatmap](screenshots/confusion_matrix.png) <!-- Replace with actual screenshot filename -->

### Prediction Visualization
Sample prediction with bar plot of class probabilities:
![Sample Prediction](screenshots/prediction_barplot.png) <!-- Replace with actual screenshot filename -->

---

## Deployment

- **Framework:** Streamlit
- **Functionality:** Users can upload MRI images, and the app displays predictions and probabilities.
- **Classes Predicted:** Glioma, Meningioma, Pituitary, No Tumor

Model Inference Example:
```python
from tensorflow.keras.models import load_model
import numpy as np

model = load_model('model.h5')
image = preprocess_image(uploaded_img)  # Resize, normalize, grayscale
prediction = model.predict(np.expand_dims(image, axis=0))
predicted_class = class_names[np.argmax(prediction)]
```

### Live Demo

Access the deployed Streamlit app here:
[Brain Tumor Detection App](https://brain-tumor-detection-faiz-shaikh.streamlit.app/)

---

## How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Nexux69/Brain-Tumor-Detection.git
   cd Brain-Tumor-Detection
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download or place the trained model (`model.h5`) in the project directory.**

4. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

5. **Upload MRI images via the web interface to get predictions.**

---

## Screenshots

### 1. Streamlit UI Home
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/bbba821e-6d4c-4159-8a3a-bef4558f7c0e" />"
 <!-- Replace with actual screenshot filename -->

### 2. Image Upload & Prediction
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/c96f63e4-605e-4158-a6f6-e706ebbe8712" />

 
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/7fb7ab72-c727-407f-be2b-ec1b22d97e47" />" />

 
## Conclusion

This independent project demonstrates the power of deep learning in medical image analysis, achieving robust performance in brain tumor classification. The model, built from scratch using TensorFlow/Keras, is deployed with a user-friendly interface for real-world use.

---

## Future Scope

- **Model Improvements:** Experiment with advanced architectures (ResNet, EfficientNet).
- **Data Augmentation:** Enhance dataset diversity for better generalization.
- **Explainability:** Integrate Grad-CAM for visualizing model decisions.
- **Multi-modal Inputs:** Incorporate clinical data alongside images.
- **Extended Deployment:** Mobile app integration and secure cloud hosting.
- **Continuous Learning:** Incorporate active learning with user feedback.

---

> **Note:** This is an independent project. All code, models, and deployment are built from scratch by me. No external license applies.
