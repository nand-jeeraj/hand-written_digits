# Handwritten Digit Recognition using Machine Learning

This project involves building a machine learning model to recognize handwritten digits (0–9) from image data. The model is trained using the popular **MNIST** dataset and demonstrates how classification algorithms can be applied to image recognition tasks.

---

## Objective

The goal is to develop an image classification model that can accurately recognize handwritten digits from pixel-based image inputs using traditional machine learning techniques.

---

## Dataset

- **Name**: MNIST Handwritten Digit Dataset  
- **Source**: Available via `sklearn.datasets` or `keras.datasets`
- **Details**:
  - 70,000 grayscale images (60,000 training, 10,000 testing)
  - Each image is 28x28 pixels representing digits 0 through 9

---

## Tools & Technologies

| Technology     | Purpose                          |
|----------------|----------------------------------|
| Python         | Programming language             |
| scikit-learn   | ML model training and evaluation |
| pandas, numpy  | Data manipulation                |
| matplotlib     | Visualization                    |
| seaborn        | Plotting confusion matrix, graphs|

---

## Model Used

- **K-Nearest Neighbors (KNN)**
  - Instance-based learning algorithm
  - Predicts based on majority class among k nearest neighbors
- Optional: Logistic Regression, SVM, or Neural Network (can be added)

---

## Workflow

1. Load and preprocess the dataset
2. Flatten the images (28x28 → 784 features)
3. Split the data into training and test sets
4. Train the KNN model
5. Evaluate using accuracy score and confusion matrix
6. 6. Visualize predictions and results

---

## How to Run bash
# Clone the repository
git clone https://github.com/nand-jeeraj/hand-written_digits/
cd handwritten-digit-recognition




Confusion matrix and misclassified digits visualized

Sample digit predictions plotted

Results
High accuracy achieved with minimal preprocessing

Model performs well on unseen test images

