# Handwritten Digit Recognition using KNN and PCA (Flask Deployment with MNIST)

This project is a web application that recognizes handwritten digits using a machine learning model built with K-Nearest Neighbors (KNN) and Principal Component Analysis (PCA). The model is trained on the MNIST dataset and deployed using the Flask web framework.

## Project Overview

The goal is to classify digits (0–9) from 28x28 pixel grayscale images using traditional machine learning techniques and provide predictions through a web interface built with Flask.

## Technologies Used

- Python
- Flask
- scikit-learn
- numpy
- pandas
- matplotlib
- pickle
- HTML/CSS (Jinja2 templates)

## Dataset

- **Dataset:** [MNIST](http://yann.lecun.com/exdb/mnist/)
- 60,000 training images and 10,000 test images
- Each image is 28x28 pixels in grayscale, flattened into a 784-dimensional vector for processing

## ML Workflow

1. Load and preprocess MNIST dataset
   - Normalize pixel values
   - Flatten 28x28 images into 1D vectors (784 features)
2. Reduce dimensions using PCA (e.g., to 50 components)
3. Train a KNN classifier
4. Evaluate model on test data
5. Save the PCA and KNN models using pickle
6. Integrate models into Flask app for prediction

## Features

- Upload or draw a digit image for real-time prediction
- Uses dimensionality reduction (PCA) for faster inference
- Simple, interactive web interface powered by Flask

## Results

- Achieved accuracy of around 96–98% using KNN after PCA
- Fast prediction due to reduced feature dimensions

## Key Learning Points

- How to use PCA for high-dimensional image data
- Training and saving ML models using pickle
- Creating a full-stack machine learning application with Flask
