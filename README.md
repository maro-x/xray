# XRay Classification

## Overview
This project implements a **chest X-ray classification system** that detects whether a patient has **Pneumonia** or is **Normal**.
The project uses deep learning models like **ResNet18, ResNet34, and MobileNetV2** and is deployed via a **FastAPI** server for inference.

The dataset is preprocessed by resizing all images to **128x128** and balancing the number of samples in each class.

---

## Features
- Resize and preprocess X-ray images.
- Train deep learning models for classification.
- Save trained models and label encoders for inference.
- Provide an API using **FastAPI** to predict classes from images.
- Track training metrics with **MLflow**.

---

## Dataset
- Dataset contains images of normal and pneumonia cases split into training, validation, and test sets.
- All images are resized to 128x128 pixels and stored in a separate folder.
- A CSV file maps each image to its class and split.

---

## Requirements
Python 3.8+ with the following libraries:
- torch
- torchvision
- pandas
- Pillow
- scikit-learn
- fastapi
- uvicorn
- mlflow

All dependencies can be installed using `pip` and a `requirements.txt` file.

---

## Training
The project supports training the models on the preprocessed dataset. Training metrics such as loss are tracked using MLflow. Trained models are saved for later use in inference.

---

## Inference
Trained models can be used to classify new X-ray images. The system supports multiple models, and predictions return the class label of the image (Normal or Pneumonia).

---

## FastAPI Deployment
The project includes a FastAPI application for serving the trained models. The API accepts images in base64 format and returns predictions from the selected model. This allows easy integration into web or mobile applications.

---

## Project Structure
