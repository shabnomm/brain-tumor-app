# Brain Tumor Classifier

This project is a deep learning-based web app that classifies brain tumor types from MRI images using a Convolutional Neural Network (CNN) built with TensorFlow and Keras. The app is built using Streamlit for an easy-to-use interface.

## Features
- Upload an MRI image (jpg, jpeg, png)
- Predicts tumor type: Glioma, Meningioma, No Tumor, Pituitary
- Simple and interactive UI using Streamlit

## Project Structure
- `train_model.py` — Script to train the CNN model on brain tumor MRI dataset
- `brain_tumor.py` — Streamlit app for tumor classification
- `brain_tumor_model.keras` — Trained model file
- `Training/` and `Testing/` — Dataset folders with MRI images (if included)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/brain-tumor.git
   cd brain-tumor
