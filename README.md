# Stroke-Classifier

This project is a simple yet effective demonstration of a Machine Learning pipeline using Scikit-learn. The main focus of this project is to showcase the use of pipelines in preprocessing data and integrating it with a model, in this case, a Random Forest Classifier.

## Dataset

The dataset used in this project is the Stroke Prediction Dataset from Kaggle. You can download it from [here](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset). After downloading, please place the `healthcare-dataset-stroke-data.csv` file in the same directory as the `main.py` script.

## Project Structure

The project is structured into two main sections:

### 1. Preprocessing

The preprocessing pipeline is designed to apply transformations only to the specified numeric features. This allows us to keep other features (for example, categorical ones) intact during the preprocessing phase.

### 2. Model Choice and Training

The model chosen for this project is a Random Forest Classifier. The classifier is part of the pipeline and is applied to the preprocessed input features.

### 3. Model Assessment

The model's performance is assessed using cross-validation. The accuracy of the model is calculated and displayed. In addition to accuracy, other metrics are also calculated using a classification report.

## Usage

To run the project, simply execute the `main.py` script. The script will preprocess the data, train the model, and print out the model's performance metrics.