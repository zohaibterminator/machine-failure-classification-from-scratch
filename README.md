# Machine Failure Prediction

This project aims to predict machine failures using logistic regression. The dataset is downloaded from Kaggle, and it contains sensor readings collected from various machines. The notebook includes data preprocessing, feature scaling, model training from scratch, evaluation, and visualization of results. Machine Failure prediction is really important for preventing uneccessary maintanence cost, helping in scheduling maintenance activities during non-operational hours, and ensuring peak productivity. Involving a ML model like this one for failure prediction can help ensure early detection of failure and utilize the data coming from the sensors in real-time.

Link to the dataset: https://www.kaggle.com/datasets/umerrtx/machine-failure-prediction-using-sensor-data

Link to my Kaggle notebook: https://www.kaggle.com/code/specialterminator69/regularized-logistic-regression-from-scratch

## Overview

The notebook contains the following key steps:
1. Data Import and Exploration
2. Data Preprocessing
3. Handling Imbalanced Dataset
4. Feature Scaling
5. Data Splitting
6. Model Training
7. Model Evaluation
8. Visualization of Results

## Prerequisites

To run the code in this notebook, you need to have the libraries mentioned in the requirements.txt file installed. Put the file in the same directory you are working in, and run the following command in the termial.

```bash
pip install -r requirements.txt
```

## Setup
* Clone this repository or download the notebook file.
* Ensure you have the required libraries installed.
* Place the data.csv file in the same directory as the notebook.

## Steps
1. Data Import and Exploration:

The data is imported using pandas, and basic exploration is conducted to understand the structure, data types, and summary statistics. Duplicate rows are identified and counted.

2. Data Preprocessing:

Any duplicate rows found in the dataset are dropped to ensure data quality.

3. Handling Imbalanced Dataset:

The dataset is checked for class imbalance in the target variable. SMOTE (Synthetic Minority Over-sampling Technique) is used to balance the dataset by oversampling the minority class.

4. Feature Scaling:

The features are scaled using MinMaxScaler to bring all values into the range between 0 and 1, which helps improve the performance of the machine learning model.

5. Data Splitting:

The dataset is split into training and testing sets. A 70:30 split is used to ensure a proper evaluation of the model.

6. Model Training:

A regularized logistic regression model is trained from scratch using gradient descent. Functions for computing cost, gradients, and applying gradient descent are defined and used to optimize the model parameters.

7. Model Evaluation:

The model is evaluated using various metrics such as accuracy, precision, recall, and F1 score. A confusion matrix and classification report are generated to provide detailed insights into the model's performance on each class.

8. Visualization of Results:

Confusion matrix and evaluation scores are visualized using matplotlib and seaborn. Bar plots are used to display accuracy, precision, recall, and F1 score for easy comparison.

## Results
The model showed exceptional performance in both classes, obtaining 92% accuracy, recall, precision and f1-score. The confusion matrix showed few examples that were incorrectly labelled, but overall, showed great performance.

## Conclusion
This notebook provides a comprehensive approach to predicting machine failures using logistic regression implemented from scratch. Follow the steps to preprocess data, train the model, evaluate its performance, and visualize the results.
