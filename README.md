# Breast Cancer Classification - Machine Assignment 2
Name: Hariharan S  |  ID: 2026AK05257  |  Course: Machine Learning (M.Tech AIML)

An end-to-end classification project: five ML models trained on the Breast Cancer Wisconsin 
dataset, served through an interactive Streamlit app that scores
user-uploaded test data (no retraining in the app).


## 1. Problem Statement

The objective of this assignment is to develop and compare five machine learning classification algorithms to predict whether a breast tumour is malignant or benign.

The project implements Logistic Regression, Decision Tree, K-Nearest Neighbors (KNN), Gaussian Naive Bayes, and Support Vector Machine (SVM) using the same dataset.

The models are evaluated using Accuracy, AUC Score, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC). An interactive Streamlit application is also developed and deployed to allow users to upload test data, select a classification model, and view its evaluation results.

## 2. Dataset Description

**Dataset Name:** Breast Cancer Wisconsin (Diagnostic) Dataset

**Dataset Source:** Scikit-learn – `load_breast_cancer()`

The Breast Cancer Wisconsin Diagnostic dataset is used for binary classification to identify whether a breast tumour is malignant or benign.

- Total Instances: 569
- Total Input Features: 30
- Target Variable: Diagnosis (Malignant / Benign)
- Classification Type: Binary Classification
- Training Samples: 455
- Testing Samples: 114
- Train-Test Split: 80% Training and 20% Testing
- Random State: 42
- Feature Scaling: StandardScaler

The dataset contains numerical features extracted from digitized images of fine needle aspirates (FNA) of breast masses. These features describe characteristics such as radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension.

The dataset was divided into training and testing sets using stratified sampling. StandardScaler was fitted on the training data and applied to both training and testing data.

## 3. GitHub Repository Link

**GitHub Repository:**  
https://github.com/2026ak05257-afk/2026AK05257_ML-ASSIGNMENT2

The repository contains the following files and folders:

- `app.py` – Streamlit application source code.
- `requirements.txt` – Required Python libraries and dependencies.
- `README.md` – Project documentation and model comparison.
- `test_data.csv` – Test dataset used for model evaluation.
- `model/` – Saved trained classification models and StandardScaler.
- `2026AK05257 ML_Assignment_2 Complete Code.ipynb` – Complete implementation, training, and evaluation of all five models.

## 4. Models Used and Performance Comparison

Five classification algorithms were implemented using the same training and testing dataset.

The following evaluation metrics were calculated for each model:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

### Model Performance Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9825 | 0.9825 | 0.9825 | 0.9623 |
| Decision Tree | 0.9123 | 0.9157 | 0.9161 | 0.9123 | 0.9130 | 0.8174 |
| KNN | 0.9561 | 0.9788 | 0.9561 | 0.9561 | 0.9560 | 0.9054 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9298 | 0.9298 | 0.9298 | 0.8492 |
| Support Vector Machine | 0.9825 | 0.9950 | 0.9825 | 0.9825 | 0.9825 | 0.9623 |

*Note: All values are based on the model evaluation results obtained from the test dataset. Metrics are rounded to four decimal places.*

### Observations on Model Performance

**1. Logistic Regression**

Logistic Regression achieved an accuracy of 98.25% and the highest AUC score of 0.9954. It also achieved strong Precision, Recall, F1 Score, and MCC values, making it the best overall-performing model.

**2. Decision Tree Classifier**

Decision Tree achieved an accuracy of 91.23%. It performed comparatively lower than the other models across most evaluation metrics.

**3. K-Nearest Neighbors (KNN)**

KNN achieved an accuracy of 95.61% and an AUC score of 0.9788. It showed good classification performance but was slightly below Logistic Regression and SVM.

**4. Gaussian Naive Bayes**

Naive Bayes achieved an accuracy of 92.98% and an AUC score of 0.9868. Although its AUC score was high, its overall accuracy and MCC were lower than those of Logistic Regression and SVM.

**5. Support Vector Machine (SVM)**

SVM achieved an accuracy of 98.25%, with strong Precision, Recall, F1 Score, and MCC values. Its performance was very close to Logistic Regression, with a slightly lower AUC score of 0.9950.

### Overall Winner for the Dataset

**Logistic Regression**

Logistic Regression is selected as the overall winner because it achieved the highest AUC score of 0.9954 while maintaining the joint-highest accuracy of 98.25%. It also achieved the same highest Precision, Recall, F1 Score, and MCC values as SVM.

Therefore, Logistic Regression demonstrated the best overall performance on the selected dataset.

## 5. Streamlit Web Application

**Live Streamlit Application:**  
https://2026ak05257-afk-2026ak05257-ml-assignment2-app-lcyrkm.streamlit.app/

The interactive web application is developed using Streamlit and deployed through Streamlit Community Cloud.

### Features Implemented

1. **Dataset Upload:** Allows users to upload a test dataset in CSV format.
2. **Model Selection:** Provides a dropdown to select any one of the five trained classification models.
3. **Evaluation Metrics:** Displays Accuracy, AUC Score, Precision, Recall, F1 Score, and MCC.
4. **Confusion Matrix:** Displays a confusion matrix to visualize classification performance.
5. **Classification Report:** Displays the classification report for the selected model.

The application loads the previously saved trained models and StandardScaler. It applies the saved preprocessing transformation to the uploaded test data and calculates evaluation metrics without retraining the models.

### Technologies and Libraries Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Seaborn
- Joblib

## 6. Conclusion

This assignment demonstrates the implementation, evaluation, comparison, and deployment of five machine learning classification algorithms using the Breast Cancer Wisconsin Diagnostic dataset.

Among the five models, Logistic Regression achieved the best overall performance with 98.25% accuracy and an AUC score of 0.9954.

The project also demonstrates an end-to-end machine learning workflow, from dataset preparation and model training to performance evaluation and deployment of an interactive Streamlit application.

