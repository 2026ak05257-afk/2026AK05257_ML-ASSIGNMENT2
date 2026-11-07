# Breast Cancer Classification - Machine Assignment 2
Name: Hariharan S  |  ID: 2026AK05257  |  Course: Machine Learning (M.Tech AIML)

# Breast Cancer Classification – Machine Learning Assignment 2

**Student Name:** Hariharan S  
**BITS ID:** 2026AK05257  
**Programme:** M.Tech (AIML)  
**Course:** Machine Learning

This project implements and compares five machine learning classification algorithms on the Breast Cancer Wisconsin (Diagnostic) dataset. The trained models are saved and used in an interactive Streamlit application to evaluate uploaded test data without retraining.

## 1. Problem Statement

The objective of this assignment is to develop and compare five machine learning classification algorithms to predict whether a breast tumour is malignant or benign.

The five classification algorithms implemented using the same dataset are:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

The models are evaluated using six metrics: Accuracy, AUC Score, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC). An interactive Streamlit application is also developed and deployed to allow users to upload test data, select a classification model, and view its evaluation results.

## 2. Dataset Description

**Dataset Name:** Breast Cancer Wisconsin (Diagnostic) Dataset  
**Dataset Source Used in Implementation:** Scikit-learn `load_breast_cancer()`  
**Dataset Type:** Binary classification  
**Target Variable:** Diagnosis (Malignant / Benign)

The dataset contains numerical features extracted from digitized images of fine needle aspirates (FNA) of breast masses. These features describe characteristics such as radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension.

| Dataset detail | Description |
|---|---:|
| Total instances | 569 |
| Number of input features | 30 |
| Number of target classes | 2 |
| Training samples | 455 |
| Testing samples | 114 |
| Train-test split | 80% training / 20% testing |
| Random state | 42 |
| Feature scaling | StandardScaler |

The dataset was divided into training and testing sets using stratified sampling. `StandardScaler` was fitted on the training data and then applied to both the training and testing data. The same prepared dataset split was used to evaluate all five classifiers.

## 3. GitHub Repository Link

**GitHub Repository:**  
https://github.com/2026ak05257-afk/2026AK05257_ML-ASSIGNMENT2

### Project Folder Structure

```text
2026AK05257_ML-ASSIGNMENT2/
│
├── app.py
│
├── requirements.txt
│
├── README.md
│
├── test_data.csv
│
├── model/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── support_vector_machine.pkl
│   └── scaler.pkl
│
└── 2026AK05257_ML_Assignment_2_Complete_Code.ipynb
```

### File Description

| File / folder | Description |
|---|---|
| `app.py` | Streamlit application source code |
| `requirements.txt` | Python libraries and dependencies required to run the project |
| `README.md` | Project documentation, model comparison, and observations |
| `test_data.csv` | Test dataset uploaded to the Streamlit application for evaluation |
| `model/` | Saved trained classification models and the `StandardScaler` preprocessing artifact |
| `2026AK05257_ML_Assignment_2_Complete_Code.ipynb` | Complete implementation, training, and evaluation of all five classification models |

**Note:** The folder and file names shown above should match the actual names in the GitHub repository. The scaler file is included because it is required to apply the same preprocessing during app-based evaluation.

## 4. Models Used and Performance Comparison

Five classification algorithms were implemented using the same training and testing dataset. Each model was evaluated using the following metrics:

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
| Naive Bayes (Gaussian) | 0.9298 | 0.9868 | 0.9298 | 0.9298 | 0.9298 | 0.8492 |
| Support Vector Machine | 0.9825 | 0.9950 | 0.9825 | 0.9825 | 0.9825 | 0.9623 |

*Note: The values above are reproduced from the model comparison results in the project notebook and are rounded to four decimal places.*

### Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Achieved 98.25% accuracy and the highest AUC score of 0.9954. It also achieved 0.9825 for Precision, Recall, and F1 Score, and 0.9623 MCC. It provided the strongest overall results among the evaluated models. |
| Decision Tree | Achieved 91.23% accuracy and an AUC score of 0.9157. It showed comparatively lower performance than the other four models on this test set. |
| KNN | Achieved 95.61% accuracy and an AUC score of 0.9788. It demonstrated good classification performance, although its results were below Logistic Regression and SVM. |
| Naive Bayes (Gaussian) | Achieved 92.98% accuracy and an AUC score of 0.9868. Its AUC score was relatively high, while its accuracy and MCC were lower than those of Logistic Regression and SVM. |
| Support Vector Machine | Achieved 98.25% accuracy and an AUC score of 0.9950. Its Precision, Recall, F1 Score, and MCC matched Logistic Regression, while its AUC score was slightly lower. |
| **Overall Winner for your dataset** | **Logistic Regression.** It achieved the highest AUC score (0.9954) and tied with SVM for the highest accuracy (0.9825), Precision, Recall, F1 Score, and MCC. Based on these evaluation results, Logistic Regression is selected as the overall winner. |

## 5. Streamlit Web Application

**Live Streamlit Application:**  
https://2026ak05257-afk-2026ak05257-ml-assignment2-app-lcyrkm.streamlit.app/

The interactive web application is developed using Streamlit and deployed through Streamlit Community Cloud.

### Features Implemented

1. **Dataset Upload:** Allows users to upload a test dataset in CSV format. The uploaded test CSV must contain all required input feature columns and the target column so that evaluation metrics can be calculated.
2. **Model Selection:** Provides a dropdown to select any one of the five trained classification models.
3. **Evaluation Metrics:** Displays Accuracy, AUC Score, Precision, Recall, F1 Score, and MCC for the selected model.
4. **Confusion Matrix:** Displays a confusion matrix to visualize the classification performance of the selected model.
5. **Classification Report:** Displays the classification report for the selected model.

The application loads the saved trained model and preprocessing artifact, applies the saved preprocessing transformation to the uploaded test data, and calculates the evaluation metrics. The models are not retrained on the uploaded test data.

## 6. Conclusion

This assignment demonstrates the implementation, evaluation, comparison, and deployment of five machine learning classification algorithms using the Breast Cancer Wisconsin (Diagnostic) dataset.

Among the five models, Logistic Regression achieved the best overall performance, with 98.25% accuracy and an AUC score of 0.9954. It tied with SVM on accuracy and the other reported classification metrics, while achieving a slightly higher AUC score.

The project demonstrates an end-to-end machine learning workflow, from dataset preparation and model training to performance evaluation and deployment of an interactive Streamlit application that evaluates uploaded test data without retraining the models.


