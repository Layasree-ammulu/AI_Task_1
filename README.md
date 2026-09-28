# Customer Churn Prediction Using Machine Learning

## 1. Project Overview

This project implements a supervised machine learning classification system to predict whether a telecom customer is likely to churn.

The project compares two machine learning algorithms:

* Logistic Regression
* Random Forest Classifier

The models are evaluated using Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

## 2. Objective

The main objective is to build and evaluate classification models that can predict customer churn based on customer demographic information, service usage, contract details, and billing information.

## 3. Dataset

The project uses the Telco Customer Churn dataset.

The dataset contains customer information such as:

* Gender
* Senior Citizen status
* Partner and dependent information
* Tenure
* Phone and internet services
* Online security and support services
* Contract type
* Payment method
* Monthly charges
* Total charges
* Churn status

The target variable is:

`Churn`

where:

* `Yes` = Customer churned
* `No` = Customer did not churn

## 4. Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

## 5. Machine Learning Models

### Logistic Regression

Logistic Regression is used as a baseline classification model for predicting whether a customer will churn.

### Random Forest

Random Forest is an ensemble learning algorithm that combines multiple decision trees to perform classification.

## 6. Data Preprocessing

The following preprocessing steps were performed:

1. Loaded the dataset using Pandas.
2. Converted the `TotalCharges` column into numeric format.
3. Removed rows with missing `TotalCharges` values.
4. Removed the `customerID` column because it is only an identifier.
5. Converted the `Churn` target into numerical values:

   * Yes → 1
   * No → 0
6. Identified numerical and categorical features.
7. Standardized numerical features using `StandardScaler`.
8. Converted categorical features into numerical representations using `OneHotEncoder`.

## 7. Train/Test Split

The dataset was divided into:

* 80% training data
* 20% testing data

Stratified splitting was used to maintain the class distribution.

## 8. Cross-Validation

5-fold Stratified Cross-Validation was performed on the training data.

Mean cross-validation accuracy:

| Model               | Mean CV Accuracy |
| ------------------- | ---------------: |
| Logistic Regression |           0.8025 |
| Random Forest       |           0.7904 |

## 9. Model Evaluation

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC

### Test Set Results

| Metric    | Logistic Regression | Random Forest |
| --------- | ------------------: | ------------: |
| Accuracy  |              0.8038 |        0.7896 |
| Precision |              0.6485 |        0.6345 |
| Recall    |              0.5722 |        0.4920 |
| F1-Score  |              0.6080 |        0.5542 |
| ROC-AUC   |              0.8359 |        0.8140 |

## 10. Visualizations

The project generates the following visualizations:

* Customer churn distribution
* Model performance comparison
* Logistic Regression confusion matrix
* Random Forest confusion matrix
* ROC curve comparison
* Random Forest feature importance

## 11. Important Features

The Random Forest model identified several features with relatively high importance, including:

* Total Charges
* Tenure
* Monthly Charges
* Contract type
* Online Security
* Payment Method
* Internet Service
* Tech Support

## 12. Conclusion

Both Logistic Regression and Random Forest were trained and evaluated for customer churn prediction.

The results show differences in their classification performance across the evaluation metrics. Cross-validation was also used to assess model performance across multiple training-validation splits.

The complete implementation and generated visualizations are included in this project.

## 13. How to Run

### Step 1: Install Python

Make sure Python is installed on your system.

### Step 2: Install Dependencies

Open a terminal in the project directory and run:

```bash
pip install -r requirements.txt
```

### Step 3: Run the Program

Run:

```bash
python customer_churn.py
```

The program trains both models, evaluates their performance, performs cross-validation, and generates the required plots and result files.

## 14. Project Files

```text
Task-1-ML-Classification/
│
├── customer_churn.py
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── requirements.txt
├── README.md
├── model_comparison.csv
├── churn_distribution.png
├── model_comparison.png
├── logistic_confusion_matrix.png
├── random_forest_confusion_matrix.png
├── roc_curve.png
└── feature_importance.png
```
