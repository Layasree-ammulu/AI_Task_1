import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("=" * 60)
print("CUSTOMER CHURN CLASSIFICATION")
print("=" * 60)

file_path = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = pd.read_csv(file_path)

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)


# ============================================================
# 2. UNDERSTAND THE DATA
# ============================================================

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nDataset information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())


# ============================================================
# 3. DATA CLEANING
# ============================================================

# TotalCharges contains some blank values.
# Convert it to numeric. Invalid values become NaN.

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Remove rows where TotalCharges is missing

df = df.dropna(subset=["TotalCharges"])

print("\nShape after cleaning:", df.shape)


# ============================================================
# 4. REMOVE UNNECESSARY COLUMN
# ============================================================

# customerID is only an identifier and is not useful
# for predicting churn.

df = df.drop("customerID", axis=1)


# ============================================================
# 5. CONVERT TARGET VARIABLE
# ============================================================

# Churn:
# Yes = 1
# No  = 0

df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})


# ============================================================
# 6. EXPLORATORY DATA ANALYSIS
# ============================================================

plt.figure(figsize=(6, 4))

sns.countplot(
    x="Churn",
    data=df
)

plt.title("Customer Churn Distribution")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.savefig("churn_distribution.png")
plt.show()


# ============================================================
# 7. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop("Churn", axis=1)

y = df["Churn"]


# ============================================================
# 8. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nNumerical features:")
print(numerical_features)

print("\nCategorical features:")
print(categorical_features)


# ============================================================
# 9. DATA PREPROCESSING
# ============================================================

# Numerical features:
# StandardScaler standardizes numerical values.

# Categorical features:
# OneHotEncoder converts categories into numerical values.

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_features
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)


# ============================================================
# 10. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# ============================================================
# 11. LOGISTIC REGRESSION MODEL
# ============================================================

logistic_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000
            )
        )
    ]
)


# ============================================================
# 12. RANDOM FOREST MODEL
# ============================================================

random_forest_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42
            )
        )
    ]
)


# ============================================================
# 13. TRAIN LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 60)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 60)

logistic_model.fit(
    X_train,
    y_train
)

logistic_predictions = logistic_model.predict(
    X_test
)

logistic_probabilities = logistic_model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 14. TRAIN RANDOM FOREST
# ============================================================

print("\n" + "=" * 60)
print("TRAINING RANDOM FOREST")
print("=" * 60)

random_forest_model.fit(
    X_train,
    y_train
)

rf_predictions = random_forest_model.predict(
    X_test
)

rf_probabilities = random_forest_model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 15. EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model_name,
    y_true,
    predictions,
    probabilities
):

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions
    )

    recall = recall_score(
        y_true,
        predictions
    )

    f1 = f1_score(
        y_true,
        predictions
    )

    roc_auc = roc_auc_score(
        y_true,
        probabilities
    )

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            predictions
        )
    )

    return [
        accuracy,
        precision,
        recall,
        f1,
        roc_auc
    ]


# ============================================================
# 16. EVALUATE BOTH MODELS
# ============================================================

logistic_results = evaluate_model(
    "Logistic Regression",
    y_test,
    logistic_predictions,
    logistic_probabilities
)

rf_results = evaluate_model(
    "Random Forest",
    y_test,
    rf_predictions,
    rf_probabilities
)


# ============================================================
# 17. MODEL COMPARISON
# ============================================================

results = pd.DataFrame(
    {
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score",
            "ROC-AUC"
        ],
        "Logistic Regression": logistic_results,
        "Random Forest": rf_results
    }
)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results)


# ============================================================
# 18. SAVE MODEL COMPARISON
# ============================================================

results.to_csv(
    "model_comparison.csv",
    index=False
)


# ============================================================
# 19. MODEL COMPARISON PLOT
# ============================================================

results_plot = results.set_index("Metric")

results_plot.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)

plt.xticks(
    rotation=0
)

plt.legend(
    title="Models"
)

plt.tight_layout()

plt.savefig(
    "model_comparison.png"
)

plt.show()


# ============================================================
# 20. CONFUSION MATRIX - LOGISTIC REGRESSION
# ============================================================

logistic_cm = confusion_matrix(
    y_test,
    logistic_predictions
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    logistic_cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title(
    "Confusion Matrix - Logistic Regression"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    "logistic_confusion_matrix.png"
)

plt.show()


# ============================================================
# 21. CONFUSION MATRIX - RANDOM FOREST
# ============================================================

rf_cm = confusion_matrix(
    y_test,
    rf_predictions
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    rf_cm,
    annot=True,
    fmt="d",
    cmap="Greens"
)

plt.title(
    "Confusion Matrix - Random Forest"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    "random_forest_confusion_matrix.png"
)

plt.show()


# ============================================================
# 22. ROC CURVE
# ============================================================

logistic_fpr, logistic_tpr, _ = roc_curve(
    y_test,
    logistic_probabilities
)

rf_fpr, rf_tpr, _ = roc_curve(
    y_test,
    rf_probabilities
)

plt.figure(figsize=(8, 6))

plt.plot(
    logistic_fpr,
    logistic_tpr,
    label=f"Logistic Regression (AUC = {roc_auc_score(y_test, logistic_probabilities):.3f})"
)

plt.plot(
    rf_fpr,
    rf_tpr,
    label=f"Random Forest (AUC = {roc_auc_score(y_test, rf_probabilities):.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve Comparison")

plt.legend()

plt.tight_layout()

plt.savefig(
    "roc_curve.png"
)

plt.show()


# ============================================================
# 23. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

# Get the trained preprocessing step

rf_preprocessor = random_forest_model.named_steps[
    "preprocessor"
]

rf_classifier = random_forest_model.named_steps[
    "classifier"
]

# Get transformed feature names

feature_names = rf_preprocessor.get_feature_names_out()

feature_importance = pd.DataFrame(
    {
        "Feature": feature_names,
        "Importance": rf_classifier.feature_importances_
    }
)

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n" + "=" * 60)
print("TOP 15 IMPORTANT FEATURES")
print("=" * 60)

print(
    feature_importance.head(15)
)


# ============================================================
# 24. FEATURE IMPORTANCE PLOT
# ============================================================

top_features = feature_importance.head(15)

plt.figure(figsize=(10, 7))

sns.barplot(
    data=top_features,
    x="Importance",
    y="Feature"
)

plt.title(
    "Top 15 Random Forest Feature Importances"
)

plt.tight_layout()

plt.savefig(
    "feature_importance.png"
)

plt.show()


# ============================================================
# 25. CROSS-VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("5-FOLD CROSS-VALIDATION")
print("=" * 60)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

logistic_cv_scores = cross_val_score(
    logistic_model,
    X_train,
    y_train,
    cv=cv,
    scoring="accuracy"
)

rf_cv_scores = cross_val_score(
    random_forest_model,
    X_train,
    y_train,
    cv=cv,
    scoring="accuracy"
)

print("\nLogistic Regression CV Scores:")
print(logistic_cv_scores)

print(
    "Logistic Regression Mean CV Accuracy:",
    logistic_cv_scores.mean()
)

print("\nRandom Forest CV Scores:")
print(rf_cv_scores)

print(
    "Random Forest Mean CV Accuracy:",
    rf_cv_scores.mean()
)


# ============================================================
# 26. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print(
    f"Logistic Regression Test Accuracy: "
    f"{logistic_results[0]:.4f}"
)

print(
    f"Random Forest Test Accuracy: "
    f"{rf_results[0]:.4f}"
)

print(
    f"Logistic Regression Mean CV Accuracy: "
    f"{logistic_cv_scores.mean():.4f}"
)

print(
    f"Random Forest Mean CV Accuracy: "
    f"{rf_cv_scores.mean():.4f}"
)

print("\nAll Task 1 requirements completed!")

print("\nGenerated files:")
print("1. churn_distribution.png")
print("2. model_comparison.png")
print("3. model_comparison.csv")
print("4. logistic_confusion_matrix.png")
print("5. random_forest_confusion_matrix.png")
print("6. roc_curve.png")
print("7. feature_importance.png")