import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import joblib
import time
import os

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
    average_precision_score,
    confusion_matrix,
    det_curve
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier


# =====================================================
# Reproducibility
# =====================================================

np.random.seed(42)
random.seed(42)


# =====================================================
# Step 1: Load Feature Files
# =====================================================

train_df = pd.read_csv("../features/features_train.csv")
dev_df   = pd.read_csv("../features/features_dev.csv")
eval_df  = pd.read_csv("../features/features_eval.csv")


# Map labels
for df in [train_df, dev_df, eval_df]:
    df["label"] = df["label"].map({"real": 0, "fake": 1})


X_train = train_df.drop(columns=["label"])
y_train = train_df["label"]

X_dev = dev_df.drop(columns=["label"])
y_dev = dev_df["label"]

X_eval = eval_df.drop(columns=["label"])
y_eval = eval_df["label"]


# =====================================================
# Dataset statistics
# =====================================================

print("\nDataset Statistics")
print("Train samples:", len(X_train))
print("Dev samples:", len(X_dev))
print("Eval samples:", len(X_eval))
print("Number of features:", X_train.shape[1])

print("\nTrain class distribution:")
print(y_train.value_counts())


# =====================================================
# Step 2: Feature Scaling
# =====================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_dev_scaled   = scaler.transform(X_dev)
X_eval_scaled  = scaler.transform(X_eval)


# =====================================================
# Step 3: Metrics
# =====================================================

def compute_eer(y_true, y_score):

    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr

    idx = np.nanargmin(np.abs(fpr - fnr))

    eer = (fpr[idx] + fnr[idx]) / 2

    return eer


def compute_tdcf(y_true, y_score):

    P_tar = 0.01
    P_non = 0.99
    C_miss = 1
    C_fa = 10

    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr

    tdcf_values = []

    for i in range(len(fpr)):
        tdcf = C_miss * P_tar * fnr[i] + C_fa * P_non * fpr[i]
        tdcf_values.append(tdcf)

    return min(tdcf_values)


# =====================================================
# Step 4: Hyperparameter Tuning
# =====================================================

print("\n🔎 Performing Hyperparameter Tuning...\n")

start_time = time.time()


# Logistic Regression
log_reg = GridSearchCV(
    LogisticRegression(class_weight="balanced", max_iter=2000),
    param_grid={"C": [0.01, 0.1, 1, 10]},
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)

log_reg.fit(X_train_scaled, y_train)


# Random Forest
rf = GridSearchCV(
    RandomForestClassifier(class_weight="balanced", random_state=42),
    param_grid={
        "n_estimators": [300, 500],
        "max_depth": [None, 20, 40],
        "min_samples_split": [2, 5]
    },
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)

rf.fit(X_train, y_train)


# SVM
svm = GridSearchCV(
    SVC(kernel="rbf", class_weight="balanced", probability=True),
    param_grid={
        "C": [0.1, 1, 10],
        "gamma": ["scale", 0.1, 0.01]
    },
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)

svm.fit(X_train_scaled, y_train)


# XGBoost
scale_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

xgb = XGBClassifier(
    scale_pos_weight=scale_weight,
    eval_metric="logloss",
    use_label_encoder=False
)

xgb.fit(X_train, y_train)


print("\nTraining time:", round(time.time() - start_time, 2), "seconds")


# =====================================================
# Print Best Hyperparameters
# =====================================================

print("\nBest Hyperparameters")
print("Logistic Regression:", log_reg.best_params_)
print("Random Forest:", rf.best_params_)
print("SVM:", svm.best_params_)


# =====================================================
# Step 5: Model Evaluation
# =====================================================

models = {
    "Logistic Regression": (log_reg.best_estimator_, True),
    "Random Forest": (rf.best_estimator_, False),
    "SVM": (svm.best_estimator_, True),
    "XGBoost": (xgb, False)
}

results = {}

plt.figure(figsize=(8,6))


for name, (model, needs_scaling) in models.items():

    if needs_scaling:
        y_prob = model.predict_proba(X_eval_scaled)[:,1]
        y_pred = model.predict(X_eval_scaled)
    else:
        y_prob = model.predict_proba(X_eval)[:,1]
        y_pred = model.predict(X_eval)

    acc = accuracy_score(y_eval, y_pred)
    auc = roc_auc_score(y_eval, y_prob)
    auprc = average_precision_score(y_eval, y_prob)
    eer = compute_eer(y_eval, y_prob)
    tdcf = compute_tdcf(y_eval, y_prob)

    results[name] = {
        "Accuracy": acc,
        "AUC": auc,
        "AUPRC": auprc,
        "EER": eer,
        "tDCF": tdcf
    }

    fpr, tpr, _ = roc_curve(y_eval, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")


# =====================================================
# ROC Curve
# =====================================================

plt.plot([0,1],[0,1],"k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Eval Set)")
plt.legend()
plt.grid(True)
plt.show()


# =====================================================
# DET Curve
# =====================================================

fpr, fnr, _ = det_curve(y_eval, y_prob)

plt.figure()
plt.plot(fpr, fnr)
plt.xlabel("False Alarm Rate")
plt.ylabel("Miss Rate")
plt.title("DET Curve")
plt.grid(True)
plt.show()


# =====================================================
# Confusion Matrix
# =====================================================

cm = confusion_matrix(y_eval, y_pred)

print("\nConfusion Matrix")
print(cm)


# =====================================================
# Step 6: Print Results
# =====================================================

print("\nFINAL MODEL COMPARISON (EVAL SET)\n")

print("{:<25} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
    "Model","Accuracy","AUC","AUPRC","EER","tDCF"
))

for model, m in results.items():

    print("{:<25} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f}".format(
        model,
        m["Accuracy"],
        m["AUC"],
        m["AUPRC"],
        m["EER"],
        m["tDCF"]
    ))


# =====================================================
# Step 7: Save Results
# =====================================================

os.makedirs("../results", exist_ok=True)

results_df = pd.DataFrame(results).T

results_df.to_csv("../results/model_results.csv", index=False)


print("\nResults saved to ../results/model_results.csv")


# =====================================================
# Step 8: Save Best Model
# =====================================================

joblib.dump(svm.best_estimator_, "../models/best_svm_model.pkl")

print("Best model saved to ../models/best_svm_model.pkl")


print("\n✅ Training & Evaluation Completed Successfully")