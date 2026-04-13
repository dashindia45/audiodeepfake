import pandas as pd
import numpy as np
import random
import joblib
import time
import os
from tqdm import tqdm  

from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.metrics import (
    roc_curve,
    accuracy_score,
    roc_auc_score,
    average_precision_score
)


# Reproducibility
np.random.seed(42)
random.seed(42)

# Load Data

train_df = pd.read_csv("../features/featurenew_train.csv")
dev_df   = pd.read_csv("../features/featurenew_dev.csv")

for df in [train_df, dev_df]:
    df["label"] = df["label"].map({"real": 0, "fake": 1})

X_train = train_df.drop(columns=["label"])
y_train = train_df["label"]

X_dev = dev_df.drop(columns=["label"])
y_dev = dev_df["label"]

# REMOVE NOISE FEATURES from ablation study

REMOVE_NOISE = True

NOISE_COLS = ["zcr_mean", "zcr_std", "energy_entropy"]

if REMOVE_NOISE:
    X_train = X_train.drop(columns=NOISE_COLS, errors="ignore")
    X_dev   = X_dev.drop(columns=NOISE_COLS, errors="ignore")


# Scaling (ONLY FOR SVM)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_dev_scaled = scaler.transform(X_dev)


# METRICS calculation

def compute_eer(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return (fpr[idx] + fnr[idx]) / 2

def evaluate_model(name, y_true, prob):
    pred = (prob >= 0.5).astype(int)

    return {
        "Accuracy": accuracy_score(y_true, pred),
        "AUC": roc_auc_score(y_true, prob),
        "AUPRC": average_precision_score(y_true, prob),
        "EER": compute_eer(y_true, prob)
    }


# Training Models

print("\n🚀 Training Models...\n")
start_time = time.time()

#  MODEL LIST FOR CLEAN LOOP
models_to_train = [
    ("Random Forest", RandomizedSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
        {
            "n_estimators": [400, 600],
            "max_depth": [None],
            "min_samples_leaf": [1, 2]
        },
        n_iter=3,
        scoring="roc_auc",
        cv=3,
        n_jobs=-1,
        random_state=42
    ), X_train, y_train),

    ("SVM", RandomizedSearchCV(
        SVC(kernel="rbf", class_weight="balanced", probability=True),
        {
            "C": [1, 5],
            "gamma": ["scale"]
        },
        n_iter=2,
        scoring="roc_auc",
        cv=3,
        n_jobs=-1,
        random_state=42
    ), X_train_scaled, y_train),

    ("XGBoost", RandomizedSearchCV(
        XGBClassifier(
            eval_metric="logloss",
            scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
            n_jobs=-1
        ),
        {
            "n_estimators": [500, 700],
            "max_depth": [5, 6],
            "learning_rate": [0.02, 0.03],
            "subsample": [0.8],
            "colsample_bytree": [0.8]
        },
        n_iter=3,
        scoring="roc_auc",
        cv=3,
        n_jobs=-1,
        random_state=42
    ), X_train, y_train)
]

trained_models = {}

#   PROGRESS BAR
for name, model, X, y in tqdm(models_to_train, desc="Training Models", unit="model"):

    print(f"\n➡️ Training {name}...")

    model.fit(X, y)
    trained_models[name] = model.best_estimator_

print("\n⏱ Training time:", round(time.time() - start_time, 2), "seconds")


# Calibration

print("\n🔧 Calibrating Models...\n")

rf_cal = CalibratedClassifierCV(trained_models["Random Forest"], method='sigmoid', cv=3)
xgb_cal = CalibratedClassifierCV(trained_models["XGBoost"], method='isotonic', cv=3)
svm_cal = CalibratedClassifierCV(trained_models["SVM"], method='sigmoid', cv=3)

calibration_steps = [
    ("RF Calibration", rf_cal, X_train, y_train),
    ("XGB Calibration", xgb_cal, X_train, y_train),
    ("SVM Calibration", svm_cal, X_train_scaled, y_train)
]

for name, model, X, y in tqdm(calibration_steps, desc="Calibration", unit="step"):

    print(f"\n➡️ {name}...")
    model.fit(X, y)


# DEV Evaluation

rf_prob = rf_cal.predict_proba(X_dev)[:, 1]
xgb_prob = xgb_cal.predict_proba(X_dev)[:, 1]
svm_prob = svm_cal.predict_proba(X_dev_scaled)[:, 1]

results = {
    "Random Forest": evaluate_model("RF", y_dev, rf_prob),
    "XGBoost": evaluate_model("XGB", y_dev, xgb_prob),
    "SVM": evaluate_model("SVM", y_dev, svm_prob)
}


# results

print("\n" + "="*80)
print("              🚀 MODEL PERFORMANCE SUMMARY")
print("="*80)

best_model = min(results, key=lambda x: results[x]["EER"])

for model, metrics in results.items():
    marker = "🔥 BEST" if model == best_model else ""

    print(f"{model:<20} | "
          f"Acc: {metrics['Accuracy']:.5f} | "
          f"AUC: {metrics['AUC']:.5f} | "
          f"AUPRC: {metrics['AUPRC']:.5f} | "
          f"EER: {metrics['EER']:.5f} {marker}")

print("="*80)
print(f"🏆 BEST MODEL: {best_model}")
print("="*80)


# SAVE MODELS

os.makedirs("../models", exist_ok=True)

joblib.dump(rf_cal, "../models/rf_newmodel.pkl")
joblib.dump(svm_cal, "../models/svm_newmodel.pkl")
joblib.dump(xgb_cal, "../models/xgb_newmodel.pkl")
joblib.dump(scaler, "../models/newscaler.pkl")

print("\n✅ Models saved successfully.")
print("\n🎯 Training Completed Successfully")