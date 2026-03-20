import pandas as pd
import numpy as np
import random
import joblib
import time
import os

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


# =====================================================
# Reproducibility
# =====================================================

np.random.seed(42)
random.seed(42)


# =====================================================
# Load Feature Files
# =====================================================

train_df = pd.read_csv("../features/features_train.csv")
dev_df   = pd.read_csv("../features/features_dev.csv")

for df in [train_df, dev_df]:
    df["label"] = df["label"].map({"real":0,"fake":1})


X_train = train_df.drop(columns=["label"])
y_train = train_df["label"]

X_dev = dev_df.drop(columns=["label"])
y_dev = dev_df["label"]


# =====================================================
# Feature Scaling
# =====================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_dev_scaled = scaler.transform(X_dev)


# =====================================================
# Hyperparameter Tuning
# =====================================================

print("\nTraining Models...\n")

start_time = time.time()


# Random Forest
rf = GridSearchCV(
    RandomForestClassifier(class_weight="balanced", random_state=42),
    param_grid={
        "n_estimators":[300,500],
        "max_depth":[None,20,40],
        "min_samples_split":[2,5]
    },
    cv=3,
    scoring="roc_auc",
    n_jobs=-1
)

rf.fit(X_train, y_train)


# SVM
svm = GridSearchCV(
    SVC(kernel="rbf", class_weight="balanced", probability=True),
    param_grid={
        "C":[0.1,1,10],
        "gamma":["scale",0.1,0.01]
    },
    cv=3,
    scoring="roc_auc",
    n_jobs=-1
)

svm.fit(X_train_scaled, y_train)


# XGBoost
scale_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])

xgb = XGBClassifier(
    scale_pos_weight=scale_weight,
    eval_metric="logloss",
    use_label_encoder=False
)

xgb.fit(X_train, y_train)


print("\nTraining time:", round(time.time()-start_time,2),"seconds")


# =====================================================
# Save Models
# =====================================================

os.makedirs("../models", exist_ok=True)

joblib.dump(rf.best_estimator_, "../models/rf_model.pkl")
joblib.dump(svm.best_estimator_, "../models/svm_model.pkl")
joblib.dump(xgb, "../models/xgb_model.pkl")
joblib.dump(scaler, "../models/scaler.pkl")

print("\nModels saved successfully.")

print("\nTraining  Completed Successfully")