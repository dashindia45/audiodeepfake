import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
    average_precision_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier


# -------------------------------------------------
# 1. Load Extracted Features
# -------------------------------------------------
# We already extracted physiology-aware features
# Now we train ML models on structured feature space

df = pd.read_csv("../features.csv")

df["label"] = df["label"].map({
    "real": 0,
    "fake": 1
})

X = df.drop(columns=["label"])
y = df["label"]


# -------------------------------------------------
# 2. Stratified Train-Test Split
# -------------------------------------------------
# Stratification ensures class imbalance ratio
# is preserved in both train and test sets

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


# -------------------------------------------------
# 3. Feature Scaling
# -------------------------------------------------
# Linear models & SVM require scaling
# Tree-based models do NOT need scaling

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# -------------------------------------------------
# 4. Proper EER Calculation
# -------------------------------------------------
def compute_eer(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2
    return eer


# -------------------------------------------------
# 5. Hyperparameter Tuning (IMPORTANT)
# -------------------------------------------------
# Instead of using default parameters,
# we tune models using 5-fold CV and optimize ROC-AUC

print("\n🔎 Performing Hyperparameter Tuning...\n")

# ---- Logistic Regression ----
log_reg = GridSearchCV(
    LogisticRegression(class_weight="balanced", max_iter=2000),
    param_grid={
        "C": [0.01, 0.1, 1, 10],
    },
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)

log_reg.fit(X_train_scaled, y_train)


# ---- Random Forest ----
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


# ---- SVM (RBF) ----
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


# ---- XGBoost (Strong Baseline) ----
# scale_pos_weight handles imbalance explicitly
scale_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

xgb = XGBClassifier(
    scale_pos_weight=scale_weight,
    eval_metric="logloss",
    use_label_encoder=False
)

xgb.fit(X_train, y_train)


# -------------------------------------------------
# 6. Model Evaluation
# -------------------------------------------------
models = {
    "Logistic Regression (Tuned)": (log_reg.best_estimator_, True),
    "Random Forest (Tuned)": (rf.best_estimator_, False),
    "SVM (Tuned)": (svm.best_estimator_, True),
    "XGBoost": (xgb, False)
}

results = {}

plt.figure(figsize=(8, 6))

for name, (model, needs_scaling) in models.items():

    # Use correct feature version
    if needs_scaling:
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        y_pred = model.predict(X_test_scaled)
    else:
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    auprc = average_precision_score(y_test, y_prob)
    eer = compute_eer(y_test, y_prob)

    results[name] = {
        "Accuracy": acc,
        "AUC": auc,
        "AUPRC": auprc,
        "EER": eer
    }

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")


# -------------------------------------------------
# 7. ROC Curve Plot
# -------------------------------------------------
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison (Tuned Models)")
plt.legend()
plt.grid(True)
plt.show()


# -------------------------------------------------
# 8. Print Final Comparison
# -------------------------------------------------
print("\n📊 FINAL MODEL COMPARISON\n")
print("{:<30} {:<10} {:<10} {:<10} {:<10}".format(
    "Model", "Accuracy", "AUC", "AUPRC", "EER"
))

for model, m in results.items():
    print("{:<30} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f}".format(
        model,
        m["Accuracy"],
        m["AUC"],
        m["AUPRC"],
        m["EER"]
    ))

print("\n✅ Training & Evaluation Completed Successfully")
