import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve


# -------------------------------------------------
# 1. Load dataset
# -------------------------------------------------
df = pd.read_csv("../features.csv")

df["label"] = df["label"].map({
    "real": 0,
    "fake": 1
})

X = df.drop(columns=["label"])
y = df["label"]

# -------------------------------------------------
# 2. Train-test split (stratified)
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# -------------------------------------------------
# 3. Feature scaling (for linear & SVM models)
# -------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------------------------
# 4. Proper EER computation (improved)
# -------------------------------------------------
def compute_eer(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr

    # Find point where FPR and FNR are closest
    idx = np.nanargmin(np.abs(fpr - fnr))

    # Interpolated EER (not just FPR)
    eer = (fpr[idx] + fnr[idx]) / 2
    eer_threshold = thresholds[idx]

    return eer, eer_threshold


# -------------------------------------------------
# 5. Define models (same as before)
# -------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced",
        max_iter=1000
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42
    ),
    "SVM (RBF)": SVC(
        kernel="rbf",
        class_weight="balanced",
        probability=True
    )
}

# -------------------------------------------------
# 6. Train, evaluate & plot
# -------------------------------------------------
results = {}

plt.figure(figsize=(8, 6))

for name, model in models.items():

    # Use scaled or unscaled features correctly
    if name == "Random Forest":
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
    else:
        model.fit(X_train_scaled, y_train)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        y_pred = model.predict(X_test_scaled)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    eer, eer_th = compute_eer(y_test, y_prob)

    results[name] = {
        "Accuracy": acc,
        "AUC": auc,
        "EER": eer,
        "EER_Threshold": eer_th
    }

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")

# -------------------------------------------------
# 7. ROC plot
# -------------------------------------------------
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison (All Models)")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------------------------
# 8. Print comparison (clean & meaningful)
# -------------------------------------------------
print("\nMODEL COMPARISON RESULTS (Improved Metrics)\n")
print("{:<20} {:<10} {:<10} {:<10}".format(
    "Model", "Accuracy", "AUC", "EER"
))

for model, m in results.items():
    print("{:<20} {:<10.3f} {:<10.3f} {:<10.3f}".format(
        model,
        m["Accuracy"],
        m["AUC"],
        m["EER"]
    ))
