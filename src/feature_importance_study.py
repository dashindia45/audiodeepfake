import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve


df = pd.read_csv("../features.csv")

df["label"] = df["label"].map({
    "real": 0,
    "fake": 1
})

X = df.drop(columns=["label"]).values
y = df["label"].values


#  Feature group indices (FIXED)

BREATHING = [0, 1, 2, 3, 4]
PAUSE = [5, 6, 7]
COUPLING = [8, 9]

ALL = list(range(10))



#  EER computation (proper)

def compute_eer(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2
    return eer


#  Single split (same for all experiments)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


#  Experiment definitions

experiments = {
    "All features": ALL,
    "No Breathing": [i for i in ALL if i not in BREATHING],
    "No Pause": [i for i in ALL if i not in PAUSE],
    "No Coupling": [i for i in ALL if i not in COUPLING]
}



# Run ablation

print("\nABLATION STUDY RESULTS (Random Forest)\n")
print("{:<20} {:<10} {:<10}".format("Experiment", "AUC", "EER"))

for name, feat_idx in experiments.items():

    X_tr = X_train[:, feat_idx]
    X_te = X_test[:, feat_idx]

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_tr, y_train)
    y_prob = model.predict_proba(X_te)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    eer = compute_eer(y_test, y_prob)

    print("{:<20} {:<10.3f} {:<10.3f}".format(name, auc, eer))
