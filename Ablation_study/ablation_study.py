import pandas as pd
import numpy as np
import os
import joblib
import json
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import roc_auc_score, roc_curve, balanced_accuracy_score

warnings.filterwarnings("ignore")

# =====================================================
# PATHS
# =====================================================

FEATURES_DIR = "../features"
MODELS_DIR = "../models/ablation_models"
RESULTS_DIR = "../results"

os.makedirs(RESULTS_DIR, exist_ok=True)

# =====================================================
# LOAD BEST MODEL
# =====================================================

with open("../models/best_model.json") as f:
    BEST_MODEL = json.load(f)["best_model"]

# =====================================================
# LOAD BASE RESULTS
# =====================================================

base_results = pd.read_csv("../results/model_results.csv")
best_row = base_results[base_results["Model"] == BEST_MODEL].iloc[0]

BASE_AUC = best_row["AUC"]
BASE_EER = best_row["EER"]
BASE_BAL = best_row["Accuracy"]

# =====================================================
# LOAD DATA
# =====================================================

eval_df = pd.read_csv(f"{FEATURES_DIR}/featurenew_eval.csv")
eval_df["label"] = eval_df["label"].map({"real":0, "fake":1})

# =====================================================
# FEATURE GROUPS
# =====================================================

PHYSIOLOGY = ["breath_count", "pause_count"]

MFCC = [c for c in eval_df.columns if c.startswith("mfcc_")]

DELTA = [c for c in eval_df.columns if c.startswith("delta_")]
DELTA2 = [c for c in eval_df.columns if c.startswith("delta2_")]

SPECTRAL = (
    [c for c in eval_df.columns if "centroid" in c or "bandwidth" in c] +
    [c for c in eval_df.columns if c.startswith("contrast_")] +
    [c for c in eval_df.columns if c.startswith("chroma_")]
)

NOISE = ["zcr_mean", "zcr_std", "energy_entropy"]

# =====================================================
# EXPERIMENTS
# =====================================================

EXPERIMENTS = {
    "all_features": [],
    "no_physiology": PHYSIOLOGY,
    "no_mfcc": MFCC,
    "no_temporal": DELTA + DELTA2,
    "no_spectral": SPECTRAL,
    "no_noise": NOISE
}

# =====================================================
# EER FUNCTION
# =====================================================

def compute_eer(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return (fpr[idx] + fnr[idx]) / 2

# =====================================================
# RUN ABLATION
# =====================================================

results = []

print("\n🔥 ABLATION RESULTS\n")
print("{:<20} {:<10} {:<10} {:<10}".format("Experiment","AUC","EER","BalAcc"))

for exp_name, remove_feats in EXPERIMENTS.items():

    if exp_name == "all_features":
        auc = BASE_AUC
        eer = BASE_EER
        bal = BASE_BAL

    else:
        model = joblib.load(f"{MODELS_DIR}/{exp_name}_model.pkl")
        scaler = joblib.load(f"{MODELS_DIR}/{exp_name}_scaler.pkl")

        df_temp = eval_df.drop(columns=remove_feats, errors="ignore")

        X = df_temp.drop(columns=["label"])
        y = df_temp["label"]

        if BEST_MODEL == "SVM":
            X_scaled = scaler.transform(X)
            y_prob = model.predict_proba(X_scaled)[:,1]
            y_pred = model.predict(X_scaled)
        else:
            y_prob = model.predict_proba(X)[:,1]
            y_pred = model.predict(X)

        auc = roc_auc_score(y, y_prob)
        eer = compute_eer(y, y_prob)
        bal = balanced_accuracy_score(y, y_pred)

    print("{:<20} {:<10.3f} {:<10.3f} {:<10.3f}".format(
        exp_name, auc, eer, bal
    ))

    results.append({
        "Experiment": exp_name,
        "AUC": auc,
        "EER": eer,
        "BalancedAccuracy": bal
    })

# =====================================================
# SAVE RESULTS
# =====================================================

results_df = pd.DataFrame(results)
results_df.to_csv(f"{RESULTS_DIR}/ablation_results.csv", index=False)

# =====================================================
# IMPORTANCE COMPUTATION (UNCHANGED)
# =====================================================

feature_counts = {
    "physiology": len(PHYSIOLOGY),
    "mfcc": len(MFCC),
    "temporal": len(DELTA + DELTA2),
    "spectral": len(SPECTRAL),
    "noise": len(NOISE)
}

base_eer = results_df.loc[results_df["Experiment"]=="all_features","EER"].values[0]

importance = []

for _, row in results_df.iterrows():

    if row["Experiment"] == "all_features":
        continue

    group = row["Experiment"].replace("no_", "")

    eer_increase = row["EER"] - base_eer

    norm_importance = eer_increase / (feature_counts.get(group, 1) + 1e-6)

    importance.append({
        "FeatureGroup": group.capitalize(),
        "Raw_EER_Increase": eer_increase,
        "Normalized_Importance": norm_importance
    })

imp_df = pd.DataFrame(importance)

# =====================================================
# 🔥 RESEARCH-LEVEL VISUALIZATION
# =====================================================

# ---- EER CHANGE (MAIN GRAPH) ----
eer_df = imp_df.copy()
eer_df["EER_Change"] = eer_df["Raw_EER_Increase"]
eer_df = eer_df.sort_values("EER_Change", ascending=True)

plt.figure(figsize=(8,5))

bars = plt.barh(eer_df["FeatureGroup"], eer_df["EER_Change"])

for bar, val in zip(bars, eer_df["EER_Change"]):
    if val > 0:
        bar.set_color("red")
    else:
        bar.set_color("green")

plt.axvline(0)
plt.xlabel("Change in EER (ΔEER)")
plt.title("Feature Contribution Analysis (Ablation Study)")

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/ablation_eer_contribution.png", dpi=300)
plt.show()

# ---- NORMALIZED IMPORTANCE ----
plt.figure(figsize=(8,5))

plt.barh(
    imp_df["FeatureGroup"],
    imp_df["Normalized_Importance"],
    color="steelblue"
)

plt.xlabel("Normalized Importance (ΔEER / Feature Count)")
plt.title("Normalized Feature Importance")

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/ablation_normalized_importance.png", dpi=300)
plt.show()

# =====================================================
# INTERPRETATION
# =====================================================

print("\n📊 Feature Interpretation:")

for _, row in eer_df.iterrows():
    if row["EER_Change"] < 0:
        print(f"🟢 {row['FeatureGroup']} → hurting performance")
    else:
        print(f"🔴 {row['FeatureGroup']} → important feature")

print("\n✅ Ablation completed successfully")