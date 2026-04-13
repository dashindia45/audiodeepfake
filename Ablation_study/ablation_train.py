import pandas as pd
import os
import json
import joblib
from tqdm import tqdm   

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# =====================================================
# PATHS
# =====================================================

FEATURES_DIR = "../features"
MODELS_DIR = "../models/ablation_models"

os.makedirs(MODELS_DIR, exist_ok=True)

# =====================================================
# LOAD BEST MODEL
# =====================================================

with open("../models/best_model.json") as f:
    best_model_info = json.load(f)

BEST_MODEL = best_model_info["best_model"]

print("\n" + "="*60)
print(f"🔥 ABLATION USING BEST MODEL: {BEST_MODEL}")
print("="*60 + "\n")

# =====================================================
# LOAD DATA
# =====================================================

train_df = pd.read_csv(f"{FEATURES_DIR}/featurenew_train.csv")
train_df["label"] = train_df["label"].map({"real": 0, "fake": 1})

# =====================================================
# FEATURE GROUPS
# =====================================================

PHYSIOLOGY = ["breath_count", "pause_count"]

MFCC = [c for c in train_df.columns if c.startswith("mfcc_")]

DELTA = [c for c in train_df.columns if c.startswith("delta_")]
DELTA2 = [c for c in train_df.columns if c.startswith("delta2_")]

SPECTRAL = (
    [c for c in train_df.columns if "centroid" in c or "bandwidth" in c] +
    [c for c in train_df.columns if c.startswith("contrast_")] +
    [c for c in train_df.columns if c.startswith("chroma_")]
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
# TRAIN MODELS
# =====================================================

print("🚀 Training Ablation Models...\n")

# ✅ CLEAN SINGLE PROGRESS BAR
for exp_name, remove_feats in tqdm(
    EXPERIMENTS.items(),
    desc="Ablation Experiments",
    unit="model",
    ncols=100
):

    print(f"\n➡️ Training: {exp_name}")

    # Remove feature group
    df_temp = train_df.drop(columns=remove_feats, errors="ignore")

    X = df_temp.drop(columns=["label"])
    y = df_temp["label"]

    scaler = StandardScaler()

    # =================================================
    # APPLY SCALING (ONLY FOR SVM)
    # =================================================

    if BEST_MODEL == "SVM":

        X_np = X.values.copy()
        cols = list(X.columns)

        # 🔥 BOOST PHYSIOLOGY
        physio_idx = [i for i, c in enumerate(cols) if c in PHYSIOLOGY]
        X_np[:, physio_idx] *= 1.8

        # 🔥 REDUCE NOISE IMPACT
        noise_idx = [i for i, c in enumerate(cols) if c in NOISE]
        X_np[:, noise_idx] *= 0.5

        X_train = scaler.fit_transform(X_np)

    else:
        X_train = X  # no scaling needed

    # =================================================
    # MODEL SELECTION
    # =================================================

    if BEST_MODEL == "SVM":

        model = SVC(
            kernel="rbf",
            C=10,
            gamma=0.005,
            class_weight="balanced",
            probability=True
        )

    elif BEST_MODEL == "Random Forest":

        model = RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )

    elif BEST_MODEL == "XGBoost":

        scale_weight = len(y[y == 0]) / len(y[y == 1])

        model = XGBClassifier(
            n_estimators=600,
            max_depth=6,
            learning_rate=0.03,
            scale_pos_weight=scale_weight,
            eval_metric="logloss",
            n_jobs=-1
        )

    # =================================================
    # TRAIN
    # =================================================

    model.fit(X_train, y)

    # =================================================
    # SAVE MODEL + SCALER
    # =================================================

    joblib.dump(model, f"{MODELS_DIR}/{exp_name}_model.pkl")
    joblib.dump(scaler, f"{MODELS_DIR}/{exp_name}_scaler.pkl")

    print(f"✅ Saved: {exp_name}")

print("\n🎯 All ablation models trained successfully!")