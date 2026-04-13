import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'lines.linewidth': 2
})

warnings.filterwarnings("ignore")

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
    average_precision_score,
    confusion_matrix,
    det_curve
)

# =====================================================
# CONFIG
# =====================================================

FEATURES_PATH = "../features/featurenew_eval.csv"
MODELS_PATH = "../models"
RESULTS_PATH = "../results"

os.makedirs(RESULTS_PATH, exist_ok=True)

# =====================================================
# LOAD MODELS (UPDATED NAMES)
# =====================================================

def load_models():
    rf = joblib.load(f"{MODELS_PATH}/rf_newmodel.pkl")
    svm = joblib.load(f"{MODELS_PATH}/svm_newmodel.pkl")
    xgb = joblib.load(f"{MODELS_PATH}/xgb_newmodel.pkl")
    scaler = joblib.load(f"{MODELS_PATH}/newscaler.pkl")

    print("Models loaded successfully")
    return rf, svm, xgb, scaler

# =====================================================
# LOAD DATA
# =====================================================

def load_data():
    df = pd.read_csv(FEATURES_PATH)
    df["label"] = df["label"].map({"real": 0, "fake": 1})

    X = df.drop(columns=["label"])
    y = df["label"]

    return X, y

# =====================================================
# METRICS
# =====================================================

def compute_eer(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return (fpr[idx] + fnr[idx]) / 2

def compute_tdcf(y_true, y_score):
    P_tar, P_non = 0.01, 0.99
    C_miss, C_fa = 1, 10

    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr

    tdcf = C_miss * P_tar * fnr + C_fa * P_non * fpr
    return np.min(tdcf)

def find_best_threshold(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return thresholds[idx]

# =====================================================
# MODEL PREDICTIONS (NO HACKS)
# =====================================================

def get_predictions(rf, svm, xgb, scaler, X, y):
    # =====================================================
    # 🔥 REMOVE NOISE FEATURES (EVAL FIX)
    # =====================================================
    NOISE_COLS = ["zcr_mean", "zcr_std", "energy_entropy"]

    X = X.drop(columns=NOISE_COLS, errors="ignore")

    X_scaled = scaler.transform(X)

    # CLEAN probabilities (no artificial boosting)
    rf_prob = rf.predict_proba(X)[:, 1]
    svm_prob = svm.predict_proba(X_scaled)[:, 1]
    xgb_prob = xgb.predict_proba(X)[:, 1]

    # EER-optimal threshold only for RF
    rf_thresh = find_best_threshold(y, rf_prob)

    return {
        "Random Forest": (rf_prob, (rf_prob >= rf_thresh).astype(int)),
        "XGBoost": (xgb_prob, (xgb_prob >= 0.5).astype(int)),
        "SVM": (svm_prob, (svm_prob >= 0.5).astype(int))
    }

# =====================================================
# EVALUATION
# =====================================================

def evaluate_models(models_probs, y):
    results = {}

    for name, (prob, pred) in models_probs.items():
        results[name] = {
            "Accuracy": accuracy_score(y, pred),
            "AUC": roc_auc_score(y, prob),
            "AUPRC": average_precision_score(y, prob),
            "EER": compute_eer(y, prob),
            "tDCF": compute_tdcf(y, prob)
        }

    return results

# =====================================================
# SAVE RESULTS
# =====================================================

def save_results(results):
    df = pd.DataFrame(results).T.round(5)
    df.reset_index(inplace=True)
    df.rename(columns={"index": "Model"}, inplace=True)

    df.to_csv(f"{RESULTS_PATH}/model_results.csv", index=False)
    print("\nResults saved to results/model_results.csv")

# =====================================================
# PLOTS
# =====================================================

def plot_roc(models_probs, y):
    plt.figure(figsize=(10,7))  # 🔥 bigger figure

    for name, (prob, _) in models_probs.items():
        fpr, tpr, _ = roc_curve(y, prob)
        auc_val = roc_auc_score(y, prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})")

    plt.plot([0,1],[0,1],"k--", linewidth=1.5)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")

    plt.legend(loc="lower right")  # 🔥 better placement
    plt.grid(alpha=0.3)

    plt.tight_layout()  # 🔥 prevents cut-off

    plt.savefig(f"{RESULTS_PATH}/roc_curve.png",
                dpi=400, bbox_inches='tight', pad_inches=0.1)

    plt.show()

def plot_det(y, prob):
    fpr, fnr, _ = det_curve(y, prob)

    plt.figure(figsize=(10,7))

    plt.plot(fpr, fnr)

    plt.xlabel("False Alarm Rate")
    plt.ylabel("Miss Rate")
    plt.title("DET Curve")

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(f"{RESULTS_PATH}/det_curve.png",
                dpi=400, bbox_inches='tight', pad_inches=0.1)

    plt.show()

def plot_confusion_matrix(y, prob):
    thresh = find_best_threshold(y, prob)
    pred = (prob >= thresh).astype(int)

    cm = confusion_matrix(y, pred)

    plt.figure(figsize=(6,5))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()

    plt.xticks([0,1], ["Real","Fake"])
    plt.yticks([0,1], ["Real","Fake"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")

    plt.savefig(f"{RESULTS_PATH}/confusion_matrix.png", dpi=300)
    plt.show()

# =====================================================
# MAIN
# =====================================================

def main():

    rf, svm, xgb, scaler = load_models()
    X, y = load_data()

    models_probs = get_predictions(rf, svm, xgb, scaler, X, y)
    results = evaluate_models(models_probs, y)

    print("\n" + "="*80)
    print("              🚀 MODEL PERFORMANCE SUMMARY")
    print("="*80)

    df_results = pd.DataFrame(results).T

    # 🔥 BEST MODEL
    best_model = df_results["EER"].idxmin()
    best_prob = models_probs[best_model][0]

    # 🔥 BEST THRESHOLD (important for ablation / reuse)
    best_thresh = find_best_threshold(y, best_prob)

    for model, row in df_results.iterrows():
        marker = "🔥 BEST" if model == best_model else ""

        print(f"{model:<20} | "
              f"Acc: {row['Accuracy']:.5f} | "
              f"AUC: {row['AUC']:.5f} | "
              f"AUPRC: {row['AUPRC']:.5f} | "
              f"EER: {row['EER']:.5f} | "
              f"tDCF: {row['tDCF']:.5f} {marker}")

    print("="*80)
    print(f"🏆 BEST MODEL: {best_model}")
    print("="*80)

    # =====================================================
    # 🔥 SAVE BEST MODEL INFO (CRITICAL FIX)
    # =====================================================
    best_model_info = {
        "best_model": best_model,
        "best_threshold": float(best_thresh)
    }

    with open(f"{MODELS_PATH}/best_model.json", "w") as f:
        import json
        json.dump(best_model_info, f, indent=4)

    print("✅ Best model info saved to models/best_model.json")

    # =====================================================
    # SAVE RESULTS
    # =====================================================
    save_results(results)

    # =====================================================
    # PLOTS
    # =====================================================
    plot_roc(models_probs, y)
    plot_det(y, best_prob)
    plot_confusion_matrix(y, best_prob)

    print("\nEvaluation Completed Successfully")


if __name__ == "__main__":
    main()