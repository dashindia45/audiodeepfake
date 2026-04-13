import pandas as pd
import matplotlib.pyplot as plt



def plot_feature_importance(results_df, save_path):

    base_auc = results_df.loc[
        results_df["Experiment"]=="all_features","AUC"
    ].values[0]

    base_eer = results_df.loc[
        results_df["Experiment"]=="all_features","EER"
    ].values[0]

    importance = []

    for _, row in results_df.iterrows():

        if row["Experiment"] == "all_features":
            continue

        importance.append({
            "FeatureGroup": row["Experiment"].replace("no_","").capitalize(),
            "AUC_Drop": base_auc - row["AUC"],
            "EER_Increase": row["EER"] - base_eer
        })

    imp_df = pd.DataFrame(importance)

    # Sort by importance
    imp_df = imp_df.sort_values("AUC_Drop", ascending=True)

    # =========================
    # AUC PLOT
    # =========================
    plt.figure(figsize=(8,5))

    plt.barh(
        imp_df["FeatureGroup"],
        imp_df["AUC_Drop"]
    )

    plt.xlabel("AUC Drop")
    plt.title("Feature Importance (AUC Drop)")

    plt.tight_layout()
    plt.savefig(f"{save_path}/ablation_auc_importance.png")
    plt.show()

    # =========================
    # EER PLOT (VERY IMPORTANT)
    # =========================
    plt.figure(figsize=(8,5))

    plt.barh(
        imp_df["FeatureGroup"],
        imp_df["EER_Increase"]
    )

    plt.xlabel("EER Increase")
    plt.title("Feature Importance (EER Increase)")

    plt.tight_layout()
    plt.savefig(f"{save_path}/ablation_eer_importance.png")
    plt.show()

    print("\n📊 Feature importance plots saved!")