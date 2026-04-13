import subprocess
import sys


def run_step(step_name, command):
    print("\n" + "=" * 70)
    print(f"STEP: {step_name}")
    print("=" * 70)

    result = subprocess.run(command, shell=True)

    if result.returncode != 0:
        print(f"\n❌ Pipeline stopped at step: {step_name}")
        sys.exit(1)

    print(f"✅ Completed: {step_name}")


if __name__ == "__main__":

    print("\n🚀 Starting End-to-End Audio Deepfake Detection Pipeline")

    # Step 1: Dataset Splitting
    run_step(
        "Dataset Splitting (Real vs Fake)",
        "python split_real_fake.py"
    )

    # Step 2: Audio Preprocessing
    run_step(
        "Audio Preprocessing",
        "python preprocess.py"
    )

    # Step 3: Feature Extraction (with built-in filtering)
    run_step(
        "Feature Extraction (Auto-clean + Skip Short Files)",
        "python feature_extraction.py"
    )

    # Step 4: Model Training
    run_step(
        "Model Training",
        "python train_model.py"
    )

    # Step 5: Model Evaluation
    run_step(
        "Model Evaluation",
        "python evaluate_model.py"
    )

    # Step 6: Ablation Study (separate module)
    run_step(
    "Ablation Training (Multiple Feature Configurations)",
    "python ablation_study/ablation_train.py"
    )

    run_step(
        "Ablation Analysis & Comparison",
        "python ablation_study/ablation_study.py"
    )

    print("\n🎉 PIPELINE EXECUTED SUCCESSFULLY")