#  Audio Deepfake Detection using Hybrid Acoustic Features

##  Overview

This project presents a lightweight and interpretable framework for detecting audio deepfakes using hybrid acoustic features. It combines multiple feature types with efficient machine learning models to achieve strong performance with low computational cost.

---

##  Key Features

* Hybrid feature extraction (MFCC, spectral, temporal, etc.)
* Lightweight ML models (Random Forest, SVM, XGBoost)
* Ablation-based feature analysis
* Real-time capable system
* Interpretable pipeline

---

## 📂 Project Structure

```
audiodeepfake/
│
├── data/  
│   ├── raw          #raw audio files after dataset split
│   │    ├── train
│   │    ├── dev
│   │    └── eval
│   └── processed    #processed after processing 
│       ├── train
│       ├── dev
│       └── eval
├── features/
├── models/
├── results/
├── LA/           # Actual Asvproof LA dataset
│
├── Ablation_study/
│   ├── ablation_train.py
│   └── ablation_study.py
│
├── src/
│   ├── split_real_fake.py
│   ├── preprocess.py
│   ├── feature_extraction.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── run_pipeline.py
│   └── sanity_check.py    
│   └── remove_badfiles.py  # Optional as we are skipping bad/crupt files in extracting feature script
│
├── venv
├── requirements.txt
├── README.md
└── setup.sh
```

---

## ⚙️ Installation & Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

###  Option 1: Run everything (Recommended)

```bash
bash setup.sh
```

---

###  Option 2: Run manually

```bash
python src/run_pipeline.py
```

---

###  Option 3: Run step-by-step

```bash
python src/split_real_fake.py
python src/preprocess.py
python src/feature_extraction.py
python src/train_model.py
python src/evaluate_model.py
python ablation_study/ablation_train.py
python ablation_study/ablation_study.py
```

---

## ⚠️ If you face package/version issues

It is recommended to use a virtual environment.

### 1. Create virtual environment

```bash
python -m venv venv
```

### 2. Activate environment

* **Windows**

```bash
venv\Scripts\activate
```

* **Linux/Mac**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 💡 Tips

* Use **setup.sh** for easiest execution
* Use **manual steps** for debugging or control
* Use **virtual environment** if dependency issues occur

```


## 🔄 Pipeline Flow

Audio → Preprocessing → Feature Extraction → Model Training → Evaluation → Ablation Study

---

## 📊 Evaluation Metrics

* Accuracy
* AUC
* EER (Equal Error Rate)
* t-DCF

---

## 🧠 Key Insight

* Feature selection is more important than complex models
* Lightweight ML can achieve strong performance
* Ablation study helps identify useful features

---

## 📌 Dataset
ASVspoof 2019 LA dataset with:
- Train → model training  
- Dev → validation & tuning  
- Eval → final testing (unseen data)

---


##  Who is cloning from github note this points
## 📌 Data Setup

This project uses the **ASVspoof 2019 Logical Access (LA)** dataset.

### 🔽 Download Dataset

Download the dataset from the official source .

After downloading, place it in the project directory as:

```
LA/
```

---

## 📂 Data Organization

### 📁 `LA/` (Raw Dataset)

* Contains original ASVspoof audio files
* Includes train, dev, and eval splits
* Not included in this repository due to large size

---

### 📁 `data/`

This folder is used during processing:

* **raw/** → intermediate audio files after initial preprocessing
* **processed/** → cleaned and standardized audio used for feature extraction

👉 These folders are automatically created during pipeline execution.

---

### 📁 `features/`

* Stores extracted feature files (CSV format)
* Used for training and evaluation

---

### 📁 `models/`

* Contains trained machine learning models
* Saved as `.pkl` files

---

### 📁 `results/`

* Stores output results such as:

  * Evaluation metrics
  * Plots and analysis results

---

## ⚠️ Important Notes

* `LA/` folder must be provided manually
* `data/`, `features/`, `models/`, and `results/` are generated automatically


---

## ▶️ After Setup

Once the dataset is placed correctly, run:

```bash
bash setup.sh
```

or

```bash
python src/run_pipeline.py
```


## 👨‍💻 Author

Dashrath Kumar
CSE, NIT Warangal

---
