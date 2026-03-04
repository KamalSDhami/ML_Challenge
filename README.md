# alrIEEEna'26 — ML Challenge

> Binary Fault Detection using Machine Learning & Deep Learning

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-latest-orange.svg)](https://xgboost.readthedocs.io)
[![LightGBM](https://img.shields.io/badge/LightGBM-latest-green.svg)](https://lightgbm.readthedocs.io)

---

## 📌 Problem Statement

Given 47 numerical features (`F01`–`F47`) captured by an embedded detection system, predict whether a device is operating **normally (Class 0)** or experiencing a **fault condition (Class 1)**.

| Property | Value |
|----------|-------|
| Task | Binary Classification |
| Features | 47 numerical |
| Train samples | ~43,777 |
| Test samples | ~10,945 |
| Target | `Class` (0 = Normal, 1 = Faulty) |

---

## 📁 Project Structure

```
├── Dataset/
│   ├── TRAIN.csv          # Training data (F01–F47 + Class)
│   ├── TEST.csv           # Test data (ID + F01–F47)
│   └── readme.txt         # Dataset description
├── main.ipynb             # Complete ML pipeline notebook
├── FINAL.csv              # Submission file (ID, Class)
├── info.txt               # Event details
├── Terms&Condition.txt    # Competition rules
└── README.md              # This file
```

---

## 🔧 Pipeline Overview

### 1. Data Loading & Sanity Checks
- Load `TRAIN.csv` and `TEST.csv`
- Verify shapes, dtypes, missing values, duplicates

### 2. Exploratory Data Analysis (EDA)
- Class distribution (bar + pie charts)
- Feature distributions (histograms, KDE by class)
- Correlation heatmap & highly correlated pairs
- Box plots for outlier detection
- PCA 2D visualization & cumulative variance

### 3. Preprocessing & Feature Engineering
- Remove duplicates
- Drop highly correlated features (`|r| > 0.95`)
- **Statistical features:** row-wise mean, std, min, max, range, median, skew, kurtosis, IQR
- **Interaction features:** multiply & divide top target-correlated feature pairs
- Drop near-zero mutual information features
- Scale with `RobustScaler`
- Stratified train/validation split (80/20)

### 4. Classical ML Models
| Model | Details |
|-------|---------|
| Logistic Regression | Baseline, `class_weight='balanced'` |
| Random Forest | 500 trees, balanced weights |
| XGBoost | Early stopping, `scale_pos_weight` for imbalance |
| LightGBM | Early stopping, `is_unbalance=True` |

- **Hyperparameter tuning:** Optuna (80 trials, 5-fold StratifiedKFold CV)
- **Metrics:** Accuracy, F1, Precision, Recall, AUC-ROC

### 5. Deep Learning (PyTorch)
- **Architecture:** Feedforward NN (256→128→64→32→1) with BatchNorm + Dropout
- **Loss:** `BCEWithLogitsLoss` with `pos_weight` for class imbalance
- **Optimizer:** Adam + ReduceLROnPlateau scheduler
- **Early stopping:** patience = 15 epochs
- `WeightedRandomSampler` for balanced batches

### 6. Ensemble
- Weighted average of probabilities from tuned LightGBM, tuned XGBoost, and Neural Network
- Automatic weight search across multiple combinations
- Best ensemble selected by validation F1

### 7. Submission
- `FINAL.csv` with columns `ID` and `Class`
- Validated: correct row count, valid class values, no duplicate IDs

---

## 🚀 Quick Start

### Local
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm optuna torch
jupyter notebook main.ipynb
```

### Google Colab
1. Upload `Dataset/TRAIN.csv` and `Dataset/TEST.csv`
2. Open `main.ipynb` in Colab
3. Set runtime to **GPU** (Runtime → Change runtime type → T4 GPU)
4. Run all cells

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `pandas`, `numpy` | Data manipulation |
| `matplotlib`, `seaborn` | Visualization |
| `scikit-learn` | Preprocessing, metrics, baseline models |
| `xgboost` | Gradient boosting |
| `lightgbm` | Gradient boosting |
| `optuna` | Hyperparameter optimization |
| `torch` | Neural network |

---

## 📊 Key Results

Results are generated in the notebook after running all cells. The final comparison table ranks all models by F1-score and identifies the best approach (typically the ensemble or a tuned tree model).

---

## 🏆 Competition

**alrIEEEna'26** — State-level ML Challenge by IEEE Student Branch, Graphic Era Hill University, Dehradun.

- **Round 1 (Online):** Submit optimized ML solutions via GitHub
- **Grand Finale (Offline):** Mar 20, 2026
- **Evaluation:** Innovation, accuracy, performance metrics, code quality

---

## 📜 License

This project is for educational purposes as part of the alrIEEEna'26 ML Challenge. No ownership is declared over the dataset.
