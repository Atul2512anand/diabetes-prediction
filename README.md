# Diabetes Prediction with Logistic Regression

![Python](https://img.shields.io/badge/Python-3-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-F37626?logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/Data-pandas-150458?logo=pandas&logoColor=white)

A binary-classification project that predicts diabetes from diagnostic measurements using logistic regression, with a full evaluation workflow: confusion matrix, classification report, ROC/AUC, sigmoid curve, and threshold tuning.

## Overview

The notebook uses the Pima Indians Diabetes dataset to demonstrate how logistic regression turns eight clinical/measurement features into a probability of diabetes. It focuses on interpretation as much as accuracy: how the sigmoid works, how the default 0.5 threshold behaves, and how lowering the threshold changes the precision/recall trade-off.

> **Educational project — not a medical device or diagnostic tool.** The dataset is small and historical; the results must not be used for clinical decisions.

## Dataset

| Item | Value |
|---|---|
| Source | [Kaggle — Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) |
| File | `diabetes.csv` |
| Rows | 768 |
| Columns | 9 |
| Target | `Outcome` (0 = no diabetes, 1 = diabetes) |

### Features

```text
Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
BMI, DiabetesPedigreeFunction, Age
```

## Method

The notebook has eight cells:

1. Load the CSV in Colab and preview it.
2. Plot the class distribution (`Outcome`).
3. Separate features (`X`) from the target (`y`).
4. Standardise all features with `StandardScaler`.
5. Split into 80% train / 20% test (`random_state=42`).
6. Fit `sklearn.linear_model.LogisticRegression`.
7. Evaluate:
   - confusion matrix
   - precision, recall, F1-score, and accuracy
   - ROC curve and AUC
   - sigmoid activation curve
8. Re-evaluate at a custom threshold of **0.30** to show the effect on the confusion matrix.

### Core Code

```python
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_scaled = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LogisticRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
```

## Results

Test set: **154 samples** (99 negative, 55 positive).

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| 0 — No diabetes | 0.81 | 0.80 | 0.81 | 99 |
| 1 — Diabetes | 0.65 | 0.67 | 0.66 | 55 |
| **Accuracy** | | | **0.75** | **154** |
| Macro average | 0.73 | 0.74 | 0.73 | 154 |
| Weighted average | 0.76 | 0.75 | 0.75 | 154 |

- **ROC-AUC:** approximately **0.83** (shown in the notebook's ROC plot).
- **Threshold 0.30:** the notebook includes a second confusion matrix. Lowering the threshold generally increases diabetes recall at the cost of more false positives, which is the relevant trade-off when missed cases are costly.

## Getting Started

### Google Colab

1. Open `Diabetes_prediction.ipynb` in Colab.
2. Run the first cell and upload `diabetes.csv`.
3. Run the remaining cells.

### Local Jupyter

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook Diabetes_prediction.ipynb
```

Skip the `google.colab.files.upload()` cell and confirm that `pd.read_csv("diabetes.csv")` resolves to the included file.

## Project Structure

```text
diabetes-prediction/
├── Diabetes_prediction.ipynb     # 8-cell notebook
├── diabetes.csv                  # Pima Indians dataset (768 × 9)
├── screenshots/                  # Exported notebook figures
└── README.md
```

## Notes and Limitations

- **Invalid zeros are not handled.** `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` contain zero values that are physiologically implausible and likely represent missing data. They are currently treated as real measurements.
- **Scaling before splitting:** `StandardScaler` is fitted on the full dataset before the train/test split. For a rigorous evaluation, fit it on the training set only and transform the test set with those parameters.
- **No cross-validation:** the reported metrics come from a single 80/20 split.
- **Class imbalance:** roughly 65% negative / 35% positive. Accuracy alone can be misleading; precision, recall, F1, and AUC are reported for that reason.
- **No feature selection or hyperparameter tuning.**
- **No calibration or decision-curve analysis.**
- **Not clinical advice:** the model is a teaching example and has not been validated for medical use.

## Data Credit

Pima Indians Diabetes Database, originally from the UCI Machine Learning Repository and distributed on Kaggle by **uciml**. Refer to the Kaggle page for licensing and usage terms.

## Author

**Atul Anand**  
B.Tech–M.Tech (Cyber Security), National Forensic Sciences University  
GitHub: [@Atul2512anand](https://github.com/Atul2512anand)
