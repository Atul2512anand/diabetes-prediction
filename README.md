# : Logistic Regression – Diabetes Prediction

This project implements **binary classification** using **Logistic Regression** to predict whether a patient has **diabetes** based on diagnostic data. The model is trained and evaluated using various metrics and enhanced with rich visualizations to improve understanding.

---

## 🎯 Objective

To build a **Logistic Regression model** that predicts diabetes presence (1 or 0) using medical attributes like:
- Glucose level
- BMI
- Age
- Blood Pressure
- Number of Pregnancies

---

## 📁 Dataset

- **Source**: [Kaggle – Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **File Used**: `diabetes.csv`
- **Target Variable**: `Outcome` (0 = No Diabetes, 1 = Diabetes)

---

## 🧰 Tools & Libraries

- `Pandas`, `NumPy` – data handling
- `Matplotlib`, `Seaborn`, `Plotly` – visualizations
- `Scikit-learn` – modeling & evaluation

---

## 🔧 Steps Performed

### 📊 1. Data Exploration
- Distribution of the target classes
- Missing or anomalous value detection
- Data standardization using `StandardScaler`

### ⚙️ 2. Model Building
- Train-Test split (80/20)
- Logistic Regression model using `sklearn.linear_model`
- Prediction and probability outputs

### 📈 3. Evaluation Metrics
- Accuracy, Precision, Recall, F1-score
- **Confusion Matrix**
- **ROC-AUC Score**
- **Sigmoid function plot**
- **Threshold tuning and visualization**

---

## 📸 Visual Results

| Metric | Visualization |
|--------|---------------|
| Class Imbalance | ![class](screenshots/class_distribution.png) |
| Confusion Matrix | ![cm](screenshots/confusion_matrix.png) |
| ROC Curve | ![roc](screenshots/roc_curve.png) |
| Sigmoid Function | ![sigmoid](screenshots/sigmoid_curve.png) |
| Custom Threshold (0.3) | ![thresh](screenshots/threshold_03_confusion.png) |

---

## 📈 Model Performance

| Metric        | Value     |
|---------------|-----------|
| Accuracy      | ~75%      |
| Precision (1) | ~70%      |
| Recall (1)    | ~65%      |
| AUC-ROC       | ~0.83     |

---

## 🧠 Concepts Covered

- **Logistic Regression** for binary classification
- **Sigmoid Function** & probability interpretation
- **Decision threshold** tuning for optimizing precision vs recall
- **ROC Curve** and **AUC score** for model evaluation

---

## 🏥 Real-World Relevance

This model simulates how machine learning can assist healthcare professionals in:
- Predicting diabetes risks
- Prioritizing diagnostic screenings
- Optimizing recall (to catch more diabetic cases)

---

## 📂 Project Structure
Task-4-LogisticRegression/
├── diabetes.csv
├── Logistic_Regression_Diabetes.ipynb
├── README.md
└── screenshots/
├── class_distribution.png
├── confusion_matrix.png
├── roc_curve.png
├── sigmoid_curve.png
└── threshold_03_confusion.png


---

## 🚀 How to Run

1. Upload `diabetes.csv` into Colab or your IDE.
2. Run all cells in `Logistic_Regression_Diabetes.ipynb`.
3. View performance metrics and plots.
4. Adjust threshold to see precision/recall tradeoffs.

---

## 🙌 Author

**Atul Anand**  
B.Tech–M.Tech (Cyber Security)  
National Forensic Sciences University  
GitHub: [@Atul2512anand](https://github.com/Atul2512anand)

---

## 📜 License

This project is open for educational use. Dataset credit: UCI / Kaggle.



