# 🏥 Health Insurance Claim Predictor

An end-to-end Machine Learning pipeline and interactive web application that estimates health insurance claim payouts based on demographic, lifestyle, and clinical health features.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)

---

## 📌 Project Overview
Predicting medical claim costs helps insurance providers evaluate risk and enables individuals to estimate out-of-pocket expenses. This project evaluates multiple linear regression architectures and deploys a tuned **Lasso Regression** model inside an interactive **Streamlit** user interface.

---

## 📊 Key Results & Performance
Models were cross-validated ($K=10$) using `GridSearchCV`. **Lasso Regression** delivered the best predictive accuracy:

| Metric | Score |
| :--- | :--- |
| **$R^2$ Score** | **~87.68%** |
| **Mean Absolute Error (MAE)** | **~$2,961** |
| **Root Mean Squared Error (RMSE)** | **~$4,213** |

---

## 🛠️ Data Preprocessing & Features
* **Continuous Features (`age`, `bmi`, `bloodpressure`):** Scaled using `StandardScaler`.
* **Discrete Features (`children`):** Retained as exact integer counts.
* **Categorical Features (`gender`, `diabetic`, `smoker`, `region`):** Binary mapped and one-hot encoded with baseline dropping (`drop_first=True`).

---

## 📁 Repository Structure
```text
├── app.py                      # Interactive Streamlit Web UI & prediction logic
├── lasso_insurance_model.pkl   # Serialized Lasso model (best estimator)
├── scaler.pkl                  # Serialized StandardScaler
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
