# 🏦 Loan Payback Probability Predictor

This project was developed for the **Kaggle Swag Competition**, where the goal was to predict the likelihood of an applicant paying back their loan. By leveraging machine learning ensemble techniques and custom feature engineering, the model achieves a high level of predictive accuracy.

## 🚀 Project Overview

Predicting loan default is a critical task for financial institutions. This project focuses on analyzing applicant data, identifying hidden patterns through feature engineering, and deploying a user-friendly web interface for real-time risk assessment.

### Key Highlights:

* **Performance:** Achieved an **AUC-ROC of 92%**.
* **Ensemble Approach:** Combined LightGBM and Random Forest for robust predictions.
* **Deployment:** Fully functional web app built with **Streamlit**.

---

## 🛠️ Tech Stack

* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn, LightGBM
* **Model Deployment:** Streamlit, Joblib

---

## 📊 Methodology

### 1. Data Preprocessing & EDA

Initially, a correlation matrix showed that only `credit_score` had a strong linear relationship with `loan_payback`. This indicated that simple linear models wouldn't suffice, and significant feature engineering was required.

* Separated features into **Categorical** and **Numerical** types.
* Handled missing values and encoded categorical variables.

### 2. Feature Engineering (The Game Changer)

To capture hidden patterns, several interaction and ratio features were created. These became the most significant predictors in the final model:

* **Debt-to-Income Ratio** & **Loan-to-Income Ratio**
* **Interest Burden:** 
* **Credit Utilization:** 
* **Income Interaction:** Derived relationships between income and existing debt.

### 3. Model Architecture

An ensemble method was used by combining two powerful classifiers with specific weights:

* **LightGBM (65% weight):** Optimized for high-speed training and accuracy.
* **Random Forest (35% weight):** Used to reduce variance and prevent overfitting.

**LGBM Hyperparameters:**

```python
learning_rate=0.03, n_estimators=1000, num_leaves=50, class_weight='balanced'

```

---

## 📈 Feature Importance

The model identified the following top 15 features as the most influential in determining loan payback probability:

| Feature | Score |
| --- | --- |
| `debt_to_income_ratio` | 9442 |
| `credit_score` | 4562 |
| `credit_burden` | 4488 |
| `interest_rate` | 4337 |
| `loan_amount` | 3180 |

---

## 💻 Web Application

The model is deployed via **Streamlit**, providing a user-friendly interface for bank officials to input applicant details and receive an instant risk verdict.

**Features of the App:**

* Automatic calculation of derived features (ratios/burdens).
* Visual probability indicator (Progress bar).
* Dynamic verdicts: ✅ Strong Candidate, 🟡 Moderate Risk, or ❌ High Risk.

---

## 🏗️ Installation and Usage

1. Clone the repository:
```bash
git clone https://github.com/your-username/loan-predictor.git

```


2. Install dependencies:
```bash
pip install -r requirements.txt

```


3. Run the app:
```bash
streamlit run app.py

```



---

## 👤 Author

**Imon Hossain**

* Kaggle Swag Participant
* CSE Student & AI Enthusiast
