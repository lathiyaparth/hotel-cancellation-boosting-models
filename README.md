# 🏨 Hotel Booking Cancellation Prediction

> End-to-end machine learning project to predict whether a hotel booking will be cancelled, comparing Decision Tree, Random Forest and five boosting algorithms (AdaBoost, Gradient Boosting, XGBoost, LightGBM, CatBoost).

**Project by:** Creative Institute of Data Science
**Author:** `<Your Name>`

---

## 📌 Table of Contents

1. [Project Overview](#-project-overview)
2. [Business Problem](#-business-problem)
3. [Dataset](#-dataset)
4. [Project Workflow](#-project-workflow)
5. [Feature Engineering](#-feature-engineering)
6. [Models Built](#-models-built)
7. [Evaluation Metrics](#-evaluation-metrics)
8. [Results](#-results)
9. [Key Business Insights](#-key-business-insights)
10. [Tech Stack](#-tech-stack)
11. [Project Structure](#-project-structure)
12. [How to Run](#-how-to-run)
13. [Concepts Explained](#-concepts-explained)
14. [Future Improvements](#-future-improvements)
15. [Author](#-author)

---

## 📖 Project Overview

Hotel booking cancellations hurt revenue and make room planning difficult. This project analyses historical booking data, performs thorough data cleaning and EDA, engineers useful features, and trains multiple classification models to predict **`is_canceled`** (1 = cancelled, 0 = not cancelled).

The project covers the complete data science pipeline:

- Data loading, cleaning and validation
- Outlier detection and handling (IQR method)
- Feature engineering (guests, stay length, family flag, room change, date features)
- Exploratory Data Analysis (EDA) on cancellation behaviour
- Data leakage detection and removal
- Encoding, train/test split and model building
- Hyperparameter tuning and 5-Fold Cross Validation
- Model comparison, feature importance and ROC analysis
- Predicting cancellation probability for new bookings

---

## 🎯 Business Problem

> *Can we predict, at the time of booking, whether a customer is likely to cancel — so that the hotel can reduce lost revenue through overbooking strategy, deposit policies and targeted customer engagement?*

---

## 📂 Dataset

- **Name:** Hotel Booking Demand dataset
- **Download:** [Google Drive link](https://drive.google.com/file/d/1DFdYLmYoZh7E4H0efMPhLao2q2n6gPpV/view?usp=sharing)
- **Target variable:** `is_canceled`
- **Hotel types:** City Hotel and Resort Hotel

**Important columns include:** `hotel`, `lead_time`, `arrival_date_year`, `arrival_date_month`, `arrival_date_day_of_month`, `stays_in_weekend_nights`, `stays_in_week_nights`, `adults`, `children`, `babies`, `country`, `market_segment`, `deposit_type`, `customer_type`, `adr`, `is_repeated_guest`, `previous_cancellations`, `reserved_room_type`, `assigned_room_type`, `total_of_special_requests`, etc.

> ⚠️ Place the downloaded CSV inside the `data/` folder before running the notebook.

---

## 🔄 Project Workflow

| Phase | Tasks |
|-------|-------|
| **1. Data Understanding** | Load data, view first 5 rows, shape, column names, data types |
| **2. Data Cleaning** | Missing values (count + %), imputation, duplicate detection & removal, invalid bookings (0 guests) |
| **3. Outlier Analysis** | IQR method on `adr`, `lead_time`, `stays_in_week_nights`, `stays_in_weekend_nights` |
| **4. Feature Engineering** | `total_guests`, `total_stay`, `is_family`, `room_changed`, `arrival_date`, date-based features |
| **5. EDA** | Cancellation % and class balance; cancellation by hotel, market segment, customer type, deposit type, lead time, ADR, special requests, previous cancellations, repeated guests, country |
| **6. Preprocessing** | Identify categorical/numerical columns, drop unnecessary columns, **remove data leakage**, encode categoricals, split X/y, train-test split |
| **7. Modeling** | Decision Tree → Random Forest → AdaBoost → Gradient Boosting → XGBoost → LightGBM → CatBoost |
| **8. Tuning & Validation** | Hyperparameter tuning, 5-Fold CV, before vs after tuning comparison |
| **9. Evaluation** | Metrics, confusion matrices, ROC curves, feature importance, overfitting check |
| **10. Deployment-style Prediction** | Predict cancellation and probability for new booking data |

---

## 🛠 Feature Engineering

| Feature | Description |
|---------|-------------|
| `total_guests` | `adults + children + babies` |
| `total_stay` | `stays_in_weekend_nights + stays_in_week_nights` |
| `is_family` | 1 if booking includes children or babies, else 0 |
| `room_changed` | 1 if `reserved_room_type` ≠ `assigned_room_type` |
| `arrival_date` | Proper datetime built from year, month and day columns |
| Date features | Arrival month, weekday, week number, quarter, weekend flag, season, etc. |

### ⚠️ Data Leakage Handling
Columns that reveal the outcome after the booking is made (for example `reservation_status` and `reservation_status_date`) are **removed before model training** to avoid unrealistically high scores.

---

## 🤖 Models Built

| # | Model | Tuned Parameters |
|---|-------|------------------|
| 1 | Decision Tree | – (overfitting check via train vs test accuracy) |
| 2 | Random Forest | – |
| 3 | AdaBoost | `n_estimators`, `learning_rate` |
| 4 | Gradient Boosting | `n_estimators`, `learning_rate`, `max_depth` |
| 5 | XGBoost | `n_estimators`, `learning_rate`, `max_depth`, `gamma`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda` |
| 6 | LightGBM | `n_estimators`, `learning_rate`, `num_leaves`, `max_depth` |
| 7 | CatBoost (with native categorical features) | `iterations`, `learning_rate`, `depth`, `l2_leaf_reg` |

---

## 📏 Evaluation Metrics

- Train Accuracy & Test Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Training Time
- Confusion Matrix
- 5-Fold Cross Validation score

---

## 📊 Results

> Fill in these tables after running the notebook.

### Model Comparison

| Model | Train Acc | Test Acc | Precision | Recall | F1 | ROC-AUC | Training Time (s) |
|-------|-----------|----------|-----------|--------|----|---------|-------------------|
| Decision Tree | | | | | | | |
| Random Forest | | | | | | | |
| AdaBoost | | | | | | | |
| Gradient Boosting | | | | | | | |
| XGBoost | | | | | | | |
| LightGBM | | | | | | | |
| CatBoost | | | | | | | |

### Before vs After Tuning

| Model | ROC-AUC (Before) | ROC-AUC (After) | Best Parameters |
|-------|------------------|-----------------|-----------------|
| XGBoost | | | |
| LightGBM | | | |
| CatBoost | | | |

### 🏆 Best Model
**`<Model name>`** — selected because of `<balanced precision/recall, highest ROC-AUC, low overfitting, reasonable training time>`.

### Top 10 Important Features
*(Add feature importance chart here)*

```
![Feature Importance](images/feature_importance.png)
![ROC Curves](images/roc_curves.png)
```

---

## 💡 Key Business Insights

Replace with your findings from the analysis. Typical areas to cover:

1. **Lead time** – longer lead times generally show higher cancellation rates.
2. **Deposit type** – bookings with non-refundable / no-deposit policies behave very differently.
3. **Market segment** – online travel agents / groups tend to cancel more than direct bookings.
4. **Previous cancellations & repeated guests** – past behaviour is a strong signal.
5. **Special requests** – guests with special requests are usually less likely to cancel.

---

## 🧰 Tech Stack

- **Language:** Python 3.x
- **Data:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **ML:** Scikit-learn, XGBoost, LightGBM, CatBoost
- **Environment:** Jupyter Notebook / Google Colab

---

## 📁 Project Structure

```
hotel-booking-cancellation-prediction/
│
├── data/
│   └── hotel_bookings.csv
├── notebooks/
│   └── hotel_cancellation_prediction.ipynb
├── images/
│   ├── roc_curves.png
│   └── feature_importance.png
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/hotel-booking-cancellation-prediction.git
cd hotel-booking-cancellation-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add the dataset to the data/ folder

# 4. Launch the notebook
jupyter notebook notebooks/hotel_cancellation_prediction.ipynb
```

**`requirements.txt`**

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
lightgbm
catboost
jupyter
```

---

## 📚 Concepts Explained

<details>
<summary><b>Bagging vs Boosting</b></summary>

- **Bagging** (e.g., Random Forest) trains many models **in parallel** on random samples and averages them → reduces **variance**.
- **Boosting** (e.g., AdaBoost, XGBoost) trains models **sequentially**, each correcting the errors of the previous one → reduces **bias**.
</details>

<details>
<summary><b>AdaBoost vs Gradient Boosting</b></summary>

AdaBoost increases the **weights of misclassified samples**; Gradient Boosting fits new trees on the **residual errors (gradients of a loss function)**, which allows custom loss functions.
</details>

<details>
<summary><b>Gradient Boosting vs XGBoost</b></summary>

XGBoost adds **L1/L2 regularization**, **parallel tree construction**, built-in missing-value handling, pruning with `gamma`, and is generally faster and more regularized.
</details>

<details>
<summary><b>XGBoost vs LightGBM</b></summary>

XGBoost grows trees **level-wise**; LightGBM grows **leaf-wise** with histogram-based splitting, making it faster and lighter on memory, though it can overfit on small data.
</details>

<details>
<summary><b>XGBoost vs CatBoost</b></summary>

CatBoost handles **categorical features natively** (ordered target statistics) and uses **symmetric (oblivious) trees**, needing less preprocessing and often less tuning; XGBoost needs encoded categoricals.
</details>

<details>
<summary><b>LightGBM vs CatBoost</b></summary>

LightGBM prioritizes **speed** and leaf-wise growth; CatBoost prioritizes **categorical handling and stability** with symmetric trees and ordered boosting to reduce target leakage.
</details>

<details>
<summary><b>Hyperparameters</b></summary>

| Parameter | Meaning |
|-----------|---------|
| `learning_rate` | Step size of each tree's contribution; lower = slower but more robust learning |
| `n_estimators` | Number of trees / boosting rounds |
| `gamma` (XGBoost) | Minimum loss reduction required to make a split; higher = more conservative |
| `reg_alpha` | L1 regularization on leaf weights (promotes sparsity) |
| `reg_lambda` | L2 regularization on leaf weights (shrinks weights) |
| `num_leaves` (LightGBM) | Maximum leaves per tree; main control of model complexity |
| `iterations` (CatBoost) | Number of boosting rounds (equivalent to `n_estimators`) |
</details>

<details>
<summary><b>Why the highest accuracy model is not always the best</b></summary>

Accuracy can be misleading on **imbalanced data**, ignores the **cost of false negatives vs false positives**, and may hide **overfitting**, slow training or poor interpretability. Precision, recall, F1, ROC-AUC, cross-validation and business cost should be considered together.
</details>

---

## 🚀 Future Improvements

- Use SHAP for model explainability
- Handle class imbalance with SMOTE / class weights if required
- Deploy as a Streamlit / Flask web app
- Add automated pipelines with `sklearn.pipeline`
- Experiment with Optuna for advanced hyperparameter search

---

## 👤 Author

**`<Parth N Lathiya>`**
📧 `<lathiyaparth2004@gmail.com>` | 🔗 [LinkedIn](https://linkedin.com/in/your-profile](https://www.linkedin.com/public-profile/settings/?trk=d_flagship3_profile_self_view_public_profile&lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base%3B%2BtrYq52vTbiak41C%2FzQHCg%3D%3D)) | 💻 [GitHub](https://github.com/lathiyaparth)

⭐ If you found this project useful, please star the repository!
