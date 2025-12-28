# 🏦 Bank Conversion Intelligence

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen?style=for-the-badge)](https://github.com/zyna-b/Bank-Conversion-Intelligence/issues)

> **Predict customer subscription to bank term deposits using Machine Learning** — An end-to-end data science project that transforms raw marketing campaign data into actionable business intelligence.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Business Problem](#-business-problem)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Key Findings](#-key-findings)
- [Model Performance](#-model-performance)
- [Feature Importance](#-feature-importance)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results & Business Recommendations](#-results--business-recommendations)
- [Technologies Used](#-technologies-used)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

**Bank Conversion Intelligence** is a machine learning project designed to predict whether a bank customer will subscribe to a term deposit based on direct marketing campaign data. This project demonstrates the complete ML pipeline from data cleaning to model deployment insights.

### What This Project Does:
- ✅ Predicts customer subscription likelihood with **85% accuracy**
- ✅ Identifies the **top 10 drivers** of customer conversion
- ✅ Provides **actionable business recommendations** for marketing optimization
- ✅ Compares multiple ML algorithms (Logistic Regression, Decision Tree, Random Forest)

---

## 💼 Business Problem

### The Challenge
Banks conduct direct marketing campaigns (phone calls) to sell term deposits to customers. However:
- **Only ~11% of customers** actually subscribe
- Cold-calling is **expensive and time-consuming**
- Sales teams waste resources on uninterested leads

### The Solution
Build a predictive model that:
1. **Identifies high-potential customers** before the call
2. **Reduces wasted calls** by filtering unlikely converters
3. **Maximizes ROI** by focusing on warm leads

---

## 📊 Dataset

| Attribute | Description |
|-----------|-------------|
| **Source** | [UCI Machine Learning Repository - Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing) |
| **Records** | 41,188 customer records |
| **Features** | 20 attributes (demographic, campaign, economic) |
| **Target** | `y` - Has the client subscribed to a term deposit? (yes/no) |

### Feature Categories

| Category | Features |
|----------|----------|
| **Demographics** | age, job, marital status, education |
| **Financial** | housing loan, personal loan |
| **Campaign Data** | contact type, month, day, duration, campaign count |
| **Previous Campaign** | pdays, previous contacts, outcome |
| **Economic Indicators** | employment rate, consumer price index, euribor rate |

---

## 🔄 Project Workflow

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Data Loading   │───▶│  Data Cleaning  │───▶│      EDA        │
│   & Inspection  │    │  & Preprocessing│    │  (Exploration)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                      │
                                                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│    Business     │◀───│     Model       │◀───│    Feature      │
│    Insights     │    │   Comparison    │    │   Engineering   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Step-by-Step Process

1. **Data Cleaning**
   - Removed 12 duplicate records
   - Dropped `default` column (99% unknown values)
   - Dropped `duration` column (prevents data leakage)
   - Imputed `education` unknowns using job-group mode

2. **Feature Engineering**
   - Created `previously_contacted` binary feature from `pdays`
   - Winsorized `campaign` outliers at 99th percentile
   - One-hot encoded all categorical variables

3. **Data Preprocessing**
   - Train/Test split: 80/20 with stratification
   - StandardScaler normalization for numerical features

4. **Model Training & Evaluation**
   - Tested 3 algorithms with `class_weight='balanced'`
   - Evaluated using Precision, Recall, F1-Score, and Accuracy

---

## 🔍 Key Findings

### Class Imbalance Discovery
```
Target Distribution:
├── No (Did not subscribe):  88.7%
└── Yes (Subscribed):        11.3%
```

### Unknown Value Analysis
| Column | Unknown % | Action Taken |
|--------|-----------|--------------|
| `default` | 20.9% | Dropped column |
| `education` | 4.2% | Imputed by job group |
| `job` | 0.8% | Dropped rows |
| `marital` | 0.2% | Dropped rows |
| `housing` & `loan` | 2.4% | Kept as category ("declined to answer") |

---

## 📈 Model Performance

### Comparison Summary

| Model | Accuracy | Precision (Yes) | Recall (Yes) | F1-Score |
|-------|----------|-----------------|--------------|----------|
| Logistic Regression | 77% | 36% | **63%** | 46% |
| Decision Tree | 79% | 30% | 33% | 31% |
| **Random Forest** 🏆 | **85%** | **40%** | **61%** | **48%** |

### Model Analysis

#### 🥉 Logistic Regression (Baseline)
- **Strategy:** Linear model with balanced class weights
- **Result:** High Recall (63%) but Low Precision (36%)
- **Business Interpretation:** "Aggressive Sales Manager" — catches most buyers but wastes resources on many non-buyers

#### 🥈 Decision Tree (Challenger)
- **Strategy:** Non-linear model to capture complex patterns
- **Result:** Failed with only 33% Recall and 30% Precision
- **Business Interpretation:** Overfitting to training noise — "fired" this model

#### 🥇 Random Forest (Champion) ✅
- **Strategy:** Ensemble of 100 trees with max_depth=10
- **Result:** Best trade-off with 61% Recall and 40% Precision
- **Business Interpretation:** "Efficient Sales Director" — maintains high buyer capture while reducing wasted calls by 4%

### Why Not 90% Precision?

The **~40% precision ceiling** exists due to:
1. **Data Limitations:** No real-time income, debt levels, or immediate financial needs
2. **Human Variance:** Emotional/external factors invisible to the model

> **To break 50% precision:** Integrate external data like credit scores or spending habits.

---

## 🎯 Feature Importance

### Top 10 Drivers of Customer Conversion

| Rank | Feature | Category | Importance |
|------|---------|----------|------------|
| 1️⃣ | `euribor3m` | Economic | ⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛ |
| 2️⃣ | `nr.employed` | Economic | ⬛⬛⬛⬛⬛⬛⬛⬛⬛ |
| 3️⃣ | `age` | Demographic | ⬛⬛⬛⬛⬛⬛⬛ |
| 4️⃣ | `pdays` | Campaign | ⬛⬛⬛⬛⬛⬛ |
| 5️⃣ | `emp.var.rate` | Economic | ⬛⬛⬛⬛⬛ |
| 6️⃣ | `cons.price.idx` | Economic | ⬛⬛⬛⬛ |
| 7️⃣ | `campaign` | Campaign | ⬛⬛⬛ |
| 8️⃣ | `cons.conf.idx` | Economic | ⬛⬛⬛ |
| 9️⃣ | `previous` | Campaign | ⬛⬛ |
| 🔟 | `previously_contacted` | Engineered | ⬛⬛ |

### Key Insight: **Macroeconomics > Demographics**

> 📊 **Interest rates and employment levels predict conversions better than customer age or job.**

---

## 💡 Results & Business Recommendations

### Strategic Recommendations

| Strategy | Action | Expected Impact |
|----------|--------|-----------------|
| **Dynamic Scheduling** | Call aggressively when interest rates are favorable | 📈 +15-20% conversion |
| **Retarget Warm Leads** | Prioritize customers contacted in last 30 days | 📈 +10% efficiency |
| **Reduce Cold Calls** | Filter out low-probability segments | 💰 -25% call costs |
| **Economic Monitoring** | Track euribor3m and employment rates weekly | ⏰ Timing optimization |

### ROI Projection

```
Before ML Model:
├── Calls per day: 1,000
├── Conversion rate: 11%
└── Successful conversions: 110

After ML Model (Top 40% leads):
├── Calls per day: 400
├── Conversion rate: 27.5% (from model precision)
└── Successful conversions: 110
└── 💰 Cost savings: 60% fewer calls for same revenue!
```

---

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or Google Colab

### Setup

```bash
# Clone the repository
git clone https://github.com/zyna-b/Bank-Conversion-Intelligence.git
cd Bank-Conversion-Intelligence

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Required Libraries

```python
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=0.24.0
```

---

## 🚀 Usage

### Run the Notebook

```bash
jupyter notebook Bank_Conversion_Intelligence.ipynb
```

### Quick Start (Python)

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Load your data
df = pd.read_csv('bank-additional-full.csv', sep=';')

# Preprocess (see notebook for full pipeline)
# ...

# Train the champion model
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',
    random_state=42
)
rf_model.fit(X_train_scaled, y_train)

# Predict
predictions = rf_model.predict(X_test_scaled)
```

---

## 🔧 Technologies Used

| Technology | Purpose |
|------------|---------|
| ![Python](https://img.shields.io/badge/-Python-3776AB?style=flat&logo=python&logoColor=white) | Core programming language |
| ![Pandas](https://img.shields.io/badge/-Pandas-150458?style=flat&logo=pandas&logoColor=white) | Data manipulation & analysis |
| ![NumPy](https://img.shields.io/badge/-NumPy-013243?style=flat&logo=numpy&logoColor=white) | Numerical computing |
| ![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557c?style=flat&logo=python&logoColor=white) | Data visualization |
| ![Seaborn](https://img.shields.io/badge/-Seaborn-3776AB?style=flat&logo=python&logoColor=white) | Statistical visualization |
| ![scikit-learn](https://img.shields.io/badge/-Scikit--Learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) | Machine learning models |
| ![Jupyter](https://img.shields.io/badge/-Jupyter-F37626?style=flat&logo=jupyter&logoColor=white) | Interactive development |

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Ideas for Contribution
- [ ] Add XGBoost/LightGBM models
- [ ] Implement SHAP explanations
- [ ] Create a Flask/Streamlit web app
- [ ] Add cross-validation analysis
- [ ] Integrate hyperparameter tuning

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📬 Contact

**Zyna B** - [GitHub](https://github.com/zyna-b)

Project Link: [https://github.com/zyna-b/Bank-Conversion-Intelligence](https://github.com/zyna-b/Bank-Conversion-Intelligence)

---

<p align="center">
  <b>⭐ If this project helped you, please give it a star! ⭐</b>
</p>

---

## 📚 References

- [UCI ML Repository - Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
- [Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014
