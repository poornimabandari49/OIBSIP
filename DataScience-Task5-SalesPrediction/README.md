# Task 5: Sales Prediction Using Python

**Oasis Infobyte Internship (OIBSIP) — Data Science Track**  
**Author:** Poornima  
**Repository Name:** `OIBSIP`  

---

## 📌 Project Overview
This project focuses on predicting product sales revenue based on advertising expenditure across three primary marketing channels: **TV**, **Radio**, and **Newspaper**. Using the Kaggle Advertising dataset (`Advertising.csv`), we perform Exploratory Data Analysis, fit multiple regression models (**Linear Regression** baseline and **Random Forest Regressor**), and quantify marketing channel ROI to guide advertising budget allocation.

---

## 🛠️ Tech Stack & Libraries
- **Language:** Python 3.13
- **Data Preprocessing & EDA:** `pandas`, `numpy`, `matplotlib`, `seaborn`
- **Machine Learning Models:** `LinearRegression`, `RandomForestRegressor`
- **Evaluation Metrics:** `mean_absolute_error` (MAE), `mean_squared_error` (MSE/RMSE), `r2_score` ($R^2$)

---

## 📊 Model Performance Comparison

| Model | MAE | RMSE | $R^2$ Score | Performance Summary |
| :--- | :---: | :---: | :---: | :--- |
| **Random Forest Regressor** | **0.6200** | **0.7662** | **98.14%** | Exceptional fit capturing non-linear interactions |
| **Linear Regression** | 1.4608 | 1.7816 | 89.94% | Strong linear baseline model |

---

## 💡 Key Marketing Insights & ROI Analysis
1. **TV Advertising is the Dominant Growth Driver:**
   - TV ad budget exhibits the highest overall correlation with sales revenue ($r > 0.78$) and feature importance weight (~80%).
2. **Radio Advertising Provides Significant Synergy:**
   - Radio spend acts as an effective secondary channel, contributing substantial sales lift per dollar spent.
3. **Newspaper Advertising Offers Minimal ROI:**
   - Newspaper spend displays a near-zero correlation ($r \approx 0.23$) and low regression coefficient ($0.0028$), indicating that budget spent on newspapers yields negligible returns.

---

## 📂 Project Structure
```text
DataScience-Task5-SalesPrediction/
├── Sales_Prediction.ipynb              # Interactive Jupyter Notebook with regression models & plots
├── sales_prediction.py                  # Standalone Python sales prediction script
├── Advertising.csv                      # Kaggle Advertising Dataset (200 records)
├── scatter_plots_spend_vs_sales.png     # Scatter plots with trendlines per channel
├── correlation_heatmap_sales.png        # Correlation matrix plot
├── residual_and_feature_importance.png  # Residual error plot & Feature Importance chart
└── README.md                            # Documentation report
```

---

## 🚀 How to Run

1. **Install Dependencies:**
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter
   ```
2. **Execute Python Script:**
   ```bash
   python sales_prediction.py
   ```
3. **Run Jupyter Notebook:**
   ```bash
   jupyter notebook Sales_Prediction.ipynb
   ```
