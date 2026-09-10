"""
Oasis Infobyte - Data Science Internship (OIBSIP)
Task 5: Sales Prediction Using Python

Author: Poornima
Track: Data Science
Task: Task 5 - Sales Prediction Using Python
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set visual style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.size'] = 11

def main():
    print("=" * 60)
    print("OASIS INFOBYTE SIP - TASK 5: SALES PREDICTION")
    print("=" * 60)

    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, "Advertising.csv")
    if not os.path.exists(data_path):
        data_path = os.path.join(base_dir, "Advertising Budget and Sales.csv")

    # 1. Load Dataset
    df = pd.read_csv(data_path)
    print(f"\n[1] Dataset Loaded Successfully! Shape: {df.shape}")

    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Handle column names if in Kaggle format "TV Ad Budget ($)", etc.
    column_mapping = {}
    for col in df.columns:
        if 'TV' in col:
            column_mapping[col] = 'TV'
        elif 'Radio' in col:
            column_mapping[col] = 'Radio'
        elif 'Newspaper' in col:
            column_mapping[col] = 'Newspaper'
        elif 'Sales' in col:
            column_mapping[col] = 'Sales'
            
    df.rename(columns=column_mapping, inplace=True)
    
    # Drop index column if present (e.g. Unnamed: 0)
    df = df[[col for col in ['TV', 'Radio', 'Newspaper', 'Sales'] if col in df.columns]].copy()

    print("\nCleaned Dataset Preview:")
    print(df.head())

    print("\nSummary Statistics:")
    print(df.describe())

    print("\nMissing Values:")
    print(df.isnull().sum())

    # 2. Exploratory Data Analysis
    # Pairplot
    g = sns.pairplot(df, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', height=4, aspect=1, kind='reg',
                     plot_kws={'line_kws': {'color': 'red'}})
    g.fig.suptitle("Advertising Spend vs Sales Revenue", y=1.03, fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(base_dir, "scatter_plots_spend_vs_sales.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved plot: scatter_plots_spend_vs_sales.png")

    # Correlation Heatmap
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(df.corr(), annot=True, cmap='Blues', fmt=".3f", linewidths=1, ax=ax)
    ax.set_title("Correlation Heatmap: Media Channels vs Sales", fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "correlation_heatmap_sales.png"), dpi=300)
    plt.close()
    print("Saved plot: correlation_heatmap_sales.png")

    # 3. Model Building & Evaluation
    X = df[['TV', 'Radio', 'Newspaper']]
    y = df['Sales']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # Models
    lr = LinearRegression()
    rf = RandomForestRegressor(n_estimators=100, random_state=42)

    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    lr_preds = lr.predict(X_test)
    rf_preds = rf.predict(X_test)

    def get_metrics(name, y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        return {"Model": name, "MAE": mae, "MSE": mse, "RMSE": rmse, "R2 Score": r2}

    results = pd.DataFrame([
        get_metrics("Linear Regression", y_test, lr_preds),
        get_metrics("Random Forest Regressor", y_test, rf_preds)
    ])

    print("\n--- Model Evaluation Results ---")
    print(results.to_string(index=False))

    # 4. Residual Plot & Feature Importance
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residual Plot for Random Forest
    residuals = y_test - rf_preds
    sns.scatterplot(x=rf_preds, y=residuals, ax=axes[0], color='purple', alpha=0.7)
    axes[0].axhline(y=0, color='red', linestyle='--')
    axes[0].set_title("Residual Plot (Random Forest)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Predicted Sales ($)")
    axes[0].set_ylabel("Residuals (Actual - Predicted)")

    # Feature Importance for Random Forest
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True)
    importances.plot(kind='barh', ax=axes[1], color='#3498db')
    axes[1].set_title("Feature Importance (Random Forest)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Importance Weight")

    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "residual_and_feature_importance.png"), dpi=300)
    plt.close()
    print("Saved plot: residual_and_feature_importance.png")

    # Linear Regression Coefficients
    coef_df = pd.DataFrame({"Feature": X.columns, "Coefficient": lr.coef_})
    print("\nLinear Regression Coefficients:")
    print(coef_df.to_string(index=False))
    print(f"Intercept: {lr.intercept_:.4f}")

    print("\n" + "=" * 60)
    print("SALES PREDICTION COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()
