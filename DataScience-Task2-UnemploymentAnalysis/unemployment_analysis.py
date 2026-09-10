"""
Oasis Infobyte - Data Science Internship (OIBSIP)
Task 2: Unemployment Analysis with Python

Author: Poornima
Track: Data Science
Task: Task 2 - Unemployment Analysis with Python
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.size'] = 11
plt.rcParams['figure.titlesize'] = 14

def main():
    print("=" * 60)
    print("OASIS INFOBYTE SIP - TASK 2: UNEMPLOYMENT ANALYSIS")
    print("=" * 60)

    base_dir = os.path.dirname(__file__)
    file1 = os.path.join(base_dir, "Unemployment_in_India.csv")
    file2 = os.path.join(base_dir, "Unemployment_Rate_upto_11_2020.csv")

    # 1. Load Datasets
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    print(f"\n[1] Primary Dataset Loaded (Unemployment_in_India.csv): Shape {df1.shape}")
    print(f"[2] Secondary Dataset Loaded (Unemployment_Rate_upto_11_2020.csv): Shape {df2.shape}")

    # Clean column headers (strip white spaces and BOM character if present)
    df1.columns = df1.columns.str.replace('\ufeff', '').str.strip()
    df2.columns = df2.columns.str.replace('\ufeff', '').str.strip()

    print("\nColumns in Primary Dataset:", df1.columns.tolist())
    
    # Clean Missing Values
    print("\nMissing values count in Primary Dataset before cleaning:")
    print(df1.isnull().sum())
    df1.dropna(inplace=True)

    # Convert Date to datetime format
    df1['Date'] = pd.to_datetime(df1['Date'].astype(str).str.strip(), format='%d-%m-%Y', errors='coerce')
    df1['Year'] = df1['Date'].dt.year
    df1['Month'] = df1['Date'].dt.strftime('%b')
    
    df2['Date'] = pd.to_datetime(df2['Date'].astype(str).str.strip(), format='%d-%m-%Y', errors='coerce')

    print("\nCleaned Data Summary:")
    print(df1.describe())

    # -------------------------------------------------------------
    # Chart 1: Time Series Trend for Key States
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    key_states = ["Delhi", "Maharashtra", "Uttar Pradesh", "Tamil Nadu", "West Bengal", "Karnataka", "Bihar"]
    df_key = df1[df1['Region'].isin(key_states)]

    sns.lineplot(data=df_key, x='Date', y='Estimated Unemployment Rate (%)', hue='Region', marker='o', ax=ax, linewidth=2.5)
    ax.set_title("Unemployment Rate Over Time Across Major Indian States (2019-2020)", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Date", labelpad=10)
    ax.set_ylabel("Unemployment Rate (%)", labelpad=10)
    ax.axvline(pd.Timestamp("2020-03-24"), color='red', linestyle='--', linewidth=2, label='COVID-19 Lockdown Announced')
    ax.legend(title="Region", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "unemployment_time_series.png"), dpi=300)
    plt.close()
    print("Saved plot: unemployment_time_series.png")

    # -------------------------------------------------------------
    # Chart 2: Top 10 States by Average Unemployment Rate
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    top10 = df1.groupby('Region')['Estimated Unemployment Rate (%)'].mean().sort_values(ascending=False).head(10).reset_index()

    sns.barplot(data=top10, x='Estimated Unemployment Rate (%)', y='Region', palette='Reds_r', ax=ax)
    ax.set_title("Top 10 Indian States with Highest Average Unemployment Rate", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Average Unemployment Rate (%)")
    ax.set_ylabel("State / Region")
    for p in ax.patches:
        width = p.get_width()
        ax.annotate(f'{width:.2f}%', (width + 0.3, p.get_y() + p.get_height() / 2),
                    ha='left', va='center', fontsize=10, color='black', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "top10_unemployment_states.png"), dpi=300)
    plt.close()
    print("Saved plot: top10_unemployment_states.png")

    # -------------------------------------------------------------
    # Chart 3: Rural vs Urban Comparison
    # -------------------------------------------------------------
    if 'Area' in df1.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(data=df1, x='Area', y='Estimated Unemployment Rate (%)', palette='Set2', ax=ax)
        ax.set_title("Unemployment Rate Distribution: Rural vs Urban Areas", fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel("Area Type", labelpad=10)
        ax.set_ylabel("Unemployment Rate (%)", labelpad=10)
        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, "rural_vs_urban_unemployment.png"), dpi=300)
        plt.close()
        print("Saved plot: rural_vs_urban_unemployment.png")

    # -------------------------------------------------------------
    # Chart 4: Pre-COVID vs Post-COVID Impact Analysis
    # -------------------------------------------------------------
    df1['COVID_Period'] = np.where(df1['Date'] < pd.Timestamp("2020-03-01"), 'Pre-COVID (Jan 2019 - Feb 2020)', 'Post-COVID Outbreak (Mar 2020 Onwards)')
    
    covid_comparison = df1.groupby('COVID_Period')['Estimated Unemployment Rate (%)'].agg(['mean', 'median', 'std', 'max']).reset_index()
    print("\n--- COVID-19 Impact Analysis ---")
    print(covid_comparison.to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=df1, x='COVID_Period', y='Estimated Unemployment Rate (%)', palette='Set1', ax=ax)
    ax.set_title("Impact of COVID-19 Lockdown on Unemployment Rates in India", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Period", labelpad=10)
    ax.set_ylabel("Unemployment Rate (%)", labelpad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "covid_impact_comparison.png"), dpi=300)
    plt.close()
    print("Saved plot: covid_impact_comparison.png")

    # -------------------------------------------------------------
    # Chart 5: Correlation Heatmap
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    corr_cols = ['Estimated Unemployment Rate (%)', 'Estimated Employed', 'Estimated Labour Participation Rate (%)']
    corr_matrix = df1[corr_cols].corr()

    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=1, ax=ax)
    ax.set_title("Correlation Heatmap: Unemployment, Employment & LPR", fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "correlation_heatmap.png"), dpi=300)
    plt.close()
    print("Saved plot: correlation_heatmap.png")

    print("\n" + "=" * 60)
    print("UNEMPLOYMENT ANALYSIS COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()
