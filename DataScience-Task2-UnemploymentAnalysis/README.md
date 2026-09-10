# Task 2: Unemployment Analysis with Python

**Oasis Infobyte Internship (OIBSIP) — Data Science Track**  
**Author:** Poornima  
**Repository Name:** `OIBSIP`  

---

## 📌 Project Overview
This project presents a data-driven exploratory analysis of unemployment rates in India before and during the COVID-19 pandemic. Using official Kaggle datasets (`Unemployment_in_India.csv` and `Unemployment_Rate_upto_11_2020.csv`), we uncover regional disparities, rural vs. urban labor dynamics, and the dramatic economic shock triggered by the March 2020 nationwide lockdown.

---

## 🛠️ Tech Stack & Libraries
- **Language:** Python 3.13
- **Data Manipulation:** `pandas`, `numpy`
- **Data Visualization:** `matplotlib`, `seaborn`
- **Environment:** Jupyter Notebook (`.ipynb`), Python script (`.py`)

---

## 📊 Key Analysis & Findings

1. **COVID-19 Lockdown Impact:**
   - **Pre-COVID Mean Unemployment Rate:** ~9.51%
   - **Post-COVID Mean Unemployment Rate:** ~17.77% (Peak rates exceeded 75% in certain severely impacted states like Puducherry & Bihar during April/May 2020).
2. **Rural vs. Urban Disparity:**
   - Urban centers experienced significantly higher median unemployment rates and volatility compared to rural areas due to service sector shutdowns and industrial pauses.
3. **Top Affected States:**
   - Puducherry, Jharkhand, Bihar, and Haryana recorded the highest average unemployment rates across the 2019–2020 timeline.
4. **Labor Force Participation:**
   - A drop in Labour Participation Rate (LPR) accompanied the unemployment spike, indicating widespread workforce discouragement during peak lockdown months.

---

## 📂 Project Structure
```text
DataScience-Task2-UnemploymentAnalysis/
├── Unemployment_Analysis.ipynb        # Interactive Jupyter Notebook with full EDA & markdown insights
├── unemployment_analysis.py            # Executable standalone Python analysis pipeline
├── Unemployment_in_India.csv           # Kaggle Primary Dataset (Rural vs. Urban breakdown)
├── Unemployment_Rate_upto_11_2020.csv # Kaggle Secondary Dataset (State & Regional coordinates)
├── unemployment_time_series.png        # Time-series plot across major states
├── top10_unemployment_states.png       # Bar chart of top 10 affected states
├── rural_vs_urban_unemployment.png    # Boxplot of rural vs urban unemployment rates
├── covid_impact_comparison.png         # Pre-COVID vs Post-COVID comparison boxplot
├── correlation_heatmap.png             # Correlation matrix plot
└── README.md                           # Documentation report
```

---

## 🚀 How to Run

1. **Activate Environment & Install Dependencies:**
   ```bash
   pip install pandas numpy matplotlib seaborn jupyter
   ```
2. **Execute Python Pipeline:**
   ```bash
   python unemployment_analysis.py
   ```
3. **Run Jupyter Notebook:**
   ```bash
   jupyter notebook Unemployment_Analysis.ipynb
   ```
