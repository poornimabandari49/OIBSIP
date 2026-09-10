# Oasis Infobyte SIP — Data Science Domain Submissions

**Internship Program:** Oasis Infobyte Student Internship Program (OIBSIP)  
**Track:** Data Science  
**Author:** Poornima  
**GitHub Account:** `poornimabandari49`  
**Repository Name:** `OIBSIP`  

---

## 📌 Repository Overview

This repository contains all completed projects for the **Data Science Domain** internship track at **Oasis Infobyte**. Each task is organized in strict accordance with Oasis Infobyte guidelines (`OIBSIP/[TrackName]-[Level/Task]-[ProjectName]/`).

---

## 📂 Completed Task Submissions

| Task Directory | Project Name | Tech Stack | Key Results & Achievements |
| :--- | :--- | :--- | :--- |
| [`DataScience-Task2-UnemploymentAnalysis/`](./DataScience-Task2-UnemploymentAnalysis/) | **Task 2: Unemployment Analysis with Python** | Python, Pandas, Matplotlib, Seaborn, Jupyter | Quantified COVID-19 lockdown impact across Indian states (pre vs post COVID comparison, rural vs urban dynamics). |
| [`DataScience-Task4-EmailSpamDetection/`](./DataScience-Task4-EmailSpamDetection/) | **Task 4: Email Spam Detection with Machine Learning** | Python, Scikit-learn, TF-IDF, Naive Bayes, WordCloud | Built NLP classification pipeline achieving **98.12% accuracy** and **98.48% precision** using Support Vector Machines & Naive Bayes. |
| [`DataScience-Task5-SalesPrediction/`](./DataScience-Task5-SalesPrediction/) | **Task 5: Sales Prediction Using Python** | Python, Scikit-learn, Linear Regression, Random Forest | Achieved **98.14% $R^2$ score** predicting sales revenue from TV, Radio, and Newspaper ad budgets. |

---

## 🛠️ Repository Directory Structure

```text
OIBSIP/
├── DataScience-Task2-UnemploymentAnalysis/
│   ├── Unemployment_Analysis.ipynb
│   ├── unemployment_analysis.py
│   ├── Unemployment_in_India.csv
│   ├── Unemployment_Rate_upto_11_2020.csv
│   ├── unemployment_time_series.png
│   ├── top10_unemployment_states.png
│   ├── rural_vs_urban_unemployment.png
│   ├── covid_impact_comparison.png
│   ├── correlation_heatmap.png
│   └── README.md
│
├── DataScience-Task4-EmailSpamDetection/
│   ├── Email_Spam_Detection.ipynb
│   ├── email_spam_detection.py
│   ├── spam.csv
│   ├── spam_ham_wordclouds.png
│   ├── model_confusion_matrices.png
│   └── README.md
│
├── DataScience-Task5-SalesPrediction/
│   ├── Sales_Prediction.ipynb
│   ├── sales_prediction.py
│   ├── Advertising.csv
│   ├── scatter_plots_spend_vs_sales.png
│   ├── correlation_heatmap_sales.png
│   ├── residual_and_feature_importance.png
│   └── README.md
│
├── .gitignore
└── README.md
```

---

## 💻 Environment Setup & Instructions

To run any of the tasks locally, clone the repository and install required packages:

```bash
git clone https://github.com/poornimabandari49/OIBSIP.git
cd OIBSIP
pip install pandas numpy scikit-learn matplotlib seaborn wordcloud jupyter
```

Each task folder contains both an interactive `.ipynb` notebook and an executable `.py` script.
