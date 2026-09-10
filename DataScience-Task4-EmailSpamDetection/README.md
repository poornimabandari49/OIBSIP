# Task 4: Email Spam Detection with Machine Learning

**Oasis Infobyte Internship (OIBSIP) — Data Science Track**  
**Author:** Poornima  
**Repository Name:** `OIBSIP`  

---

## 📌 Project Overview
This project builds a Natural Language Processing (NLP) machine learning classifier to accurately distinguish between **Ham** (legitimate) and **Spam** (unsolicited / phishing) messages. Using the standard Kaggle SMS Spam Collection dataset (`spam.csv`), we perform text cleaning, feature extraction via **TF-IDF Vectorization**, model training across multiple algorithms, and evaluate model performance with a focus on Precision and Recall trade-offs.

---

## 🛠️ Tech Stack & Libraries
- **Language:** Python 3.13
- **NLP & Feature Extraction:** `scikit-learn` (`TfidfVectorizer`), `re` (Regular Expressions), `wordcloud`
- **Machine Learning Models:** `MultinomialNB`, `LogisticRegression`, `SVC` (Support Vector Classifier)
- **Evaluation Metrics:** `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`
- **Visualization:** `matplotlib`, `seaborn`, `wordcloud`

---

## 📊 Key Results & Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Support Vector Machine (SVC)** | **98.12%** | **98.48%** | **87.25%** | **0.9253** |
| **Multinomial Naive Bayes** | **97.22%** | **99.17%** | **79.87%** | **0.8848** |
| **Logistic Regression** | **96.23%** | **99.08%** | **72.48%** | **0.8372** |

---

## 💡 Precision vs. Recall Discussion
In email spam classification, **Precision** measures the ratio of true spam detected among all messages flagged as spam, while **Recall** measures how much of the total spam was caught.

> [!IMPORTANT]
> **Why Precision is Paramount in Spam Detection:**  
> A **False Positive** (flagging a legitimate, important email as spam) causes severe user disruption (e.g., missing job offers or billing alerts). A **False Negative** simply leaves an unwanted promo message in the inbox. Therefore, models like **Multinomial Naive Bayes** and **SVC** are optimized for high precision (98.5%+ - 99.1%+) to ensure legitimate emails are never misclassified as spam.

---

## 📂 Project Structure
```text
DataScience-Task4-EmailSpamDetection/
├── Email_Spam_Detection.ipynb       # Interactive Jupyter Notebook with NLP pipeline & visual plots
├── email_spam_detection.py           # Standalone Python classification script
├── spam.csv                          # Kaggle Dataset (5,572 labeled messages)
├── spam_ham_wordclouds.png           # WordCloud visual comparing Spam vs Ham keywords
├── model_confusion_matrices.png      # Confusion matrix plots for all models
└── README.md                         # Documentation report
```

---

## 🚀 How to Run

1. **Install Dependencies:**
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn wordcloud jupyter
   ```
2. **Execute Python Pipeline:**
   ```bash
   python email_spam_detection.py
   ```
3. **Run Jupyter Notebook:**
   ```bash
   jupyter notebook Email_Spam_Detection.ipynb
   ```
