"""
Oasis Infobyte - Data Science Internship (OIBSIP)
Task 4: Email Spam Detection with Machine Learning

Author: Poornima
Track: Data Science
Task: Task 4 - Email Spam Detection with Machine Learning
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from wordcloud import WordCloud

# Set visual style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.size'] = 11

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove non-alphabet characters and punctuation
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    print("=" * 60)
    print("OASIS INFOBYTE SIP - TASK 4: EMAIL SPAM DETECTION")
    print("=" * 60)

    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, "spam.csv")

    # 1. Load Dataset
    try:
        df = pd.read_csv(data_path, encoding='latin-1')
    except Exception:
        df = pd.read_csv(data_path, encoding='utf-8')

    print(f"\n[1] Dataset Loaded Successfully! Shape: {df.shape}")

    # Keep relevant columns (v1 = label, v2 = text)
    if 'v1' in df.columns and 'v2' in df.columns:
        df = df[['v1', 'v2']].copy()
        df.columns = ['Category', 'Message']
    elif 'Category' not in df.columns or 'Message' not in df.columns:
        df = df.iloc[:, :2].copy()
        df.columns = ['Category', 'Message']

    print("\nDataset Info:")
    print(df.head())

    # Check Class Distribution
    print("\nClass Distribution:")
    class_counts = df['Category'].value_counts()
    class_pct = df['Category'].value_counts(normalize=True) * 100
    dist_df = pd.DataFrame({'Count': class_counts, 'Percentage (%)': class_pct.round(2)})
    print(dist_df)

    # Convert target labels: ham -> 0, spam -> 1
    df['Target'] = df['Category'].map({'ham': 0, 'spam': 1})
    df.dropna(subset=['Target', 'Message'], inplace=True)
    df['Target'] = df['Target'].astype(int)

    # 2. Text Preprocessing
    print("\n[2] Preprocessing Text Messages...")
    df['Clean_Message'] = df['Message'].apply(preprocess_text)

    # WordCloud Visualizations
    spam_words = " ".join(df[df['Target'] == 1]['Clean_Message'])
    ham_words = " ".join(df[df['Target'] == 0]['Clean_Message'])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    wc_spam = WordCloud(width=600, height=400, background_color='black', colormap='Reds').generate(spam_words)
    wc_ham = WordCloud(width=600, height=400, background_color='white', colormap='Greens').generate(ham_words)

    axes[0].imshow(wc_spam, interpolation='bilinear')
    axes[0].set_title("Most Frequent Words in Spam Messages", fontsize=14, fontweight='bold', pad=10)
    axes[0].axis('off')

    axes[1].imshow(wc_ham, interpolation='bilinear')
    axes[1].set_title("Most Frequent Words in Ham Messages", fontsize=14, fontweight='bold', pad=10)
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "spam_ham_wordclouds.png"), dpi=300)
    plt.close()
    print("Saved plot: spam_ham_wordclouds.png")

    # 3. TF-IDF Feature Extraction
    print("\n[3] Extracting Features via TF-IDF Vectorizer...")
    tfidf = TfidfVectorizer(max_features=3000, stop_words='english')
    X = tfidf.fit_transform(df['Clean_Message']).toarray()
    y = df['Target'].values

    # Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    print(f"Training Set: {X_train.shape[0]} samples | Testing Set: {X_test.shape[0]} samples")

    # 4. Model Training & Evaluation
    models = {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Support Vector Machine (SVC)": SVC(kernel='linear')
    }

    results = []

    plt.figure(figsize=(15, 4))
    for idx, (name, model) in enumerate(models.items(), 1):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1
        })

        plt.subplot(1, 3, idx)
        cm = confusion_matrix(y_test, preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
        plt.title(f"{name}\nConfusion Matrix", fontsize=12, fontweight='bold')
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")

    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "model_confusion_matrices.png"), dpi=300)
    plt.close()
    print("Saved plot: model_confusion_matrices.png")

    results_df = pd.DataFrame(results)
    print("\n--- Model Performance Comparison ---")
    print(results_df.to_string(index=False))

    print("\n" + "=" * 60)
    print("EMAIL SPAM DETECTION COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()
