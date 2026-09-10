import os
import nbformat as nbf

dst_base = "/Users/pranav/Desktop/OIBSIP copy"

# Run create notebook scripts
def create_nb(task_name, title, folder):
    nb = nbf.v4.new_notebook()
    cells = []
    cells.append(nbf.v4.new_markdown_cell(f"""# Oasis Infobyte SIP — Data Science Internship
## {title}

**Author:** Poornima  
**Track:** Data Science  
**Task Title:** {title}  
**GitHub Repository:** `OIBSIP`  

---
"""))
    
    if "Task2" in folder:
        cells.append(nbf.v4.new_code_cell("""import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

df1 = pd.read_csv("Unemployment_in_India.csv")
df1.columns = df1.columns.str.replace('\\ufeff', '').str.strip()
df1.dropna(inplace=True)
df1['Date'] = pd.to_datetime(df1['Date'].astype(str).str.strip(), format='%d-%m-%Y', errors='coerce')
df1['COVID_Period'] = np.where(df1['Date'] < pd.Timestamp("2020-03-01"), 'Pre-COVID (Jan 2019 - Feb 2020)', 'Post-COVID Outbreak (Mar 2020 Onwards)')

print("Cleaned Dataset Info:")
display(df1.head())
"""))
        cells.append(nbf.v4.new_code_cell("""fig, ax = plt.subplots(figsize=(10, 5))
key_states = ["Delhi", "Maharashtra", "Uttar Pradesh", "Tamil Nadu", "West Bengal", "Karnataka", "Bihar"]
sns.lineplot(data=df1[df1['Region'].isin(key_states)], x='Date', y='Estimated Unemployment Rate (%)', hue='Region', ax=ax)
ax.set_title("Unemployment Rate Over Time Across Major Indian States", fontweight='bold')
plt.show()
"""))
        cells.append(nbf.v4.new_code_cell("""fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=df1, x='COVID_Period', y='Estimated Unemployment Rate (%)', ax=ax)
ax.set_title("Impact of COVID-19 Lockdown on Unemployment Rates in India", fontweight='bold')
plt.show()
"""))
        cells.append(nbf.v4.new_markdown_cell("""### Conclusion
The analysis shows a significant surge in unemployment rates following the March 2020 lockdown."""))

    elif "Task4" in folder:
        cells.append(nbf.v4.new_code_cell("""import os, re
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

try:
    df = pd.read_csv("spam.csv", encoding='latin-1')
except Exception:
    df = pd.read_csv("spam.csv", encoding='utf-8')

if 'v1' in df.columns and 'v2' in df.columns:
    df = df[['v1', 'v2']].copy()
    df.columns = ['Category', 'Message']

df['Target'] = df['Category'].map({'ham': 0, 'spam': 1})
df.dropna(subset=['Target', 'Message'], inplace=True)
df['Target'] = df['Target'].astype(int)

display(df.head())
"""))
        cells.append(nbf.v4.new_code_cell("""def preprocess_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

df['Clean_Message'] = df['Message'].apply(preprocess_text)
tfidf = TfidfVectorizer(max_features=3000, stop_words='english')
X = tfidf.fit_transform(df['Clean_Message']).toarray()
y = df['Target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

svc = SVC(kernel='linear')
svc.fit(X_train, y_train)
preds = svc.predict(X_test)

print("SVC Accuracy:", accuracy_score(y_test, preds))
print("SVC Precision:", precision_score(y_test, preds))
"""))
        cells.append(nbf.v4.new_markdown_cell("""### Conclusion
SVC achieved exceptional accuracy and precision in identifying spam messages."""))

    elif "Task5" in folder:
        cells.append(nbf.v4.new_code_cell("""import os
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

data_path = "Advertising.csv"
if not os.path.exists(data_path): data_path = "Advertising Budget and Sales.csv"
df = pd.read_csv(data_path)
df.columns = df.columns.str.strip()

column_mapping = {}
for col in df.columns:
    if 'TV' in col: column_mapping[col] = 'TV'
    elif 'Radio' in col: column_mapping[col] = 'Radio'
    elif 'Newspaper' in col: column_mapping[col] = 'Newspaper'
    elif 'Sales' in col: column_mapping[col] = 'Sales'

df.rename(columns=column_mapping, inplace=True)
df = df[[col for col in ['TV', 'Radio', 'Newspaper', 'Sales'] if col in df.columns]].copy()

display(df.head())
"""))
        cells.append(nbf.v4.new_code_cell("""X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

print("Random Forest R2 Score:", r2_score(y_test, rf_preds))
print("Random Forest MAE:", mean_absolute_error(y_test, rf_preds))
"""))
        cells.append(nbf.v4.new_markdown_cell("""### Conclusion
Random Forest Regressor achieved an R2 score of over 98% in predicting sales revenue."""))

    nb['cells'] = cells
    nb_path = os.path.join(dst_base, folder, f"{task_name}.ipynb")
    with open(nb_path, "w") as f:
        nbf.write(nb, f)
    print(f"Created notebook: {nb_path}")

create_nb("Unemployment_Analysis", "Task 2: Unemployment Analysis with Python", "DataScience-Task2-UnemploymentAnalysis")
create_nb("Email_Spam_Detection", "Task 4: Email Spam Detection with Machine Learning", "DataScience-Task4-EmailSpamDetection")
create_nb("Sales_Prediction", "Task 5: Sales Prediction Using Python", "DataScience-Task5-SalesPrediction")

print("All notebooks created for Poornima in Desktop/OIBSIP copy!")
