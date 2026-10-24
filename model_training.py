import pandas as pd
import urllib.parse
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

def extract_csic_parameters(url_string):
    """Strips the HTTP wrapper and extracts only the URL parameters."""
    try:
        url_part = str(url_string).rsplit(' HTTP/', 1)[0]
        if '?' in url_part:
            return url_part.split('?', 1)[1]
        else:
            return ""
    except Exception:
        return ""

def waf_tokenizer(text):
    """Custom tokenizer to capture special SQL/XSS characters."""
    text = urllib.parse.unquote(str(text)).lower()
    return re.findall(r'\w+|[^\w\s]', text)

if __name__ == "__main__":
    print("Loading datasets...")
    # NOTE: These paths assume the CSVs are in a Datasets/ folder relative to this script
    df_sql = pd.read_csv("../Datasets/Modified_SQL_Dataset.csv")
    df_csic = pd.read_csv("../Datasets/csic_database.csv")

    df_sql = df_sql[['Query', 'Label']].rename(columns={'Query': 'payload', 'Label': 'label'})
    df_csic = df_csic[['URL', 'classification']].rename(columns={'URL': 'payload', 'classification': 'label'})

    print("Cleaning CSIC data...")
    df_csic['payload'] = df_csic['payload'].apply(extract_csic_parameters)
    df_csic = df_csic[~((df_csic['payload'] == "") & (df_csic['label'] == 1))]

    df_combined = pd.concat([df_sql, df_csic], ignore_index=True).dropna(subset=['payload'])

    print("Vectorizing payloads...")
    vectorizer = TfidfVectorizer(tokenizer=waf_tokenizer, ngram_range=(1, 2), max_features=10000)
    X = vectorizer.fit_transform(df_combined['payload'])
    y = df_combined['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    print("Exporting model and vectorizer...")
    joblib.dump(model, 'waf_logistic_regression.pkl')
    joblib.dump(vectorizer, 'waf_tfidf_vectorizer.pkl')
    print("Training complete. Files saved.")