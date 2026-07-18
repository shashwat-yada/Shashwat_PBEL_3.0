"""
train_model.py
---------------
Trains a Fake News Detection model.

Pipeline:  raw text -> clean_text() -> TF-IDF vectorizer -> PassiveAggressiveClassifier
Also prints accuracy, confusion matrix, and classification report,
and saves the trained model + vectorizer to the model/ folder.

Run:
    python train_model.py
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from utils import clean_text

DATA_PATH = "data/news.csv"
MODEL_PATH = "model/fake_news_model.pkl"
VECTORIZER_PATH = "model/tfidf_vectorizer.pkl"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Make the script tolerant of the common Kaggle "Fake.csv / True.csv"
    # column names (title, text, subject, date) as well as our simple
    # (text, label) demo format.
    if "label" not in df.columns:
        raise ValueError(
            "Expected a 'label' column (REAL/FAKE) in your CSV. "
            "See README.md Step 5 for dataset format instructions."
        )
    if "text" not in df.columns and "title" in df.columns:
        df["text"] = df["title"]

    df = df.dropna(subset=["text", "label"])
    return df


def main():
    print("1) Loading dataset...")
    df = load_data(DATA_PATH)
    print(f"   Loaded {len(df)} rows. Label counts:\n{df['label'].value_counts()}\n")

    print("2) Cleaning text...")
    df["clean_text"] = df["text"].apply(clean_text)

    print("3) Splitting into train/test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    print("4) Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("5) Training LogisticRegression classifier...")
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X_train_tfidf, y_train)

    print("6) Evaluating on test set...\n")
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f"   Accuracy: {acc * 100:.2f}%\n")
    print("   Confusion Matrix (rows=actual, cols=predicted):")
    print("  ", confusion_matrix(y_test, y_pred, labels=["REAL", "FAKE"]), "\n")
    print("   Classification Report:")
    print(classification_report(y_test, y_pred))

    print("7) Saving model and vectorizer to model/ ...")
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"   Saved: {MODEL_PATH}")
    print(f"   Saved: {VECTORIZER_PATH}")
    print("\nDone! You can now run: python predict.py")


if __name__ == "__main__":
    main()
