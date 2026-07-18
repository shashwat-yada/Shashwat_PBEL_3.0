"""
predict.py
----------
Loads the trained model + vectorizer and lets you check whether a
news headline/article is REAL or FAKE, right from the terminal.

Run:
    python predict.py
"""

import sys
import joblib
from utils import clean_text

MODEL_PATH = "model/fake_news_model.pkl"
VECTORIZER_PATH = "model/tfidf_vectorizer.pkl"


def load_artifacts():
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
    except FileNotFoundError:
        print("ERROR: Model files not found.")
        print("Please run 'python train_model.py' first to train and save the model.")
        sys.exit(1)
    return model, vectorizer


def predict_news(text: str, model, vectorizer) -> tuple[str, float]:
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    confidence = max(proba) * 100
    return pred, confidence


def main():
    model, vectorizer = load_artifacts()
    print("=" * 60)
    print(" FAKE NEWS DETECTOR — type a headline/article, or 'quit' to exit")
    print("=" * 60)

    while True:
        text = input("\nEnter news text: ").strip()
        if text.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not text:
            print("Please enter some text.")
            continue

        label, confidence = predict_news(text, model, vectorizer)
        tag = "🟢 REAL" if label == "REAL" else "🔴 FAKE"
        print(f"Prediction: {tag}   (confidence: {confidence:.2f}%)")


if __name__ == "__main__":
    main()
