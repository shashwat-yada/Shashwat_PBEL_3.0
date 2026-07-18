"""
app.py
------
A simple, presentable web UI for the Fake News Detector, built with Streamlit.

Run:
    streamlit run app.py

Then your browser will open automatically at http://localhost:8501
"""

import streamlit as st
import joblib
import os
from utils import clean_text

MODEL_PATH = "model/fake_news_model.pkl"
VECTORIZER_PATH = "model/tfidf_vectorizer.pkl"

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.title("📰 AI-Based Fake News Detection System")
st.write(
    "This tool uses **Machine Learning (TF-IDF + Logistic Regression)** "
    "to classify a news headline or article as **REAL** or **FAKE**."
)

if not (os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)):
    st.error(
        "Model not found. Please run `python train_model.py` in your terminal "
        "first, then restart this app."
    )
    st.stop()


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


model, vectorizer = load_artifacts()

user_text = st.text_area(
    "Enter a news headline or article text:",
    height=180,
    placeholder="Paste or type a news headline/article here...",
)

col1, col2 = st.columns([1, 3])
with col1:
    analyze = st.button("🔍 Analyze", type="primary")

if analyze:
    if not user_text.strip():
        st.warning("Please enter some text first.")
    else:
        cleaned = clean_text(user_text)
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]
        confidence = max(proba) * 100

        st.divider()
        if prediction == "REAL":
            st.success(f"### 🟢 This looks like REAL news")
        else:
            st.error(f"### 🔴 This looks like FAKE news")

        st.metric("Confidence", f"{confidence:.2f}%")
        st.progress(min(int(confidence), 100))

        with st.expander("See class probabilities"):
            classes = model.classes_
            for c, p in zip(classes, proba):
                st.write(f"**{c}**: {p * 100:.2f}%")

st.divider()
st.caption(
    "Project: AI-Based Fake News Detection | Built with Python, scikit-learn "
    "(TF-IDF Vectorizer + Logistic Regression) and Streamlit."
)
