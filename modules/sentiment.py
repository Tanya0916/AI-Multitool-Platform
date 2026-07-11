"""
Module 3: Sentiment Analysis
Uses a TF-IDF + Logistic Regression pipeline (scikit-learn) 
"""
import os
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "sentiment_pipeline.pkl")

EMOJI = {"positive": "It's Positive", "negative": "It's  Negative", "neutral": "It's  Neutral"}


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def predict_sentiment(text: str, model):
    proba = model.predict_proba([text])[0]
    classes = model.classes_
    label = classes[proba.argmax()]
    confidence = proba.max()
    breakdown = dict(zip(classes, proba))
    return label, confidence, breakdown


def render(log_usage_fn=None, username: str = "guest"):
    st.subheader("💬 Sentiment Analysis")
    st.caption("TF-IDF + Logistic Regression classifier (scikit-learn), trained on labeled review-style text.")

    model = load_model()
    if model is None:
        st.error(
            "Model not found. Run `python train_sentiment_model.py` first "
            "to train and save the sentiment model."
        )
        return

    tab1, tab2 = st.tabs(["Single Text", "Batch (CSV)"])

    with tab1:
        text = st.text_area(
            "Enter text to analyze",
            placeholder="e.g. The delivery was late and the packaging was damaged.",
            height=120,
        )
        if st.button("Analyze Sentiment", type="primary", key="sentiment_btn"):
            if not text.strip():
                st.warning("Please enter some text first.")
            else:
                label, confidence, breakdown = predict_sentiment(text, model)
                st.markdown(f"### Result: {EMOJI.get(label, label)}")
                st.progress(float(confidence))
                st.write(f"Confidence: **{confidence:.1%}**")

                bdf = pd.DataFrame({
                    "sentiment": list(breakdown.keys()),
                    "probability": list(breakdown.values()),
                }).sort_values("probability", ascending=False)
                st.bar_chart(bdf.set_index("sentiment"))

                if log_usage_fn:
                    log_usage_fn(username, "Sentiment Analysis")

    with tab2:
        st.write("Upload a CSV with a `text` column to classify sentiment in bulk.")
        uploaded = st.file_uploader("Upload CSV", type=["csv"], key="sentiment_csv")
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            if "text" not in df.columns:
                st.error("CSV must contain a column named 'text'.")
            else:
                if st.button("Run Batch Analysis"):
                    results = df["text"].astype(str).apply(
                        lambda t: predict_sentiment(t, model)[0]
                    )
                    df["predicted_sentiment"] = results
                    st.dataframe(df, use_container_width=True)
                    st.download_button(
                        "Download results as CSV",
                        df.to_csv(index=False).encode("utf-8"),
                        "sentiment_results.csv",
                        "text/csv",
                    )
                    if log_usage_fn:
                        log_usage_fn(username, "Sentiment Analysis (Batch)")
