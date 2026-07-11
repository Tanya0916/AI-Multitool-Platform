"""
Module 2: Text Summarizer
Extractive summarization using TF-IDF sentence scoring (scikit-learn).

"""
import re
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer


def split_sentences(text: str):
    # Lightweight sentence splitter 
    text = re.sub(r"\s+", " ", text.strip())
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [s.strip() for s in sentences if s.strip()]


def summarize(text: str, num_sentences: int = 3):
    sentences = split_sentences(text)
    if len(sentences) <= num_sentences:
        return sentences, sentences

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(sentences)
    scores = np.asarray(tfidf_matrix.mean(axis=1)).ravel()

    ranked_idx = np.argsort(scores)[::-1][:num_sentences]
    ranked_idx_sorted = sorted(ranked_idx)  # preserve original order

    summary_sentences = [sentences[i] for i in ranked_idx_sorted]
    return summary_sentences, sentences


def render(log_usage_fn=None, username: str = "guest"):
    st.subheader("📝 Text Summarizer")
    st.caption("Extractive summarization via TF-IDF sentence ranking (scikit-learn).")

    text = st.text_area(
        "Paste the text you want summarized",
        height=250,
        placeholder="Paste an article, report, or long paragraph here...",
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        num_sentences = st.slider("Number of sentences in summary", 1, 10, 3)
    with col2:
        st.write("")
        st.write("")
        run = st.button("Summarize", type="primary")

    if run:
        if not text.strip():
            st.warning("Please paste some text first.")
        else:
            summary_sentences, all_sentences = summarize(text, num_sentences)
            summary = " ".join(summary_sentences)

            st.markdown("### Summary")
            st.info(summary)

            reduction = 100 * (1 - len(summary_sentences) / max(len(all_sentences), 1))
            c1, c2, c3 = st.columns(3)
            c1.metric("Original sentences", len(all_sentences))
            c2.metric("Summary sentences", len(summary_sentences))
            c3.metric("Reduction", f"{reduction:.0f}%")

            st.download_button(
                "Download summary as .txt",
                summary,
                "summary.txt",
                "text/plain",
            )

            if log_usage_fn:
                log_usage_fn(username, "Text Summarizer")
