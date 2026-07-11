"""
Module 1: AI Chat Assistant

A fully self-contained chatbot — no API key, no internet call, no
configuration. Uses TF-IDF similarity matching (scikit-learn) over a
bank of intents/example phrases to pick the best-matching response,
with regex rules for small talk and simple pattern replies as backup.
"""
import random
import re
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------------------------------------------------- knowledge base
# Each intent has several example phrases (for matching) and possible replies.
INTENTS = [
    {
        "tag": "greeting",
        "examples": ["hello", "hi", "hey", "good morning", "good evening", "hiya", "yo", "what's up"],
        "responses": [
            "Hey! How can I help you today?",
            "Hello! What's on your mind?",
            "Hi there! What can I do for you?",
        ],
    },
    {
        "tag": "how_are_you",
        "examples": ["how are you", "how's it going", "how are you doing", "you good", "how do you feel"],
        "responses": [
            "I'm doing well, thanks for asking! How about you?",
            "All good on my end. What's up with you?",
        ],
    },
    {
        "tag": "name",
        "examples": ["what is your name", "who are you", "what should I call you"],
        "responses": [
            "I'm your AI Chat Assistant — one of the modules in this platform.",
            "I'm a chatbot built for this app. Ask me anything!",
        ],
    },
    {
        "tag": "capabilities",
        "examples": [
            "what can you do", "help me", "what are your features",
            "how does this app work", "what is this app", "what modules do you have",
        ],
        "responses": [
            "This platform has three tools: I'm the Chat Assistant, and there's also a "
            "Text Summarizer and a Sentiment Analysis tool in the sidebar — check them out!",
            "I can chat with you here. For deeper text tasks, try the Text Summarizer or "
            "Sentiment Analysis modules from the sidebar navigation.",
        ],
    },
    {
        "tag": "thanks",
        "examples": ["thank you", "thanks", "appreciate it", "thanks a lot", "much appreciated"],
        "responses": ["You're welcome!", "Anytime!", "Happy to help!"],
    },
    {
        "tag": "goodbye",
        "examples": ["bye", "goodbye", "see you later", "talk to you later", "gotta go", "farewell"],
        "responses": ["Goodbye! Come back anytime.", "See you later!", "Take care!"],
    },
    {
        "tag": "mood_good",
        "examples": ["I am happy", "feeling great", "I'm doing good", "life is good", "I feel amazing"],
        "responses": ["That's great to hear! Glad things are going well.", "Love that energy — keep it up!"],
    },
    {
        "tag": "mood_bad",
        "examples": ["I am sad", "feeling down", "I'm stressed", "having a bad day", "I feel tired", "I'm frustrated"],
        "responses": [
            "Sorry to hear that. Want to talk through what's going on?",
            "That sounds tough — I'm here if you want to vent or think it through.",
        ],
    },
    {
        "tag": "joke",
        "examples": ["tell me a joke", "make me laugh", "say something funny", "know any jokes"],
        "responses": [
            "Why do programmers prefer dark mode? Because light attracts bugs.",
            "I told my computer I needed a break, and now it won't stop sending me KitKats.",
            "Why did the developer go broke? Because they used up all their cache.",
        ],
    },
    {
        "tag": "weather",
        "examples": ["what's the weather", "is it raining", "how's the weather today"],
        "responses": [
            "I don't have live weather access in this offline mode — try a weather app or site for that!",
        ],
    },
]

# Flatten examples for vectorizer fitting, keep a parallel list of tags
ALL_EXAMPLES = []
ALL_TAGS = []
for intent in INTENTS:
    for ex in intent["examples"]:
        ALL_EXAMPLES.append(ex)
        ALL_TAGS.append(intent["tag"])

TAG_TO_RESPONSES = {intent["tag"]: intent["responses"] for intent in INTENTS}

MATCH_THRESHOLD = 0.35  # below this similarity, use a generic fallback reply

GENERIC_FALLBACKS = [
    "Interesting — tell me a bit more about that?",
    "I hear you. What would you like to explore about that?",
    "Got it. Could you rephrase or add a little more detail?",
    "Not sure I fully caught that — could you say it differently?",
]


@st.cache_resource
def load_vectorizer():
    # Note: no stop-word removal here — these example phrases are short
    # ("how are you", "what can you do"), and stripping stop words like
    # "how"/"what"/"you" can leave nothing left to match on.
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(ALL_EXAMPLES)
    return vectorizer, matrix


def get_reply(user_text: str) -> str:
    text = user_text.strip()
    if not text:
        return "Say something and I'll respond!"

    vectorizer, matrix = load_vectorizer()
    user_vec = vectorizer.transform([text])

    similarities = (matrix @ user_vec.T).toarray().ravel()
    best_idx = int(np.argmax(similarities))
    best_score = float(similarities[best_idx])

    if best_score >= MATCH_THRESHOLD:
        tag = ALL_TAGS[best_idx]
        return random.choice(TAG_TO_RESPONSES[tag])

    # Light rule-based backstop for anything the intent matcher misses
    if "?" in text:
        return "Good question! I'm a simple built-in chatbot, so I can hold everyday " \
               "conversation but I don't have deep knowledge lookup here."
    return random.choice(GENERIC_FALLBACKS)


def render(log_usage_fn=None, username: str = "guest"):
    st.subheader("🤖 AI Chat Assistant")
    st.caption("A self-contained conversational bot — no API key or setup required.")

    with st.sidebar:
        st.markdown("---")
        st.markdown("**Chat Assistant**")
        if st.button("Clear chat history"):
            st.session_state.chat_history = []
            st.rerun()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Type a message...")
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            reply = get_reply(prompt)
            st.markdown(reply)

        st.session_state.chat_history.append({"role": "assistant", "content": reply})

        if log_usage_fn:
            log_usage_fn(username, "AI Chat Assistant")
