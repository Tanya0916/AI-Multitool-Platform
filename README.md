#  AI Multi-Tool Platform

An advanced multi-module AI application built with **Streamlit** and
**scikit-learn**, combining three AI capabilities behind a real
authentication system and a usage-analytics dashboard.



##  Modules Implemented :

| # | Module | Technique |
|---|--------|-----------|
| 1 | **AI Chat Assistant** | Fully self-contained — no API key, no external calls. Uses TF-IDF similarity matching (scikit-learn) over a bank of intents (greetings, small talk, jokes, mood check-ins, app help) to pick the best-matching reply. |
| 2 | **Text Summarizer** | Extractive summarization — TF-IDF sentence scoring (scikit-learn) ranks sentences by information density and returns the top‑N in original order. |
| 3 | **Sentiment Analysis** | TF-IDF + Logistic Regression classifier (scikit-learn), trained on an included labeled dataset of ~900 review-style sentences (positive / negative / neutral). Supports single-text and batch CSV analysis. |

## ✨ Platform Features :

- **Authentication system** — username/password accounts stored in SQLite with salted SHA-256 password hashing (no plaintext passwords).
- **Dashboard** — total interactions, active users, per-module usage bar chart, activity-over-time line chart, and a recent-activity log.
- **Usage analytics** — every module interaction is logged (user, module, timestamp) and surfaced on the dashboard.
- **Responsive UI** — custom CSS tightens layout and padding on narrow/mobile viewports; Streamlit's native layout is responsive by default.
- **Batch processing** — sentiment analysis supports CSV upload for bulk scoring, with a downloadable results file.

## 📁 Project Structure :

```
ai-multitool-platform/
├── app.py                      # Main app: auth, navigation, dashboard
├── modules/
│   ├── chat_assistant.py       # Module 1
│   ├── summarizer.py           # Module 2
│   └── sentiment.py            # Module 3
├── utils/
│   ├── auth.py                 # SQLite-backed auth (hashing, sessions)
│   └── analytics.py            # Usage logging + aggregation for dashboard
├── data/
│   ├── generate_dataset.py     # Builds the sentiment training dataset
│   └── sentiment_dataset.csv   # ~900 labeled rows (positive/negative/neutral)
├── models/
│   └── sentiment_pipeline.pkl  # Trained TF-IDF + LogisticRegression pipeline
├── train_sentiment_model.py    # Retrains and saves the sentiment model
├── requirements.txt
├── .streamlit/config.toml      # Theme
└── README.md
```

##  Run Locally :

```bash
# 1. Clone the repo
git clone https://github.com/Tanya0916/AI-Multitool-Platform.git
cd ai-multitool-platform

# 2. Install dependencies
pip install -r requirements.txt

# 3. Optional as pre-trained model is already included
python data/generate_dataset.py
python train_sentiment_model.py

# 4. Launch the app
streamlit run app.py
```


### Demo login :
A demo account is auto-created on first run:
```
username: demo
password: demo123
```
Or use "Create Account" to register your own.



## 📊 Dataset :

`data/sentiment_dataset.csv` — ~900 labeled sentences generated from a
combination of 30 subject templates × 45 sentiment-bearing phrase
patterns (positive/negative/neutral), designed for demonstrating a
clean, reproducible scikit-learn training pipeline. Regenerate anytime
with `python data/generate_dataset.py`, or swap in your own labeled
CSV (columns: `text`, `label`) and rerun `train_sentiment_model.py`.

## 🛠️ Technologies Used :

- **Streamlit** — UI framework and app hosting
- **scikit-learn** — TF-IDF vectorization, Logistic Regression, model pipeline
- **pandas / numpy** — data handling
- **SQLite** — authentication + usage-analytics storage

##  Possible Extensions (not implemented) :
- Image Classification module (TensorFlow/PyTorch + OpenCV)
- Resume Analyzer module (NLP entity extraction + scoring)
- AI Content Generator module (Hugging Face text-generation models)
- OAuth-based authentication in place of local username/password
- PostgreSQL instead of SQLite for multi-instance deployments


For educational/demo purposes.
