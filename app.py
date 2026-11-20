# ============================================================
#  Amazon Phone Review Sentiment Analyzer  (P719)
#  Model: Logistic Regression + TF-IDF + TextBlob polarity
# ============================================================
import re
import pandas as pd
import streamlit as st
from textblob import TextBlob
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

st.set_page_config(page_title="Review Sentiment Analyzer", page_icon="📱")

# ---------- 1. Helper functions (same steps as the notebook) ----------
def clean_text(text):
    text = str(text).lower()                      # lowercase
    text = re.sub(r"[^a-z\s]", " ", text)         # keep only letters
    return re.sub(r"\s+", " ", text).strip()      # remove extra spaces

def polarity_feature(cleaned_text):
    return (TextBlob(cleaned_text).sentiment.polarity + 1) / 2   # scale -1..1 to 0..1

def rating_to_sentiment(rating):
    if rating >= 4:
        return "Positive"
    if rating == 3:
        return "Neutral"
    return "Negative"

# ---------- 2. Train the model once (cached, so the app stays fast) ----------
@st.cache_resource
def train_model():
    df = pd.read_excel("P719-Dataset.xlsx")
    df["sentiment"] = df["rating"].apply(rating_to_sentiment)
    df["clean_text"] = (df["title"].fillna("").astype(str) + " " + df["body"].fillna("").astype(str)).apply(clean_text)
    df["polarity"] = df["clean_text"].apply(polarity_feature)

    X_train_text, X_test_text, y_train, y_test, pol_train, pol_test = train_test_split(
        df["clean_text"], df["sentiment"], df["polarity"],
        test_size=0.2, random_state=42, stratify=df["sentiment"])

    tfidf = TfidfVectorizer(stop_words="english", max_features=3000, ngram_range=(1, 2))
    X_train = hstack([tfidf.fit_transform(X_train_text), csr_matrix(pol_train.values).T]).tocsr()
    X_test = hstack([tfidf.transform(X_test_text), csr_matrix(pol_test.values).T]).tocsr()

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    scores = {
        "accuracy": accuracy_score(y_test, predictions),
        "f1": f1_score(y_test, predictions, average="weighted"),
        "reviews": len(df),
    }
    return model, tfidf, scores

model, tfidf, scores = train_model()

def predict(reviews):
    """Takes a list of reviews, returns predicted labels and probabilities."""
    cleaned = [clean_text(r) for r in reviews]
    polarity = csr_matrix([[polarity_feature(c)] for c in cleaned])
    features = hstack([tfidf.transform(cleaned), polarity]).tocsr()
    return model.predict(features), model.predict_proba(features)

# ---------- 3. Page layout ----------
st.title("📱 Phone Review Sentiment Analyzer")
st.caption("Paste a customer review and find out if it is Positive, Neutral or Negative.")

with st.sidebar:
    st.header("About the model")
    st.write("**Logistic Regression** trained on Amazon phone reviews using TF-IDF and TextBlob polarity.")
    st.metric("Test accuracy", f"{scores['accuracy']:.1%}")
    st.metric("Weighted F1-score", f"{scores['f1']:.1%}")
    st.metric("Reviews used", scores["reviews"])
    st.info("Neutral reviews are the hardest to predict, because they often mix good and bad opinions.")

COLORS = {"Positive": "#16a34a", "Neutral": "#f59e0b", "Negative": "#dc2626"}
EMOJIS = {"Positive": "😊", "Neutral": "😐", "Negative": "😠"}

tab_single, tab_batch = st.tabs(["✍️ Single review", "📂 Many reviews (file)"])

# ---------- Tab 1: one review ----------
with tab_single:
    st.write("Try an example:")
    examples = {
        "😊 Positive": "Amazing phone, battery lasts all day and the camera is superb!",
        "😠 Negative": "Worst phone ever, the screen stopped working in a week.",
        "😐 Mixed": "Camera is good but the battery drains very fast. Average phone overall.",
    }
    columns = st.columns(3)
    for column, (label, text) in zip(columns, examples.items()):
        if column.button(label, use_container_width=True):
            st.session_state["review"] = text

    review = st.text_area("Your review", key="review", height=150,
                          placeholder="Type or paste a phone review here...")

    if st.button("Analyze sentiment", type="primary"):
        if not review.strip():
            st.warning("Please enter a review first.")
        else:
            labels, probabilities = predict([review])
            result = labels[0]
            chances = dict(zip(model.classes_, probabilities[0]))

            st.markdown(
                f"""<div style="padding:20px; border-radius:12px; text-align:center;
                     background-color:{COLORS[result]}22; border:2px solid {COLORS[result]};">
                     <div style="font-size:42px;">{EMOJIS[result]}</div>
                     <div style="font-size:26px; font-weight:700; color:{COLORS[result]};">{result}</div>
                     <div>Confidence: {chances[result]:.0%}</div></div>""",
                unsafe_allow_html=True)

            st.write("")
            st.subheader("How sure is the model?")
            for name in ["Positive", "Neutral", "Negative"]:
                st.write(f"{EMOJIS[name]} {name}: **{chances[name]:.0%}**")
                st.progress(float(chances[name]))

            if chances[result] < 0.5:
                st.info("The model is not very confident about this review.")

# ---------- Tab 2: a file with many reviews ----------
with tab_batch:
    st.write("Upload a CSV or Excel file that has a column of reviews.")
    file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

    if file is not None:
        data = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
        review_column = st.selectbox("Which column contains the reviews?", data.columns)

        if st.button("Analyze all reviews", type="primary"):
            labels, _ = predict(data[review_column].fillna("").astype(str).tolist())
            data["predicted_sentiment"] = labels

            st.subheader("Summary")
            st.bar_chart(data["predicted_sentiment"].value_counts())
            st.dataframe(data, use_container_width=True)
            st.download_button("⬇️ Download results", data.to_csv(index=False),
                               file_name="sentiment_results.csv", mime="text/csv")
