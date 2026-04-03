import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from supabase import create_client, Client

st.set_page_config(page_title="Spam Detector Dashboard", layout="wide")

# -------------------------------
# Supabase
# -------------------------------
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# -------------------------------
# Training data
# -------------------------------
messages = [
    "win money now",
    "free prize",
    "claim your reward",
    "click here for cash",
    "call me later",
    "see you tomorrow",
    "let us meet today",
    "please send homework"
]

labels = [1, 1, 1, 1, 0, 0, 0, 0]

# -------------------------------
# Train model
# -------------------------------
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(messages)

model = MultinomialNB()
model.fit(X, labels)

# -------------------------------
# Helpers
# -------------------------------
def label_to_text(label: int) -> str:
    return "Spam" if label == 1 else "Not Spam"

def save_feedback(message: str, predicted_label: str, correct_label: str, spam_probability: float) -> None:
    supabase.table("spam_feedback").insert({
        "message": message,
        "predicted_label": predicted_label,
        "correct_label": correct_label,
        "spam_probability": spam_probability
    }).execute()

def load_feedback():
    response = (
        supabase
        .table("spam_feedback")
        .select("*")
        .order("created_at", desc=True)
        .execute()
    )
    return response.data if response.data else []

# -------------------------------
# UI
# -------------------------------
st.title("Spam Detector + Data Dashboard")

tab1, tab2 = st.tabs(["Predict", "Dashboard"])

# -------------------------------
# TAB 1: Predict
# -------------------------------
with tab1:
    user_message = st.text_input("Enter a message:")

    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    if st.button("Predict"):
        X_msg = vectorizer.transform([user_message])
        pred = model.predict(X_msg)[0]
        prob = model.predict_proba(X_msg)[0]

        st.session_state.last_result = {
            "message": user_message,
            "pred": int(pred),
            "spam_probability": float(prob[1]),
        }

    if st.session_state.last_result:
        result = st.session_state.last_result
        predicted_text = label_to_text(result["pred"])

        if result["pred"] == 1:
            st.error(f"Prediction: {predicted_text}")
        else:
            st.success(f"Prediction: {predicted_text}")

        st.write(f"Spam probability: {result['spam_probability']:.2f}")
        st.write("### Teach the model")

        col1, col2 = st.columns(2)

        if col1.button("This is Spam"):
            save_feedback(
                message=result["message"],
                predicted_label=predicted_text,
                correct_label="Spam",
                spam_probability=result["spam_probability"],
            )
            st.success("Saved online as: Spam")

        if col2.button("This is Not Spam"):
            save_feedback(
                message=result["message"],
                predicted_label=predicted_text,
                correct_label="Not Spam",
                spam_probability=result["spam_probability"],
            )
            st.success("Saved online as: Not Spam")

# -------------------------------
# TAB 2: Dashboard
# -------------------------------
with tab2:
    st.subheader("Collected User Feedback")

    feedback_data = load_feedback()

    if not feedback_data:
        st.info("No feedback saved yet.")
    else:
        df = pd.DataFrame(feedback_data)

        total_rows = len(df)
        spam_rows = (df["correct_label"] == "Spam").sum()
        not_spam_rows = (df["correct_label"] == "Not Spam").sum()

        c1, c2, c3 = st.columns(3)
        c1.metric("Total feedback", total_rows)
        c2.metric("Spam labels", int(spam_rows))
        c3.metric("Not Spam labels", int(not_spam_rows))

        st.write("### Label distribution")
        chart_data = df["correct_label"].value_counts()
        st.bar_chart(chart_data)

        st.write("### Spam probability trend")
        if "spam_probability" in df.columns:
            trend_df = df[["created_at", "spam_probability"]].copy()
            trend_df["created_at"] = pd.to_datetime(trend_df["created_at"])
            trend_df = trend_df.sort_values("created_at")
            trend_df = trend_df.set_index("created_at")
            st.line_chart(trend_df)

        st.write("### Saved feedback table")
        st.dataframe(df, use_container_width=True)
