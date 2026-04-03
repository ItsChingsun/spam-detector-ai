import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from supabase import create_client, Client

st.set_page_config(page_title="Spam Detector Dashboard", layout="wide")

# -------------------------------
# Secrets / Config
# -------------------------------
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

ALLOWED_USERS = [
    "chingvong26@gmail.com"
]

# -------------------------------
# Supabase
# -------------------------------
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# -------------------------------
# Training data
# -------------------------------
messages = [
    "win money now",
    "free prize",
    "claim your reward",
    "click here for cash",
    "limited offer act now",
    "urgent you have won",
    "call me later",
    "see you tomorrow",
    "let us meet today",
    "please send homework",
    "can we talk tonight",
    "meeting starts at 3pm"
]

labels = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]

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


def get_current_user_email():
    return getattr(st.user, "email", None)


def save_feedback(
    message: str,
    predicted_label: str,
    correct_label: str,
    spam_probability: float,
) -> None:
    user_email = get_current_user_email()

    supabase.table("spam_feedback").insert(
        {
            "message": message,
            "predicted_label": predicted_label,
            "correct_label": correct_label,
            "spam_probability": spam_probability,
            "user_email": user_email,
        }
    ).execute()


def load_feedback():
    try:
        response = (
            supabase.table("spam_feedback")
            .select("*")
            .order("created_at", desc=True)
            .execute()
        )
        return response.data if response.data else []
    except Exception:
        st.error("Database error. Check Supabase setup and key.")
        return []


def require_dashboard_login():
    if not getattr(st.user, "is_logged_in", False):
        st.warning("Please log in with Google to view the dashboard.")
        st.login("google")
        st.stop()

    user_email = get_current_user_email()

    if user_email not in ALLOWED_USERS:
        st.error("Access denied. Your account is not authorized for the dashboard.")
        st.stop()

    st.success(f"Logged in as: {user_email}")


# -------------------------------
# UI
# -------------------------------
st.title("Spam Detector + Data Dashboard")
st.caption("Predict messages, collect feedback, and review results.")

tab1, tab2 = st.tabs(["Predict", "Dashboard"])

# -------------------------------
# TAB 1: Predict
# -------------------------------
with tab1:
    st.subheader("Spam Prediction")

    user_message = st.text_input("Enter a message:")

    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    if st.button("Predict", type="primary"):
        if not user_message.strip():
            st.warning("Please enter a message first.")
        else:
            X_msg = vectorizer.transform([user_message])
            pred = int(model.predict(X_msg)[0])
            prob = model.predict_proba(X_msg)[0]

            st.session_state.last_result = {
                "message": user_message,
                "pred": pred,
                "spam_probability": float(prob[1]),
            }

    if st.session_state.last_result:
        result = st.session_state.last_result
        predicted_text = label_to_text(result["pred"])

        st.write("### Prediction Result")
        if result["pred"] == 1:
            st.error(f"Prediction: {predicted_text}")
        else:
            st.success(f"Prediction: {predicted_text}")

        st.write(f"Spam probability: {result['spam_probability']:.2%}")
        st.write("### Teach the model")

        if getattr(st.user, "is_logged_in", False):
            st.info(f"Signed in as: {get_current_user_email()}")
        else:
            st.info("Feedback can still be saved without login, but user email will be blank.")

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
# TAB 2: Dashboard (Protected)
# -------------------------------
with tab2:
    require_dashboard_login()

    st.subheader("Collected User Feedback")

    feedback_data = load_feedback()

    if not feedback_data:
        st.info("No feedback saved yet.")
    else:
        df = pd.DataFrame(feedback_data)

        total_rows = len(df)
        spam_rows = int((df["correct_label"] == "Spam").sum())
        not_spam_rows = int((df["correct_label"] == "Not Spam").sum())

        c1, c2, c3 = st.columns(3)
        c1.metric("Total feedback", total_rows)
        c2.metric("Spam labels", spam_rows)
        c3.metric("Not Spam labels", not_spam_rows)

        st.write("### Label distribution")
        chart_data = df["correct_label"].value_counts()
        st.bar_chart(chart_data)

        st.write("### Spam probability trend")
        if "spam_probability" in df.columns and "created_at" in df.columns:
            trend_df = df[["created_at", "spam_probability"]].copy()
            trend_df["created_at"] = pd.to_datetime(trend_df["created_at"])
            trend_df = trend_df.sort_values("created_at")
            trend_df = trend_df.set_index("created_at")
            st.line_chart(trend_df)

        if "user_email" in df.columns:
            st.write("## User Analytics")

            unique_users = df["user_email"].dropna().nunique()
            st.metric("Unique users", unique_users)

            st.write("### Activity per user")
            user_counts = df["user_email"].fillna("Anonymous").value_counts()
            st.bar_chart(user_counts)

            st.write("### Most active labelers")
            active_users_df = user_counts.reset_index()
            active_users_df.columns = ["user_email", "feedback_count"]
            st.dataframe(active_users_df, use_container_width=True)

            st.write("### Signed-in users")
            users_df = pd.DataFrame(
                {"user_email": sorted(df["user_email"].dropna().unique())}
            )
            if not users_df.empty:
                st.dataframe(users_df, use_container_width=True)

        st.write("### Saved feedback table")
        st.dataframe(df, use_container_width=True)
