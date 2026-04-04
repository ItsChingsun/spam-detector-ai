import time
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from supabase import create_client, Client

st.set_page_config(page_title="Spam Detector Chatbot", layout="wide")

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
st.title("Spam Detector Chatbot + Dashboard")
st.caption("Chat with the spam detector, submit feedback, and review results.")

tab1, tab2 = st.tabs(["Chatbot", "Dashboard"])

# -------------------------------
# TAB 1: Chatbot
# -------------------------------
with tab1:
    st.subheader("Chat with the Spam Detector")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "last_prediction" not in st.session_state:
        st.session_state.last_prediction = None

    # Show previous chat messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Type a message to analyze...")

    if prompt:
        # Show user message immediately
        st.session_state.chat_history.append(
            {"role": "user", "content": prompt}
        )

        # Rerender current history first
        with st.chat_message("user"):
            st.markdown(prompt)

        # Show typing / thinking effect
        with st.chat_message("assistant"):
            thinking_box = st.empty()
            thinking_box.markdown("Typing...")

            time.sleep(1.0)

            X_msg = vectorizer.transform([prompt])
            pred = int(model.predict(X_msg)[0])
            prob = model.predict_proba(X_msg)[0]
            predicted_text = label_to_text(pred)
            spam_probability = float(prob[1])

            bot_reply = (
                f"That message looks like **{predicted_text}** to me.\n\n"
                f"My confidence is **{spam_probability:.2%}**.\n\n"
                f"Please tell me if I got it right so I can improve."
            )

            thinking_box.markdown(bot_reply)

        st.session_state.chat_history.append(
            {"role": "assistant", "content": bot_reply}
        )

        st.session_state.last_prediction = {
            "message": prompt,
            "predicted_label": predicted_text,
            "spam_probability": spam_probability,
        }

    if st.session_state.last_prediction is not None:
        st.write("### Give feedback")

        if not getattr(st.user, "is_logged_in", False):
            st.warning("Please log in with Google to submit feedback with your email.")
            if st.button("Login with Google"):
                st.login("google")
            st.stop()

        st.success(f"Signed in as: {get_current_user_email()}")

        col1, col2 = st.columns(2)

        if col1.button("Yes, this is Spam"):
            with st.chat_message("assistant"):
                save_box = st.empty()
                save_box.markdown("Saving feedback...")
                time.sleep(0.8)
                save_box.markdown("Thanks — I saved your feedback as **Spam**.")

            save_feedback(
                message=st.session_state.last_prediction["message"],
                predicted_label=st.session_state.last_prediction["predicted_label"],
                correct_label="Spam",
                spam_probability=st.session_state.last_prediction["spam_probability"],
            )

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": "Thanks — I saved your feedback as **Spam**."
                }
            )
            st.session_state.last_prediction = None
            st.rerun()

        if col2.button("No, this is Not Spam"):
            with st.chat_message("assistant"):
                save_box = st.empty()
                save_box.markdown("Saving feedback...")
                time.sleep(0.8)
                save_box.markdown("Thanks — I saved your feedback as **Not Spam**.")

            save_feedback(
                message=st.session_state.last_prediction["message"],
                predicted_label=st.session_state.last_prediction["predicted_label"],
                correct_label="Not Spam",
                spam_probability=st.session_state.last_prediction["spam_probability"],
            )

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": "Thanks — I saved your feedback as **Not Spam**."
                }
            )
            st.session_state.last_prediction = None
            st.rerun()

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
