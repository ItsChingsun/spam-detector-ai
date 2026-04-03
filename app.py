import streamlit as st
import os
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

DATA_FILE = "data.pkl"

# Load or create dataset
if os.path.exists(DATA_FILE):
    data = joblib.load(DATA_FILE)
    messages = data["messages"]
    labels = data["labels"]
else:
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
    labels = [1,1,1,1,0,0,0,0]

# Train model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(messages)

model = MultinomialNB()
model.fit(X, labels)

# UI
st.title("Spam Detector (Learning AI)")

user_message = st.text_input("Enter a message:")

if st.button("Predict"):
    X_msg = vectorizer.transform([user_message])
    pred = model.predict(X_msg)[0]
    prob = model.predict_proba(X_msg)[0]

    if pred == 1:
        st.error("Prediction: Spam")
    else:
        st.success("Prediction: Not Spam")

    st.write(f"Spam probability: {prob[1]:.2f}")

    st.write("### Teach the model")

    col1, col2 = st.columns(2)

    if col1.button("This is Spam"):
        messages.append(user_message)
        labels.append(1)
        joblib.dump({"messages": messages, "labels": labels}, DATA_FILE)
        st.success("Learned: Spam!")

    if col2.button("This is Not Spam"):
        messages.append(user_message)
        labels.append(0)
        joblib.dump({"messages": messages, "labels": labels}, DATA_FILE)
        st.success("Learned: Not Spam!")

# Show memory
st.write("### Model Memory")

for msg, label in zip(messages, labels):
    if label == 1:
        st.write(f"🔴 Spam: {msg}")
    else:
        st.write(f"🟢 Not Spam: {msg}")
