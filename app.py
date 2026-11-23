import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("spam.csv")

# Input and output
X = df["message"]
y = df["label"]

# Convert text into numbers
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Train model
model = LogisticRegression()
model.fit(X_vectorized, y)

# Streamlit interface
st.title("📩 Spam Message Detector")

st.write("Enter a message below and check whether it is spam or not spam.")

message = st.text_area("Enter your message:")

if st.button("Check Message"):

    if message.strip() == "":
        st.warning("Please enter a message.")

    else:
        message_vectorized = vectorizer.transform([message])

        prediction = model.predict(message_vectorized)[0]

        if prediction == "spam":
            st.error("🚨 SPAM MESSAGE")
        else:
            st.success("✅ NOT SPAM")