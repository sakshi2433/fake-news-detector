import streamlit as st
import joblib

model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.title("📰 Fake News Detector")
st.write("Enter a news article or headline and we'll tell you if it's **Fake** or **Real**.")

user_input = st.text_area("Enter News Text", height=200)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        transformed_input = vectorizer.transform([user_input])
        prediction = model.predict(transformed_input)[0]

        if prediction == 'FAKE':
            st.error("🚨 This news is most likely **FAKE**.")
        else:
            st.success("✅ This news is most likely **REAL**.")
