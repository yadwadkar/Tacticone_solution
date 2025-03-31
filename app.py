import streamlit as st
import joblib
import google.generativeai as genai

# Load trained model and vectorizer
model = joblib.load("query_classifier.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Configure Gemini API
genai.configure(api_key="YOUR_API_KEY_HERE")

def classify_query(query):
    query_tfidf = vectorizer.transform([query])
    category = model.predict(query_tfidf)[0]
    return category

def generate_ai_response(user_query):
    gemini_model = genai.GenerativeModel("gemini-1.5-pro-latest")
    response = gemini_model.generate_content(user_query)
    return response.text  

# Streamlit UI
st.title("AI-Powered Customer Support Assistant")

user_query = st.text_input("Enter your query:")

if st.button("Get Response"):
    if user_query:
        category = classify_query(user_query)
        ai_response = generate_ai_response(user_query)
        st.write("**Predicted Category:**", category)
        st.write("**AI Response:**", ai_response)
    else:
        st.write("Please enter a query.")
