import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
try:
    df = pd.read_csv("cleaned_period_pal_samples (1).csv")
except FileNotFoundError:
    st.error("Please upload the file 'cleaned_period_pal_samples (1).csv'")
    st.stop()

prompts = df['prompt'].tolist()
responses = df['model response'].tolist()

# Train TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(prompts)

def get_bot_response(user_input):
    user_vector = tfidf_vectorizer.transform([user_input])
    cosine_similarities = cosine_similarity(user_vector, tfidf_matrix)
    most_similar_index = cosine_similarities.argmax()
    return responses[most_similar_index]

st.title("PeriodPal")
st.write("Hey there! I'm your personal AI-powered period buddy! Get cycle tracking, pain relief tips, and emotional support—all in a friendly chat. Whether you need cramps remedies, mood-boosting advice, or just someone to check in, I'm here for you.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(message["content"])

if prompt := st.chat_input("Go ahead, say anything!"):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    bot_response = get_bot_response(prompt)
    st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
    with st.chat_message("assistant"):
        st.markdown(bot_response)
