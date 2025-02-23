import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
data_path = "cleaned_period_pal_samples (1).csv"
df = pd.read_csv(data_path)
df.columns = ["prompt", "response"]
df = df.iloc[1:].reset_index(drop=True)

def clean_text(text):
    return text.strip("'").lower()

df["prompt"] = df["prompt"].apply(clean_text)
df["response"] = df["response"].apply(clean_text)

# Vectorize prompts
vectorizer = TfidfVectorizer()
prompt_vectors = vectorizer.fit_transform(df["prompt"])

def get_response(user_input):
    user_vector = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vector, prompt_vectors)
    best_match_idx = similarities.argmax()
    return df.loc[best_match_idx, "response"]

# Streamlit UI
st.title("Period Pal Chatbot")
st.write("Ask any question about periods and get an answer!")

user_input = st.text_input("Your question:")
if user_input:
    response = get_response(clean_text(user_input))
    st.write("*Response:*", response)
