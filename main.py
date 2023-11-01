import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("openai_secret_key")



# Load the pre-trained model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("finbert_finetuned")
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert", cache_dir='finbert')
st.session_state.model = model
st.session_state.tokenizer = tokenizer


# Function to predict sentiment
def predict_sentiment(text):
    encoded_input = st.session_state.tokenizer(text, return_tensors="pt")
    outputs = st.session_state.model(**encoded_input)
    predictions = outputs.logits.argmax(dim=-1)
    predicted_sentiment = predictions.item()
    label = st.session_state.model.config.id2label[predicted_sentiment]
    return predicted_sentiment, label


def decision(text, sentiment):
    prompt = f"""
    Make a decision summary for the following scenario based on the predicted sentiment of the news:
    News:
    {text}
    Sentiment:
    {sentiment}
    """
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return completion["choices"][0]["message"]["content"]


# Streamlit app
st.title("Finance News Sentiment Analysis")

# Text input for user to enter news
news_text = st.text_area("Enter finance news text:")

if st.button("Predict Sentiment"):
    if news_text:
        predicted_sentiment, label = predict_sentiment(news_text)
        st.write(f"The predicted sentiment is: {label}")
        st.write(decision(news_text, label))

