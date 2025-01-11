import streamlit as st
from transformers import pipeline

# Initialize pipelines
summarizer = pipeline("summarization", model="t5-small")
sentiment_analyzer = pipeline("sentiment-analysis")

# Streamlit app interface
st.title("Social Media Post Summarizer & Sentiment Analyzer")
st.write("This app summarizes social media posts and analyzes their sentiment (Positive or Negative).")

# Text input for the social media post
input_text = st.text_area("Enter the social media post below:", "")

if st.button("Analyze"):
    if input_text.strip():
        # Summarize the text
        summary = summarizer(input_text, max_length=50, min_length=25, do_sample=False)[0]['summary_text']

        # Analyze sentiment
        sentiment = sentiment_analyzer(input_text)[0]['label']

        # Display results
        st.subheader("Summary of the Post:")
        st.write(summary)

        st.subheader("Sentiment Analysis:")
        st.write(f"Sentiment: **{sentiment}**")
    else:
        st.warning("Please enter some text to analyze.")

st.write("Powered by Hugging Face Transformers 🌟")
