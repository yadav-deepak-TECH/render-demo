import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk

# Download stopwords once, using streamlit's caching
@st.cache_resource
def load_stopwords():
    nltk.download('stopwords')
    return stopwords.words('english')

# Load model and vectorizer once
@st.cache_resource
def load_model_and_vectorizer():
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

# Define sentiment prediction function
def predict_sentiment(text, model, vectorizer, stop_words):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [word for word in text if word not in stop_words]
    text = ' '.join(text)
    text_vector = vectorizer.transform([text])
    sentiment = model.predict(text_vector)[0]
    return "Negative" if sentiment == 0 else "Positive"

# Dummy initializer for scraper (replace with actual if needed)
@st.cache_resource
def initialize_scraper():
    class DummyScraper:
        def get_tweets(self, username, model='user', number=5):
            # Simulate tweets
            return {
                'tweets': [{'text': f"Sample tweet {i} from @{username}"} for i in range(1, number + 1)]
            }
    return DummyScraper()

# Function to create a color card
def create_card(tweet_text, sentiment):
    colour = "green" if sentiment == "Positive" else "red"
    card_html = f"""
    <div style="background-color: {colour}; padding: 10px; border-radius: 5px; margin: 10px 0;">
        <h5 style="color: white;">{sentiment} Sentiment</h5>
        <p style="color: white;">{tweet_text}</p>
    </div>
    """
    return card_html

# Main app logic
def main():
    st.title("Twitter Sentiment Analyses")

    stop_words = load_stopwords()
    model, vectorizer = load_model_and_vectorizer()
    scraper = initialize_scraper()

    option = st.selectbox("Choose an option", ["input text", "get tweets from user"])

    if option == "input text":
        text_input = st.text_area("Enter text to analyze sentiment:")
        if st.button("Analyze"):
            sentiment = predict_sentiment(text_input, model, vectorizer, stop_words)
            st.write(f"Sentiment: {sentiment}")

    elif option == "get tweets from user":
        username = st.text_input("Enter the Twitter username:")
        if st.button("Fetch Tweets"):
            tweets_data = scraper.get_tweets(username, model='user', number=5)
            if 'tweets' in tweets_data:
                for tweet in tweets_data['tweets']:
                    tweet_text = tweet['text']
                    sentiment = predict_sentiment(tweet_text, model, vectorizer, stop_words)
                    card_html = create_card(tweet_text, sentiment)
                    st.markdown(card_html, unsafe_allow_html=True)
            else:
                st.write("No tweets found or an error occurred.")

if __name__ == "__main__":
    main()
