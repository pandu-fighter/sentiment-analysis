# Sentiment Analysis on Social Media Posts
# Author: Pandu

import nltk
import string
from nltk.corpus import twitter_samples, stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Download required datasets
nltk.download('twitter_samples')
nltk.download('stopwords')
nltk.download('punkt')

# Load dataset
positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')

# Labels
pos_labels = [1] * len(positive_tweets)
neg_labels = [0] * len(negative_tweets)

# Combine
tweets = positive_tweets + negative_tweets
labels = pos_labels + neg_labels

# Preprocessing function
stop_words = stopwords.words('english')
def preprocess(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in string.punctuation and t not in stop_words]
    return " ".join(tokens)

tweets_clean = [preprocess(t) for t in tweets]

# Split data
X_train, X_test, y_train, y_test = train_test_split(tweets_clean, labels, test_size=0.2, random_state=42)

# Feature extraction
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Predictions
y_pred = model.predict(X_test_vec)

# Accuracy
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# Custom prediction
while True:
    text = input("Enter a sentence to analyze sentiment (or 'exit' to quit): ")
    if text.lower() == 'exit':
        break
    text_clean = preprocess(text)
    text_vec = vectorizer.transform([text_clean])
    pred = model.predict(text_vec)[0]
    sentiment = "Positive" if pred == 1 else "Negative"
    print("Sentiment:", sentiment)
