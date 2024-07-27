import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import nltk
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
reviews = pd.read_csv('Reviews.csv',nrows=5000)  
def preprocess_text(text):
    tokens = word_tokenize(text)
    filtered_tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word.lower() not in stop_words and word.isalpha()]
    return ' '.join(filtered_tokens)
reviews['filtered_text'] = reviews['Text'].apply(preprocess_text)
def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity
reviews['sentiment'] = reviews['filtered_text'].apply(analyze_sentiment)
def generate_comment(polarity):
    if polarity > 0.5:
        return "Great product! I love it."
    elif polarity < -0.5:
        return "Disappointing product. I wouldn't recommend it."
    elif polarity >= -0.5 and polarity <= 0.5:
        return "Neutral product. It's average."
    else:
        return "Average product. It's okay."
reviews['comment'] = reviews['sentiment'].apply(generate_comment)
print(reviews.loc[10:15, ['Text', 'sentiment', 'comment']])
