import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


data = pd.read_csv('Reviews.csv')
data = data.head(50000)
data = data[['Score','Summary', 'Text']].dropna()

#dropna() is risky, means if the row has only 1 missing entry and the rest of column have values, It gets dropped no warning, concise and useful data might be lost
#Safe way dropna(subset=["Text"]) -> drop row only if text is empty

# Setup
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# Preprocessing
def preprocess(text):

    text = text.lower()

    text = re.sub(r'http\S+|www\S+',' ',text)

    text = re.sub(r'<.*?>',' ',text)

    text = re.sub(r'[^a-z\s]','',text)

    tokens = word_tokenize(text)

    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and len(word) > 2
    ]


    return ' '.join(tokens)

print("Applying Preprocessing.....")
data['cleaned'] = data['Text'].apply(preprocess)

def get_sentiment(score):
    if score >= 4:
        return 'positive'
    if score <= 2:
        return 'negative'
    else:
        return None

data['sentiment'] = data["Score"].apply(get_sentiment)

print("Before Preprocessing.....")
print(data['Text'].iloc[0][:200])

print("After Preprocessing.....")
print(data['cleaned'].iloc[0][:200])

print("Sentiment distribution.....")
print(data['sentiment'].value_counts())

data.to_csv("cleaned_reviews.csv",index = False)