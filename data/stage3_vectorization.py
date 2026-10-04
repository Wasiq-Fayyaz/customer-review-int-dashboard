import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from gensim.models import Word2Vec
import warnings
warnings.filterwarnings('ignore')


# Loading Cleaned Data
df = pd.read_csv('cleaned_reviews.csv')
print(f"Loaded  {len(df)}")
print(f"Columns  {df.columns}")
print("Sample Cleaned Review: ")
print(df['cleaned'].iloc[0])

# Bag of Words
print("\n--------------------- Executing BOW ---------------------\n")
bow_vectorizer = CountVectorizer(
    max_features= 5000,
    min_df= 5,
    max_df= 0.95
)


bow_matrix = bow_vectorizer.fit_transform(df['cleaned'])

print(f"Bag of Words matrix shape: {bow_matrix.shape}")
print(f"Vocabulary size: {len(bow_vectorizer.vocabulary_)}")
print(f"\nSample Vocab (first 20 words): ")

vocab = list(bow_vectorizer.vocabulary_.keys())[:20]
for word in vocab:
    print(f"   {word}")


# II - TF-IDF
print("Executing TF-IDF........")

tfidf_vectorizer = TfidfVectorizer(
    max_features= 5000,
    min_df= 5,
    max_df= 0.95,
    ngram_range= (1,2) #include single word and 2 word phrases
)

tfidf_matrix = tfidf_vectorizer.fit_transform(df['cleaned'])
print(f"\n tfidf matrix shape: {tfidf_matrix.shape}")
print(f"\n Vocabulary Size: {tfidf_vectorizer.vocabulary_}")

features_name = tfidf_vectorizer.get_feature_names_out()
first_review_scores = tfidf_matrix[0].toarray()[0]
top_indices = first_review_scores.argsort()[-10:][::-1]

print("\n Top 10 TF-IDF words in review")
for idx in top_indices:
    print(f"   {features_name[idx] < 30} score: {first_review_scores[idx]:.4f}")