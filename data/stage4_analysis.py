import ast
import os
import warnings
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from gensim.models import Word2Vec
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score
from textblob import TextBlob
from wordcloud import WordCloud

warnings.filterwarnings('ignore')

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# Loading data (Stage 1 output). Neutral reviews (Score 3) have no sentiment label, so they are
# excluded from sentiment comparisons but kept for the rating distribution.
df = pd.read_csv("cleaned_reviews.csv")
df['cleaned'] = df['cleaned'].fillna("").astype(str)
df = df[df['cleaned'].str.strip() != ""].reset_index(drop=True)
labeled = df.dropna(subset=['sentiment']).reset_index(drop=True)
print(f"Loaded {len(df)} reviews ({len(labeled)} with positive/negative label)")

# I - Rating & sentiment distribution
print("\n--------------------- Rating Distribution ---------------------\n")
print(df['Score'].value_counts().sort_index())

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.countplot(x='Score', data=df, palette='viridis', ax=ax[0])
ax[0].set_title("Rating Distribution")
sns.countplot(x='sentiment', data=labeled, order=['positive', 'negative'],
              palette=['#2a9d8f', '#e76f51'], ax=ax[1])
ax[1].set_title("Sentiment Distribution")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/rating_sentiment_distribution.png", dpi=150)
plt.close()

# II - Lexicon based sentiment (TextBlob polarity on original text) vs rating-based labels
print("\n--------------------- TextBlob Sentiment ---------------------\n")
sample = labeled.sample(n=min(10000, len(labeled)), random_state=42).copy()
sample['polarity'] = sample['Text'].astype(str).apply(lambda t: TextBlob(t[:1000]).sentiment.polarity)
sample['subjectivity'] = sample['Text'].astype(str).apply(lambda t: TextBlob(t[:1000]).sentiment.subjectivity)
sample['textblob_label'] = np.where(sample['polarity'] >= 0, 'positive', 'negative')

print(sample.groupby('sentiment')[['polarity', 'subjectivity']].mean().round(3))
tb_acc = accuracy_score(sample['sentiment'], sample['textblob_label'])
print(f"\nTextBlob agreement with rating-based label: {tb_acc:.4f}")

cm = confusion_matrix(sample['sentiment'], sample['textblob_label'], labels=['positive', 'negative'])
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['positive', 'negative'], yticklabels=['positive', 'negative'])
plt.xlabel("TextBlob prediction")
plt.ylabel("Rating-based label")
plt.title("TextBlob vs Rating Labels")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/textblob_confusion_matrix.png", dpi=150)
plt.close()

plt.figure(figsize=(7, 4))
sns.boxplot(x='Score', y='polarity', data=sample, palette='coolwarm')
plt.title("TextBlob Polarity by Rating")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/polarity_by_rating.png", dpi=150)
plt.close()

# III - Word frequency per sentiment + Word Clouds
print("\n--------------------- Top Words per Sentiment ---------------------\n")
top_words = {}
for label in ['positive', 'negative']:
    words = " ".join(labeled.loc[labeled['sentiment'] == label, 'cleaned']).split()
    counts = Counter(words)
    top_words[label] = counts
    print(f"Top 15 words in {label} reviews:")
    for word, count in counts.most_common(15):
        print(f"   {word:<20} {count}")
    print()

    wc = WordCloud(width=900, height=450, background_color='white',
                   colormap='Greens' if label == 'positive' else 'Reds',
                   max_words=100).generate_from_frequencies(counts)
    plt.figure(figsize=(9, 4.5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"{label.capitalize()} Reviews Word Cloud")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/wordcloud_{label}.png", dpi=150)
    plt.close()

# Words most distinctive for each sentiment (log ratio of relative frequencies)
pos_total = sum(top_words['positive'].values())
neg_total = sum(top_words['negative'].values())
distinct = []
for word in set(top_words['positive']) | set(top_words['negative']):
    p, n = top_words['positive'].get(word, 0), top_words['negative'].get(word, 0)
    if p + n >= 50:
        distinct.append((word, np.log(((p + 1) / pos_total) / ((n + 1) / neg_total))))
distinct.sort(key=lambda x: x[1])
print("Most negative-specific words:", [w for w, _ in distinct[:15]])
print("Most positive-specific words:", [w for w, _ in distinct[-15:][::-1]])

# IV - Topic Modeling (LDA) to surface complaint / praise themes
print("\n--------------------- LDA Topic Modeling ---------------------\n")


def run_lda(texts, n_topics, name):
    vec = CountVectorizer(max_features=3000, min_df=10, max_df=0.5)
    matrix = vec.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=15)
    lda.fit(matrix)
    names = vec.get_feature_names_out()
    print(f"Topics in {name} reviews:")
    for i, comp in enumerate(lda.components_):
        print(f"   Topic {i + 1}: {', '.join(names[comp.argsort()[-10:][::-1]])}")
    print()


run_lda(labeled.loc[labeled['sentiment'] == 'negative', 'cleaned'], 5, "negative")
run_lda(labeled.loc[labeled['sentiment'] == 'positive', 'cleaned'].head(15000), 5, "positive")

# V - Word2Vec (the model stage 3 prepared the sentences for)
print("\n--------------------- Word2Vec Similarity ---------------------\n")
sentences = [t.split() for t in df['cleaned']]
w2v = Word2Vec(sentences, vector_size=100, window=5, min_count=5, workers=4, epochs=5, seed=42)
w2v.save("word2vec.model")
print(f"Word2Vec vocabulary size: {len(w2v.wv)}")
for word in ['delicious', 'terrible', 'coffee', 'price', 'shipping']:
    if word in w2v.wv:
        similar = ", ".join(f"{w} ({s:.2f})" for w, s in w2v.wv.most_similar(word, topn=6))
        print(f"   {word:<10} -> {similar}")

# VI - Adjectives from Stage 2
print("\n--------------------- Stage 2 Linguistic Insights ---------------------\n")
ling = pd.read_csv("linguistic_reviews.csv")
ling['adjectives'] = ling['adjectives'].apply(ast.literal_eval)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for a, label, color in zip(ax, ['positive', 'negative'], ['#2a9d8f', '#e76f51']):
    adjs = Counter(x for row in ling.loc[ling['sentiment'] == label, 'adjectives'] for x in row).most_common(10)
    print(f"Top adjectives ({label}): {adjs}")
    a.barh([w for w, _ in adjs][::-1], [c for _, c in adjs][::-1], color=color)
    a.set_title(f"Top Adjectives - {label}")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/top_adjectives.png", dpi=150)
plt.close()

print("\nEnding Stage 4")
