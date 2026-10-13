import ast
import json
import os
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

warnings.filterwarnings('ignore')

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# Collects every number the dashboard / PDF report needs into one JSON file, so the API never
# has to re-run the heavy NLP stages.
df = pd.read_csv("cleaned_reviews.csv")
df['cleaned'] = df['cleaned'].fillna("").astype(str)
df = df[df['cleaned'].str.strip() != ""].reset_index(drop=True)
labeled = df.dropna(subset=['sentiment'])

data = {}

data['overview'] = {
    "total_reviews": int(len(df)),
    "labeled_reviews": int(len(labeled)),
    "positive": int((labeled['sentiment'] == 'positive').sum()),
    "negative": int((labeled['sentiment'] == 'negative').sum()),
    "neutral": int(df['sentiment'].isna().sum()),
    "average_score": round(float(df['Score'].mean()), 3),
}
data['rating_distribution'] = {str(k): int(v) for k, v in df['Score'].value_counts().sort_index().items()}

# Top words per sentiment and the words most specific to each sentiment
counts = {}
for label in ['positive', 'negative']:
    counts[label] = Counter(" ".join(labeled.loc[labeled['sentiment'] == label, 'cleaned']).split())
data['top_words'] = {l: [{"word": w, "count": c} for w, c in counts[l].most_common(20)] for l in counts}

pos_total, neg_total = sum(counts['positive'].values()), sum(counts['negative'].values())
distinct = []
for word in set(counts['positive']) | set(counts['negative']):
    p, n = counts['positive'].get(word, 0), counts['negative'].get(word, 0)
    if p + n >= 50:
        distinct.append((word, float(np.log(((p + 1) / pos_total) / ((n + 1) / neg_total)))))
distinct.sort(key=lambda x: x[1])
data['distinctive_words'] = {
    "negative": [w for w, _ in distinct[:15]],
    "positive": [w for w, _ in distinct[-15:][::-1]],
}


# LDA topics
def lda_topics(texts, n_topics=5):
    vec = CountVectorizer(max_features=3000, min_df=10, max_df=0.5)
    matrix = vec.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=15).fit(matrix)
    names = vec.get_feature_names_out()
    return [[names[i] for i in comp.argsort()[-8:][::-1]] for comp in lda.components_]


data['topics'] = {
    "negative": lda_topics(labeled.loc[labeled['sentiment'] == 'negative', 'cleaned']),
    "positive": lda_topics(labeled.loc[labeled['sentiment'] == 'positive', 'cleaned'].head(15000)),
}

# Adjectives from stage 2
ling = pd.read_csv("linguistic_reviews.csv")
ling['adjectives'] = ling['adjectives'].apply(ast.literal_eval)
data['top_adjectives'] = {
    l: [{"word": w, "count": c} for w, c in Counter(
        a for row in ling.loc[ling['sentiment'] == l, 'adjectives'] for a in row).most_common(10)]
    for l in ['positive', 'negative']
}

# Model comparison from stage 5
comparison = pd.read_csv(f"{OUT_DIR}/model_comparison.csv", index_col=0)
data['model_comparison'] = [{"model": m, **row} for m, row in comparison.round(4).to_dict('index').items()]
data['best_model'] = comparison['macro_f1'].idxmax()

# Most negative reviews with long, informative text (useful as a "worst reviews" table)
neg = labeled[(labeled['sentiment'] == 'negative') & (labeled['Score'] == 1)]
neg = neg[neg['Text'].str.len().between(80, 400)].head(10)
data['sample_negative_reviews'] = [{"summary": r.Summary, "text": r.Text} for r in neg.itertuples()]

with open(f"{OUT_DIR}/dashboard_data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)
print(f"Saved {OUT_DIR}/dashboard_data.json")
print("Ending Stage 6")
