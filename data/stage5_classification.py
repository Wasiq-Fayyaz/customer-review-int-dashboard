import os
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

warnings.filterwarnings('ignore')

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# Loading cleaned data. Neutral (3 star) reviews have no label, so only positive / negative are used.
df = pd.read_csv("cleaned_reviews.csv")
df['cleaned'] = df['cleaned'].fillna("").astype(str)
df = df.dropna(subset=['sentiment'])
df = df[df['cleaned'].str.strip() != ""].reset_index(drop=True)
print(f"Loaded {len(df)} labeled reviews")
print(df['sentiment'].value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned'], df['sentiment'], test_size=0.2, random_state=42, stratify=df['sentiment']
)
print(f"\nTrain size: {len(X_train)}  Test size: {len(X_test)}")

# Same TF-IDF settings as Stage 3; fit on the training split only to avoid leakage
tfidf = TfidfVectorizer(max_features=5000, min_df=5, max_df=0.95, ngram_range=(1, 2))
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)
print(f"TF-IDF shape (train): {X_train_vec.shape}")

# Data is heavily skewed to positive, so class_weight='balanced' is used where supported
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "Linear SVM": LinearSVC(class_weight='balanced'),
}

results = {}
for name, model in models.items():
    print(f"\n--------------------- {name} ---------------------\n")
    model.fit(X_train_vec, y_train)
    preds = model.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)
    f1_neg = f1_score(y_test, preds, pos_label='negative')
    f1_macro = f1_score(y_test, preds, average='macro')
    results[name] = {"model": model, "preds": preds, "accuracy": acc, "f1_negative": f1_neg, "f1_macro": f1_macro}
    print(f"Accuracy: {acc:.4f}   Macro F1: {f1_macro:.4f}   Negative-class F1: {f1_neg:.4f}\n")
    print(classification_report(y_test, preds, digits=3))

# Model comparison (macro F1 used for picking, because accuracy hides weak negative-class performance)
summary = pd.DataFrame({
    n: {"accuracy": r["accuracy"], "macro_f1": r["f1_macro"], "negative_f1": r["f1_negative"]}
    for n, r in results.items()
}).T.round(4)
print("\n--------------------- Model Comparison ---------------------\n")
print(summary)
summary.to_csv(f"{OUT_DIR}/model_comparison.csv")

best_name = summary['macro_f1'].idxmax()
best = results[best_name]
print(f"\nBest model: {best_name}")

# Confusion matrix of best model
labels = ['positive', 'negative']
cm = confusion_matrix(y_test, best['preds'], labels=labels)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix - {best_name}")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/classification_confusion_matrix.png", dpi=150)
plt.close()

# Model comparison chart
summary.plot(kind='bar', figsize=(8, 4), colormap='viridis')
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.title("Model Comparison")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/model_comparison.png", dpi=150)
plt.close()

# Most influential words (Logistic Regression coefficients; classes sorted alphabetically,
# so a positive coefficient pushes towards 'positive' and a negative one towards 'negative')
lr = results["Logistic Regression"]["model"]
features = tfidf.get_feature_names_out()
order = np.argsort(lr.coef_[0])
neg_words = [features[i] for i in order[:15]]
pos_words = [features[i] for i in order[::-1][:15]]
print(f"\nTop words predicting {lr.classes_[0]}: {neg_words}")
print(f"Top words predicting {lr.classes_[1]}: {pos_words}")

top_idx = np.concatenate([order[:10], order[::-1][:10]])
plt.figure(figsize=(8, 6))
plt.barh([features[i] for i in top_idx], lr.coef_[0][top_idx],
         color=['#e76f51'] * 10 + ['#2a9d8f'] * 10)
plt.title("Most Influential Words (Logistic Regression)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/top_predictive_words.png", dpi=150)
plt.close()

# Save best model + vectorizer for reuse (e.g. by the report stage)
joblib.dump({"vectorizer": tfidf, "model": best["model"], "name": best_name}, "sentiment_model.joblib")
print("\nSaved sentiment_model.joblib")

print("\n--------------------- Sample Predictions ---------------------\n")
samples = [
    "Absolutely delicious, great flavor and I will buy again",
    "Terrible taste, arrived stale and I want a refund",
    "Arrived on time but the product was just okay",
]
for s in samples:
    pred = best["model"].predict(tfidf.transform([s.lower()]))[0]
    print(f"   {pred:<9} <- {s}")

print("\nEnding Stage 5")
