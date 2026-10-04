import pandas as pd
import spacy
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from collections import Counter

nltk.download('averaged_perceptron_tagger_eng')

nlp = spacy.load("en_core_web_sm")

df = pd.read_csv("cleaned_reviews.csv")

# Trying linguistic analysis on first 5000 rows instead of 45k
sub_df = df.head(5000).copy()

# I - POS Tagging (Using NLTK on cleaned text)

print("Executing POS tagging \n")


# str was used just incase. Pandas might treat some values as float
def get_pos_tags(text):
    tokens = word_tokenize(str(text))

    return pos_tag(tokens)


sub_df['pos_tags'] = sub_df['cleaned'].apply(get_pos_tags)


def get_adjectives(pos_tags):
    return [word for word, tag in pos_tags if tag in ['JJ', 'JJR', 'JJS']]


sub_df['adjectives'] = sub_df['pos_tags'].apply(get_adjectives)

# Most Common Adjectives Overall
all_adjectives = [adj for adjs in sub_df['adjectives'] for adj in adjs]
top_adjectives = Counter(all_adjectives).most_common(15)
print("Top 15 Objectives in Review:")

for word, count in top_adjectives:
    print(f"   {word:<20} {count}")

# II - NER (using spaCy on original text)
print("\n Executing NER on original text...")


def get_entities(text):
    doc = nlp(str(text)[:500])
    return [(ent.text, ent.label_) for ent in doc.ents]


sub_df['entities'] = sub_df['Text'].apply(get_entities)

# Counting entity types.....
all_entities = [ent for ents in sub_df['entities'] for ent in ents]
entity_types = Counter([label for _, label in all_entities]).most_common()
print("\n Entity types found: \n")
for label, count in entity_types:
    print(f"   {label:<15} {count}")

# Most common organizations & products
orgs = [text for text, label in all_entities if label == 'ORG']
top_orgs = Counter(orgs).most_common(10)

print("\n Top 10 organisations mentioned: \n")
for org, count in top_orgs:
    print(f"   {org:<30} {count}")

products = [text for text, label in all_entities if label == 'PRODUCT']
top_products = Counter(products).most_common(10)
for product, count in top_products:
    print(f"  {product:<30} {count}")

# III - Dependency Parsing
print("\nDependency Parsing on sample reviews")

for i in range(3):
    text = sub_df['Text'].iloc[i][:200]
    doc = nlp(text)

    print(f"\nReview {i + 1}: {text} \n")
    print("Dependencies")

    found = False
    for token in doc:
        if token.dep_ in ['nsubj', 'dobj', 'amod', 'neg']:
            print(f"   {token.text:<15} : {token.dep_:<10}")
            found = True
    if not found:
        print("  No key dependencies found in first 150 chars")

sub_df.to_csv("linguistic_reviews.csv", index=False)
print("Ending State 2 ")
