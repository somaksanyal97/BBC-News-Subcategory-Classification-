#!/usr/bin/env python
# coding: utf-8

# In[2]:


import nltk
nltk.download('punkt_tab')


# In[3]:


import os
import urllib.request
import zipfile
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import spacy
from transformers import pipeline
import torch
from sklearn.datasets import load_files

# ---------------------- DOWNLOAD & EXTRACT BBC DATASET ----------------------
url = "http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip"
dataset_path = "bbc"
zip_path = "bbc.zip"

if not os.path.exists(dataset_path):
    print("Downloading BBC dataset...")
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(".")

# ---------------------- NLTK SETUP ----------------------
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# ---------------------- CONFIG ----------------------
subcategory_lists = {
    'entertainment': ['music', 'cinema', 'theatre', 'television', 'celebrity_news', 'film_awards', 'book_reviews', 'TV_series', 'concerts', 'festival'],
    'business': ['stock_market', 'company_news', 'mergers_acquisitions', 'economic_policy', 'oil_energy', 'employment', 'housing_market', 'global_trade', 'technology_business', 'financial_markets'],
    'sport': ['football', 'cricket', 'rugby', 'tennis', 'olympics', 'athletics', 'motorsport', 'golf', 'boxing', 'cycling'],
    'politics': ['elections', 'government_policy', 'party_politics', 'international_relations', 'parliament_news', 'health_policy', 'education_policy', 'defense_policy', 'immigration', 'environment_policy'],
    'tech': ['mobile_tech', 'software', 'internet', 'hardware', 'gaming', 'cybersecurity', 'AI_ML', 'social_media', 'telecom', 'gadgets']
}

# ---------------------- LOAD BBC DATASET ----------------------
print("Loading BBC dataset...")
dataset = load_files("bbc", encoding="latin1", decode_error="replace")
df = pd.DataFrame({'text': dataset.data, 'category': dataset.target})
df['category'] = df['category'].apply(lambda i: dataset.target_names[i])

print(f"Original shape: {df.shape}")
df = df.drop_duplicates(subset='text')
print(f"After removing duplicates: {df.shape}")

# ---------------------- TEXT CLEANING ----------------------
print("Cleaning text...")
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    tokens = word_tokenize(text.lower())
    filtered = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(filtered)

df['clean_text'] = df['text'].apply(clean_text)

# ---------------------- SUBCATEGORY CLASSIFICATION USING DISTILBERT ----------------------
print("Loading DistilBERT classifier...")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def classify_with_distilbert(text, category):
    subcategories = subcategory_lists.get(category, [])
    if not subcategories:
        return f"{category}_topic_other"
    try:
        result = classifier(text[:1000], candidate_labels=subcategories)
        top_score = result['scores'][0]
        if top_score < 0.5:
            return f"{category}_topic_other"
        return result['labels'][0]
    except Exception as e:
        print(f"[CLASSIFICATION ERROR] {e}")
        return "error"

print("Classifying articles...")
tqdm.pandas()
df['subcategory'] = df.progress_apply(lambda row: classify_with_distilbert(row['clean_text'], row['category']), axis=1)

# ---------------------- NAMED ENTITY RECOGNITION + ROLES ----------------------
print("Extracting media personalities using spaCy...")
nlp = spacy.load("en_core_web_sm")
role_keywords = {
    "politician": ["minister", "mp", "senator", "president", "prime minister", "government"],
    "musician": ["singer", "band", "musician", "album", "song"],
    "actor": ["actor", "actress", "film", "movie", "tv"],
    "tv_personality": ["tv", "host", "presenter", "show"]
}

def extract_named_entities_with_roles(text):
    doc = nlp(text)
    persons = set([ent.text for ent in doc.ents if ent.label_ == "PERSON"])
    roles = {}
    for person in persons:
        if person in roles:
            continue
        lower_text = text.lower()
        assigned_role = "unknown"
        for role, keywords in role_keywords.items():
            if any(keyword in lower_text for keyword in keywords):
                assigned_role = role
                break
        roles[person] = assigned_role
    return roles

df['entities_with_roles'] = df['text'].apply(extract_named_entities_with_roles)

# ---------------------- APRIL EVENT SUMMARY ----------------------
print("Extracting April event summaries...")
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def summarize_april_event(text):
    try:
        summary = summarizer(text[:1024], max_length=60, min_length=10, do_sample=False)[0]['summary_text']
        return summary
    except Exception as e:
        print(f"[SUMMARY ERROR] {e}")
        return ""

def extract_april_summary(text):
    if "april" in text.lower():
        return summarize_april_event(text)
    else:
        return ""

df['april_summary'] = df['text'].apply(extract_april_summary)

# ---------------------- DIAGNOSTICS ----------------------
print("Total articles processed:", len(df))
print("Subcategories assigned:", df['subcategory'].notna().sum())
print("Named entities extracted:", df['entities_with_roles'].notna().sum())
print("April summaries extracted:", (df['april_summary'] != "").sum())

# ---------------------- SAVE RESULTS ----------------------
df[['text', 'category', 'subcategory', 'entities_with_roles', 'april_summary']].to_csv("classified_bbc_articles_distilbert.csv", index=False)
print("Done. Results saved to 'classified_bbc_articles_distilbert.csv'.")


# In[7]:


get_ipython().system('pip install gensim')


# In[5]:


from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from gensim.utils import simple_preprocess

print("Calculating coherence scores for BERT-assigned subcategories...")

coherence_results = []

for category in df['category'].unique():
    subcats = df[df['category'] == category]['subcategory'].unique()

    for subcat in subcats:
        texts = df[(df['category'] == category) & (df['subcategory'] == subcat)]['clean_text'].tolist()
        processed_texts = [simple_preprocess(text, deacc=True) for text in texts if text.strip() != '']

        if len(processed_texts) < 5:
            print(f"⚠️ Skipping '{subcat}' (too few documents: {len(processed_texts)})")
            continue

        dictionary = Dictionary(processed_texts)
        corpus = [dictionary.doc2bow(text) for text in processed_texts]

        # Extract most common words in the subcategory as a pseudo-topic
        word_freq = dictionary.dfs  # document frequency of each word
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        topn = 10
        topics = [[dictionary[id] for id, freq in sorted_words[:topn]]]

        coherence_model = CoherenceModel(
            topics=topics,
            texts=processed_texts,
            dictionary=dictionary,
            coherence='c_v'
        )

        coherence_score = coherence_model.get_coherence()
        print(f"✅ Coherence for [{category} → {subcat}]: {coherence_score:.4f}")
        coherence_results.append((category, subcat, coherence_score))


# In[6]:


# --- Aggregate and print average coherence per main category ---
from collections import defaultdict
import numpy as np

category_to_scores = defaultdict(list)

for cat, subcat, score in coherence_results:
    category_to_scores[cat].append(score)

print("\n📊 Average Coherence Scores by Main Category:")
for cat, scores in category_to_scores.items():
    avg_score = np.mean(scores)
    print(f"🔸 {cat}: {avg_score:.4f}")


# In[7]:


counts = df[['category', 'subcategory']].value_counts().reset_index(name='count')

# Sort by 'category' and then 'subcategory'
counts_sorted = counts.sort_values(by=['category', 'subcategory'], ascending=[True, True]).reset_index(drop=True)

counts_sorted


# In[8]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Prepare Seaborn theme for a modern look
sns.set_theme(style="whitegrid", font_scale=1.1, palette="muted")

# Create individual plots per category
categories = counts_sorted['category'].unique()

for cat in categories:
    plt.figure(figsize=(10, 6))

    # Filter data for the category
    data_cat = counts_sorted[counts_sorted['category'] == cat].sort_values(by='count', ascending=True)

    # Barplot with color gradients
    colors = sns.color_palette("crest", len(data_cat))
    ax = sns.barplot(
        x='count',
        y='subcategory',
        data=data_cat,
        palette=colors
    )

    # Titles and labels
    ax.set_title(f"{cat.capitalize()} Subcategory Distribution", fontsize=18, weight='bold', pad=15)
    ax.set_xlabel('Article Count', fontsize=14)
    ax.set_ylabel('Subcategory', fontsize=14)

    # Add value labels for clarity
    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f', label_type='edge', fontsize=10, padding=3)

    # Remove spines for a cleaner aesthetic
    sns.despine(left=True, bottom=True)

    # Tight layout and display
    plt.tight_layout()
    plt.show()


# In[9]:


import pandas as pd
import plotly.express as px

# Use your counts_sorted DataFrame here

category_colors = {
    'business': '#636EFA',
    'entertainment': '#EF553B',
    'politics': '#00CC96',
    'sport': '#AB63FA',
    'tech': '#FFA15A'
}

fig = px.sunburst(
    counts_sorted,
    path=['category', 'subcategory'],
    values='count',
    color='category',
    color_discrete_map=category_colors,
    title='Interactive BBC Category and Subcategory Distribution (Distinct Rings)'
)

fig.update_layout(
    margin=dict(t=50, l=0, r=0, b=0),
    title_font_size=22,
    uniformtext=dict(minsize=10, mode='hide')
)

fig.show()


# In[ ]:




