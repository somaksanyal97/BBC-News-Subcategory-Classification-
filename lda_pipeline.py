import os

def main():
    # coding: utf-8
    
    # In[3]:
    
    
    get_ipython().system('pip install --user torch --upgrade')
    
    
    # In[7]:
    
    
    get_ipython().system('pip install wordcloud')
    
    
    # In[10]:
    
    
    import os
    import pandas as pd
    from bs4 import BeautifulSoup
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud
    import re
    from nltk.corpus import stopwords, wordnet
    from nltk.stem import WordNetLemmatizer
    import nltk
    from sklearn.feature_extraction.text import CountVectorizer
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    
    # Load BBC dataset
    def load_bbc_dataset(dataset_path):
        data = []
        for category in os.listdir(dataset_path):
            cat_path = os.path.join(dataset_path, category)
            if not os.path.isdir(cat_path):
                continue
            for file in os.listdir(cat_path):
                with open(os.path.join(cat_path, file), 'r', encoding='latin1') as f:
                    text = f.read()
                    text = BeautifulSoup(text, "html.parser").get_text()
                    data.append({"category": category, "text": text})
        return pd.DataFrame(data)
    
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        stop_words = set(stopwords.words('english'))
        additional_stopwords = set(['said', 'mr', 'new', 'year', 'years', 'like', 'make', 'use', 'people', 'just', 'also', 'could', 'would', 'one', 'time'])
        all_stopwords = stop_words.union(additional_stopwords)
        words = [word for word in text.split() if word not in all_stopwords]
        return ' '.join(words)
    
    def lemmatize_text(text):
        lemmatizer = WordNetLemmatizer()
        words = text.split()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
        return ' '.join(lemmatized_words)
    
    # Load and clean data
    df = load_bbc_dataset(r"C:\Users\User\OneDrive\Desktop\BBC\data\bbc-fulltext\bbc")
    df['clean_text'] = df['text'].apply(clean_text)
    df['clean_text'] = df['clean_text'].apply(lemmatize_text)
    
    # Generate WordCloud per category on cleaned + lemmatized text
    for category in df['category'].unique():
        text = " ".join(df[df['category'] == category]['clean_text'])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'WordCloud for {category} (Cleaned + Lemmatized)')
        plt.show()
    
    
    # Prepare stopwords
    stop_words = set(stopwords.words('english'))
    additional_stopwords = set(['said', 'mr', 'new', 'year', 'years', 'like', 'make', 'use', 'people', 'just'])
    all_stopwords = stop_words.union(additional_stopwords)
    
    # Use list instead of set
    vectorizer = CountVectorizer(stop_words=list(all_stopwords), max_features=50)
    
    for category in df['category'].unique():
        texts = df[df['category'] == category]['clean_text']
        X = vectorizer.fit_transform(texts)
        print(f"\nTop 50 terms found for category '{category}':")
        print(vectorizer.get_feature_names_out())
    
    
    
    # In[19]:
    
    
    import os
    import pandas as pd
    from bs4 import BeautifulSoup
    import re
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    import nltk
    from gensim import corpora, models
    from gensim.utils import simple_preprocess
    from gensim.models import CoherenceModel
    import spacy
    
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    
    # Load BBC dataset
    def load_bbc_dataset(dataset_path):
        data = []
        for category in os.listdir(dataset_path):
            cat_path = os.path.join(dataset_path, category)
            if not os.path.isdir(cat_path):
                continue
            for file in os.listdir(cat_path):
                with open(os.path.join(cat_path, file), 'r', encoding='latin1') as f:
                    text = f.read()
                    text = BeautifulSoup(text, "html.parser").get_text()
                    data.append({"category": category, "text": text})
        return pd.DataFrame(data)
    
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        stop_words = set(stopwords.words('english'))
        additional_stopwords = set(['said', 'mr', 'new', 'year', 'years', 'like', 'make', 'use', 'people', 'just', 'also', 'could', 'would', 'one', 'time', 'government', 'company', 'minister', 'team'])
        all_stopwords = stop_words.union(additional_stopwords)
        words = [word for word in text.split() if word not in all_stopwords]
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
        return ' '.join(lemmatized_words)
    
    # Load and clean data
    df = load_bbc_dataset(r"C:\Users\User\OneDrive\Desktop\BBC\data\bbc-fulltext\bbc")
    print(f"Original shape: {df.shape}")
    df = df.drop_duplicates(subset='text')
    print(f"After removing duplicates: {df.shape}")
    df['clean_text'] = df['text'].apply(clean_text)
    
    # Fit LDA models with enhanced settings
    lda_models = {}
    dictionaries = {}
    categories = ['business', 'entertainment', 'sport', 'politics', 'tech']
    
    for category in categories:
        print(f"Fitting LDA for {category}")
        texts = df[df['category'] == category]['clean_text'].tolist()
        processed_texts = [simple_preprocess(doc, deacc=True) for doc in texts]
        dictionary = corpora.Dictionary(processed_texts)
        corpus = [dictionary.doc2bow(text) for text in processed_texts]
        num_topics = 12 if category in ['business', 'politics'] else 10
        lda_model = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=42, passes=20)
        lda_models[category] = lda_model
        dictionaries[category] = dictionary
        print(f"Top topics for {category}:")
        print(lda_model.print_topics())
    
    # Compute coherence scores
    for category in categories:
        texts = df[df['category'] == category]['clean_text'].tolist()
        processed_texts = [simple_preprocess(doc, deacc=True) for doc in texts]
        dictionary = dictionaries[category]
        corpus = [dictionary.doc2bow(text) for text in processed_texts]
        lda_model = lda_models[category]
        coherence_model = CoherenceModel(model=lda_model, texts=processed_texts, dictionary=dictionary, coherence='c_v')
        coherence = coherence_model.get_coherence()
        print(f"Coherence Score for {category}: {coherence:.4f}")
    
    # Subcategory mapping
    subcategory_mapping = {
        'entertainment': {0: 'music', 1: 'cinema', 2: 'theatre', 3: 'television', 4: 'celebrity_news', 5: 'film_awards', 6: 'book_reviews', 7: 'TV_series', 8: 'concerts', 9: 'festival'},
        'business': {0: 'stock_market', 1: 'company_news', 2: 'mergers_acquisitions', 3: 'economic_policy', 4: 'oil_energy', 5: 'employment', 6: 'housing_market', 7: 'global_trade', 8: 'technology_business', 9: 'financial_markets'},
        'sport': {0: 'football', 1: 'cricket', 2: 'rugby', 3: 'tennis', 4: 'olympics', 5: 'athletics', 6: 'motorsport', 7: 'golf', 8: 'boxing', 9: 'cycling'},
        'politics': {0: 'elections', 1: 'government_policy', 2: 'party_politics', 3: 'international_relations', 4: 'parliament_news', 5: 'health_policy', 6: 'education_policy', 7: 'defense_policy', 8: 'immigration', 9: 'environment_policy'},
        'tech': {0: 'mobile_tech', 1: 'software', 2: 'internet', 3: 'hardware', 4: 'gaming', 5: 'cybersecurity', 6: 'AI_ML', 7: 'social_media', 8: 'telecom', 9: 'gadgets'}
    }
    
    # Assign subcategories
    def assign_subcategory_labels(df, lda_models, dictionaries, subcategory_mapping):
        subcategories = []
        for idx, row in df.iterrows():
            category = row['category']
            tokens = row['clean_text'].split()
            bow = dictionaries[category].doc2bow(tokens)
            topic_probs = lda_models[category].get_document_topics(bow)
            if topic_probs:
                top_topic = max(topic_probs, key=lambda x: x[1])[0]
                subcat = subcategory_mapping[category].get(top_topic, f"{category}_topic_{top_topic}")
            else:
                subcat = "unknown"
            subcategories.append(subcat)
        df['subcategory'] = subcategories
        return df
    
    df = assign_subcategory_labels(df, lda_models, dictionaries, subcategory_mapping)
    
    # Named Entity Extraction with Roles
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
            lower_text = text.lower()
            assigned_role = "unknown"
            for role, keywords in role_keywords.items():
                if any(keyword in lower_text for keyword in keywords):
                    assigned_role = role
                    break
            roles[person] = assigned_role
        return roles
    
    df['entities_with_roles'] = df['text'].apply(extract_named_entities_with_roles)
    
    # Extract April event summaries
    def extract_april_summary(text):
        sentences = re.split(r'(?<=[.!?]) +', text)
        april_sentences = [sent for sent in sentences if "april" in sent.lower()]
        return " ".join(april_sentences[:2]) if april_sentences else "None"
    
    df['april_summary'] = df['text'].apply(extract_april_summary)
    
    # Save final enriched dataset
    df.to_csv("bbc_lda_final_named_subcategories2.csv", index=False)
    print("Full LDA pipeline with precise subcategories, NER, and April summaries completed. Saved as 'bbc_lda_final_named_subcategories.csv'.")
    
    
    # In[20]:
    
    
    df
    
    
    # In[21]:
    
    
    counts = df[['category', 'subcategory']].value_counts().reset_index(name='count')
    
    # Sort by 'category' and then 'subcategory'
    counts_sorted = counts.sort_values(by=['category', 'subcategory'], ascending=[True, True]).reset_index(drop=True)
    
    counts_sorted
    
    
    # In[24]:
    
    
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
    
    
    # In[11]:
    
    
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
    
    
    # In[3]:
    
    
    import os
    import pandas as pd
    from sklearn.datasets import load_files
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    from tqdm import tqdm
    import spacy
    import ollama
    
    # ---------------------- SETUP ----------------------
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
    dataset = load_files(r"C:\Users\User\OneDrive\Desktop\BBC\data\bbc-fulltext\bbc", encoding="latin1", decode_error="replace")
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
    
    # ---------------------- SUBCATEGORY CLASSIFICATION ----------------------
    def extract_subcategory(model_output, subcategories):
        output = model_output.lower()
        for subcat in subcategories:
            if subcat.lower() in output:
                return subcat
        return "unknown"
    

if __name__ == "__main__":
    main()
