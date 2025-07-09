import os

def main():
    def classify_article_with_gemma(text, category):
        subcategories = subcategory_lists.get(category, [])
        if not subcategories:
            return "unknown"
    
        prompt = (
            f"Given the following BBC news article under the category '{category}', "
            f"classify it into ONE of the following subcategories: {', '.join(subcategories)}.\n\n"
            f"Return ONLY the subcategory name. No explanation. No punctuation.\n\n"
            f"Article:\n{text[:3000]}"
        )
    
        try:
            response = ollama.chat(model='gemma:2b', messages=[
                {"role": "system", "content": "You are a helpful classification assistant."},
                {"role": "user", "content": prompt}
            ])
            raw_output = response['message']['content'].strip()
            subcategory = extract_subcategory(raw_output, subcategories)
    
        except Exception as e:
            print(f"[ERROR] {e}")
            subcategory = "error"
    
        return subcategory
    
    print("Classifying articles with Gemma 2B...")
    tqdm.pandas()
    
    df['subcategory'] = df.progress_apply(lambda row: classify_article_with_gemma(row['clean_text'], row['category']), axis=1)
    
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
            if person in roles:  # Avoid repeats
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
    
    def summarize_april_event(text):
        prompt = (
            f"Summarize this BBC news article, focusing only on what took place or is scheduled to take place in April. "
            f"Return only a brief summary without any introductory phrases or explanations.\n\n{text}"
        )
        try:
            response = ollama.chat(model='gemma:2b', messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes news articles."},
                {"role": "user", "content": prompt}
            ])
            return response['message']['content'].strip().replace('\n', ' ')
        except Exception as e:
            print(f"[SUMMARY ERROR] {e}")
            return ""
    
    def extract_april_summary(text):
        if "april" in text.lower():
            return summarize_april_event(text)
        else:
            return ""
    
    df['april_summary'] = df['text'].apply(extract_april_summary)
    
    # ---------------------- SAVE RESULTS ----------------------
    df[['text', 'category', 'subcategory', 'entities_with_roles', 'april_summary']].to_csv("classified_bbc_articles_with_entities_final.csv", index=False)
    print("✅ Done. Results saved to 'classified_bbc_articles_with_entities_final.csv'.")
    
    
    # In[10]:
    
    
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L12-v2')
    print("Model loaded ✅")
    
    
    # In[4]:
    
    
    # ---------------------- COHERENCE SCORE EVALUATION ----------------------
    from sentence_transformers import SentenceTransformer, util
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    
    print("\n🔍 Evaluating semantic coherence for LLM subcategory predictions...")
    
    # Reload from CSV to be safe
    df = pd.read_csv(r"C:\Users\User\OneDrive\Desktop\BBC\src\classified_bbc_articles_with_entities_final.csv")
    
    # Load sentence embedding model
    embedding_model = SentenceTransformer('all-MiniLM-L12-v2')  # Or 'all-MiniLM-L6-v2' for faster
    
    def compute_coherence(text, subcategory):
        try:
            if not isinstance(subcategory, str) or subcategory.lower() == "unknown":
                return np.nan
            # Convert label to readable format
            label_text = subcategory.replace('_', ' ')
            # Encode text and label
            text_emb = embedding_model.encode(text[:512], convert_to_tensor=True)
            label_emb = embedding_model.encode(label_text, convert_to_tensor=True)
            # Cosine similarity as coherence score
            score = util.cos_sim(text_emb, label_emb).item()
            return round(score, 4)
        except Exception as e:
            print(f"[COHERENCE ERROR] {e}")
            return np.nan
    
    # Use tqdm progress_apply for progress bar
    tqdm.pandas()
    
    df['coherence_score'] = df.progress_apply(lambda row: compute_coherence(row['text'], row['subcategory']), axis=1)
    
    # Print average coherence per top-level category
    print("\n📈 Average coherence scores by category:")
    for category in df['category'].unique():
        avg_score = df[df['category'] == category]['coherence_score'].mean()
        print(f" - {category}: {avg_score:.4f}")
    
    # Save updated CSV with coherence scores
    df.to_csv("classified_bbc_articles_with_coherence.csv", index=False)
    print("✅ Coherence scores saved to 'classified_bbc_articles_with_coherence.csv'.")
    
    
    # In[5]:
    
    
    df
    
    
    # In[15]:
    
    
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Load the saved results if not already in memory
    df = pd.read_csv("classified_bbc_articles_with_entities_final.csv")
    
    # Count articles per subcategory within each category
    counts = df.groupby(['category', 'subcategory']).size().reset_index(name='count')
    
    # Sort subcategories by count within each category
    counts_sorted = counts.sort_values(by=['category', 'count'], ascending=[True, False])
    
    # Seaborn theme
    sns.set_theme(style="whitegrid", font_scale=1.1, palette="muted")
    
    # Plot each category
    categories = counts_sorted['category'].unique()
    
    for cat in categories:
        plt.figure(figsize=(10, 6))
        data_cat = counts_sorted[counts_sorted['category'] == cat].sort_values(by='count', ascending=True)
    
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
    
        # Add value labels
        for container in ax.containers:
            ax.bar_label(container, fmt='%.0f', label_type='edge', fontsize=10, padding=3)
    
        sns.despine(left=True, bottom=True)
        plt.tight_layout()
        plt.show()
    
    
    # In[16]:
    
    
    counts_gemma2 = df[['category', 'subcategory']].value_counts().reset_index(name='count')
    
    # Sort by 'category' and then 'subcategory'
    counts_gemma2_sorted = counts_gemma2.sort_values(by=['category', 'subcategory'], ascending=[True, True]).reset_index(drop=True)
    
    counts_gemma2_sorted
    
    
    # In[17]:
    
    
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
        counts_gemma2_sorted,
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
    
    
    # In[21]:
    
    
    """import pandas as pd
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # Load your dataset
    df_lda = pd.read_csv("bbc_lda_final_named_subcategories2.csv")
    df_llm = pd.read_csv("classified_bbc_articles_with_entities_final.csv")
    
    # Ensure relevant columns exist
    print(df.columns)
    
    # Drop potential 'error' rows from LLM classification
    df_llm = df_llm[df_llm['subcategory'] != 'error']
    
    # Compare only where both methods have labels
    lda_labels = df_lda['subcategory']
    llm_labels = df_llm['subcategory']
    
    # Calculate accuracy (label match %)
    accuracy = accuracy_score(lda_labels, llm_labels)
    print(f"Agreement Accuracy between LDA and LLM: {accuracy:.4f}")
    
    # Generate classification report
    print("Classification Report (LDA vs LLM):")
    print(classification_report(lda_labels, llm_labels, zero_division=0))
    
    # Generate confusion matrix
    cm = confusion_matrix(lda_labels, llm_labels, labels=sorted(df['subcategory'].unique()))
    
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=sorted(df['subcategory'].unique()), yticklabels=sorted(df['subcategory'].unique()), cmap='Blues')
    plt.xlabel("LLM Predicted Labels")
    plt.ylabel("LDA Labels")
    plt.title("Confusion Matrix: LDA vs LLM Subcategory Assignment")
    plt.tight_layout()
    plt.show()
    """
    

if __name__ == "__main__":
    main()
