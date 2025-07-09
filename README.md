# 📰 BBC News Subcategory Classification

Fine-grained BBC News categorization for the **HMLR Data Science Challenge** using LLMs, LDA, NER with role tagging, and April-focused event summarization.

---
## 🎯 Objective

This project aims to:

- 🔹 **Break down broad BBC News categories** (e.g., `Business`, `Entertainment`, `Sport`) into more granular **subcategories**, such as:
  - `Business` → *stock market*, *company news*, *M&A*
  - `Entertainment` → *film*, *music*, *literature*, *TV personalities*
  - `Sport` → *cricket*, *football*, *Olympics*, etc.

- 🔹 **Extract named entities** and classify their **roles** (e.g., `Politician`, `Musician`, `TV Personality`).

- 🔹 **Summarize articles** mentioning events:
  - That occurred in **April**
  - That were **scheduled for April**

---

## ✨ Key Features

- 🧩 **Topic Modeling (LDA)**  
  Subcategory discovery per category using Latent Dirichlet Allocation.

- 🤖 **Transformer-Based Text Classification**  
  Fine-tuned LLMs (Gemma 2B + Olama, DistilBERT/BART) for multi-class subcategory classification.

- 🏷️ **NER with Role Classification**  
  Extracted entities are tagged with specific societal roles using SpaCy and rule-based heuristics.

- 📰 **Summarization**  
  April-related summaries using Hugging Face Transformers (e.g., `DistilBART CNN`) and `Gemma:2B`.

- 📊 **Evaluation**  
  - LDA: Topic **Coherence Score**
  - LLM: Semantic alignment using **MiniLM** sentence embeddings and semantic coherence score

---
## 📁 Dataset

- **Source:** [BBC Dataset – UCD](http://mlg.ucd.ie/datasets/bbc.html)
- Raw `.txt` news articles grouped by 5 top-level categories: `business`, `entertainment`, `politics`, `sport`, and `tech`.

Preprocessing is handled manually for full control.

---
## 🧪 BBC News NLP Pipelines

This repo includes **three modular pipelines**, each handling a major NLP component:

| Pipeline | Script | Description |
|---------|--------|-------------|
| 🧠 Topic Modeling | `lda_pipeline.py` | Runs category-wise LDA to generate subcategories |
| 🤖 LLM Classification + Summarization | `gemma_olama_pipeline.py` | Uses Gemma 2B via Olama for role-tagged entity extraction, subcategory classification, and April-related summarization |
| 🔬 BERT/NLI-based Classification | `bert-pipeline.py` | Classifies articles using `BART-large-MNLI`, summarizes using `DistilBART` |

---


## 🛠️ Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Ensure the `data/` folder is unzipped after cloning the repo.

---

### 📂 Structure

```
BBC-News-Subcategory-Classification-/
├── data/                   # BBC Dataset in zip file. Please unzip the data locally after cloning
├── results/                # Outputs (CSV files with the final outputs to be stored here)
├── lda_pipeline.py
├── gemma_olama_pipeline.py
├── bert-pipeline.py
├── requirements.txt
├── README.md
└── .gitignore
```

### ✅ Usage

Run a pipeline script (example):

```bash
python lda_pipeline.py
```

### Preprocessing and Analysis
- Articles are grouped by category
- Duplicates removed, text cleaned (stopwords, lemmatization via NLTK)
- Visualized using **WordClouds** per category
- Top 50 frequent terms per category using `CountVectorizer`

The goal is to uncover the key subcategories discussed within each news category to use for classification in later stages.

<!-- First row: 3 images -->
<p align="center">
  <img src="https://github.com/somaksanyal97/BBC-News-Subcategory-Classification-/blob/main/Plots%20and%20Visualisation/WordCloud%20Visualisations/business.png" width="350" />
  <img src="https://github.com/somaksanyal97/BBC-News-Subcategory-Classification-/blob/main/Plots%20and%20Visualisation/WordCloud%20Visualisations/entertainment.png" width="350" />
</p>

<p align="center">
  <img src="https://github.com/somaksanyal97/BBC-News-Subcategory-Classification-/blob/main/Plots%20and%20Visualisation/WordCloud%20Visualisations/sports.png" width="350" />
  <img src="https://github.com/somaksanyal97/BBC-News-Subcategory-Classification-/blob/main/Plots%20and%20Visualisation/WordCloud%20Visualisations/tech.png" width="350" />
</p>

<p align="center">
  <img src="https://github.com/somaksanyal97/BBC-News-Subcategory-Classification-/blob/main/Plots%20and%20Visualisation/WordCloud%20Visualisations/politics.png" width="350" />
</p>

---

### 🔠 Top 50 Terms per Category

| Category       | Top 50 Terms |
|----------------|--------------|
| **Business**   | analyst, bank, bn, business, chief, china, company, cost, country, cut, deal, december, dollar, economic, economy, euro, executive, expected, figure, financial, firm, government, group, growth, however, investment, job, last, market, may, month, oil, price, profit, rate, report, rise, sale, say, share, since, state, stock, tax, two, uk, world, yukos, âbn, âm |
| **Entertainment** | actor, actress, album, award, band, bbc, best, british, chart, comedy, day, director, festival, film, first, hit, including, last, life, made, million, movie, music, nomination, number, oscar, place, play, prize, record, rock, role, sale, series, show, singer, single, song, star, three, top, tv, two, uk, week, well, win, winner, world, âm |
| **Politics**   | bbc, blair, britain, british, brown, campaign, chancellor, claim, conservative, council, country, election, general, get, government, home, howard, issue, labour, last, law, leader, lib, lord, made, minister, mp, next, party, plan, police, prime, public, report, right, say, secretary, service, spokesman, tax, told, tony, tory, two, uk, vote, want, way, week, work |
| **Sport**      | added, back, best, champion, chance, chelsea, club, coach, cup, england, final, first, france, game, get, go, goal, going, good, got, great, im, injury, ireland, last, made, match, minute, open, play, player, rugby, season, second, set, side, six, take, team, think, three, two, victory, wale, want, week, well, win, world, yearold |
| **Tech**       | broadband, company, computer, consumer, data, device, digital, firm, first, gadget, game, get, home, information, internet, many, market, medium, microsoft, million, mobile, month, music, net, network, number, online, pc, phone, player, program, say, search, security, service, site, software, system, take, technology, tv, uk, used, user, using, video, way, website, work, world |

---

## 1. Unsupervised Topic Modeling (LDA)

- Each category has its **own LDA model**
- Reveals **latent subtopics** (e.g., in `Business`: *M&A*, *banking*, *stock market*)
- No manual labeling required

**[View the results here](https://github.com/somaksanyal97/BBC-News-Subcategory-Classification-/blob/main/results/bbc_lda_final_named_subcategories2.csv)**

---

### 📊 Coherence Scores (LDA)

| Category      | Coherence Score |
|---------------|-----------------|
| Business      | 0.4282          |
| Entertainment | 0.3636          |
| Sport         | 0.4817          |
| Politics      | 0.3402          |
| Tech          | 0.3685          |

---

#### Subcategory Distribution

<p align="center">
  <img src="https://github.com/somaksanyal97/BBC-News-Subcategory-Classification-/blob/main/Plots%20and%20Visualisation/LDA/lda_barplot.png" width="350" />
  <img src="https://github.com/somaksanyal97/BBC-News-Subcategory-Classification-/blob/main/Plots%20and%20Visualisation/LDA/lda_barplot1.png" width="350" />
</p>

<p align="center">
  <img src="https://github.com/somaksanyal97/BBC-News-Subcategory-Classification-/blob/main/Plots%20and%20Visualisation/LDA/lda_barplot2.png" width="350" />
  <img src="https://github.com/somaksanyal97/BBC-News-Subcategory-Classification-/blob/main/Plots%20and%20Visualisation/LDA/lda_barplot3.png" width="350" />
</p>

<p align="center">
  <img src="https://github.com/somaksanyal97/BBC-News-Subcategory-Classification-/blob/main/Plots%20and%20Visualisation/LDA/lda_barplot4.png" width="350" />
</p>

## 2. LLM Classification + Summarization (Gemma 2B via Olama)

- Runs locally with **Gemma 2B** using **Olama**
- Predicts subcategories, tags named entities with roles (e.g., `Politician`, `Musician`)
- Summarizes articles if:
  - Event occurred in April
  - Event is scheduled for April

**[View the results here](https://github.com/somaksanyal97/BBC-News-Subcategory-Classification-/blob/main/results/classified_bbc_articles_with_entities_final.csv)**

Initially, GPT 4 and Mistral:7B were tried, but due to hardware restrictions, ultimately the Gemma:2B model is implemented.

⚠️ **Token limit:**  
Only first 3000 tokens are used per article due to prompt size constraints. Increasing this may improve performance.

### 📊 Semantic Coherence (MiniLM)

| Category      | Coherence Score |
|---------------|-----------------|
| Business      | 0.2287          |
| Entertainment | 0.1960          |
| Sport         | 0.2696          |
| Politics      | 0.2587          |
| Tech          | 0.2262          |


### 📉 Subcategory Distribution (Gemma)

<p align="center">
  <img src="https://github.com/somaksanyal97/BBC-News-Subcategory-Classification-/blob/main/Plots%20and%20Visualisation/Gemma/gemma_b.png" width="350" />
  <img src="https://github.com/somaksanyal97/BBC-News-Subcategory-Classification-/blob/main/Plots%20and%20Visualisation/Gemma/gemma_e.png" width="350" />
</p>

<p align="center">
  <img src="https://github.com/somaksanyal97/BBC-News-Subcategory-Classification-/blob/main/Plots%20and%20Visualisation/Gemma/gemma_p.png" width="350" />
  <img src="https://github.com/somaksanyal97/BBC-News-Subcategory-Classification-/blob/main/Plots%20and%20Visualisation/Gemma/gemma_s.png" width="350" />
</p>

<p align="center">
  <img src="https://github.com/somaksanyal97/BBC-News-Subcategory-Classification-/blob/main/Plots%20and%20Visualisation/Gemma/gemma_t.png" width="350" />
</p>

## 3. Subcategory Classification & Summarisation using `DistilBERT (BART-large-MNLI)`

- Uses `BART-large-MNLI` for classification:
  - If no subcategory has a confidence ≥ 0.5 → assigns `{category}_topic_other`
- April event summarization via `DistilBART CNN`
- Input truncated to **5000 characters** for performance

**[View the results here](https://github.com/somaksanyal97/BBC-News-Subcategory-Classification-/blob/main/results/classified_bbc_articles_distilbert.csv)**

⚠️ **Token limit:**
Limiting input to **5000 characters** may reduce classification accuracy for longer articles.  
Increasing token context could yield more accurate subcategory predictions.

📊 Subcategory Coherence Evaluation (Gensim)

| Category      | Coherence Score |
|---------------|-----------------|
| Business      | 0.3538          |
| Entertainment | 0.3632          |
| Sport         | 0.4143          |
| Politics      | 0.3636          |
| Tech          | 0.3279          |

📊 Coherence scored using `gensim`’s `C_V` metric  
🔁 Skips subcategories with < 5 samples

### 📉 Subcategory Distribution (BART-large-MNLI)

<p align="center">
  <img src="https://github.com/somaksanyal97/BBC-News-Subcategory-Classification-/blob/main/Plots%20and%20Visualisation/bert/bert_b.png" width="350" />
  <img src="https://github.com/somaksanyal97/BBC-News-Subcategory-Classification-/blob/main/Plots%20and%20Visualisation/bert/bert_e.png" width="350" />
</p>

<p align="center">
  <img src="https://github.com/somaksanyal97/BBC-News-Subcategory-Classification-/blob/main/Plots%20and%20Visualisation/bert/bert_p.png" width="350" />
  <img src="https://github.com/somaksanyal97/BBC-News-Subcategory-Classification-/blob/main/Plots%20and%20Visualisation/bert/bert_s.png" width="350" />
</p>

<p align="center">
  <img src="https://github.com/somaksanyal97/BBC-News-Subcategory-Classification-/blob/main/Plots%20and%20Visualisation/bert/bert_t.png" width="350" />
</p>

## **NER with Role Classification**  
The code extracts named entities from BBC articles using SpaCy and assigns them specific societal roles like Politician, Musician, or TV Personality. Role classification is enhanced using context-based rules and LLM prompts (in the Gemma pipeline) for better accuracy.

## 📚 Citation

If you make use of the BBC dataset, please consider citing the following publication:

> D. Greene and P. Cunningham.  
> *Practical Solutions to the Problem of Diagonal Dominance in Kernel Document Clustering*,  
> Proceedings of the 23rd International Conference on Machine Learning (ICML), 2006.  
> [PDF](http://mlg.ucd.ie/files/publications/greene06icml.pdf) | [BibTeX](http://mlg.ucd.ie/files/publications/greene06icml.bib)

