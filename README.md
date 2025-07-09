# BBC-News-Subcategory-Classification-
BBC News subcategory classification for the HMLR Data Science Challenge using LLMs and LDA. Includes fine-grained topic modeling, transformer-based classification, named entity recognition with role tagging (e.g. politicians, musicians), and summarization of events related to April.

## 🎯 Objective

Given the BBC News dataset, the goals of this project are to:

- 🔹 **Break down broad categories** (e.g., `Business`, `Entertainment`, `Sport`) into more meaningful **subcategories** such as:
  - `Business` → *stock market*, *company news*, *mergers and acquisitions*
  - `Entertainment` → *cinema*, *theatre*, *music*, *literature*, *personalities*
  - `Sport` → *cricket*, *football*, *Olympics*, etc.

- 🔹 **Extract named entities** from the text and identify their **roles** (e.g., `Politician`, `TV/Film Personality`, `Musician`).

- 🔹 **Summarize articles** that describe events which:
  - Took place in **April**
  - Were **scheduled** to occur in April

## ✨ Key Features

- 🧩 **Topic Modeling:**  
  Unsupervised subcategory detection using **Latent Dirichlet Allocation (LDA)**.

- 🤖 **Text Classification:**  
  Fine-tuned transformer-based **Large Language Models (LLMs)** (e.g., `DistilBERT`, `BERT`) for multi-class subcategory classification.

- 🏷️ **NER & Role Identification:**  
  Named Entity Recognition with role classification using **SpaCy** and custom rule-based mappings.

- 📰 **Summarization:**  
  Extractive and abstractive summaries of **April-related articles** using **Hugging Face Transformers**.

- 📊 **Evaluation Metrics:**  
  Performance measured using **Coherence score**.

## 📁 Dataset

- **Source:** [BBC Dataset – UCD](http://mlg.ucd.ie/datasets/bbc.html)

  This project uses the **raw text files** from the dataset. All preprocessing steps were implemented from scratch to ensure full control and customization.

# BBC News NLP Pipelines

This repository contains three Python scripts converted from Jupyter notebooks. They perform various NLP tasks on the BBC news dataset, focusing on topic modeling (LDA), classification and summarization using Gemma 2B + Olama, and exploratory analysis.

---

## 🚀 Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 🛠️ Scripts

### 1️⃣ `lda_pipeline.py`
- Performs topic modeling using LDA.

### 2️⃣ `gemma_olama_pipeline.py`
- Uses Gemma 2B + Olama for classification and summarization.

### 3️⃣ `bert-pipeline.py`
- 

---

## 📂 Structure

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

---

## ✅ Usage

Run a pipeline script (example):

```bash
python lda_pipeline.py
```

Ensure the data folder is unzipped locally after cloning. 

---



## Preprocessing and Analysis
- Initial pre-processing includes:

- 📂 Loading and organizing news articles by category  
- 🧹 Removing duplicates and cleaning text (lowercasing, removing punctuation, custom stopwords, etc.)  
- 🧠 Lemmatizing words using NLTK  
- 🌥️ Visualizing top words per category using WordClouds  
- 📈 Extracting top 50 frequent terms per category with `CountVectorizer`  

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

## 🔠 Top 50 Terms per Category

| Category       | Top 50 Terms |
|----------------|--------------|
| **Business**   | analyst, bank, bn, business, chief, china, company, cost, country, cut, deal, december, dollar, economic, economy, euro, executive, expected, figure, financial, firm, government, group, growth, however, investment, job, last, market, may, month, oil, price, profit, rate, report, rise, sale, say, share, since, state, stock, tax, two, uk, world, yukos, âbn, âm |
| **Entertainment** | actor, actress, album, award, band, bbc, best, british, chart, comedy, day, director, festival, film, first, hit, including, last, life, made, million, movie, music, nomination, number, oscar, place, play, prize, record, rock, role, sale, series, show, singer, single, song, star, three, top, tv, two, uk, week, well, win, winner, world, âm |
| **Politics**   | bbc, blair, britain, british, brown, campaign, chancellor, claim, conservative, council, country, election, general, get, government, home, howard, issue, labour, last, law, leader, lib, lord, made, minister, mp, next, party, plan, police, prime, public, report, right, say, secretary, service, spokesman, tax, told, tony, tory, two, uk, vote, want, way, week, work |
| **Sport**      | added, back, best, champion, chance, chelsea, club, coach, cup, england, final, first, france, game, get, go, goal, going, good, got, great, im, injury, ireland, last, made, match, minute, open, play, player, rugby, season, second, set, side, six, take, team, think, three, two, victory, wale, want, week, well, win, world, yearold |
| **Tech**       | broadband, company, computer, consumer, data, device, digital, firm, first, gadget, game, get, home, information, internet, many, market, medium, microsoft, million, mobile, month, music, net, network, number, online, pc, phone, player, program, say, search, security, service, site, software, system, take, technology, tv, uk, used, user, using, video, way, website, work, world |

### 1. Unsupervised Topic Modeling (LDA)

To derive **fine-grained subcategories** within each top-level news category (e.g., Business, Politics), we use **Latent Dirichlet Allocation (LDA)** — a powerful unsupervised topic modeling technique.

Each category has its **own LDA model** trained on its respective articles. These models uncover latent topics that represent thematic subclusters within that category (e.g., within *Business*: stock market, mergers, economic policy).

This enables **automatic, interpretable subcategorization** of thousands of news articles without manual labeling.

---

### 📊 Coherence Scores by Category

| Category      | Coherence Score |
|---------------|-----------------|
| Business      | 0.4282          |
| Entertainment | 0.3636          |
| Sport         | 0.4817          |
| Politics      | 0.3402          |
| Tech          | 0.3685          |

---

### 📈 Visualization Examples

#### Subcategory Distribution Barplot

<p align="center">
  <img src="https://github.com/somaksanyal97/BBC-News-Subcategory-Classification-/blob/main/Plots%20and%20Visualisation/LDA/lda_barplot.png" width="350" />
  <img src="https://github.com/somaksanyal97/BBC-News-Subcategory-Classification-/blob/main/Plots%20and%20Visualisation/LDA/lda_barplot1.png" width="350" />
</p>

<p align="center">
  <img src="https://github.com/somaksanyal97/BBC-News-Subcategory-Classification-/blob/main/Plots%20and%20Visualisation/LDA/lda_barplot2.png" width="350" />
  <img src="https://github.com/somaksanyal97/BBC-News-Subcategory-Classification-/blob/main/Plots%20and%20Visualisation/LDA/lda_barplot4.png" width="350" />
</p>

<p align="center">
  <img src="https://github.com/somaksanyal97/BBC-News-Subcategory-Classification-/blob/main/Plots%20and%20Visualisation/LDA/lda_barplot4.png" width="350" />
</p>

### 2. Subcategory Classification and Summarisation using `Gemma:2B`

Here the Gemma:2B model is used for the sub-categorisation. The model was run locally using Olama. 

Initially, GPT 4 and Mistral:7B were tried, but due to hardware restrictions, ultimately the Gemma:2B model is implemented.

### 📊 Semantic Coherence Scoring

To assess **how well LLM-predicted subcategories semantically align** with article content:

- Sentence embeddings (`MiniLM`) are generated for:
  - The article (`text`)
  - The predicted subcategory label (`subcategory`)
- Cosine similarity is computed → **Coherence Score**
- Higher scores mean better alignment.

| Category      | Coherence Score |
|---------------|-----------------|
| Business      | 0.2287          |
| Entertainment | 0.1960          |
| Sport         | 0.2696          |
| Politics      | 0.2587          |
| Tech          | 0.2262          |

### 📈 Visualization Examples

#### Subcategory Distribution Barplot

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





## 📚 Citation

If you make use of the BBC dataset, please consider citing the following publication:

> D. Greene and P. Cunningham.  
> *Practical Solutions to the Problem of Diagonal Dominance in Kernel Document Clustering*,  
> Proceedings of the 23rd International Conference on Machine Learning (ICML), 2006.  
> [PDF](http://mlg.ucd.ie/files/publications/greene06icml.pdf) | [BibTeX](http://mlg.ucd.ie/files/publications/greene06icml.bib)

