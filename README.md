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

 



## 📚 Citation

If you make use of the BBC dataset, please consider citing the following publication:

> D. Greene and P. Cunningham.  
> *Practical Solutions to the Problem of Diagonal Dominance in Kernel Document Clustering*,  
> Proceedings of the 23rd International Conference on Machine Learning (ICML), 2006.  
> [PDF](http://mlg.ucd.ie/files/publications/greene06icml.pdf) | [BibTeX](http://mlg.ucd.ie/files/publications/greene06icml.bib)

