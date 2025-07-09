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
- Saves processed CSVs and visualizations to `results/`.

### 2️⃣ `gemma_olama_pipeline.py`
- Uses Gemma 2B + Olama for classification and summarization.
- Prints and/or saves summaries in the `results/` folder.

### 3️⃣ `untitled2_1.py`
- Contains additional exploratory data analysis and NLP experiments.

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

Run a specific pipeline script:

```bash
python lda_pipeline.py
```

Ensure required data is placed in the `data/` folder.

---

## 🤝 Contributions

Feel free to fork, test, and raise issues or pull requests to improve or adapt this repository for other datasets.




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
  Performance measured using **accuracy**, **F1-score**, **topic coherence scores**, and supported by **sample outputs**.

## 📁 Dataset

- **Source:** [BBC Dataset – UCD](http://mlg.ucd.ie/datasets/bbc.html)

- **Usage:**  
  This project uses the **raw text files** from the dataset. All preprocessing steps were implemented from scratch to ensure full control and customization.
  
- **Preprocessing Steps:**
  - 🧹 **Text Cleaning:** Removing duplicates, lowercasing, punctuation removal, stopword removal, etc.
  - 🔠 **Tokenization:** Splitting text into tokens for analysis.
  - 📄 **LDA Corpus Formatting:** Creating document-term matrices and preparing text for topic modeling.
  - 🏷️ **Label Encoding:** Converting category/subcategory labels into numerical form for LLM training.

## 📚 Citation

If you make use of the BBC dataset, please consider citing the following publication:

> D. Greene and P. Cunningham.  
> *Practical Solutions to the Problem of Diagonal Dominance in Kernel Document Clustering*,  
> Proceedings of the 23rd International Conference on Machine Learning (ICML), 2006.  
> [PDF](http://mlg.ucd.ie/files/publications/greene06icml.pdf) | [BibTeX](http://mlg.ucd.ie/files/publications/greene06icml.bib)

