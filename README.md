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
  Performance measured using **accuracy**, **F1-score**, **topic coherence scores**, and supported by **sample outputs**.

## 📁 Dataset

- **Source:** [BBC Dataset – UCD](http://mlg.ucd.ie/datasets/bbc.html)

- **Usage:**  
  This project uses the **raw text files** from the dataset. All preprocessing steps were implemented from scratch to ensure full control and customization.
  
- **Preprocessing Steps:**
  - 🧹 **Text Cleaning:** Lowercasing, punctuation removal, stopword removal, etc.
  - 🔠 **Tokenization:** Splitting text into tokens for analysis.
  - 📄 **LDA Corpus Formatting:** Creating document-term matrices and preparing text for topic modeling.
  - 🏷️ **Label Encoding:** Converting category/subcategory labels into numerical form for LLM training.

## 📚 Citation

If you make use of the BBC dataset, please consider citing the following publication:

> D. Greene and P. Cunningham.  
> *Practical Solutions to the Problem of Diagonal Dominance in Kernel Document Clustering*,  
> Proceedings of the 23rd International Conference on Machine Learning (ICML), 2006.  
> [PDF](http://mlg.ucd.ie/files/publications/greene06icml.pdf) | [BibTeX](http://mlg.ucd.ie/files/publications/greene06icml.bib)

