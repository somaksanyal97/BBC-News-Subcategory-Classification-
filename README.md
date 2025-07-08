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
