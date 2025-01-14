# Systematic Review Article Summarizer

This project automates the summarization of scientific articles issued from the references of a systematic review, using natural language processing (NLP) techniques. 

It builds an embedding index from a set of scientific articles, performs similarity search based on a user query, and generates a summary of the most relevant excerpts using a Large Language Model (LLM).

---

## 📖 Project Overview

The goal of this project is to:
1. **Input Data:** Process a set of PDF scientific articles from the references of a published systematic review.  
2. **User Query:** Accept a user-provided query and the original conclusion of the systematic review.  
3. **Embedding Index:** Build a vectorized embedding index from the articles and store it for fast similarity searches.  
4. **Similarity Search:** Perform a similarity search comparing the vectorized user query with the embeddings of the articles.  
5. **Summarization:** Select the most relevant excerpts and generate a summary using a pre-trained Large Language Model (LLM).  
6. **Comparison:** Save the generated summary alongside the original author's conclusion for comparison purposes.  

---

## 📂 Project Structure

```plaintext
project-root/
│
├── source/REVIEW_NAME/          # Input data folder
│   ├── articles/                # PDF scientific articles
│   ├── query.txt                # User query in plain text
│   └── original_summary.txt     # Original systematic review conclusion
│
├── output/REVIEW_NAME/          # Generated results folder
│   ├── indexed_database.        # Vectorized embeddings stored here
│   ├── answer.txt               # Generated summary
│   └── selected_excerpts.csv    # Summary and original conclusion comparison
│
├── scripts/                     # Python scripts for each stage of the pipeline
│   ├── functions.py             # functions called by the main python script
│   └── main_script.py           # Main script to run the entire pipeline
│
├── environment.yml              # Conda environment file for package management
├── README.md                    # Project documentation
└── .gitignore                   # Ignore unnecessary files from committing


## 🚀 How to Run the Project

1. Create and activate the environment

 - This project uses a Conda environment for package management, defined in the environment.yml file. To set up the environment, run:

conda env create -f environment.yml
conda activate LLM_articles_synthesis_env


2. **Add an open AI API key:**

 - for now, this code uses the Open AI API for embedding (ext-embedding-3-small) and generating the response (gpt-4o-mini).
 - so you need a (paying) Open AI API key https://platform.openai.com/api-keys
 - once you have it, create an .env file in which you put OPENAI_API_KEY=YOUR_KEY. The main script will read it

3. **Run the main_script.py:**
