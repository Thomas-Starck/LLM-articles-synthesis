# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 10:40:21 2025

@author: thomasstarck
"""


# %% Import librairies from environment

# pip install pymupdf # to better read pdf when calling SimpleDirectoryReader()
# pip install llama_index
# pip install python-dotenv # to handle API key in .env file 


import os


from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding

from functions import load_pdf_documents, build_or_load_vector_index, generate_llm_answer, save_answers, generate_and_return_response, save_excerpts_to_csv # functions defined and used for the code




#from dotenv import load_dotenv # to load API key from .env file
# Load the .env file containing API key
# load_dotenv()
# api_key = os.getenv("OPENAI_API_KEY") # Retrieve the API key securely from environment variables
# if not api_key:
#     raise ValueError("API key is missing. Set the OPENAI_API_KEY environment variable.")
    

from dotenv import load_dotenv
# in a .env file (do not commit it !), enter your API key into OPENAI_API_KEY=your_API_key (no "")
load_dotenv(override=True)  # Loads variables from .env into environment
print(os.getenv("OPENAI_API_KEY"))




# %% Set Variables

# Review name (can be found in the source folder)
review_folder = "retracted_vaccine_autism"

# Directory containing the review articles
input_dir = os.path.join("source", review_folder, "articles")

# Directory where we will store the indexed vectorized database
output_dir = os.path.join("output", review_folder)




# Text chunks size and overlap
chunk_size = 512 # length (in tokens) of text chunks: about 2000 characters or 400 words
chunk_overlap = 50 # length (in tokens) of overlap between 2 chunks:  about 200 characters or 40 words

# Features of chunks retrieval with query
similarity_cutoff = 0.85 # minimum similarity [0,1] between user query and text chunks vectors, to select text chunks
similarity_top_k = 100 # maximum number of chunks retrieved after similarity selection

# Embedding model settings
embedding_model_name = "text-embedding-3-small" # embedding model https://platform.openai.com/docs/guides/embeddings#embedding-models
embedding_size_limit = 8191 # tokens limit of the embedding model in 1 query (has to be changed by hand if mebedding chnaged, see URL above)
batch_size = embedding_size_limit // chunk_size # optimize batch size to minimize queries

# Define embedding model
embedding_model = OpenAIEmbedding(
    embedding_model_name = embedding_model_name, # select our desired embedding model
    strip_newlines=True, # transforms \n charcters into empty space
    batch_size=batch_size  # uses optimized batches based on embedding max tokens and chunk size
    )

# Create a ServiceContext to pass custom embedding model & other configs
Settings.embed_model = embedding_model

# LLM settings
llm_model = "gpt-4o-mini" # LLM to use to generate answer https://platform.openai.com/docs/models
context_length_llm = 128000 # context length (in tokens) of the LLM (has to be changed by hand if LLM chnaged, see URL above)
temperature = 0 # ensures no stochasticity in the LLM answer, to minimize hallucination and for reproducibility




# Retrieve the user's query
with open(os.path.join("source", review_folder, "query.txt"), "r", encoding="utf-8") as file:
    user_query = file.read()
    
# Retrieve the original review summary
with open(os.path.join("source", review_folder, "original_summary.txt"), "r", encoding="utf-8") as file:
    original_summary = file.read()

# The context given to the LLM to generate answer
prompt_context = (
    "You are a research assistant. Here is our query:" +
    user_query +
    " Here are some relevant excerpts:\n\n"
)

# The instructions to the LLM to generate answer
prompt_instructions = (
    "Based on the excerpts above, please provide an answer to our query based on these excerpts, and ONLY these excertps"
    "and reference of the excerpts supporting your statements."
    "If there is no or too fex excerpts above, answer 'I cannot answer based on the available evidence.' "
)





# %% Read documents
documents = load_pdf_documents(input_dir) #rRead each article and make a list of strings of articles content


# %% Build index or load existing index

# Create and save index out of the documents only if output directory does NOT already exist; otherwise only loads existing index
index = build_or_load_vector_index(output_dir, chunk_size, chunk_overlap, documents, embedding_model) 
    

# %% Matches indexed vectors to user query and saves selected excerpts

# Generate the response
response = generate_and_return_response(index, user_query, similarity_top_k, similarity_cutoff)

# Save the excerpts ith source to a CSV file
save_excerpts_to_csv(response, output_dir)


# %% Generate LLM answer based on excerpts and context

# Extract relevant excerpts
relevant_excerpts = [node.get_content() for node in response.source_nodes]

answer = generate_llm_answer(
    llm_model, temperature, context_length_llm, # LLM parameters
    prompt_context, relevant_excerpts, prompt_instructions # prompt given to the LLM
)


# %% Save the synthesized response (response.response) and LLM-generated answer to a text file.   

save_answers(response, answer, original_summary, output_dir)



