# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 19:19:55 2025

@author: thomasstarck
"""

import os
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.llms.openai import OpenAI
import tiktoken #to count number of tokens
import fitz  # PyMuPDF
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter 
import csv
from llama_index.core.schema import NodeRelationship
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine




def load_pdf_documents(input_dir: str) -> list[Document]:
    """
    Reads all PDF files from the given directory using PyMuPDF, extracts text,
    and returns a list of Document objects (suitable for LlamaIndex).
    
    Args:
        input_dir (str): The directory containing PDF files.
    
    Returns:
        list[Document]: A list of Document objects, where each contains text from a single PDF.
    """
    documents = []
    for file_name in os.listdir(input_dir):
        if file_name.lower().endswith(".pdf"):
            pdf_path = os.path.join(input_dir, file_name)
            with fitz.open(pdf_path) as doc:
                text_content = ""
                for page in doc:
                    text_content += page.get_text()

            # Create a LlamaIndex Document for each PDF
            documents.append(
                Document(
                    text=text_content,
                    doc_id=file_name  # optional
                )
            )
    
    return documents



def build_or_load_vector_index(output_dir, chunk_size, chunk_overlap, documents, embedding_model):
    """
    Create a vector store index from a collection of documents or load an existing index from storage.

    If the specified output directory does not exist, the function will:
    - Split the provided documents into smaller text chunks using a sentence splitter.
    - Create a vector store index based on these text chunks.
    - Save the index to the specified output directory for future use.

    If the directory already exists, the function will:
    - Load the previously saved vector store index from the directory.

    Parameters:
    - output_dir (str): Path to the directory where the index should be stored or loaded from.
    - chunk_size (int): The size of each text chunk (number of characters or tokens).
    - chunk_overlap (int): The number of overlapping characters/tokens between consecutive chunks.
    - documents (list): A list of documents to be split and indexed.
    - embedding_model (OpenAIEmbedding): embedding model as defined

    Returns:
    - index (VectorStoreIndex): The created or loaded vector store index.
    """
    output_dir = os.path.join(output_dir, "indexed_database")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
        # Parse documents into smaller text chunks (nodes)
        parser = SentenceSplitter( 
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap
            )
        nodes = parser.get_nodes_from_documents(documents)
        
        # Build the vector store index
        index = VectorStoreIndex(nodes=nodes, embed_model=embedding_model)
        
        # Persist the index in the output directory
        index.storage_context.persist(persist_dir=output_dir)

        print(f"Unexisting index created and stored in: {output_dir}")
        
    else:
        # If the directory already exists, load the existing index
        storage_context = StorageContext.from_defaults(persist_dir=output_dir)
        index = load_index_from_storage(storage_context)

        print(f"Index loaded from existing index in: {output_dir}")
    
    return index



def generate_and_return_response(index, user_query, similarity_top_k, similarity_cutoff):
    """
    Generates a response for the user query using the provided index and parameters.
    
    Parameters:
    - index (object): The index object used for retrieving relevant nodes. 
                       It should have an 'as_retriever' method that accepts a 'similarity_top_k' parameter.
    - user_query (str): The query string entered by the user.
    - similarity_top_k (int): The number of top similar results to consider for the retrieval process.
    - similarity_cutoff (float): The similarity threshold used by the SimilarityPostprocessor to filter results.

    Returns:
    - response (object): The query response containing the relevant nodes and similarity scores.
    
    Description:
    This function initializes a retriever with the specified similarity_top_k parameter, creates a 
    SimilarityPostprocessor with the given cutoff, and executes a query to generate the response. 
    The response includes the relevant nodes that match the query based on similarity.
    """
    # Initialize the retriever with a high similarity_top_k
    retriever = index.as_retriever(similarity_top_k=similarity_top_k)

    # Create a SimilarityPostprocessor with the defined cutoff
    similarity_postprocessor = SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)

    # Create a query engine using the retriever and the similarity postprocessor
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=[similarity_postprocessor]
    )

    # Execute the query and return the response
    return query_engine.query(user_query)



def save_excerpts_to_csv(response, output_dir):
    """
    Saves the query response containing the relevant nodes and similarity scores to a CSV file.
    
    Parameters:
    - response (object): The query response containing the relevant nodes and similarity scores.
    - output_dir (str): The path where the resulting data will be saved as a CSV file. The CSV file
                           will contain columns: "Node Index", "Node Score", "Node Content", and "PDF Origin".
    
    Description:
    This function processes the query response and saves the relevant nodes, scores, content, and PDF origin 
    (if available) to a CSV file at the specified path.
    """
    csv_file_path = os.path.join(output_dir, "selected_excerpts.csv")
    
    
    # Open the CSV file and write the results
    with open(csv_file_path, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header row
        writer.writerow(["Excerpt_ID", "Source", "Similarity_to_query", "Content"])
        
        # Process and save the results with the similarity of each chunk
        for idx, node_with_score in enumerate(response.source_nodes):
            node = node_with_score.node
            score = node_with_score.score
            
            # Get the node content
            content = node.get_content()
            
            # Retrieve the PDF origin from the node's relationships,
            # assuming it's stored in the SOURCE relationship
            pdf_origin = None
            if NodeRelationship.SOURCE in node.relationships:
                pdf_origin = node.relationships[NodeRelationship.SOURCE].node_id
            
            # Write the row to the CSV
            writer.writerow([idx + 1, pdf_origin, score, content])

    print(f"Saved source nodes information to {csv_file_path}")

        

def generate_llm_answer(llm_model, temperature, context_length_llm, prompt_context, relevant_excerpts, prompt_instructions):
    """
    Generate an LLM answer based on provided context and excerpts.

    Parameters:
    - llm_model (str): The name of the LLM model to use.
    - temperature (float): The temperature setting for the LLM.
    - prompt_context (str): The context to include in the prompt.
    - relevant_excerpts (list): A list of passages to include in the prompt.
    - prompt_instructions (str): Instructions to append at the end of the prompt.
    - context_length_llm (int): The maximum context length of the LLM.

    Returns:
    - answer on success
    - None if the prompt is too long or any other issue occurred.
    """

    # Set the OpenAI LLM features
    llm = OpenAI(model=llm_model, temperature=temperature)

    # Construct the prompt for the LLM: context, then selected excerpts, and finally instruction
    prompt = prompt_context
    for i, passage in enumerate(relevant_excerpts, 1):
        prompt += f"[Excerpt {i}]:\n{passage}\n\n"
    prompt += prompt_instructions

    # Encode the prompt to get the number of tokens
    encoding = tiktoken.encoding_for_model(llm_model)
    num_tokens = len(encoding.encode(prompt))

    # Check if the prompt exceeds the model's context length
    if num_tokens > context_length_llm:
        error_msg = (
            f"Prompt length ({num_tokens} tokens) exceeds the llm {llm_model}'s "
            f"context length of max ({context_length_llm} tokens)."
            f"Diminish the number of selected excerpts (e.g. with similarity_cutoff or similarity_top_k)"
        )
        print(error_msg)
        return None


    # Query and get generated answer from the LLM
    answer = llm.complete(prompt)
    
    # Print success message
    print(
        f"OK: {num_tokens} tokens in the prompt, which is smaller than "
        f"the context length of {llm_model} ({context_length_llm} tokens)."
    )

    return answer



def save_answers(response, answer, original_summary, output_dir):
    """
    Save the synthesized response and LLM-generated answer to a text file.

    Parameters:
    response (Response): The response object containing the synthesized response.
    answer (CompletionResponse): The LLM-generated answer object.
    output_dir (str): The path to the file where the content will be saved.
    original_summary (str): Orginal author's stytematic review summary'
    """
    file_path = os.path.join(output_dir, "answer.txt")
    
    with open(file_path, 'w', encoding='utf-8') as file:
        # Write the synthesized response
        file.write("Embedded summary of the selected excerpts:\n\n")
        file.write(response.response + '\n\n\n\n')  # Ensure there's a newline after the response

        # Write the LLM-generated answer
        file.write("LLM-Generated Answer based on selected excerpts and context:\n\n")
        file.write(answer.text + '\n\n\n\n')  # Ensure the answer ends with a newline
        
        # Write the original review summary
        file.write("Original author's systematic review summary':\n\n")
        file.write(original_summary + '\n')  # Ensure the answer ends with a newline
