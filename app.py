
import os
import sys
import logging
import json
import numpy as np

import streamlit as st
from dotenv import load_dotenv

from src.pdf_processor import extract_sections
from src.custom_algorithm import enhance_scientific_notations
from src.embeddings import build_vector_store, EmbeddingModel
from src.rag_model import get_answer
from src.evaluation import (
    precision_at_k, recall_at_k, mean_reciprocal_rank, ndcg_at_k,
    compute_bleu, compute_rouge, compute_bert_score
)

# Configure logging
logging.basicConfig(
    filename='logs/app.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s'
)
logger = logging.getLogger(__name__)

import openai

# Load environment variables
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def main():
    st.title("Nebula Nexus: A Scientific Paper Query Tool")

    # Mode Selection
    mode = st.sidebar.selectbox("Select Mode", options=["Interactive Mode", "Evaluation Mode"])

    if mode == "Interactive Mode":
        run_interactive_mode()
    elif mode == "Evaluation Mode":
        run_evaluation_mode()

def run_interactive_mode():
    # Embedding Model Selection
    embedding_option = st.selectbox(
        "Select Embedding Model",
        options=["OpenAI Embeddings", "Open-Source Embedding Model"]
    )

    if embedding_option == "OpenAI Embeddings":
        selected_model = 'openai'
        # Check if API key is set
        if not openai.api_key:
            st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable in your .env file.")
            st.stop()
    else:
        # Provide options for open-source models
        open_source_models = ["sentence-transformers/gtr-t5-large", "sentence-transformers/all-MiniLM-L6-v2"]
        selected_model = st.selectbox("Select Open-Source Embedding Model", options=open_source_models)

    # Proceed with the rest of the application
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file is not None:
        try:
            with st.spinner("Processing PDF..."):
                # Extract abstract and methods sections
                sections = extract_sections(uploaded_file)
                if not sections['abstract'] and not sections['methods']:
                    st.error("Could not find 'Abstract' or 'Methods' sections in the PDF.")
                    st.stop()

                # Enhance scientific notations
                enhanced_sections = enhance_scientific_notations(sections)

                # Initialize the embedding model
                embedding_model = EmbeddingModel(model_name=selected_model)

                # Create embeddings and build vector store
                vector_store = build_vector_store(enhanced_sections, embedding_model)
                st.success("PDF processed successfully!")
        except Exception as e:
            logger.exception("Error processing PDF: %s", e)
            st.error(f"An error occurred while processing the PDF: {e}")
            st.stop()

        # Input for user query
        query = st.text_input("Enter your query:")
        if query:
            try:
                with st.spinner("Generating answer..."):
                    answer = get_answer(query, vector_store)
                    st.write("**Answer:**")
                    st.write(answer)
            except Exception as e:
                logger.exception("Error generating answer: %s", e)
                st.error(f"An error occurred while generating the answer: {e}")
                st.stop()

def run_evaluation_mode():
    st.header("Evaluation Mode")

    # Load test dataset
    test_data = load_test_data()  # Function to load test queries and ground truth

    # Embedding Model Selection
    embedding_option = st.selectbox(
        "Select Embedding Model",
        options=["OpenAI Embeddings", "Open-Source Embedding Model"]
    )

    if embedding_option == "OpenAI Embeddings":
        selected_model = 'openai'
        if not openai.api_key:
            st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable in your .env file.")
            st.stop()
    else:
        # Provide options for open-source models
        open_source_models = ["sentence-transformers/gtr-t5-large", "sentence-transformers/all-MiniLM-L6-v2"]
        selected_model = st.selectbox("Select Open-Source Embedding Model", options=open_source_models)

    if st.button("Run Evaluation"):
        try:
            with st.spinner("Running evaluation..."):
                # Initialize the embedding model
                embedding_model = EmbeddingModel(model_name=selected_model)

                # Perform retrieval and response evaluation
                retrieval_metrics, response_metrics = evaluate_system(test_data, embedding_model)

                # Display metrics
                st.subheader("Retrieval Evaluation Metrics")
                st.write(retrieval_metrics)

                st.subheader("Response Evaluation Metrics")
                st.write(response_metrics)

        except Exception as e:
            logger.exception("Error during evaluation: %s", e)
            st.error(f"An error occurred during evaluation: {e}")



def load_test_data():
    with open('test_data/test_data.json', 'r') as f:
        test_data = json.load(f)
    return test_data

def evaluate_system(test_data, embedding_model):
    """
    Evaluates the system on the test data.
    """
    precision_list = []
    recall_list = []
    ndcg_list = []
    mrr_list = []
    bleu_scores = []
    rouge_scores = []
    bert_scores = []

    retrieved_indices_list = []
    relevant_indices_list = []

    for data in test_data:
        query = data['query']
        relevant_indices = data['relevant_indices']
        reference_answer = data['reference_answer']

        # Build vector store (In practice, use preprocessed data)
        # For demonstration, we'll assume vector_store is already built
        sections = {
            'abstract': 'Abstract text goes here.',
            'methods': 'Methods text goes here.'
        }
        vector_store = build_vector_store(sections, embedding_model)

        # Retrieve documents
        query_embedding = np.array(embedding_model.get_embedding(query))
        similarities = []
        for embedding in vector_store['embeddings']:
            embedding = np.array(embedding)
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            similarities.append(similarity)

        # Get ranked list of document indices
        retrieved_indices = np.argsort(similarities)[::-1].tolist()

        # Append for MRR calculation
        retrieved_indices_list.append(retrieved_indices)
        relevant_indices_list.append(relevant_indices)

        # Compute retrieval metrics
        precision = precision_at_k(relevant_indices, retrieved_indices, k=1)
        recall = recall_at_k(relevant_indices, retrieved_indices, k=1)
        ndcg = ndcg_at_k(relevant_indices, retrieved_indices, k=1)

        precision_list.append(precision)
        recall_list.append(recall)
        ndcg_list.append(ndcg)

        # Generate answer
        answer = get_answer(query, vector_store)

        # Compute response metrics
        bleu = compute_bleu(reference_answer, answer)
        rouge = compute_rouge(reference_answer, answer)
        F1_scores, avg_f1 = compute_bert_score([reference_answer], [answer])

        bleu_scores.append(bleu)
        rouge_scores.append(rouge)
        bert_scores.append(avg_f1)

    # Compute Mean Reciprocal Rank
    mrr = mean_reciprocal_rank(relevant_indices_list, retrieved_indices_list)

    # Aggregate metrics
    retrieval_metrics = {
        'Average Precision@1': np.mean(precision_list),
        'Average Recall@1': np.mean(recall_list),
        'Average nDCG@1': np.mean(ndcg_list),
        'MRR': mrr
    }

    # Aggregate ROUGE scores
    avg_rouge1 = np.mean([score['rouge1'].fmeasure for score in rouge_scores])
    avg_rougeL = np.mean([score['rougeL'].fmeasure for score in rouge_scores])

    response_metrics = {
        'Average BLEU Score': np.mean(bleu_scores),
        'Average ROUGE-1 F1 Score': avg_rouge1,
        'Average ROUGE-L F1 Score': avg_rougeL,
        'Average BERTScore F1': np.mean(bert_scores)
    }

    return retrieval_metrics, response_metrics

if __name__ == "__main__":
    main()
