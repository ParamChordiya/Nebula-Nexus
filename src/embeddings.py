
# embeddings.py

import os
import logging
import numpy as np
from typing import List, Dict

from src.preprocessing import preprocess_text

# Import necessary libraries
import openai
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Set the OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

class EmbeddingModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        if model_name == 'openai':
            pass  # OpenAI embeddings will be called directly
        else:
            # Load the open-source embedding model using SentenceTransformer
            self.model = SentenceTransformer(model_name)
        
    def get_embedding(self, text: str) -> List[float]:
        if self.model_name == 'openai':
            response = openai.Embedding.create(
                input=text,
                model="text-embedding-ada-002"
            )
            embedding = response['data'][0]['embedding']
            return embedding
        else:
            # Generate the embedding using SentenceTransformer
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()

def create_embeddings(text_list: List[str], embedding_model: EmbeddingModel) -> List[List[float]]:
    """
    Creates embeddings for a list of texts using the specified embedding model.

    Args:
        text_list (List[str]): List of texts to embed.
        embedding_model (EmbeddingModel): The embedding model to use.

    Returns:
        List[List[float]]: List of embeddings.
    """
    embeddings = []
    try:
        for text in text_list:
            # Preprocess the text
            preprocessed_text = preprocess_text(text)
            # Generate the embedding
            embedding = embedding_model.get_embedding(preprocessed_text)
            embeddings.append(embedding)
        return embeddings
    except Exception as e:
        logger.exception("Error creating embeddings: %s", e)
        raise

def build_vector_store(sections: Dict[str, str], embedding_model: EmbeddingModel) -> Dict[str, List]:
    """
    Builds a vector store from the given text sections using the specified embedding model.

    Args:
        sections (Dict[str, str]): Dictionary containing text sections.
        embedding_model (EmbeddingModel): The embedding model to use.

    Returns:
        Dict[str, List]: A vector store with embeddings and texts.
    """
    try:
        texts = [sections['abstract'], sections['methods']]
        embeddings = create_embeddings(texts, embedding_model)
        vector_store = {
            'embeddings': embeddings,
            'texts': texts,
            'embedding_model_name': embedding_model.model_name  # Store the model name
        }
        return vector_store
    except Exception as e:
        logger.exception("Error building vector store: %s", e)
        raise
