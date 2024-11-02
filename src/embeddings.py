import openai
import numpy as np
import os
import logging

from typing import List, Dict
from src.preprocessing import preprocess_text

logger = logging.getLogger(__name__)

# Set the OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

def create_embeddings(text_list: List[str]) -> List[List[float]]:
    """
    Creates embeddings for a list of texts.

    Args:
        text_list (List[str]): List of texts to embed.

    Returns:
        List[List[float]]: List of embeddings.
    """
    embeddings = []
    try:
        for text in text_list:
            # Preprocess the text
            preprocessed_text = preprocess_text(text)

            # Generate the embedding
            response = openai.Embedding.create(
                input=preprocessed_text,
                model="text-embedding-ada-002"
            )
            embedding = response['data'][0]['embedding']
            embeddings.append(embedding)
        return embeddings
    except Exception as e:
        logger.exception("Error creating embeddings: %s", e)
        raise

def build_vector_store(sections: Dict[str, str]) -> Dict[str, List]:
    """
    Builds a vector store from the given text sections.

    Args:
        sections (Dict[str, str]): Dictionary containing text sections.

    Returns:
        Dict[str, List]: A vector store with embeddings and texts.
    """
    try:
        texts = [sections['abstract'], sections['methods']]
        embeddings = create_embeddings(texts)
        vector_store = {
            'embeddings': embeddings,
            'texts': texts
        }
        return vector_store
    except Exception as e:
        logger.exception("Error building vector store: %s", e)
        raise
