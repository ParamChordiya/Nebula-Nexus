import openai
import numpy as np
import os
import logging

from typing import Dict

logger = logging.getLogger(__name__)

# Set the OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_answer(query: str, vector_store: Dict[str, list]) -> str:
    """
    Generates an answer to the user's query using the RAG model with GPT-4.

    Args:
        query (str): The user's query.
        vector_store (Dict[str, list]): The vector store containing embeddings and texts.

    Returns:
        str: The generated answer.
    """
    try:
        # Create embedding for the query
        response = openai.Embedding.create(
            input=query,
            model="text-embedding-ada-002"
        )
        query_embedding = np.array(response['data'][0]['embedding'])

        # Calculate cosine similarities
        similarities = []
        for embedding in vector_store['embeddings']:
            embedding = np.array(embedding)
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            similarities.append(similarity)

        # Get the most similar context
        most_similar_idx = int(np.argmax(similarities))
        context = vector_store['texts'][most_similar_idx]

        # Use OpenAI GPT-4 model to generate answer using ChatCompletion
        messages = [
            {"role": "system", "content": "You are an assistant that answers questions in the simplest possible way based on the provided context. These questions are based on different research papers and need deep understanding of scientific knowledge."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"}
        ]

        completion = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=500,
            temperature=0.7,
            n=1
        )
        answer = completion['choices'][0]['message']['content'].strip()
        return answer
    except Exception as e:
        logger.exception("Error generating answer: %s", e)
        raise
