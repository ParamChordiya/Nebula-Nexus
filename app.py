import os
import sys
import logging
# import nltk
# nltk.download()

import streamlit as st
from dotenv import load_dotenv

from src.pdf_processor import extract_sections
from src.custom_algorithm import enhance_scientific_notations
from src.embeddings import build_vector_store
from src.rag_model import get_answer

# Configure logging
logging.basicConfig(
    filename='logs/app.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s'
)
logger = logging.getLogger(__name__)
import openai
# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load environment variables
load_dotenv()

def main():
    st.title("Scientific Paper Query Application")

    # Get the OpenAI API key from environment variables
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Check if API key is set
    if not openai_api_key:
        st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable in your .env file.")
        st.stop()

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

                # Create embeddings and build vector store
                vector_store = build_vector_store(enhanced_sections)
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

if __name__ == "__main__":
    main()
