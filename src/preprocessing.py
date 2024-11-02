import re
import nltk
import spacy
import string
import logging

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

logger = logging.getLogger(__name__)

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Download NLTK data files (if not already downloaded)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

def preprocess_text(text: str) -> str:
    """
    Preprocesses the input text by applying NLP techniques.

    Args:
        text (str): The text to preprocess.

    Returns:
        str: The preprocessed text.
    """
    try:
        # Lowercase the text
        text = text.lower()

        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Tokenize the text
        tokens = nltk.word_tokenize(text)

        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]

        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]

        # Reconstruct the text
        preprocessed_text = ' '.join(tokens)
        return preprocessed_text
    except Exception as e:
        logger.exception("Error in text preprocessing: %s", e)
        raise
