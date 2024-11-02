import io
import logging

from PyPDF2 import PdfReader

logger = logging.getLogger(__name__)

def extract_sections(uploaded_file):
    """
    Extracts the abstract and methods sections from a PDF file.

    Args:
        uploaded_file (BytesIO): The uploaded PDF file.

    Returns:
        dict: A dictionary containing the 'abstract' and 'methods' text.
    """
    try:
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""

        # Initialize variables
        abstract = ""
        methods = ""
        in_abstract = False
        in_methods = False
        lines = text.split('\n')

        for line in lines:
            line_lower = line.lower().strip()
            if 'abstract' == line_lower:
                in_abstract = True
                in_methods = False
                continue
            if line_lower in ('methods', 'methodology'):
                in_methods = True
                in_abstract = False
                continue
            if line_lower in ('introduction', 'background'):
                in_abstract = False
                continue
            if in_abstract:
                abstract += line + ' '
            if in_methods:
                methods += line + ' '

        sections = {'abstract': abstract.strip(), 'methods': methods.strip()}
        return sections
    except Exception as e:
        logger.exception("Error extracting sections: %s", e)
        raise
