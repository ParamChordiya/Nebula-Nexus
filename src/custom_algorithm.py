import re
import logging

logger = logging.getLogger(__name__)

def enhance_scientific_notations(sections):
    """
    Enhances the handling of scientific notations in text.

    Args:
        sections (dict): Dictionary containing text sections.

    Returns:
        dict: Dictionary with enhanced text sections.
    """
    try:
        def normalize_notations(text):
            # Replace patterns like 1×10^3 or 1x10^3 with numerical equivalents
            pattern = re.compile(r'(\d+(?:\.\d+)?)\s*[×xX*]\s*10\^(-?\d+)')
            text = pattern.sub(lambda m: str(float(m.group(1)) * (10 ** int(m.group(2)))), text)
            return text

        sections['abstract'] = normalize_notations(sections['abstract'])
        sections['methods'] = normalize_notations(sections['methods'])
        return sections
    except Exception as e:
        logger.exception("Error enhancing scientific notations: %s", e)
        raise
