# from transformers import pipeline

# # Initialize the zero-shot-classification pipeline
# classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# def classify_query(query):
#     candidate_labels = ["general", "rulebook"]
#     result = classifier(query, candidate_labels)
#     # Get the label with the highest score
#     classification = result["labels"][0]
#     print(classification)
#     return "rulebook"
#     #return classification

from transformers import pipeline
from functools import lru_cache
from config import config
from logger import setup_logger

logger = setup_logger(__name__)

# Initialize the zero-shot-classification pipeline with caching
@lru_cache(maxsize=1)
def get_classifier():
    """Get or create the classifier model (cached)."""
    logger.info(f"Loading classifier model: {config.classifier_model}")
    return pipeline("zero-shot-classification", model=config.classifier_model)

def classify_query(query):
    """
    Classify whether a query is general D&D chat or a rulebook question.

    Args:
        query: User query to classify

    Returns:
        "general" for casual conversation
        "rulebook" for rules-related questions (default fallback)
    """
    try:
        if not query or not query.strip():
            logger.warning("Empty query received, defaulting to rulebook")
            return "rulebook"

        classifier = get_classifier()
        candidate_labels = config.classifier_labels

        if not candidate_labels or len(candidate_labels) < 2:
            logger.error("Invalid classifier labels configuration")
            return "rulebook"

        result = classifier(query, candidate_labels)

        # Get the label with the highest score
        classification = result["labels"][0]
        confidence = result["scores"][0]

        logger.debug(f"Classification: {classification} (confidence: {confidence:.2f})")

        # Map back to our agent names using first keyword from labels
        if "casual" in classification.lower() and confidence > config.confidence_threshold:
            return "general"
        else:
            return "rulebook"

    except Exception as e:
        logger.error(f"Classification error: {e}. Defaulting to rulebook agent.")
        # Default to rulebook agent on error (safer for D&D queries)
        return "rulebook"