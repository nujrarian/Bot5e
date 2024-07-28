from transformers import pipeline

# Initialize the zero-shot-classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def classify_query(query):
    candidate_labels = ["general", "rulebook"]
    result = classifier(query, candidate_labels)
    # Get the label with the highest score
    classification = result["labels"][0]
    print(classification)
    return "rulebook"
    #return classification
