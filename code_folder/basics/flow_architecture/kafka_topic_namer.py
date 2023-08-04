import spacy

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

# Business domain-specific keywords
business_keywords = ["finance", "sales", "marketing", "hr", "inventory"]

# Environment-specific keywords
environment_keywords = ["dev", "qa", "prod", "sandbox"]


# Function to suggest topic names based on NLP processing
def suggest_topic_names(sentence):
    doc = nlp(sentence)

    business_domain = None
    environment = None

    # Extract business domain and environment from sentence
    for token in doc:
        if token.text.lower() in business_keywords:
            business_domain = token.text.lower()
        if token.text.lower() in environment_keywords:
            environment = token.text.lower()

    # Generate topic name suggestions based on conventions
    topic_suggestions = []
    if business_domain and environment:
        topic_suggestions.append(f"{business_domain}.{environment}")
        topic_suggestions.append(f"{environment}.{business_domain}")
    elif business_domain:
        topic_suggestions.append(business_domain)
    elif environment:
        topic_suggestions.append(environment)

    return topic_suggestions


# Example usage
sentence = "This Kafka topic is related to finance in the production environment"
suggestions = suggest_topic_names(sentence)
print("Topic name suggestions:")
for suggestion in suggestions:
    print(suggestion)
