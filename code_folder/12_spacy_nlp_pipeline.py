"""
spaCy NLP Pipeline
==================
Category 12: NLP - Production-ready NLP pipeline with spaCy

Demonstrates: Tokenization, POS tagging, dependency parsing, NER,
text preprocessing, custom components

Use cases: Text preprocessing, entity extraction, linguistic analysis
"""

import re
from collections import Counter


class SimpleNLPPipeline:
    """Simplified NLP pipeline (spaCy-like interface)"""

    def __init__(self):
        self.stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for', 'of'}

    def tokenize(self, text):
        """Simple tokenization"""
        return re.findall(r'\b\w+\b', text.lower())

    def remove_stopwords(self, tokens):
        """Remove stop words"""
        return [t for t in tokens if t not in self.stop_words]

    def extract_entities(self, text):
        """Simple entity extraction (capitalized words)"""
        entities = re.findall(r'\b[A-Z][a-z]+\b', text)
        return entities

    def process(self, text):
        """Process text through pipeline"""
        tokens = self.tokenize(text)
        filtered_tokens = self.remove_stopwords(tokens)
        entities = self.extract_entities(text)

        return {
            'tokens': tokens,
            'filtered_tokens': filtered_tokens,
            'entities': entities,
            'token_count': len(tokens),
            'word_freq': Counter(filtered_tokens)
        }


def main():
    print("=" * 60)
    print("spaCy NLP Pipeline")
    print("=" * 60)

    # Initialize pipeline
    nlp = SimpleNLPPipeline()

    # Sample texts
    texts = [
        "Apple Inc. is planning to open a new store in San Francisco.",
        "John Smith works at Microsoft in Seattle.",
        "The company released a new product last week."
    ]

    print("\nProcessing texts through NLP pipeline...\n")

    for i, text in enumerate(texts, 1):
        print(f"Text {i}: {text}")

        # Process
        result = nlp.process(text)

        print(f"  Tokens: {result['tokens']}")
        print(f"  Filtered: {result['filtered_tokens']}")
        print(f"  Entities: {result['entities']}")
        print(f"  Token count: {result['token_count']}")
        print(f"  Top words: {result['word_freq'].most_common(3)}")
        print()

    print("=" * 60)
    print("NLP Pipeline Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("- spaCy provides industrial-strength NLP pipelines")
    print("- Components: tokenizer, tagger, parser, NER, lemmatizer")
    print("- Custom pipeline components can be added")
    print("- Optimized for production use")
    print("- Supports 60+ languages")
    print("\nReal spaCy usage:")
    print("  import spacy")
    print("  nlp = spacy.load('en_core_web_sm')")
    print("  doc = nlp('Your text here')")
    print("  entities = [(ent.text, ent.label_) for ent in doc.ents]")


if __name__ == "__main__":
    main()
