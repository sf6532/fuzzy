import re
from collections import Counter
from fuzzywuzzy import fuzz

def preprocess_text(text):
    # Remove punctuation and convert to lowercase
    return re.findall(r'\w+', text.lower())

def extract_potential_companies(text, min_word_length=5):
    words = preprocess_text(text)
    # Consider words or phrases that might be company names
    potential_companies = [word for word in words if len(word) >= min_word_length]
    # Also consider consecutive capitalized words (for multi-word company names)
    potential_companies += [' '.join(words[i:i+2]) for i in range(len(words)-1) 
                            if words[i][0].isupper() and words[i+1][0].isupper()]
    return Counter(potential_companies)

def optimized_fuzzy_match(document, company_db, threshold=70):
    potential_companies = extract_potential_companies(document)
    matches = []

    for company in company_db:
        company_words = set(preprocess_text(company))
        for potential, count in potential_companies.items():
            # Use token set ratio for better partial matching
            similarity = fuzz.token_set_ratio(potential, company)
            if similarity >= threshold:
                matches.append((company, similarity, count))

    # Sort by similarity and count
    return sorted(matches, key=lambda x: (x[1], x[2]), reverse=True)

# Example usage
company_db = [
    "Apple Inc.",
    "Microsoft Corporation",
    "Amazon.com, Inc.",
    "Alphabet Inc.",
    "Meta Platforms, Inc.",
    "Tesla, Inc.",
    "NVIDIA Corporation",
    "JPMorgan Chase & Co.",
    "Johnson & Johnson",
    "Visa Inc."
]

document = """
In today's tech landscape, companies like Microsoft and Amazon are leading the way.
The search giant Alphbet continues to innovate, while social media titan Meta Pltforms 
faces new challenges. Electric vehicle manufacturer Telsa is revolutionizing transportation.
"""

matches = optimized_fuzzy_match(document, company_db, threshold=70)

print("Fuzzy matches found in the document:")
for company, similarity, count in matches:
    print(f"{company}: Similarity {similarity}%, Occurrences: {count}")