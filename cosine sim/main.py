import re
from collections import Counter
from math import sqrt

def preprocess_text(text):
    # Remove punctuation and convert to lowercase
    return re.sub(r'[^\w\s]', '', text.lower())

def get_ngrams(text, n=2):
    return [text[i:i+n] for i in range(len(text) - n + 1)]

def cosine_similarity(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])
    
    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = sqrt(sum1) * sqrt(sum2)
    
    if not denominator:
        return 0.0
    return float(numerator) / denominator

def fuzzy_match_cosine(query, choices, threshold=0.5):
    query = preprocess_text(query)
    query_ngrams = Counter(get_ngrams(query))
    
    results = []
    for choice in choices:
        choice_clean = preprocess_text(choice)
        choice_ngrams = Counter(get_ngrams(choice_clean))
        similarity = cosine_similarity(query_ngrams, choice_ngrams)
        if similarity >= threshold:
            results.append((choice, similarity))
    
    return sorted(results, key=lambda x: x[1], reverse=True)

def find_companies_in_document(document, company_db, threshold=0.5):
    words = preprocess_text(document).split()
    potential_companies = [' '.join(words[i:i+3]) for i in range(len(words) - 2)]
    
    matches = []
    for potential in potential_companies:
        company_matches = fuzzy_match_cosine(potential, company_db, threshold)
        matches.extend(company_matches)
    
    # Deduplicate and sort results
    unique_matches = {}
    for company, similarity in matches:
        if company not in unique_matches or similarity > unique_matches[company]:
            unique_matches[company] = similarity
    
    return sorted(unique_matches.items(), key=lambda x: x[1], reverse=True)

# Example usage
company_db = [
    "Apple Inc.",
    "Microsoft Corporation",
    "Amazon.com, Inc.",
    "Alphabet Inc.",
    "Meta Platforms, Inc.",
    "Tesla",
    "NVIDIA Corporation",
    "JPMorgan Chase & Co.",
    "Johnson & Johnson",
    "Visa Inc."
]

document = """
In the fast-paced world of technology, companies like Microsft and Amazon are at the forefront of innovation. 
The search engine giant Alfabit continues to dominate the market, while social media titan Meta Platfroms 
faces new challenges in the evolving digital landscape. Electric vehicle manufacturer Tesla, Inc. is revolutionizing 
the automotive industry with its cutting-edge technologies.
"""

matches = find_companies_in_document(document, company_db, threshold=0.5)

print("Fuzzy matches found in the document:")
for company, similarity in matches:
    print(f"{company}: Similarity {similarity:.2f}")