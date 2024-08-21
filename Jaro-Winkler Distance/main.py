import re
import csv
import os
from collections import Counter
from tqdm import tqdm

def jaro_distance(s1, s2):
    # If the strings are equal
    if s1 == s2:
        return 1.0

    # Find the matching characters
    len1, len2 = len(s1), len(s2)
    max_dist = (max(len1, len2) // 2) - 1
    match = 0
    hash_s1 = [0] * len(s1)
    hash_s2 = [0] * len(s2)

    for i in range(len1):
        for j in range(max(0, i - max_dist), min(len2, i + max_dist + 1)):
            if s1[i] == s2[j] and hash_s2[j] == 0:
                hash_s1[i] = 1
                hash_s2[j] = 1
                match += 1
                break

    if match == 0:
        return 0.0

    # Count transpositions
    t = 0
    point = 0
    for i in range(len1):
        if hash_s1[i]:
            while hash_s2[point] == 0:
                point += 1
            if s1[i] != s2[point]:
                point += 1
                t += 1
            else:
                point += 1
    
    t //= 2

    # Jaro Distance
    return ((match / len1) + (match / len2) + ((match - t) / match)) / 3.0

def jaro_winkler_distance(s1, s2, p=0.1):
    jaro_dist = jaro_distance(s1, s2)
    
    # If the jaro distance is above the threshold
    if jaro_dist > 0.7:
        prefix = 0
        for i in range(min(len(s1), len(s2))):
            if s1[i] == s2[i]:
                prefix += 1
            else:
                break
        prefix = min(4, prefix)
        jaro_dist += prefix * p * (1 - jaro_dist)
    
    return jaro_dist

def preprocess_text(text):
    # Remove punctuation and convert to lowercase
    return re.sub(r'[^\w\s]', '', text.lower())

def fuzzy_match_jaro_winkler(query, choices, threshold=0.8):
    query = preprocess_text(query)
    results = []
    for choice in choices:
        choice_clean = preprocess_text(choice)
        similarity = jaro_winkler_distance(query, choice_clean)
        if similarity >= threshold:
            results.append((choice, similarity))
    return sorted(results, key=lambda x: x[1], reverse=True)

def find_companies_in_document(document, company_db, threshold=0.8):
    words = preprocess_text(document).split()
    potential_companies = [' '.join(words[i:i+3]) for i in range(len(words) - 2)]
    
    matches = []
    for potential in tqdm(potential_companies, desc="Processing document", unit="chunk"):
        company_matches = fuzzy_match_jaro_winkler(potential, company_db, threshold)
        matches.extend(company_matches)
    
    # Deduplicate and sort results
    unique_matches = {}
    for company, similarity in matches:
        if company not in unique_matches or similarity > unique_matches[company]:
            unique_matches[company] = similarity
    
    return sorted(unique_matches.items(), key=lambda x: x[1], reverse=True)

def load_companies(file_path):
    with open(file_path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        return [row[0] for row in reader]  # Assuming company names are in the first column

def load_document(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Main execution
if __name__ == "__main__":
    # Get the directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)

    # Load company database
    companies_path = os.path.join(parent_dir, 'companies.csv')
    company_db = load_companies(companies_path)
    print(f"Loaded {len(company_db)} companies from the database.")

    # Load document
    document_path = os.path.join(parent_dir, 'document.csv')
    document = load_document(document_path)
    print(f"Loaded document with {len(document.split())} words.")

    print("\nStarting document-wide company detection:")
    doc_matches = find_companies_in_document(document, company_db, threshold=0.8)
    
    print("\nCompanies found in the document:")
    for company, similarity in doc_matches:
        print(f"{company}: Similarity {similarity:.2f}")

    print("\nProcessing complete.")