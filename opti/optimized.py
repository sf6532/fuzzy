import csv
import os
import time
import sys
from collections import defaultdict
import multiprocessing as mp

def preprocess_text(text):
    return ''.join(c.lower() for c in text if c.isalnum())

def create_ngrams(text, n=3):
    return [text[i:i+n] for i in range(len(text) - n + 1)]

def create_inverted_index(companies):
    index = defaultdict(set)
    for i, company in enumerate(companies):
        preprocessed = preprocess_text(company)
        ngrams = create_ngrams(preprocessed)
        for ngram in ngrams:
            index[ngram].add(i)
    return index

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def fuzzy_match_optimized(chunk, company_db, inverted_index, threshold=0.8):
    chunk_preprocessed = preprocess_text(chunk)
    chunk_ngrams = set(create_ngrams(chunk_preprocessed))
    
    candidate_indices = set()
    for ngram in chunk_ngrams:
        candidate_indices.update(inverted_index.get(ngram, set()))
    
    matches = []
    for idx in candidate_indices:
        company = company_db[idx]
        company_preprocessed = preprocess_text(company)
        max_len = max(len(chunk_preprocessed), len(company_preprocessed))
        if max_len == 0:
            continue
        distance = levenshtein_distance(chunk_preprocessed, company_preprocessed)
        similarity = 1 - (distance / max_len)
        if similarity >= threshold:
            matches.append((company, similarity))
    
    return matches

def process_chunk(args):
    chunk, company_db, inverted_index, threshold = args
    return fuzzy_match_optimized(chunk, company_db, inverted_index, threshold)

def find_companies_in_document(document, company_db, inverted_index, threshold=0.8):
    words = document.split()
    chunks = [' '.join(words[i:i+3]) for i in range(len(words) - 2)]
    total_chunks = len(chunks)
    
    pool = mp.Pool(processes=mp.cpu_count())
    
    matches = []
    start_time = time.time()
    
    for i, chunk_matches in enumerate(pool.imap(process_chunk, ((chunk, company_db, inverted_index, threshold) for chunk in chunks))):
        matches.extend(chunk_matches)
        if i % 1000 == 0 or i == total_chunks - 1:
            progress = (i + 1) / total_chunks * 100
            elapsed_time = time.time() - start_time
            estimated_total_time = elapsed_time / (i + 1) * total_chunks
            remaining_time = estimated_total_time - elapsed_time
            progress_bar = f"[{'=' * int(progress // 2)}{' ' * (50 - int(progress // 2))}]"
            print(f"\rProgress: {progress_bar} {progress:.2f}% - Est. time remaining: {remaining_time:.2f}s", end='')
            sys.stdout.flush()
    
    pool.close()
    pool.join()
    
    print("\nProcessing complete.")
    
    unique_matches = {}
    for company, similarity in matches:
        if company not in unique_matches or similarity > unique_matches[company]:
            unique_matches[company] = similarity
    
    return sorted(unique_matches.items(), key=lambda x: x[1], reverse=True)

def load_companies(file_path):
    with open(file_path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        return [row[0] for row in reader]

def load_document(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)

    companies_path = os.path.join(parent_dir, 'companies.csv')
    company_db = load_companies(companies_path)
    print(f"Loaded {len(company_db)} companies from the database.")

    print("Creating inverted index...")
    inverted_index = create_inverted_index(company_db)
    print("Inverted index created.")

    document_path = os.path.join(parent_dir, 'document.csv')
    document = load_document(document_path)
    print(f"Loaded document with {len(document.split())} words.")

    print("\nStarting document-wide company detection:")
    doc_matches = find_companies_in_document(document, company_db, inverted_index, threshold=0.8)

    print("\nTop 20 companies found in the document:")
    for company, similarity in doc_matches[:20]:
        print(f"{company}: Similarity {similarity:.2f}")

    print(f"\nTotal matches found: {len(doc_matches)}")
    print("\nProcessing complete.")