import pandas as pd
import os
import re
import time
import multiprocessing as mp
from collections import defaultdict
from name_matching.name_matcher import NameMatcher
from tqdm import tqdm

def clean_text(text):
    """Clean text by removing non-printable characters and replacing with space."""
    cleaned_text = re.sub(r'[^\x20-\x7E]', ' ', str(text))
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    return cleaned_text.strip()

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

def load_companies(file_path):
    """Load companies from a CSV file without headers."""
    with open(file_path, 'r', encoding='utf-8') as file:
        companies = [clean_text(line.strip()) for line in file if line.strip()]
    df = pd.DataFrame(companies, columns=['Company'])
    print(f"Loaded {len(df)} companies from the file.")
    return df

def load_document(file_path):
    """Load document from a CSV file."""
    df = pd.read_csv(file_path)
    print(f"Columns in document file: {df.columns.tolist()}")
    return df

def get_valid_column(df, preferred_columns):
    """Get the first valid column name from a list of preferred columns."""
    for col in preferred_columns:
        if col in df.columns:
            return col
    raise ValueError(f"None of the preferred columns {preferred_columns} found in DataFrame. Available columns are: {df.columns.tolist()}")

def combined_match(chunk, company_db, inverted_index, name_matcher, threshold=0.8):
    # Step 1: Use inverted index for initial candidate selection
    chunk_preprocessed = preprocess_text(chunk)
    chunk_ngrams = set(create_ngrams(chunk_preprocessed))
    
    candidate_indices = set()
    for ngram in chunk_ngrams:
        candidate_indices.update(inverted_index.get(ngram, set()))
    
    candidates = [company_db[i] for i in candidate_indices]
    
    # Step 2: Use NameMatcher for refined matching
    if candidates:
        df_candidates = pd.DataFrame({'Company': candidates})
        matches = name_matcher.match_names(to_be_matched=pd.DataFrame({'Text': [chunk]}), 
                                           df_matching_data=df_candidates,
                                           column_matching='Text',
                                           column='Company')
        
        # Step 3: Apply Levenshtein distance for final similarity score
        final_matches = []
        for _, match in matches.iterrows():
            company = match['Company']
            company_preprocessed = preprocess_text(company)
            max_len = max(len(chunk_preprocessed), len(company_preprocessed))
            if max_len == 0:
                continue
            distance = levenshtein_distance(chunk_preprocessed, company_preprocessed)
            similarity = 1 - (distance / max_len)
            if similarity >= threshold:
                final_matches.append((company, similarity))
        
        return final_matches
    return []

def process_chunk(args):
    chunk, company_db, inverted_index, name_matcher, threshold = args
    return combined_match(chunk, company_db, inverted_index, name_matcher, threshold)

def find_companies_in_document(document, company_db, inverted_index, name_matcher, threshold=0.8):
    words = document.split()
    chunks = [' '.join(words[i:i+3]) for i in range(len(words) - 2)]
    total_chunks = len(chunks)
    
    pool = mp.Pool(processes=mp.cpu_count())
    
    matches = []
    start_time = time.time()
    
    for i, chunk_matches in enumerate(pool.imap(process_chunk, ((chunk, company_db, inverted_index, name_matcher, threshold) for chunk in chunks))):
        matches.extend(chunk_matches)
        if i % 1000 == 0 or i == total_chunks - 1:
            progress = (i + 1) / total_chunks * 100
            elapsed_time = time.time() - start_time
            estimated_total_time = elapsed_time / (i + 1) * total_chunks
            remaining_time = estimated_total_time - elapsed_time
            print(f"\rProgress: {progress:.2f}% - Est. time remaining: {remaining_time:.2f}s", end='', flush=True)
    
    pool.close()
    pool.join()
    
    print("\nProcessing complete.")
    
    unique_matches = {}
    for company, similarity in matches:
        if company not in unique_matches or similarity > unique_matches[company]:
            unique_matches[company] = similarity
    
    return sorted(unique_matches.items(), key=lambda x: x[1], reverse=True)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)

    # Load companies
    companies_path = os.path.join(parent_dir, 'companies.csv')
    print("Loading companies...")
    df_companies = load_companies(companies_path)
    company_column = 'Company'

    # Create inverted index
    print("Creating inverted index...")
    inverted_index = create_inverted_index(df_companies[company_column].tolist())
    print("Inverted index created.")

    # Initialize NameMatcher
    name_matcher = NameMatcher(number_of_matches=5, 
                               legal_suffixes=True, 
                               common_words=False, 
                               top_n=50, 
                               verbose=True)
    name_matcher.set_distance_metrics(['bag', 'typo', 'refined_soundex'])
    name_matcher.load_and_process_master_data(column=company_column,
                                              df_matching_data=df_companies, 
                                              transform=True)

    # Load document
    document_path = os.path.join(parent_dir, 'document.csv')
    print("Loading document...")
    df_document = load_document(document_path)
    document_column = get_valid_column(df_document, ['Text', 'text', 'content', 'document'])
    print(f"Using '{document_column}' as the document text column.")

    # Match companies in the document
    print("\nStarting company matching process:")
    doc_matches = find_companies_in_document(df_document[document_column].iloc[0], 
                                             df_companies[company_column].tolist(), 
                                             inverted_index, 
                                             name_matcher, 
                                             threshold=0.8)

    # Display results
    print("\nTop 20 companies found in the document:")
    for company, similarity in doc_matches[:20]:
        print(f"{company}: Similarity {similarity:.2f}")

    print(f"\nTotal matches found: {len(doc_matches)}")
    print("\nProcessing complete.")

if __name__ == "__main__":
    main()