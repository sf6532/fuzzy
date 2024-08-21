import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import re
from fuzzywuzzy import fuzz
from tqdm import tqdm
import time

def load_companies(file_path):
    """Load companies from a CSV file."""
    df = pd.read_csv(file_path, header=None, names=['Company'])
    print(f"Loaded {len(df)} companies from the file.")
    return df

def load_document(file_path):
    """Load document from a CSV file with error handling."""
    try:
        df = pd.read_csv(file_path)
    except pd.errors.ParserError:
        df = pd.read_csv(file_path, on_bad_lines='warn', quoting=3)
    
    print(f"Loaded {len(df)} rows from the document file.")
    print(f"Columns in document file: {df.columns.tolist()}")
    
    df.columns = df.columns.str.strip('"')
    print(f"Cleaned column names: {df.columns.tolist()}")
    
    return df

def get_valid_column(df, preferred_columns):
    """Get the first valid column name from a list of preferred columns."""
    for col in preferred_columns:
        if col in df.columns:
            return col
    raise ValueError(f"None of the preferred columns {preferred_columns} found in DataFrame. Available columns are: {df.columns.tolist()}")

def vectorize_text(company_series, doc_series):
    """Convert both company names and document text to TF-IDF vectors using a single vectorizer."""
    # Combine company names and document text
    all_text = pd.concat([company_series, doc_series]).fillna('').astype(str)
    
    vectorizer = TfidfVectorizer()
    all_vectors = vectorizer.fit_transform(all_text)
    
    # Split the vectors back into companies and documents
    company_vectors = all_vectors[:len(company_series)]
    doc_vectors = all_vectors[len(company_series):]
    
    return company_vectors, doc_vectors

def get_context(text, company, threshold=50, max_words_to_check=100):
    """Get the context around the company name in the text using a more efficient matching approach."""
    words = text.lower().split()[:max_words_to_check]  # Limit the number of words to check
    company_words = company.lower().split()
    
    best_match_ratio = 0
    best_match_pos = -1
    best_match_length = 0

    for i in range(len(words)):
        # Only check sequences up to the length of the company name plus 2
        for j in range(i+1, min(i+len(company_words)+2, len(words)+1)):
            substring = ' '.join(words[i:j])
            ratio = fuzz.partial_ratio(substring, company.lower())
            if ratio > best_match_ratio:
                best_match_ratio = ratio
                best_match_pos = i
                best_match_length = j - i

    if best_match_ratio < threshold:
        return f"Context not found (Best match: {best_match_ratio}%)", best_match_ratio, None

    start = max(0, best_match_pos - 5)
    end = min(len(words), best_match_pos + best_match_length + 5)
    context = ' '.join(words[start:end])
    matched_text = ' '.join(words[best_match_pos:best_match_pos+best_match_length])
    
    return f"Context (Match: {best_match_ratio}%): ...{context}...", best_match_ratio, matched_text

def find_companies_in_document(document_text, companies, min_similarity=80, context_words=10):
    """Find companies in the document using fuzzy string matching."""
    matches = []
    words = document_text.split()
    
    for company in companies:
        best_ratio = 0
        best_match = ""
        best_start = -1
        
        for i in range(len(words)):
            for j in range(i, min(i + 10, len(words))):  # Check phrases up to 10 words long
                phrase = " ".join(words[i:j+1])
                ratio = fuzz.partial_ratio(company.lower(), phrase.lower())
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_match = phrase
                    best_start = i
        
        if best_ratio >= min_similarity:
            start = max(0, best_start - context_words)
            end = min(len(words), best_start + len(best_match.split()) + context_words)
            context = " ".join(words[start:end])
            matches.append((company, best_ratio, context, best_match))
    
    return sorted(matches, key=lambda x: x[1], reverse=True)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)

    # Load companies
    companies_path = os.path.join(parent_dir, 'companies.csv')
    print("Loading companies...")
    df_companies = load_companies(companies_path)
    print(f"Loaded {len(df_companies)} companies.")

    # Load document
    document_path = os.path.join(parent_dir, 'document.csv')
    print("Loading document...")
    df_document = load_document(document_path)
    
    if df_document.empty:
        print("The document DataFrame is empty. Please check your CSV file.")
        return
    
    document_column = get_valid_column(df_document, ['Text', 'text', 'content', 'document'])
    print(f"Using '{document_column}' as the document text column.")
    print(f"Number of documents to process: {len(df_document)}")

    # Find companies in each document
    print("\nStarting company matching process:")
    all_matches = []
    start_time = time.time()
    
    for i, doc_text in enumerate(df_document[document_column]):
        if i % 10 == 0:  # Print progress every 10 documents
            elapsed_time = time.time() - start_time
            docs_per_second = (i + 1) / elapsed_time if elapsed_time > 0 else 0
            print(f"Processing document {i+1}/{len(df_document)} ({docs_per_second:.2f} docs/sec)")
        
        matches = find_companies_in_document(doc_text, df_companies['Company'].tolist())
        all_matches.extend([(company, similarity, context, matched_text, i) for company, similarity, context, matched_text in matches])

    print("\nMatching process completed.")
    print(f"Total processing time: {time.time() - start_time:.2f} seconds")

    # Sort and display results
    all_matches.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 50 companies found across all documents:")
    for company, similarity, context, matched_text, doc_index in all_matches[:50]:
        print(f"{company}: Match {similarity:.2f}% (Document {doc_index})")
        print(f"Context: ...{context}...")
        print(f"Matched text: '{matched_text}'")
        print()

    # Analysis of matches
    print(f"\nTotal matches found: {len(all_matches)}")

    # Display some statistics
    similarity_scores = [match[1] for match in all_matches]
    print(f"\nSimilarity score statistics:")
    print(f"Min: {min(similarity_scores):.2f}%, Max: {max(similarity_scores):.2f}%, Mean: {np.mean(similarity_scores):.2f}%")

    print("\nProcessing complete.")

if __name__ == "__main__":
    main()