
from fuzzywuzzy import fuzz
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import re
from tqdm import tqdm

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

def find_companies_in_document(doc_vector, company_vectors, companies, document_text, threshold=0.1):
    """Find companies in the document using cosine similarity and return context."""
    similarities = cosine_similarity(doc_vector, company_vectors)
    matches = []
    for i, sim in enumerate(similarities[0]):
        if sim >= threshold:
            company = companies[i]
            context = get_context(document_text, company)
            matches.append((company, sim, context))
    return sorted(matches, key=lambda x: x[1], reverse=True)



def get_context(text, company, threshold=70):
    """Get the context (up to 10 words) around the company name in the text using fuzzy matching."""
    words = re.findall(r'\b\w+\b', text.lower())
    company_words = re.findall(r'\b\w+\b', company.lower())
    
    best_match_pos = -1
    best_match_ratio = 0
    for i in range(len(words) - len(company_words) + 1):
        substring = ' '.join(words[i:i+len(company_words)])
        ratio = fuzz.ratio(substring, ' '.join(company_words))
        if ratio > best_match_ratio:
            best_match_ratio = ratio
            best_match_pos = i
    
    if best_match_ratio < threshold:
        return None  # Return None if match is below threshold
    
    start = max(0, best_match_pos - 5)
    end = min(len(words), best_match_pos + len(company_words) + 5)
    context = ' '.join(words[start:end])
    
    return f"Context (Match: {best_match_ratio}%): ...{context}..."


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)

    # Load companies
    companies_path = os.path.join(parent_dir, 'companies.csv')
    print("Loading companies...")
    df_companies = load_companies(companies_path)

    # Load document
    document_path = os.path.join(parent_dir, 'document.csv')
    print("Loading document...")
    df_document = load_document(document_path)
    
    if df_document.empty:
        print("The document DataFrame is empty. Please check your CSV file.")
        return
    
    document_column = get_valid_column(df_document, ['Text', 'text', 'content', 'document'])
    print(f"Using '{document_column}' as the document text column.")

    # Check for null values
    null_count = df_document[document_column].isnull().sum()
    if null_count > 0:
        print(f"Warning: Found {null_count} null values in the '{document_column}' column.")
        print("These will be treated as empty strings for vectorization.")

    # Vectorize companies and documents together
    print("Vectorizing companies and documents...")
    company_vectors, doc_vectors = vectorize_text(df_companies['Company'], df_document[document_column])

    # Find companies in each document
    print("\nStarting company matching process:")
    all_matches = []
    for i, (doc_vector, doc_text) in tqdm(enumerate(zip(doc_vectors, df_document[document_column])), total=len(df_document)):
        matches = find_companies_in_document(doc_vector, company_vectors, df_companies['Company'].tolist(), doc_text)
        all_matches.extend([(company, similarity, context, i) for company, similarity, context in matches])

    # Sort and display results
    all_matches.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 20 companies found across all documents (showing only matches above 75%):")
    for company, similarity, context, doc_index in all_matches[:20]:
        if context and int(re.search(r'Match: (\d+)%', context).group(1)) > 75:
            print(f"{company}: Similarity {similarity:.4f} (Document {doc_index})")
            print(f"{context}")
            print()

    print(f"\nTotal matches found: {len(all_matches)}")
    print("\nProcessing complete.")

if __name__ == "__main__":
    main()