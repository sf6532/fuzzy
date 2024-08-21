import csv
import os
import time
import openai
from dotenv import load_dotenv
import tiktoken

# Load environment variables
load_dotenv()

# Set up Azure OpenAI credentials
openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = "2024-02-01"
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")

def load_companies(file_path):
    with open(file_path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        return [row[0] for row in reader]

def load_document(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def chunk_document(document, max_tokens=4000):
    words = document.split()
    chunks = []
    current_chunk = []
    current_token_count = 0

    for word in words:
        word_tokens = num_tokens_from_string(word, "cl100k_base")
        if current_token_count + word_tokens > max_tokens:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_token_count = word_tokens
        else:
            current_chunk.append(word)
            current_token_count += word_tokens

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def create_company_chunks(companies, max_tokens=1000):
    chunks = []
    current_chunk = []
    current_token_count = 0

    for company in companies:
        company_tokens = num_tokens_from_string(company, "cl100k_base")
        if current_token_count + company_tokens > max_tokens:
            chunks.append(current_chunk)
            current_chunk = [company]
            current_token_count = company_tokens
        else:
            current_chunk.append(company)
            current_token_count += company_tokens

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def find_companies_in_chunk(chunk, company_chunk):
    prompt = f"""
    You are an AI assistant specialized in identifying company names in text.
    Below is a list of companies followed by a chunk of text.
    Your task is to identify which companies from the list appear in the text chunk.
    Only return companies that are explicitly mentioned.
    
    Companies:
    {', '.join(company_chunk)}
    
    Text chunk:
    {chunk}
    
    Please return your answer as a Python list of tuples, where each tuple contains 
    the company name and a confidence score between 0 and 1. For example:
    [("Company A", 0.95), ("Company B", 0.8)]
    """

    response = openai.ChatCompletion.create(
        engine="steven-gpt-4-32k",  # Replace with your actual deployment name
        messages=[
            {"role": "system", "content": "You are a company name detection AI."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=150
    )

    # Parse the response
    try:
        return eval(response.choices[0].message.content.strip())
    except:
        print("Error parsing OpenAI response. Returning empty list.")
        return []

def find_companies_in_document(document, company_db):
    doc_chunks = chunk_document(document)
    company_chunks = create_company_chunks(company_db)
    total_chunks = len(doc_chunks) * len(company_chunks)
    all_matches = []
    
    start_time = time.time()
    chunk_count = 0
    for doc_chunk in doc_chunks:
        for company_chunk in company_chunks:
            chunk_matches = find_companies_in_chunk(doc_chunk, company_chunk)
            all_matches.extend(chunk_matches)
            
            # Update progress
            chunk_count += 1
            progress = chunk_count / total_chunks * 100
            elapsed_time = time.time() - start_time
            estimated_total_time = elapsed_time / chunk_count * total_chunks
            remaining_time = estimated_total_time - elapsed_time
            print(f"\rProgress: {progress:.2f}% - Est. time remaining: {remaining_time:.2f}s", end='', flush=True)
    
    print("\nProcessing complete.")
    
    # Deduplicate and sort results
    unique_matches = {}
    for company, confidence in all_matches:
        if company not in unique_matches or confidence > unique_matches[company]:
            unique_matches[company] = confidence
    
    return sorted(unique_matches.items(), key=lambda x: x[1], reverse=True)

# Main execution
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)

    companies_path = os.path.join(parent_dir, 'companies.csv')
    company_db = load_companies(companies_path)
    print(f"Loaded {len(company_db)} companies from the database.")

    document_path = os.path.join(parent_dir, 'document.csv')
    document = load_document(document_path)
    print(f"Loaded document with {len(document.split())} words.")

    total_tokens = num_tokens_from_string(document, "cl100k_base")
    print(f"Estimated total tokens in document: {total_tokens}")

    print("\nStarting Azure OpenAI-based company detection:")
    doc_matches = find_companies_in_document(document, company_db)

    print("\nTop 20 companies found in the document:")
    for company, confidence in doc_matches[:20]:
        print(f"{company}: Confidence {confidence:.2f}")

    print(f"\nTotal matches found: {len(doc_matches)}")
    print("\nProcessing complete.")