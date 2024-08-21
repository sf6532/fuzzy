import csv
import os
import time
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def load_companies(file_path):
    with open(file_path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        return [row[0] for row in reader]

def load_document(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def vectorize_text(text, model, tokenizer, max_length=512):
    inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

def find_companies_in_document(document, company_db, model, tokenizer, threshold=0.7):
    # Vectorize companies
    company_vectors = np.array([vectorize_text(company, model, tokenizer)[0] for company in company_db])
    
    # Process document in chunks
    chunk_size = 256  # Adjust based on your needs and model's max input size
    words = document.split()
    total_chunks = len(words) // chunk_size + 1
    matches = []
    
    start_time = time.time()
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i+chunk_size])
        chunk_vector = vectorize_text(chunk, model, tokenizer)[0]
        
        # Calculate similarities
        similarities = cosine_similarity([chunk_vector], company_vectors)[0]
        
        # Find matches above threshold
        for j, sim in enumerate(similarities):
            if sim > threshold:
                matches.append((company_db[j], sim))
        
        # Update progress
        progress = (i + chunk_size) / len(words) * 100
        elapsed_time = time.time() - start_time
        estimated_total_time = elapsed_time / (i + chunk_size) * len(words)
        remaining_time = estimated_total_time - elapsed_time
        print(f"\rProgress: {progress:.2f}% - Est. time remaining: {remaining_time:.2f}s", end='', flush=True)
    
    print("\nProcessing complete.")
    
    # Deduplicate and sort results
    unique_matches = {}
    for company, similarity in matches:
        if company not in unique_matches or similarity > unique_matches[company]:
            unique_matches[company] = similarity
    
    return sorted(unique_matches.items(), key=lambda x: x[1], reverse=True)

# Main execution
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)

    # Load model and tokenizer
    model_name = "bert-base-uncased"  # You can change this to a model with larger context window
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    companies_path = os.path.join(parent_dir, 'companies.csv')
    company_db = load_companies(companies_path)
    print(f"Loaded {len(company_db)} companies from the database.")

    document_path = os.path.join(parent_dir, 'document.csv')
    document = load_document(document_path)
    print(f"Loaded document with {len(document.split())} words.")

    print("\nStarting LLM-based company detection:")
    doc_matches = find_companies_in_document(document, company_db, model, tokenizer)

    print("\nTop 20 companies found in the document:")
    for company, similarity in doc_matches[:20]:
        print(f"{company}: Similarity {similarity:.2f}")

    print(f"\nTotal matches found: {len(doc_matches)}")
    print("\nProcessing complete.")