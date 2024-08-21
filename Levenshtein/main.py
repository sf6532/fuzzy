import csv
import os
import time
import sys

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

def normalize_string(s):
    return ''.join(c.lower() for c in s if c.isalnum())

def fuzzy_match_levenshtein(query, choices, threshold=0.8):
    query = normalize_string(query)
    results = []
    for choice in choices:
        choice_normalized = normalize_string(choice)
        max_len = max(len(query), len(choice_normalized))
        if max_len == 0:
            similarity = 1.0
        else:
            distance = levenshtein_distance(query, choice_normalized)
            similarity = 1 - (distance / max_len)
        if similarity >= threshold:
            results.append((choice, similarity))
    return sorted(results, key=lambda x: x[1], reverse=True)

def find_companies_in_document(document, company_db, threshold=0.8):
    words = document.split()
    total_chunks = len(words) - 2
    matches = []
    start_time = time.time()

    for i in range(total_chunks):
        if i % 10 == 0 or i == total_chunks - 1:  # Update progress more frequently
            progress = (i + 1) / total_chunks * 100
            elapsed_time = time.time() - start_time
            estimated_total_time = elapsed_time / (i + 1) * total_chunks
            remaining_time = estimated_total_time - elapsed_time
            progress_bar = f"[{'=' * int(progress // 2)}{' ' * (50 - int(progress // 2))}]"
            print(f"\rProgress: {progress_bar} {progress:.2f}% - Est. time remaining: {remaining_time:.2f}s", end='')
            sys.stdout.flush()  # Force the output to be displayed immediately

        potential = ' '.join(words[i:i+3])
        company_matches = fuzzy_match_levenshtein(potential, company_db, threshold)
        matches.extend(company_matches)

    print("\nProcessing complete.")

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
    print("\nStarting document-wide company detection:")
    doc_matches = find_companies_in_document(document, company_db, threshold=0.8)

    print("\nCompanies found in the document:")
    for company, similarity in doc_matches:
        print(f"{company}: Similarity {similarity:.2f}")

    print("\nProcessing complete.")