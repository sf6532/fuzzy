import pandas as pd
import os, io
import re
from name_matching.name_matcher import NameMatcher
from tqdm import tqdm
import csv
import traceback
import sys

def clean_text(text):
    """Clean text by removing non-printable characters and replacing with space."""
    # Remove non-printable characters
    cleaned_text = re.sub(r'[^\x20-\x7E]', ' ', str(text))
    # Replace multiple spaces with a single space
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    return cleaned_text.strip()

def load_companies(file_path):
    """Load companies from a CSV file without headers."""
    with open(file_path, 'r', encoding='utf-8') as file:
        companies = [clean_text(line.strip()) for line in file if line.strip()]
    df = pd.DataFrame(companies, columns=['Company'])
    print(f"Loaded {len(df)} companies from the file.")
    return df

def load_document(file_path):
    """Load document from a CSV file with robust error handling."""
    print(f"Attempting to load document from {file_path}")
    
    # Try reading with default settings
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded document with {len(df)} rows and {len(df.columns)} columns.")
        print(f"Columns in document file: {df.columns.tolist()}")
        return df
    except pd.errors.ParserError as e:
        print(f"Error parsing CSV with default settings: {e}")
    
    # If that fails, try reading the file manually and process it chunk by chunk
    try:
        chunks = []
        problem_rows = []
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            headers = next(reader)  # Read the header row
            for i, row in enumerate(reader, start=1):
                try:
                    if len(row) != len(headers):
                        raise ValueError(f"Mismatched number of columns in row {i+1}")
                    chunks.append(row)
                except Exception as row_error:
                    problem_rows.append((i+1, row, str(row_error)))
                    if len(problem_rows) < 10:  # Print first 10 problem rows
                        print(f"Problem in row {i+1}: {row_error}")
                        print(f"Row content: {row}")
                if i % 1000 == 0:
                    print(f"Processed {i} rows...")

        if chunks:
            df = pd.DataFrame(chunks, columns=headers)
            print(f"Successfully loaded document with {len(df)} rows and {len(df.columns)} columns.")
            print(f"Columns in document file: {df.columns.tolist()}")
            print(f"Total problem rows encountered: {len(problem_rows)}")
            return df
        else:
            raise ValueError("No valid data found in the file.")

    except Exception as e:
        print(f"Error reading file manually: {e}")
        if problem_rows:
            print("\nDetailed information about problem rows:")
            for row_num, content, error in problem_rows[:10]:  # Show details for up to 10 problem rows
                print(f"Row {row_num}:")
                print(f"Content: {content}")
                print(f"Error: {error}")
                print()
        raise

def get_valid_column(df, preferred_columns):
    """Get the first valid column name from a list of preferred columns."""
    for col in preferred_columns:
        if col in df.columns:
            return col
    raise ValueError(f"None of the preferred columns {preferred_columns} found in DataFrame. Available columns are: {df.columns.tolist()}")

def match_companies(df_companies, df_document, company_column, document_column):
    """Match companies in the document using NameMatcher."""
    # Initialize the name matcher
    matcher = NameMatcher(number_of_matches=1, 
                          legal_suffixes=True, 
                          common_words=False, 
                          top_n=50, 
                          verbose=True)

    # Adjust the distance metrics to use
    matcher.set_distance_metrics(['bag', 'typo', 'refined_soundex'])

    # Load the master data (companies)
    matcher.load_and_process_master_data(column=company_column,
                                         df_matching_data=df_companies, 
                                         transform=True)

    # Perform the name matching on the document data
    total_rows = len(df_document)
    matches = []

    for _, row in tqdm(df_document.iterrows(), total=total_rows, desc="Matching companies"):
        cleaned_text = clean_text(row[document_column])
        match = matcher.match_names(to_be_matched=pd.DataFrame({document_column: [cleaned_text]}), 
                                    column_matching=document_column)
        matches.append(match)

    return pd.concat(matches, ignore_index=True)

def main():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)

        # Load companies
        companies_path = os.path.join(parent_dir, 'companies.csv')
        df_companies = load_companies(companies_path)
        company_column = 'Company'

        # Load document
        document_path = os.path.join(parent_dir, 'document.csv')
        df_document = load_document(document_path)
        document_column = get_valid_column(df_document, ['Text', 'text', 'content', 'document'])

        # Match companies in the document
        print("Starting company matching process...")
        matches = match_companies(df_companies, df_document, company_column, document_column)

        # Combine the datasets based on the matches
        combined = pd.merge(df_companies, matches, how='left', left_index=True, right_on='match_index')
        combined = pd.merge(combined, df_document, how='left', left_index=True, right_index=True)

        # Filter out rows with NaN scores
        combined = combined.dropna(subset=['score'])

        if combined.empty:
            print("No matches found with valid scores.")
            return

        # Normalize scores
        min_score = combined['score'].min()
        max_score = combined['score'].max()
        combined['normalized_score'] = (combined['score'] - min_score) / (max_score - min_score)

        # Group by company and select the best score for each
        best_matches = combined.groupby(company_column).agg({
            'score': 'max',
            'normalized_score': 'max',
            document_column: 'first'  # Include the matched document text
        }).reset_index()

        # Sort by score and get top 20 unique companies
        top_20_unique = best_matches.sort_values('score', ascending=False).head(20)

        # Display results
        print("\nTop 20 unique companies found in the document:")
        for _, row in top_20_unique.iterrows():
            company = row[company_column]
            raw_score = row['score']
            norm_score = row['normalized_score']
            matched_text = row[document_column]
            
            print(f"{company}:")
            print(f"  Raw Score: {raw_score:.2f}")
            print(f"  Normalized Score: {norm_score:.2f}")
            if pd.notna(matched_text):
                print(f"  Matched Text: {matched_text[:100]}...")  # Display first 100 characters of matched text
            else:
                print("  Matched Text: No matching text found")
            print()

        print(f"\nTotal unique companies matched: {len(best_matches)}")
        print("Processing complete.")

    except Exception as e:
        print(f"An error occurred in the main function: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()