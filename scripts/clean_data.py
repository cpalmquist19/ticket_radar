import pandas as pd
import re
from pathlib import Path

def remove_pii(text):
    """
    Remove PII (emails and phone numbers) from text.
    
    Args:
        text: String to clean
        
    Returns:
        Cleaned string with PII removed
    """
    if pd.isna(text):
        return text
    
    text = str(text)
    
    # Remove email addresses
    # Pattern matches: word@domain.com, user.name@domain.co.uk, etc.
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    text = re.sub(email_pattern, '[EMAIL_REMOVED]', text)
    
    # Remove phone numbers
    # Pattern matches various formats:
    # - <tel_num> placeholders
    # - (123) 456-7890
    # - 123-456-7890
    # - 123.456.7890
    # - +1 123 456 7890
    # - 1234567890
    phone_patterns = [
        r'<tel_num>',  # Placeholder pattern
        r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',  # US format
        r'\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}',  # International format
        r'\b\d{10,}\b',  # Long digit sequences (likely phone numbers)
    ]
    
    for pattern in phone_patterns:
        text = re.sub(pattern, '[PHONE_REMOVED]', text)
    
    return text

def count_words(text):
    """
    Count words in text.
    
    Args:
        text: String to count words in
        
    Returns:
        Number of words
    """
    if pd.isna(text):
        return 0
    
    # Split on whitespace and filter out empty strings
    words = str(text).split()
    return len(words)

def clean_dataset(input_path, output_path=None):
    """
    Load CSV, remove PII, and filter tickets with less than 10 words.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to save cleaned CSV (default: adds '_cleaned' to input filename)
    """
    # Set default output path if not provided
    if output_path is None:
        input_path_obj = Path(input_path)
        output_path = str(input_path_obj.parent / f"{input_path_obj.stem}_cleaned{input_path_obj.suffix}")
    
    print(f"Loading dataset from: {input_path}")
    df = pd.read_csv(input_path)
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Original row count: {len(df)}")
    
    # Remove PII from text columns
    text_columns = ['subject', 'body', 'answer']
    for col in text_columns:
        if col in df.columns:
            print(f"Removing PII from '{col}' column...")
            df[col] = df[col].apply(remove_pii)
    
    # Count words in body column (main ticket content)
    if 'body' in df.columns:
        print("Counting words in 'body' column...")
        df['word_count'] = df['body'].apply(count_words)
        
        # Filter out tickets with less than 10 words
        original_count = len(df)
        df = df[df['word_count'] >= 10].copy()
        removed_count = original_count - len(df)
        
        print(f"Removed {removed_count} tickets with less than 10 words")
        print(f"Remaining tickets: {len(df)}")
        
        # Drop the word_count column before saving
        df = df.drop(columns=['word_count'])
    else:
        print("Warning: 'body' column not found. Skipping word count filtering.")
    
    # Save cleaned dataset
    print(f"Saving cleaned dataset to: {output_path}")
    df.to_csv(output_path, index=False)
    
    print(f"Cleaned dataset shape: {df.shape}")
    print("Done!")

if __name__ == "__main__":
    input_file = r"C:\Code\ml\ticket_intent\training\aa_dataset-tickets-multi-lang-5-2-50-version.csv"
    clean_dataset(input_file)

