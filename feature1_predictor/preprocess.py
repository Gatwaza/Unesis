import pandas as pd
from transformers import AutoTokenizer

def preprocess_data(dataset_path):
    df = pd.read_csv(dataset_path)

    # Ensure the dataset contains columns for text and lab results
    # Update column names based on the actual dataset
    text_column = 'text'  # EDIT: Replace with the actual column name for text
    lab_results_columns = ['col1', 'col2', 'col3']  # EDIT: Replace with actual lab results columns
    label_column = 'label'  # EDIT: Replace with the actual column name for labels

    # Tokenization using a suitable tokenizer from Hugging Face Transformers
    tokenizer = AutoTokenizer.from_pretrained('allenai/llama-3-7B')

    # Tokenize text data
    tokenized_texts = tokenizer(df[text_column].tolist(), padding=True, truncation=True, return_tensors='pt')

    # Normalize numerical data (example: scaling lab results)
    lab_results = df[lab_results_columns].values
    normalized_lab_results = normalize_lab_results(lab_results)

    # Prepare labels (example: convert labels to integers)
    labels = df[label_column].astype(int).values

    return tokenized_texts, normalized_lab_results, labels

def normalize_lab_results(lab_results):
    min_value = lab_results.min(axis=0)
    max_value = lab_results.max(axis=0)
    normalized_results = (lab_results - min_value) / (max_value - min_value)
    return normalized_results

if __name__ == '__main__':
    dataset_path = '/mnt/data/heart_data.csv'  # Path to the dataset
    tokenized_texts, normalized_lab_results, labels = preprocess_data(dataset_path)

    # Save preprocessed data or use it directly for training
    tokenized_texts.save_pretrained('preprocessed/tokenized_texts')
    pd.DataFrame(normalized_lab_results).to_csv('preprocessed/lab_results.csv', index=False)
    pd.DataFrame(labels).to_csv('preprocessed/labels.csv', index=False)
