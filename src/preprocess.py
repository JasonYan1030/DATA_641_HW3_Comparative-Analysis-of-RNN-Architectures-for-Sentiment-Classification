import pandas as pd
import re
import numpy as np
import nltk
from collections import Counter
import torch
from sklearn.model_selection import train_test_split

# Download the punkt tokenizer model
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

def preprocess(file_path, sequence_length=25):

    # Load dataset
    IMDB_review_data = pd.read_csv(file_path)

    # Lowercase all text.
    IMDB_review_data['review'] = IMDB_review_data['review'].str.lower()

    # Remove punctuation and special characters.
    IMDB_review_data['review'] = IMDB_review_data['review'].apply(lambda x: re.sub(r'<.*?>', ' ', x))
    IMDB_review_data['review'] = IMDB_review_data['review'].apply(lambda x: re.sub(r'[^a-z\s]', '', x))
    IMDB_review_data['review'] = IMDB_review_data['review'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())

    # Tokenize sentences (using nltk.word_tokenize)
    tokenized = IMDB_review_data['review'].apply(nltk.word_tokenize)

    # Keep only the top 10,000 most frequent words.
    all_tokens = [word for review in tokenized for word in review]
    frequency = Counter(all_tokens)
    vocabulary = {word: i+2 for i, (word, _) in enumerate(frequency.most_common(10000))}
    vocabulary['<infrequent>'] = 0 
    vocabulary['<frequent>'] = 1

    # Convert each review to a sequence of token IDs.
    sequences = []
    for review in tokenized:
        encoded = [vocabulary.get(word, vocabulary['<frequent>']) for word in review]
        sequences.append(torch.tensor(encoded, dtype=torch.long))

    # Pad or truncate sequences to fixed lengths
    padded = torch.zeros((len(sequences), sequence_length), dtype=torch.long)
    i = 0
    while i < len(sequences):
        seq = sequences[i]
        length = min(len(seq), sequence_length)
        padded[i, :length] = seq[:length]
        i += 1
    y = IMDB_review_data['sentiment']

    # Split train-test sets (50%-50%)
    X_train, X_test, y_train, y_test = train_test_split(
        padded, y, test_size=0.5, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, vocabulary