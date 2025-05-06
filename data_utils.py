"""
Data Utilities for Emotion Detection Models

This module provides functions for loading and preprocessing the GoEmotion dataset
for RNN, BERT, and RoBERTa models.
"""

import numpy as np
import datasets
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, RobertaTokenizer

# Constants
MAX_SEQUENCE_LENGTH = 100
NUM_CLASSES = 28  # GoEmotion has 28 emotion categories

def load_goemotion_dataset():
    """
    Load the GoEmotion dataset using the Hugging Face datasets library.

    Returns:
        train_texts, train_labels, val_texts, val_labels, test_texts, test_labels
    """
    print("Loading GoEmotion dataset...")

    # Load the GoEmotion dataset
    goemotion = datasets.load_dataset("go_emotions")

    # Extract train, validation, and test sets
    train_data = goemotion["train"]
    val_data = goemotion["validation"]
    test_data = goemotion["test"]

    # Extract texts and labels
    train_texts = train_data["text"]
    train_labels = train_data["labels"]

    val_texts = val_data["text"]
    val_labels = val_data["labels"]

    test_texts = test_data["text"]
    test_labels = test_data["labels"]

    print(f"Dataset loaded: {len(train_texts)} training, {len(val_texts)} validation, {len(test_texts)} test samples")

    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels

def process_labels(label_lists):
    """
    Process the labels for multi-label classification.

    Args:
        label_lists: List of lists containing label indices

    Returns:
        Multi-hot encoded labels
    """
    # Initialize a zero matrix
    labels = np.zeros((len(label_lists), NUM_CLASSES))

    # For each sample, set the corresponding label indices to 1
    for i, label_list in enumerate(label_lists):
        for label in label_list:
            labels[i, label] = 1

    return labels

def preprocess_data_for_rnn(train_texts, train_labels, val_texts, val_labels, test_texts, test_labels):
    """
    Preprocess the text data for the RNN model.

    Args:
        train_texts, train_labels, val_texts, val_labels, test_texts, test_labels: Dataset splits

    Returns:
        Preprocessed data ready for the RNN model
    """
    print("Preprocessing data for RNN model...")

    # Create and fit tokenizer on training data
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_texts)
    word_index = tokenizer.word_index
    print(f"Found {len(word_index)} unique tokens")

    # Convert texts to sequences
    train_sequences = tokenizer.texts_to_sequences(train_texts)
    val_sequences = tokenizer.texts_to_sequences(val_texts)
    test_sequences = tokenizer.texts_to_sequences(test_texts)

    # Pad sequences to ensure uniform length
    train_data = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    val_data = pad_sequences(val_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

    # Process labels - convert to one-hot encoding
    # GoEmotion has multi-label classification, so we need to handle this appropriately
    train_labels_processed = process_labels(train_labels)
    val_labels_processed = process_labels(val_labels)
    test_labels_processed = process_labels(test_labels)

    return (train_data, train_labels_processed, val_data, val_labels_processed, 
            test_data, test_labels_processed, word_index)

def preprocess_data_for_bert(train_texts, train_labels, val_texts, val_labels, test_texts, test_labels):
    """
    Preprocess the text data for the BERT model.

    Args:
        train_texts, train_labels, val_texts, val_labels, test_texts, test_labels: Dataset splits

    Returns:
        Preprocessed data ready for the BERT model
    """
    print("Preprocessing data for BERT model...")

    # Load BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenize and prepare input for BERT
    train_encodings = tokenizer(list(train_texts), truncation=True, padding='max_length', 
                               max_length=MAX_SEQUENCE_LENGTH, return_tensors='tf')
    val_encodings = tokenizer(list(val_texts), truncation=True, padding='max_length', 
                             max_length=MAX_SEQUENCE_LENGTH, return_tensors='tf')
    test_encodings = tokenizer(list(test_texts), truncation=True, padding='max_length', 
                              max_length=MAX_SEQUENCE_LENGTH, return_tensors='tf')

    # Process labels
    train_labels_processed = process_labels(train_labels)
    val_labels_processed = process_labels(val_labels)
    test_labels_processed = process_labels(test_labels)

    return (train_encodings, train_labels_processed, val_encodings, val_labels_processed, 
            test_encodings, test_labels_processed)

def preprocess_data_for_roberta(train_texts, train_labels, val_texts, val_labels, test_texts, test_labels):
    """
    Preprocess the text data for the RoBERTa model.

    Args:
        train_texts, train_labels, val_texts, val_labels, test_texts, test_labels: Dataset splits

    Returns:
        Preprocessed data ready for the RoBERTa model
    """
    print("Preprocessing data for RoBERTa model...")

    # Load RoBERTa tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    # Tokenize and prepare input for RoBERTa
    train_encodings = tokenizer(list(train_texts), truncation=True, padding='max_length', 
                               max_length=MAX_SEQUENCE_LENGTH, return_tensors='tf')
    val_encodings = tokenizer(list(val_texts), truncation=True, padding='max_length', 
                             max_length=MAX_SEQUENCE_LENGTH, return_tensors='tf')
    test_encodings = tokenizer(list(test_texts), truncation=True, padding='max_length', 
                              max_length=MAX_SEQUENCE_LENGTH, return_tensors='tf')

    # Process labels
    train_labels_processed = process_labels(train_labels)
    val_labels_processed = process_labels(val_labels)
    test_labels_processed = process_labels(test_labels)

    return (train_encodings, train_labels_processed, val_encodings, val_labels_processed, 
            test_encodings, test_labels_processed)
