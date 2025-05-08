"""
Emotion Detection Models using GoEmotion Dataset

This program implements three emotion detection models:
1. An RNN-based model
2. A BERT-based model
3. A RoBERTa-based model

All models are trained and evaluated on the GoEmotion dataset.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# Configure GPU for training
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print(f"Found {len(physical_devices)} GPU(s):")
    for device in physical_devices:
        print(f"  - {device.name}")
        # Enable memory growth to avoid allocating all GPU memory at once
        tf.config.experimental.set_memory_growth(device, True)

    # Set TensorFlow to use the first GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print("GPU training enabled")
else:
    print("No GPU found. Training will use CPU.")

# Import from custom modules
from data_utils import load_goemotion_dataset, preprocess_data_for_rnn, preprocess_data_for_bert, preprocess_data_for_roberta
from rnn_model import train_and_evaluate_rnn_model
from bert_model import train_and_evaluate_bert_model
from robert_model import train_and_evaluate_roberta_model
from visualization import compare_models

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def main():
    """
    Main function to orchestrate the entire process.
    """
    print("Starting emotion detection model training and evaluation...")

    # Load the GoEmotion dataset
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = load_goemotion_dataset()

    # RNN Model
    print("\n" + "="*50)
    print("RNN MODEL PIPELINE")
    print("="*50)

    # Preprocess data for RNN
    rnn_train_data, rnn_train_labels, rnn_val_data, rnn_val_labels, rnn_test_data, rnn_test_labels, word_index = preprocess_data_for_rnn(
        train_texts, train_labels, val_texts, val_labels, test_texts, test_labels
    )

    # Train and evaluate RNN model
    rnn_model, rnn_history, rnn_accuracy = train_and_evaluate_rnn_model(
        rnn_train_data, rnn_train_labels, rnn_val_data, rnn_val_labels, rnn_test_data, rnn_test_labels, word_index
    )

    # BERT Model
    print("\n" + "="*50)
    print("BERT MODEL PIPELINE")
    print("="*50)

    # Preprocess data for BERT
    bert_train_encodings, bert_train_labels, bert_val_encodings, bert_val_labels, bert_test_encodings, bert_test_labels = preprocess_data_for_bert(
        train_texts, train_labels, val_texts, val_labels, test_texts, test_labels
    )

    # Train and evaluate BERT model
    bert_model, bert_history, bert_accuracy = train_and_evaluate_bert_model(
        bert_train_encodings, bert_train_labels, bert_val_encodings, bert_val_labels, bert_test_encodings, bert_test_labels
    )

    # RoBERTa Model
    print("\n" + "="*50)
    print("RoBERTa MODEL PIPELINE")
    print("="*50)

    # Preprocess data for RoBERTa
    roberta_train_encodings, roberta_train_labels, roberta_val_encodings, roberta_val_labels, roberta_test_encodings, roberta_test_labels = preprocess_data_for_roberta(
        train_texts, train_labels, val_texts, val_labels, test_texts, test_labels
    )

    # Train and evaluate RoBERTa model
    roberta_model, roberta_history, roberta_accuracy = train_and_evaluate_roberta_model(
        roberta_train_encodings, roberta_train_labels, roberta_val_encodings, roberta_val_labels, roberta_test_encodings, roberta_test_labels
    )

    # Compare models
    compare_models(rnn_history, bert_history, roberta_history, rnn_accuracy, bert_accuracy, roberta_accuracy)

    print("\nEmotion detection models training and evaluation completed!")

if __name__ == "__main__":
    main()
