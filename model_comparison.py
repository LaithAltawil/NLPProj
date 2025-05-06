"""
Model Comparison for Emotion Detection

This module provides functions for comparing the performance of different
emotion detection models (RNN, BERT, and RoBERTa) on the GoEmotion dataset.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns

# Import from custom modules
from data_utils import load_goemotion_dataset, preprocess_data_for_rnn, preprocess_data_for_bert, preprocess_data_for_roberta
from rnn_model import train_and_evaluate_rnn_model
from bert_model import train_and_evaluate_bert_model
from robert_model import train_and_evaluate_roberta_model

def load_and_evaluate_models():
    """
    Load and evaluate all three models (RNN, BERT, and RoBERTa) on the GoEmotion dataset.
    
    Returns:
        Dictionary containing model names, accuracies, and other metrics
    """
    print("Loading and evaluating models...")
    
    # Load the GoEmotion dataset
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = load_goemotion_dataset()
    
    # Dictionary to store results
    results = {
        'model_names': ['RNN', 'BERT', 'RoBERTa'],
        'accuracies': [],
        'histories': []
    }
    
    # RNN Model
    print("\nEvaluating RNN model...")
    rnn_train_data, rnn_train_labels, rnn_val_data, rnn_val_labels, rnn_test_data, rnn_test_labels, word_index = preprocess_data_for_rnn(
        train_texts, train_labels, val_texts, val_labels, test_texts, test_labels
    )
    rnn_model, rnn_history, rnn_accuracy = train_and_evaluate_rnn_model(
        rnn_train_data, rnn_train_labels, rnn_val_data, rnn_val_labels, rnn_test_data, rnn_test_labels
    )
    results['accuracies'].append(rnn_accuracy)
    results['histories'].append(rnn_history)
    
    # BERT Model
    print("\nEvaluating BERT model...")
    bert_train_encodings, bert_train_labels, bert_val_encodings, bert_val_labels, bert_test_encodings, bert_test_labels = preprocess_data_for_bert(
        train_texts, train_labels, val_texts, val_labels, test_texts, test_labels
    )
    bert_model, bert_history, bert_accuracy = train_and_evaluate_bert_model(
        bert_train_encodings, bert_train_labels, bert_val_encodings, bert_val_labels, bert_test_encodings, bert_test_labels
    )
    results['accuracies'].append(bert_accuracy)
    results['histories'].append(bert_history)
    
    # RoBERTa Model
    print("\nEvaluating RoBERTa model...")
    roberta_train_encodings, roberta_train_labels, roberta_val_encodings, roberta_val_labels, roberta_test_encodings, roberta_test_labels = preprocess_data_for_roberta(
        train_texts, train_labels, val_texts, val_labels, test_texts, test_labels
    )
    roberta_model, roberta_history, roberta_accuracy = train_and_evaluate_roberta_model(
        roberta_train_encodings, roberta_train_labels, roberta_val_encodings, roberta_val_labels, roberta_test_encodings, roberta_test_labels
    )
    results['accuracies'].append(roberta_accuracy)
    results['histories'].append(roberta_history)
    
    return results

def visualize_model_comparison(results):
    """
    Create visualizations to compare the performance of different models.
    
    Args:
        results: Dictionary containing model names, accuracies, and other metrics
    """
    print("Creating model comparison visualizations...")
    
    # Accuracy comparison
    plt.figure(figsize=(10, 6))
    plt.bar(results['model_names'], results['accuracies'], color=['blue', 'green', 'red'])
    plt.title('Model Accuracy Comparison')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    for i, v in enumerate(results['accuracies']):
        plt.text(i, v + 0.02, f"{v:.4f}", ha='center')
    plt.tight_layout()
    plt.savefig('model_accuracy_comparison.png')
    plt.close()
    
    # Training history comparison
    plt.figure(figsize=(15, 10))
    
    # Loss comparison
    plt.subplot(2, 1, 1)
    for i, model_name in enumerate(results['model_names']):
        plt.plot(results['histories'][i].history['val_loss'], label=f'{model_name} Validation Loss')
    plt.title('Validation Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy comparison
    plt.subplot(2, 1, 2)
    for i, model_name in enumerate(results['model_names']):
        plt.plot(results['histories'][i].history['val_accuracy'], label=f'{model_name} Validation Accuracy')
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('model_training_comparison.png')
    plt.close()
    
    # Print best model
    best_model_index = np.argmax(results['accuracies'])
    print(f"\nBest Model: {results['model_names'][best_model_index]} with accuracy {results['accuracies'][best_model_index]:.4f}")

def main():
    """
    Main function to orchestrate the model comparison process.
    """
    print("Starting model comparison...")
    
    # Load and evaluate models
    results = load_and_evaluate_models()
    
    # Visualize model comparison
    visualize_model_comparison(results)
    
    print("Model comparison completed!")

if __name__ == "__main__":
    main()