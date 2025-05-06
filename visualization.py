"""
Visualization Utilities for Emotion Detection Models

This module provides functions for visualizing and comparing the performance
of different emotion detection models.
"""

import matplotlib.pyplot as plt

def compare_models(rnn_history, bert_history, roberta_history, rnn_accuracy, bert_accuracy, roberta_accuracy):
    """
    Compare the performance of RNN, BERT, and RoBERTa models.

    Args:
        rnn_history, bert_history, roberta_history: Training history of the models
        rnn_accuracy, bert_accuracy, roberta_accuracy: Test accuracy of the models
    """
    print("\nModel Comparison:")
    print(f"RNN Model Test Accuracy: {rnn_accuracy:.4f}")
    print(f"BERT Model Test Accuracy: {bert_accuracy:.4f}")
    print(f"RoBERTa Model Test Accuracy: {roberta_accuracy:.4f}")

    # Plot comparison
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(rnn_history.history['val_loss'], label='RNN Validation Loss')
    plt.plot(bert_history.history['val_loss'], label='BERT Validation Loss')
    plt.plot(roberta_history.history['val_loss'], label='RoBERTa Validation Loss')
    plt.title('Validation Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(rnn_history.history['val_accuracy'], label='RNN Validation Accuracy')
    plt.plot(bert_history.history['val_accuracy'], label='BERT Validation Accuracy')
    plt.plot(roberta_history.history['val_accuracy'], label='RoBERTa Validation Accuracy')
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()

    # Determine the best model
    best_accuracy = max(rnn_accuracy, bert_accuracy, roberta_accuracy)
    if best_accuracy == rnn_accuracy:
        print("The RNN model outperformed the BERT and RoBERTa models.")
    elif best_accuracy == bert_accuracy:
        print("The BERT model outperformed the RNN and RoBERTa models.")
    else:
        print("The RoBERTa model outperformed the RNN and BERT models.")
