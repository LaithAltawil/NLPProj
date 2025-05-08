"""
RNN Model for Emotion Detection

This module provides functions for building, training, and evaluating
an RNN-based model for emotion detection using the GoEmotion dataset.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Import constants from data_utils
from data_utils import MAX_SEQUENCE_LENGTH, NUM_CLASSES

# Constants specific to RNN model
EMBEDDING_DIM = 300
BATCH_SIZE = 32
EPOCHS = 10

def build_rnn_model(word_index):
    """
    Build an RNN-based model for emotion detection.

    Args:
        word_index: Dictionary mapping words to indices or an integer representing vocabulary size

    Returns:
        Compiled an RNN model
    """
    print("Building RNN model...")

    # Input layer
    input_layer = Input(shape=(MAX_SEQUENCE_LENGTH,))

    # Determine vocabulary size
    if isinstance(word_index, int):
        vocab_size = word_index
    else:
        vocab_size = len(word_index) + 1

    # Embedding layer
    embedding_layer = Embedding(vocab_size,
                               EMBEDDING_DIM,
                               input_length=MAX_SEQUENCE_LENGTH)(input_layer)

    # Bidirectional LSTM layers
    # Use implementation=1 to avoid CuDNN compatibility issues
    lstm_layer = Bidirectional(LSTM(128, return_sequences=True, implementation=1))(embedding_layer)
    lstm_layer = Bidirectional(LSTM(64, implementation=1))(lstm_layer)

    # Dense layers with dropout for regularization
    dense_layer = Dense(128, activation='relu')(lstm_layer)
    dense_layer = Dropout(0.5)(dense_layer)
    dense_layer = Dense(64, activation='relu')(dense_layer)
    dense_layer = Dropout(0.3)(dense_layer)

    # Output layer with sigmoid activation for multi-label classification
    output_layer = Dense(NUM_CLASSES, activation='sigmoid')(dense_layer)
    output_layer = Dense(NUM_CLASSES, activation='sigmoid')(dense_layer)

    # Create and compile a model
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='binary_crossentropy',
                 optimizer='adam',
                 metrics=['accuracy'])

    print(model.summary())
    return model

def train_and_evaluate_rnn_model(train_data, train_labels, val_data, val_labels, test_data, test_labels, word_index=None):
    """
    Train and evaluate the RNN model.

    Args:
        train_data, train_labels: Training data and labels
        val_data, val_labels: Validation data and labels
        test_data, test_labels: Test data and labels
        word_index: Dictionary mapping words to indices (optional)

    Returns:
        Trained model and evaluation metrics
    """
    print("Training RNN model...")

    # Build the RNN model
    if word_index is None:
        # Fallback if word_index is not provided
        vocab_size = len(np.unique(train_data)) + 1
        model = build_rnn_model(vocab_size)
    else:
        model = build_rnn_model(word_index)

    # Define callbacks for early stopping and model checkpointing
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ModelCheckpoint('rnn_model.keras', monitor='val_loss', save_best_only=True)
    ]

    # Train the model
    history = model.fit(
        train_data, train_labels,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(val_data, val_labels),
        callbacks=callbacks
    )

    # Evaluate the model on test data
    print("Evaluating RNN model on test data...")
    test_loss, test_accuracy = model.evaluate(test_data, test_labels)
    print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}")

    # Make predictions on test data
    y_pred = model.predict(test_data)
    y_pred_binary = (y_pred > 0.5).astype(int)  # Convert probabilities to binary predictions

    # Print classification report
    print("Classification Report for RNN Model:")
    # For multi-label classification, we need to handle the report differently
    # Here's a simplified approach for demonstration
    for i in range(NUM_CLASSES):
        print(f"Emotion {i}:")
        print(classification_report(test_labels[:, i], y_pred_binary[:, i], zero_division=0))

    # Plot training history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('RNN Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('RNN Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('rnn_training_history.png')
    plt.close()

    return model, history, test_accuracy
