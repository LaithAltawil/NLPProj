"""
BERT Model for Emotion Detection

This module provides functions for building, training, and evaluating
a BERT-based model for emotion detection using the GoEmotion dataset.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from transformers import TFBertModel, AdamW

# Import constants from data_utils
from data_utils import MAX_SEQUENCE_LENGTH, NUM_CLASSES

# Constants specific to BERT model
BATCH_SIZE = 32

def build_bert_model():
    """
    Build a BERT-based model for emotion detection.

    Returns:
        Compiled BERT model
    """
    print("Building BERT model...")

    # Load pre-trained BERT model and tokenizer
    bert_model = TFBertModel.from_pretrained('bert-base-uncased')

    # Input layers for BERT
    input_ids = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name="input_ids")
    attention_mask = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name="attention_mask")

    # BERT layer
    bert_output = bert_model(input_ids, attention_mask=attention_mask)[0]

    # Use the [CLS] token output (first token)
    cls_output = bert_output[:, 0, :]

    # Dense layers
    dense_layer = Dense(256, activation='relu')(cls_output)
    dense_layer = Dropout(0.5)(dense_layer)
    dense_layer = Dense(128, activation='relu')(dense_layer)
    dense_layer = Dropout(0.3)(dense_layer)

    # Output layer with sigmoid activation for multi-label classification
    output_layer = Dense(NUM_CLASSES, activation='sigmoid')(dense_layer)

    # Create and compile model
    model = Model(inputs=[input_ids, attention_mask], outputs=output_layer)

    # Use AdamW optimizer which is recommended for BERT
    optimizer = AdamW(learning_rate=2e-5, weight_decay=0.01)

    model.compile(loss='binary_crossentropy',
                 optimizer=optimizer,
                 metrics=['accuracy'])

    print(model.summary())
    return model

def train_and_evaluate_bert_model(train_encodings, train_labels, val_encodings, val_labels, test_encodings, test_labels):
    """
    Train and evaluate the BERT model.

    Args:
        train_encodings, train_labels: Training data and labels
        val_encodings, val_labels: Validation data and labels
        test_encodings, test_labels: Test data and labels

    Returns:
        Trained model and evaluation metrics
    """
    print("Training BERT model...")

    # Build the BERT model
    model = build_bert_model()

    # Define callbacks for early stopping and model checkpointing
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True),
        ModelCheckpoint('bert_model.keras', monitor='val_loss', save_best_only=True)
    ]

    # Create TensorFlow datasets for efficient training
    train_dataset = tf.data.Dataset.from_tensor_slices((
        {'input_ids': train_encodings['input_ids'], 'attention_mask': train_encodings['attention_mask']},
        train_labels
    )).batch(BATCH_SIZE)

    val_dataset = tf.data.Dataset.from_tensor_slices((
        {'input_ids': val_encodings['input_ids'], 'attention_mask': val_encodings['attention_mask']},
        val_labels
    )).batch(BATCH_SIZE)

    test_dataset = tf.data.Dataset.from_tensor_slices((
        {'input_ids': test_encodings['input_ids'], 'attention_mask': test_encodings['attention_mask']},
        test_labels
    )).batch(BATCH_SIZE)

    # Train the model
    history = model.fit(
        train_dataset,
        epochs=5,  # Fewer epochs for BERT due to computational intensity
        validation_data=val_dataset,
        callbacks=callbacks
    )

    # Evaluate the model on test data
    print("Evaluating BERT model on test data...")
    test_loss, test_accuracy = model.evaluate(test_dataset)
    print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}")

    # Make predictions on test data
    y_pred = model.predict(test_dataset)
    y_pred_binary = (y_pred > 0.5).astype(int)  # Convert probabilities to binary predictions

    # Print classification report
    print("Classification Report for BERT Model:")
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
    plt.title('BERT Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('BERT Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('bert_training_history.png')
    plt.close()

    return model, history, test_accuracy
