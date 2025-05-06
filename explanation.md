# Detailed Explanation of Emotion Detection System

## Project Overview

This project implements an emotion detection system using two different approaches:
1. A Recurrent Neural Network (RNN) based model
2. A BERT Transformer-based model

Both models are trained and evaluated on the GoEmotion dataset, which contains text samples labeled with 28 different emotion categories. The project is structured in a modular way, with separate files for different components of the system.

## Project Structure

The project consists of the following files:

1. **Main.py**: The entry point of the program that orchestrates the entire process
2. **data_utils.py**: Contains functions for loading and preprocessing the GoEmotion dataset
3. **rnn_model.py**: Implements the RNN-based emotion detection model
4. **bert_model.py**: Implements the BERT-based emotion detection model
5. **visualization.py**: Contains functions for visualizing and comparing model performance

## Detailed Explanation of Each Module

### 1. data_utils.py

This module handles all data-related operations, including loading the GoEmotion dataset and preprocessing it for both the RNN and BERT models.

#### Key Components:

- **Constants**:
  - `MAX_SEQUENCE_LENGTH = 100`: Defines the maximum length of text sequences
  - `NUM_CLASSES = 28`: The number of emotion categories in the GoEmotion dataset

- **load_goemotion_dataset()**: 
  - Loads the GoEmotion dataset using the Hugging Face datasets library
  - Extracts train, validation, and test sets
  - Returns text and label data for each set

- **process_labels(label_lists)**:
  - Converts the multi-label format of GoEmotion (where each sample can have multiple emotion labels) into a multi-hot encoded format
  - Creates a binary matrix where each row represents a sample and each column represents an emotion category
  - For each sample, sets the corresponding emotion indices to 1

- **preprocess_data_for_rnn()**:
  - Creates and fits a tokenizer on the training data
  - Converts texts to sequences of token indices
  - Pads sequences to ensure uniform length
  - Processes labels using the process_labels function
  - Returns preprocessed data ready for the RNN model

- **preprocess_data_for_bert()**:
  - Uses the BERT tokenizer to encode the text data
  - Handles truncation and padding to ensure uniform length
  - Processes labels using the process_labels function
  - Returns preprocessed data ready for the BERT model

### 2. rnn_model.py

This module implements the RNN-based emotion detection model using bidirectional LSTM layers.

#### Key Components:

- **Constants**:
  - `EMBEDDING_DIM = 300`: Dimension of word embeddings
  - `BATCH_SIZE = 32`: Batch size for training
  - `EPOCHS = 10`: Maximum number of training epochs

- **build_rnn_model(word_index)**:
  - Creates an RNN model architecture with:
    - An embedding layer to convert token indices to dense vectors
    - Two bidirectional LSTM layers (128 and 64 units)
    - Two dense layers with dropout for regularization
    - An output layer with sigmoid activation for multi-label classification
  - Compiles the model with binary crossentropy loss and Adam optimizer

- **train_and_evaluate_rnn_model()**:
  - Builds the RNN model
  - Sets up callbacks for early stopping and model checkpointing
  - Trains the model on the training data
  - Evaluates the model on test data
  - Generates classification reports for each emotion category
  - Plots training history (loss and accuracy)
  - Returns the trained model, training history, and test accuracy

### 3. bert_model.py

This module implements the BERT-based emotion detection model using a pre-trained BERT transformer.

#### Key Components:

- **Constants**:
  - `BATCH_SIZE = 32`: Batch size for training

- **build_bert_model()**:
  - Loads a pre-trained BERT model ('bert-base-uncased')
  - Creates a model architecture with:
    - Input layers for token IDs and attention mask
    - The pre-trained BERT layer
    - Extraction of the [CLS] token output
    - Two dense layers with dropout for regularization
    - An output layer with sigmoid activation for multi-label classification
  - Compiles the model with binary crossentropy loss and AdamW optimizer (recommended for BERT)

- **train_and_evaluate_bert_model()**:
  - Builds the BERT model
  - Sets up callbacks for early stopping and model checkpointing
  - Creates TensorFlow datasets for efficient training
  - Trains the model on the training data (with fewer epochs due to computational intensity)
  - Evaluates the model on test data
  - Generates classification reports for each emotion category
  - Plots training history (loss and accuracy)
  - Returns the trained model, training history, and test accuracy

### 4. visualization.py

This module provides functions for visualizing and comparing the performance of the RNN and BERT models.

#### Key Components:

- **compare_models(rnn_history, bert_history, rnn_accuracy, bert_accuracy)**:
  - Prints the test accuracy of both models
  - Creates plots comparing:
    - Validation loss of both models
    - Validation accuracy of both models
  - Saves the comparison plot as 'model_comparison.png'
  - Determines and prints which model performed better

### 5. Main.py

This is the entry point of the program that orchestrates the entire process.

#### Key Components:

- **Imports from custom modules**:
  - Imports functions from data_utils.py, rnn_model.py, bert_model.py, and visualization.py

- **Random seed setting**:
  - Sets random seeds for reproducibility

- **main() function**:
  - Loads the GoEmotion dataset
  - Preprocesses data for the RNN model
  - Trains and evaluates the RNN model
  - Preprocesses data for the BERT model
  - Trains and evaluates the BERT model
  - Compares the performance of both models

## How the Modules Interact

1. **Main.py** calls functions from other modules in a sequential manner:
   - First, it calls `load_goemotion_dataset()` from data_utils.py to load the dataset
   - Then, it calls `preprocess_data_for_rnn()` from data_utils.py to preprocess data for the RNN model
   - Next, it calls `train_and_evaluate_rnn_model()` from rnn_model.py to train and evaluate the RNN model
   - Similarly, it calls `preprocess_data_for_bert()` and `train_and_evaluate_bert_model()` for the BERT model
   - Finally, it calls `compare_models()` from visualization.py to compare the performance of both models

2. **data_utils.py** provides preprocessed data to both rnn_model.py and bert_model.py

3. **rnn_model.py** and **bert_model.py** return trained models and evaluation metrics to Main.py

4. **visualization.py** receives training history and accuracy from both models to create comparison visualizations

## Key Design Decisions and Rationale

1. **Modular Structure**:
   - The code is organized into separate modules based on functionality
   - This makes the code more maintainable, readable, and reusable
   - Each module has a clear responsibility and can be modified independently

2. **Multi-label Classification**:
   - The GoEmotion dataset is a multi-label dataset (each text can have multiple emotion labels)
   - We use sigmoid activation in the output layer instead of softmax
   - We use binary crossentropy loss instead of categorical crossentropy

3. **RNN Model Architecture**:
   - Uses bidirectional LSTM layers to capture context from both directions
   - Includes dropout layers to prevent overfitting
   - Uses early stopping to prevent overfitting and save training time

4. **BERT Model Architecture**:
   - Uses a pre-trained BERT model to leverage transfer learning
   - Fine-tunes the model for the specific task of emotion detection
   - Uses the AdamW optimizer which is recommended for BERT
   - Uses fewer training epochs due to the computational intensity of BERT

5. **Data Preprocessing**:
   - Different preprocessing approaches for RNN and BERT models
   - RNN uses a simple tokenizer and word embeddings
   - BERT uses its own specialized tokenizer that handles subword tokenization

6. **Evaluation and Visualization**:
   - Comprehensive evaluation including accuracy and classification reports
   - Visualization of training history to understand model learning
   - Direct comparison of models to determine which approach is better

## Conclusion

This project demonstrates two different approaches to emotion detection: an RNN-based approach and a BERT-based approach. The modular structure of the code makes it easy to understand, maintain, and extend. The comprehensive evaluation and visualization help in comparing the performance of both approaches and understanding their strengths and weaknesses.