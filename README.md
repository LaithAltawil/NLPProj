# Emotion Detection in Text: RNNs vs. Transformers

## Overview
This project compares traditional Recurrent Neural Networks (RNNs) with modern Transformer-based models for emotion detection in text, with a focus on social media content. The research explores how these architectures handle the unique challenges of informal language, including sarcasm, mixed sentiments, and linguistic noise.

## Key Features

- **Comparative Analysis**: Direct performance comparison between RNN (LSTM/GRU) and Transformer (BERT) architectures
- **Social Media Focus**: Specialized evaluation on noisy, user-generated text with informal language patterns
- **Comprehensive Metrics**: Evaluation across accuracy, F1-score, computational efficiency, and memory requirements
- **Practical Insights**: Identification of optimal use cases for each architecture based on deployment constraints

## Technical Approach

### Data Processing
- Specialized text cleaning for social media content (handling emojis, slang, typos)
- Emotion label normalization across datasets

### Model Architectures
- **RNN Baseline**: Bidirectional LSTM with attention mechanism
- **Transformer Model**: Fine-tuned BERT-base with emotion classification head

### Evaluation Framework
- Standard metrics (precision, recall, F1) across emotion categories
- Computational efficiency benchmarks (training time, inference speed)
- Error analysis on challenging cases (sarcasm, ambiguous expressions)
