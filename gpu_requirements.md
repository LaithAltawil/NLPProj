# GPU Training Requirements

This document outlines the requirements and changes made to enable training on NVIDIA GPUs.

## Changes Made

The following changes have been made to enable GPU training:

1. Added GPU configuration code in `Main.py`:
   - Detection of available GPUs
   - Memory growth configuration to avoid memory allocation issues
   - Setting the CUDA_VISIBLE_DEVICES environment variable
   - Logging of GPU information

## Requirements

To train the models on an NVIDIA GPU, you need:

1. **Hardware Requirements**:
   - An NVIDIA GPU with CUDA support
   - Sufficient GPU memory (at least 8GB recommended for BERT and RoBERTa models)

2. **Software Requirements**:
   - NVIDIA GPU drivers
   - CUDA Toolkit (compatible with your TensorFlow version)
   - cuDNN library (compatible with your CUDA version)

3. **Python Package Requirements**:
   - TensorFlow with GPU support
   - Hugging Face Transformers library
   - Other dependencies as listed in the project requirements

## Installation Guide

1. **Install NVIDIA GPU Drivers**:
   - Download and install the latest drivers from the [NVIDIA website](https://www.nvidia.com/Download/index.aspx)

2. **Install CUDA Toolkit**:
   - Download and install CUDA from the [NVIDIA CUDA website](https://developer.nvidia.com/cuda-downloads)
   - Make sure to install a version compatible with your TensorFlow version

3. **Install cuDNN**:
   - Download cuDNN from the [NVIDIA cuDNN website](https://developer.nvidia.com/cudnn)
   - Follow the installation instructions provided by NVIDIA

4. **Install TensorFlow with GPU Support**:
   ```bash
   pip install tensorflow
   ```
   
   Note: Recent versions of TensorFlow automatically include GPU support if CUDA is available.

5. **Verify GPU Support**:
   ```python
   import tensorflow as tf
   print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
   ```

## Troubleshooting

If you encounter issues with GPU training:

1. **Check GPU Detection**:
   - The script will print information about detected GPUs at startup
   - If no GPUs are detected, check your drivers and CUDA installation

2. **Memory Issues**:
   - If you encounter out-of-memory errors, try reducing batch sizes in the model files
   - The code already enables memory growth to help with memory management

3. **Version Compatibility**:
   - Ensure your CUDA, cuDNN, and TensorFlow versions are compatible
   - Refer to the [TensorFlow compatibility matrix](https://www.tensorflow.org/install/source#gpu)

4. **Other Issues**:
   - Check the TensorFlow GPU guide: https://www.tensorflow.org/guide/gpu
   - Check NVIDIA documentation for your specific GPU model