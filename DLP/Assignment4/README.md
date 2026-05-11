# Deep Learning Assignment: RNNs, LSTMs, and Transformers
## Muhammad Sabeeh 23K-0002

This repository contains the implementation for a university assignment on Deep Learning, focusing on sequence modeling using Recurrent Neural Networks (RNNs), Long Short-Term Memory (LSTM) networks, Attention mechanisms, and Transformers.

## Project Structure

The assignment is divided into two main parts:

### Part 1: Image Captioning with RNN/LSTM + Attention
- **File**: `rnn_lstm_captioning.py` (Implementation)
- **Notebook**: `rnn_lstm_captioning.ipynb` (Execution and Analysis)
- **Key Tasks**:
    - Vanilla RNN forward/backward steps.
    - Word embedding layers.
    - Temporal softmax loss (ignoring padding).
    - LSTM architecture with four-gate logic.
    - Dot-product attention mechanism over spatial image features.
    - `CaptioningRNN` model that integrates an image encoder (RegNet) with the sequence decoder.
    - Greedy sampling for caption generation.

### Part 2: Transformer for Arithmetic Operations
- **File**: `transformers.py` (Implementation)
- **Notebook**: `Transformers.ipynb` (Execution and Analysis)
- **Key Tasks**:
    - Vocabulary mapping and preprocessing for arithmetic expressions.
    - Scaled dot-product attention (loop-based and vectorized).
    - `SelfAttention` and `MultiHeadAttention` modules.
    - `LayerNormalization` and `FeedForwardBlock`.
    - `EncoderBlock` and `DecoderBlock` with residual connections.
    - Positional encodings (Linear and Sinusoidal).
    - Full `Transformer` architecture for sequence-to-sequence arithmetic.

## Requirements

- Python 3.7+
- PyTorch
- Matplotlib
- NumPy
- (Optional) CUDA-enabled GPU for faster training.

## How to Run

### Using Jupyter Notebooks (Recommended)
1. Open `rnn_lstm_captioning.ipynb` for the Image Captioning part.
2. Open `Transformers.ipynb` for the Transformer arithmetic part.
3. Follow the cells to load data, run unit tests, and train models.

### Running Implementation Files Directly
The `.py` files contain the logic but not the training loops or data loading (which are in the notebooks). You can run unit tests or import these modules into your own scripts:

```python
from rnn_lstm_captioning import CaptioningRNN
# Initialize and use the model
```

## Implementation Details

- **Optimization**: The `temporal_softmax_loss` is implemented as a single-line vectorized operation for maximum efficiency.
- **Attention**: The spatial attention in the LSTM decoder allows the model to "focus" on specific regions of the image (16 grid cells) while generating each word.
- **Transformers**: The transformer implementation strictly follows the original paper, including the scaled dot-product formula and the alternating sine/cosine positional encodings.
- **Initialization**: Linear layers use a Xavier-style uniform initialization $U(-c, c)$ where $c = \sqrt{6/(D_{in} + D_{out})}$.

## Training Status

The training process was terminated by the user.
- **Part 2 (Transformer Arithmetic)**: Completed. Model saved as `transformer_model.pt`.
- **Part 1 (Attention LSTM)**: Partially completed (2 epochs).

You can review the partial training logs here: [training_log.txt](file:///d:/dlp_assignment4/training_log.txt).
