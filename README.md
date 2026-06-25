# Positional Encoding Implementations in PyTorch

A PyTorch repository implementing both fixed **Sinusoidal Positional Encodings** (from *Attention Is All You Need*, Vaswani et al., 2017) and **Learned Positional Encodings** (as used in GPT-style models), complete with attention modules, training scripts, and comparative experiments.

## Project Structure
- `src/positional_encodings.py`: Core PyTorch modules (`SinusoidalPositionalEncoding` and `LearnedPositionalEncoding`).
- `src/multihead_attention.py`: Custom PyTorch implementation of Multi-Head Attention.
- `src/model.py`: A `ToyModel` integrating embeddings, configurable positional encoding types, Multi-Head Attention, and a final classification layer.
- `src/train.py`: Training loop that measures loss, accuracy, speed, and GPU memory usage across epochs.
- `src/test.py`: Evaluation helper measuring model accuracy.
- `notebooks/pe_visualization.ipynb`: Jupyter notebook for pattern visualization of sinusoidal encodings.
- `notebooks/run_experiments.ipynb`: Jupyter notebook setting up copy and reversal tasks, training models, and generating comparison plots.
- `experiments/`: Saved metrics (JSON), heatmap visualizations, and performance comparison plots (`pe_vs_no_pe.png` and `sinusoidal_vs_learned.png`).
- `math-notes/`: Directory for handwritten mathematical notes (`sinusoidal_pe_formula.md`) and scanned proof images (`images/`).
- `NOTES.md`: Conceptual revision notes summarizing positional encoding properties, the shape alignment trap, memorization, and length generalization.

## Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/ianMuchesia/positional-encoding-implementation.git
   ```
2. Install dependencies:
   ```bash
   pip install torch matplotlib seaborn jupyter
   ```

## Usage

### 1. Model Initialization
You can initialize the `ToyModel` with any of the positional encoding variants:
```python
from src.model import ToyModel

# 1. Sinusoidal Positional Encoding
model_sin = ToyModel(d_model=512, vocab_size=1000, max_len=100, pe_type="sinusoidal")

# 2. Learned Positional Encoding
model_learned = ToyModel(d_model=512, vocab_size=1000, max_len=100, pe_type="learned")

# 3. No Positional Encoding
model_none = ToyModel(d_model=512, vocab_size=1000, max_len=100, pe_type=None)
```

### 2. Training and Testing
Run the training function and evaluate accuracy:
```python
import torch
from src.train import train_pe_variants
from src.test import test_model

# Dummy data
X = torch.randint(0, 1000, (32, 10))
Y = X.clone() # Copy task
X_val = torch.randint(0, 1000, (32, 10))
Y_val = X_val.clone()

# Train
history = train_pe_variants(model_sin, X, X_val, Y, Y_val, pe_type="sinusoidal")

# Test on longer sequence length (e.g. 50) to evaluate generalization
X_test = torch.randint(0, 1000, (32, 50))
Y_test = X_test.clone()
accuracy = test_model(model_sin, X_test, Y_test)
print(f"Accuracy: {accuracy:.2f}%")
```

## Experiment Results (Copy Task Length Generalization)
When trained on sequences of length 10 and tested on sequences of length 50, the positional encoding variants perform as follows on the Copy Task:

1. **No Positional Encoding (89.62% Test Accuracy)**: Best performance. Attention operates as an identity mirror map for word embeddings, and omitting positional noise allows the model to copy perfectly.
2. **Sinusoidal Positional Encoding (80.62% Test Accuracy)**: Generalizes well as the mathematical waves can be calculated dynamically for indices 11-50.
3. **Learned Positional Encoding (75.00% Test Accuracy)**: Lowest accuracy. Since the positional lookup table was only trained up to index 10, evaluating on length 50 adds untrained random weight noise, degrading performance.
