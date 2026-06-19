# Sinusoidal Positional Encoding Implementation

A PyTorch implementation of Sinusoidal Positional Encodings as described in the seminal paper *Attention Is All You Need* (Vaswani et al., 2017).

## Project Structure
- `src/positional_encodings.py`: Core PyTorch `nn.Module` implementing the encoding matrix.
- `notebooks/pe_visualization.ipynb`: Jupyter notebook for pattern visualization.
- `experiments/pe_heatmap.png`: Heatmap visualization of the encoding matrix.
- `math-notes/`: Directory for mathematical derivations.

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
Import the module in your PyTorch pipeline:
```python
from src.positional_encodings import SinusoidalPositionalEncoding

pe = SinusoidalPositionalEncoding(d_model=128, max_len=1000)
output = pe(x)  # Adds encodings to input x of shape (batch, seq_len, d_model)
```
