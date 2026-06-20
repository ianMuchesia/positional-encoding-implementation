# Positional Encoding Implementations in PyTorch

A PyTorch repository implementing both fixed **Sinusoidal Positional Encodings** (from *Attention Is All You Need*, Vaswani et al., 2017) and **Learned Positional Encodings** (as used in GPT-style models).

## Project Structure
- `src/positional_encodings.py`: Core PyTorch modules (`SinusoidalPositionalEncoding` and `LearnedPositionalEncoding`).
- `notebooks/pe_visualization.ipynb`: Jupyter notebook for visualizing the sinusoidal positional encoding matrix patterns.
- `experiments/pe_heatmap.png`: Heatmap visualization of the sinusoidal encoding matrix.
- `math-notes/`: Directory for mathematical derivations and notes.
- `NOTES.md`: Comprehensive conceptual and mathematical analysis of positional encodings for revision purposes.

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

### 1. Sinusoidal Positional Encoding (Fixed)
The fixed sinusoidal encoding uses alternating sine and cosine waves of varying frequencies.
```python
import torch
from src.positional_encodings import SinusoidalPositionalEncoding

# Instantiate the encoding module
pe = SinusoidalPositionalEncoding(d_model=128, max_len=1000)

# Input tensor shape: (batch_size, seq_len, d_model)
x = torch.randn(32, 50, 128)
output = pe(x)  # Returns token embeddings + sinusoidal positional encodings
```

### 2. Learned Positional Encoding (Dynamic)
The learned encoding uses an embedding layer (`nn.Embedding`) to learn position-specific vectors via gradient descent.
```python
import torch
from src.positional_encodings import LearnedPositionalEncoding

# Instantiate the encoding module
pe = LearnedPositionalEncoding(d_model=128, max_len=1000)

# Input tensor shape: (batch_size, seq_len, d_model)
x = torch.randn(32, 50, 128)
output = pe(x)  # Returns token embeddings + learned positional encodings
```
