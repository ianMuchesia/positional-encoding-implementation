# Mathematical and Conceptual Analysis of Positional Encodings

Positional encodings are a fundamental component of Transformer-based models. Since self-attention treats tokens as an unordered set (permutation invariance), sequence order must be explicitly injected into the input representation. 

This document analyzes two primary approaches to positional encoding: **Fixed Sinusoidal Positional Encoding** and **Learned Positional Encoding**.

---

## 1. Fixed Sinusoidal Positional Encoding

Introduced in *Attention Is All You Need* (Vaswani et al., 2017), this method computes fixed positional representations using trigonometric functions of varying frequencies.

### Mathematical Formulation
For a token position $pos$ and dimension index $i$ in an embedding space of size $d_{\text{model}}$:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

Where:
* $pos \in [0, \text{max\_len}-1]$ represents the position of the token in the sequence.
* $i \in [0, d_{\text{model}}/2 - 1]$ represents the dimension index.
* $10000^{2i/d_{\text{model}}}$ is the scaling factor (denominator) regulating the wave's frequency.

---

### Structural Analysis & Visual Observations

A heatmap visualization of the sinusoidal encoding matrix (e.g., $1000 \times 128$) reveals key properties of this mathematical representation:

```
          Left Side (High Freq)                 Right Side (Low Freq)
         ┌──────────────────────────────────────────────────────────┐
Pos 0    │ █ ░ █ ░ █ ░ █ ░ █                      │                 │
         │ ░ █ ░ █ ░ █ ░ █ ░                      │                 │
         │ █ ░ █ ░ █ ░ █ ░ █      Swoosh          │                 │
         │ ░ █ ░ █ ░ █ ░ █ ░      Curve           │                 │
         │                        \               │                 │
         │                         \              │                 │
         │                          \             │                 │
         │                           ▼            │                 │
Pos 999  │                                        │                 │
         └──────────────────────────────────────────────────────────┘
          Dim 0                                                     Dim d_model
```

#### A. Sequence Length (Y-Axis)
The Y-axis represents the progression of word positions in the input document (e.g., from index $0$ to $\text{max\_len}-1$). Moving vertically down the matrix corresponds to scanning through a document word-by-word.

#### B. Local Position Encoding (Left Side / Lower Dimensions)
* **Visual Pattern**: Rapid alternating oscillations of color (violently flashing between $+1$ and $-1$).
* **Mechanism**: In the early dimensions (e.g., $i \approx 0$), the denominator $10000^{2i/d_{\text{model}}}$ is close to $1$. The wave function simplifies to $\sin(pos)$, which has a very short wavelength.
* **Purpose**: High-frequency dimensions change rapidly from word to word. This allows the model's self-attention mechanism to detect local relationships and fine-grained offsets (e.g., "Is Word A immediately adjacent to Word B?").

#### C. Global Position Encoding (Right Side / Higher Dimensions)
* **Visual Pattern**: Broad, static vertical bands of color that do not repeat across the maximum sequence length.
* **Mechanism**: For dimensions near $d_{\text{model}}$, the denominator is massive ($10000$). The formula approximates $\sin(pos / 10000)$. Over a sequence length of 1,000, this wave completes less than a quarter of a full cycle.
* **Purpose**: Low-frequency dimensions change slowly, providing monotonic signals. This enables the model to resolve absolute global positions (e.g., "Is this word located near the beginning, middle, or end of the document?").

#### D. The Exponential Frequency Sweep ("Swoosh" Curve)
* **Visual Pattern**: A smooth, curved transition boundary separating the highly active left side from the stable right side.
* **Mechanism**: The scaling factor $10000^{2i/d_{\text{model}}}$ scales the wavelength exponentially as dimension index $i$ increases. The curve is the visual manifestation of this exponential rate of change.

#### E. Sine-Cosine Pairing (Vertical Stripes)
* **Visual Pattern**: Noticeable pairing of adjacent columns.
* **Mechanism**: Alternating columns compute sine and cosine functions of the same frequency. 
* **Purpose**: This pairing ensures that the model can represent relative position as a linear function. Specifically, for any fixed offset $k$, $PE_{pos+k}$ can be computed as a linear transformation of $PE_{pos}$ via a rotation matrix:
  $$PE_{pos+k} = R_k \cdot PE_{pos}$$
  This makes it mathematically straightforward for the attention mechanism to learn relative distance patterns.

---

## 2. Learned Positional Encoding

Rather than using predefined trigonometric formulas, learned positional encodings treat sequence positions as parameters to be optimized during training.

### Core Concept
Each sequence index (from $0$ to $\text{max\_len}-1$) is assigned a dedicated embedding vector of size $d_{\text{model}}$. These vectors are initialized randomly and updated via backpropagation along with the rest of the model's parameters.

### PyTorch Implementation via `nn.Embedding`
In PyTorch, this is implemented using `nn.Embedding(max_len, d_model)`.
* **Concept**: `nn.Embedding` is conceptually a **lookup table** (dictionary) rather than a linear projection layer. It does not perform matrix multiplications.
* **Discrete Indexing**: The embedding layer requires integer values as keys (e.g., position index $0, 1, 2, \dots$) rather than floating-point decimals.
* **Parameter Matrix**: It instantiates a weight matrix of shape $(\text{max\_len}, d_{\text{model}})$. Over the course of training, gradient descent updates these weights to represent position-specific semantic footprints.

### Execution Pipeline
1. **Index Generation**: For an input sequence of length $L$, generate a 1D tensor of integers representing the position index of each token: 
   $$\mathbf{t} = [0, 1, 2, \dots, L-1]$$
2. **Lookup Query**: Pass the index tensor to the positional embedding layer to retrieve the corresponding rows:
   $$E_{\text{pos}} = \text{pos\_emb}(\mathbf{t}) \in \mathbb{R}^{L \times d_{\text{model}}}$$
3. **Combination**: Add the retrieved positional embeddings element-wise to the standard token/word embeddings:
   $$X_{\text{final}} = X_{\text{word}} + E_{\text{pos}}$$

---

## 3. Comparison of Fixed and Learned Encodings

| Feature | Fixed Sinusoidal Encoding | Learned Positional Encoding |
| :--- | :--- | :--- |
| **Parameters** | 0 (computed on-the-fly or cached) | $max\_len \times d_{model}$ (learnable parameters) |
| **Extrapolation** | Can theoretically generalize to sequence lengths $> max\_len$ (though performance may degrade). | Cannot extrapolate. The model cannot handle sequences longer than $max\_len$ because no embeddings exist for indices $\ge max\_len$. |
| **Relative Distance Representation** | Exploits sine-cosine rotation properties to mathematically preserve relative distances. | Relies on the optimizer to learn relative spatial relations from data. |
| **Data Requirements** | Works well even with limited data because geometric relations are hardcoded. | Requires sufficient training data to learn optimal representations for each position index. |
| **Flexibility** | Static; does not adapt to the specific dataset or downstream task. | Dynamic; adapts parameters directly to capture dataset-specific spatial hierarchies. |