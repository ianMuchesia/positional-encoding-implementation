# Positional Encoding Revision Notes

## 1. Sinusoidal Positional Encodings (Fixed)

Here is a breakdown of what the encoding matrix heatmap represents:

* **The Y-Axis (Sequence Length)**: Moving from the top (0) to the bottom is like walking through the sequence one token at a time.
* **The Left Side (Local Position / Fast Switches)**:
  * **Visual**: Rapidly alternating colors in the leftmost columns (Dimensions 0 to ~20).
  * **Math**: The frequency denominator is close to $1$, so you are mostly plotting $\sin(pos)$.
  * **Purpose**: Allows the model to detect local, close-range relationships (e.g., "Is Word A next to Word B?").
* **The Right Side (Global Position / Slow Switches)**:
  * **Visual**: Static vertical stripes in the rightmost columns (Dimensions 100 to 126).
  * **Math**: The denominator is huge ($10000$), so you are plotting $\sin(pos / 10000)$. The wave has a very long cycle.
  * **Purpose**: Enables the model to capture global absolute position (e.g., "Is this word near the beginning or the end?").
* **The "Swoosh" Curve**: The visual curve transitioning from the jagged left to the smooth right reflects the frequency slowing down exponentially across the embedding vector dimensions: `torch.pow(10000, even_nums / self.d_model)`.
* **The Vertical Stripes**: Columns are paired in twos (Sine/Cosine pairs) to help the model distinguish different positions on the wave cycle without ambiguity.

---

## 2. Learned Positional Encodings (Dynamic)

* **The Core Concept**: Instead of a fixed mathematical formula, we let the network learn the best position representations from scratch via gradient descent.
* **The `nn.Embedding` Lookup**:
  * It is not a linear projection (`nn.Linear`) and does not do matrix multiplication.
  * It acts as a **simple dictionary lookup**. It takes integer indices (like Row 0, Row 1, Row 2) and returns corresponding vectors.
  * By initializing `nn.Embedding(max_len, d_model)`, we create a table of weights updated during training to learn the optimal embedding for each position index.
* **The Pipeline**:
  1. Generate a 1D tensor of indices representing token positions (e.g., `[0, 1, 2, ..., seq_len-1]`).
  2. Pass these indices to the embedding dictionary to fetch the respective position vectors.
  3. Add these position vectors element-wise to the word/token embeddings.