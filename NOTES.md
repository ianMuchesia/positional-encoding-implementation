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

---

## 3. Architecture & Training Lessons

### A. The "Silent Trap" of Shapes
* **The Architecture**: Multi-Head Attention (MHA) aggregates contextual representations across words but retains the feature size (`d_model`). A final `nn.Linear` classifier is required to project the output from `d_model` back to `vocab_size` to calculate the vocabulary probability distribution.
* **The CrossEntropyLoss Trap**: If you accidentally pass the `d_model` dimension (e.g. 512) into the loss function instead of the `vocab_size` (e.g. 100), PyTorch will not crash as long as your target labels fit within the larger dimension size.
* **The Consequence**: Passing output dimensions that don't match the true vocabulary size mathematically inflates the denominator of the Softmax equation. This artificially depresses probabilities, inflates the loss, and sends chaotic penalty gradients back through the network, degrading the learned embeddings.
* **Key Takeaway**: Language modeling is fundamentally a multi-class classification task; the classification projection dimension must strictly match the vocabulary size.

### B. The Memorization Experiment (Overfitting 1 Batch)
To verify the capability of each positional encoding variant, we trained on a single batch of 32 sentences (length 10) to test memorization:
* **Learned PE (100% Accuracy)**: Reached perfect training accuracy by memorizing the dedicated position representations for the 32 sentences.
* **Sinusoidal PE (99.6% Accuracy)**: Learned the task successfully but required slightly more effort to align with the predefined mathematical wave representations.
* **No PE (~60% Accuracy)**: Hit a strict mathematical ceiling. Without positional signals, the model could not resolve the positions of identical words occurring at different indices in the sentence, resulting in confusion.

---

## 4. Length Generalization & Evaluation Results

We trained models on sequence lengths of 10 and evaluated them on longer sequences of length 50 to see how they generalize.

### A. The Sequence Reversal Generalization Trap
When trained on the **Sequence Reversal** task, all models crashed to random guessing (~1.8% accuracy) when tested on length 50.
* **Why it failed**: The reversal task forces the model to learn a relative position rule dependent on the specific length (e.g., "look exactly 9 positions to my right"). When evaluated on length 50, this static rule fails completely because the target position is now much further away.

### B. The Copy Task Verdict
To measure generalization without length-dependent target offsets, we switched to the **Copy Task** ($Y = X$), where the rule is simply "token at index $i$ attends to itself."

Results on testing with sequence length 50:
1. 🥇 **No PE (89.6% Accuracy)**: Performed the best. Because self-attention acts as a perfect mirror, a word representation is always most similar to itself. Adding positional signals actually acted as noise here.
2. 🥈 **Sinusoidal PE (80.6% Accuracy)**: Generalizes well because it uses fixed mathematical waves. The model can dynamically compute sinusoidal positional vectors for indices 11 to 50 even though it has never seen them during training.
3. 🥉 **Learned PE (75.0% Accuracy)**: Performed the worst. Because the embedding lookup table was only trained on indices 0 to 9, the indices from 10 to 49 contained random, untrained initialization weights. Adding this untrained noise to the token embeddings degraded the model's performance.

---

## 5. Visualizations & Scanned Mathematical Notes

### A. Experiment Visualizations
Two plots were generated under `experiments/` to visually analyze training performance over 40 epochs:
* **[experiments/pe_vs_no_pe.png](file:///home/msodoki/Desktop/Mathematics/positional-encoding-impl/experiments/pe_vs_no_pe.png)**: Shows training and validation performance curves when comparing models with positional encodings (PE) versus models without positional encodings (No PE).
* **[experiments/sinusoidal_vs_learned.png](file:///home/msodoki/Desktop/Mathematics/positional-encoding-impl/experiments/sinusoidal_vs_learned.png)**: Compares the training/validation loss and accuracy trajectories of Sinusoidal vs. Learned positional encodings.

### B. Scanned Handwritten Math Notes
Scanned pages containing detailed handwritten derivations, proofs, and explanations of the sinusoidal positional encoding math are available in the project:
* Refer to **[math-notes/sinusoidal_pe_formula.md](file:///home/msodoki/Desktop/Mathematics/positional-encoding-impl/math-notes/sinusoidal_pe_formula.md)** to view links to the notes:
  * [Page 1](file:///home/msodoki/Desktop/Mathematics/positional-encoding-impl/math-notes/images/page-1.jpeg): Formulas, setup, and dimension coordinates.
  * [Page 2](file:///home/msodoki/Desktop/Mathematics/positional-encoding-impl/math-notes/images/page-2.jpeg): Frequency variations and wavelengths.
  * [Page 3](file:///home/msodoki/Desktop/Mathematics/positional-encoding-impl/math-notes/images/page-3.jpeg): Rotation properties and relative position transformation proofs.

