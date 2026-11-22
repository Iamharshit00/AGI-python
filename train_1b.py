import numpy as np
import pandas as pd
import test  # Imports your custom test.py module

# --- 1. 1B Parameter Hyperparameters ---
seq_len = 512            # Longer context window for a 1B model
d_model = 1792           # Width of the vector space
vocab_size = 32000       # Large standard vocabulary size
num_heads = 16           # Number of attention heads (1792 / 16 = 112 dim per head)
hidden_dim = 7168        # 4 * d_model for the Feed-Forward Network
num_layers = 24          # 24 stacked Transformer layers
learning_rate = 0.0001
epochs = 1
total_samples = 100      # Keep sample count low initially to test memory limits

print("Loading dataset...")
df = pd.read_csv('llm_dataset_1.csv')

print("Initializing 1-Billion Parameter Weight Matrices...")
print(f"Target Architecture: d_model={d_model}, layers={num_layers}, vocab={vocab_size}")

# --- 2. Initialize 1B Weights across 24 Layers ---
# Embedding & Positional Encoding
embedding_table = np.random.randn(vocab_size, d_model) * 0.02
pos_encoding = test.get_positional_encoding(seq_len, d_model)

# Layer-by-layer weights storage
layers_weights = []
for layer_idx in range(num_layers):
    layer = {
        "W_q": np.random.randn(d_model, d_model) * 0.02,
        "W_k": np.random.randn(d_model, d_model) * 0.02,
        "W_v": np.random.randn(d_model, d_model) * 0.02,
        "W_o": np.random.randn(d_model, d_model) * 0.02,
        "W1": np.random.randn(d_model, hidden_dim) * 0.02,
        "b1": np.zeros(hidden_dim),
        "W2": np.random.randn(hidden_dim, d_model) * 0.02,
        "b2": np.zeros(d_model)
    }
    layers_weights.append(layer)

# LM Head
W_lm = np.random.randn(d_model, vocab_size) * 0.02
b_lm = np.zeros(vocab_size)

print("Weights initialized successfully! Starting training loop...")

# --- 3. Training Loop ---
for epoch in range(epochs):
    total_loss = 0
    for i in range(total_samples):
        row = df.iloc[i]
        text = str(row['prompt']) + " " + str(row['answer'])
        target_token_id = abs(hash(text)) % vocab_size
        
        # --- FORWARD PASS (Multi-Layer) ---
        tokens = test.tokenize(text)
        padded_tokens = test.padding(tokens, seq_len)
        
        x = embedding_table[padded_tokens] + pos_encoding  # (seq_len, d_model)
        
        # Pass through all 24 layers sequentially
        for layer_idx, layer in enumerate(layers_weights):
            # Multi-Head Attention + Residual Connection & Layer Norm
            mha_out = test.multi_head_attention(x, layer["W_q"], layer["W_k"], layer["W_v"], layer["W_o"], num_heads)
            x = test.layer_norm(x + mha_out)
            
            # Feed-Forward Network + Residual Connection & Layer Norm
            ffn_out = test.feed_forward_network(x, layer["W1"], layer["b1"], layer["W2"], layer["b2"])
            x = test.layer_norm(x + ffn_out)
            
        # Final LM Head Projection
        logits = np.matmul(x, W_lm) + b_lm  # (seq_len, vocab_size)
        last_token_logits = logits[-1]
        
        # --- LOSS & BACKPROP (LM Head approximation) ---
        exp_logits = np.exp(last_token_logits - np.max(last_token_logits))
        probs = exp_logits / np.sum(exp_logits)
        
        loss = -np.log(probs[target_token_id] + 1e-9)
        total_loss += loss
        
        d_logits = probs.copy()
        d_logits[target_token_id] -= 1.0
        
        dW_lm = np.outer(x[-1], d_logits)
        db_lm = d_logits
        
        # Weight Update
        W_lm -= learning_rate * dW_lm
        b_lm -= learning_rate * db_lm
        
    print(f"Epoch {epoch + 1} completed! Average Loss: {total_loss / total_samples:.4f}")

# --- 4. Save Weights via test module ---
weights_dict = {
    "embedding_table": embedding_table,
    "pos_encoding": pos_encoding,
    "W_lm": W_lm, "b_lm": b_lm
}
# Save base components
test.save_weights_to_txt(weights_dict, folder="transformer_1b_weights")
print("1B Model training complete and weights saved successfully!")