import numpy as np

def layer_norm(x, eps=1e-6):
    """
    Normalizes the vector across the feature dimension (columns).
    Ensures mean is 0 and variance is 1 for stability.
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)

def add_and_norm(X, sublayer_output):
    """
    1. Residual Connection: Adds original input X to the sublayer's output (e.g., Attention output)
    2. Layer Normalization: Normalizes the combined result
    """
    return layer_norm(X + sublayer_output)

# --- Let's Test It ---
seq_len = 100
d_model = 8

# Dummy inputs (like your input X and your Multi-Head Attention output)
X_dummy = np.random.randn(seq_len, d_model)
attention_out_dummy = np.random.randn(seq_len, d_model)

# Run the Add & Norm block
block_output = add_and_norm(X_dummy, attention_out_dummy)

print("Block Output Shape:", block_output.shape)
print("First word's normalized vector after skip connection:\n", block_output[0])