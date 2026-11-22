import numpy as np

def relu(x):
    return np.maximum(0, x)

def feed_forward_network(X, W1, b1, W2, b2):
    """
    X: shape (seq_len, d_model) -> e.g., (100, 8)
    W1: shape (d_model, hidden_dim) -> e.g., (8, 32)
    b1: shape (hidden_dim,) -> e.g., (32,)
    W2: shape (hidden_dim, d_model) -> e.g., (32, 8)
    b2: shape (d_model,) -> e.g., (8,)
    """
    # 1. Expand layer + ReLU activation
    hidden = relu(np.matmul(X, W1) + b1)  # Shape: (seq_len, hidden_dim)
    
    # 2. Project back to original d_model dimension
    output = np.matmul(hidden, W2) + b2  # Shape: (seq_len, d_model)
    
    return output

# --- Let's Test It ---
seq_len = 100
d_model = 8
hidden_dim = 32 # 4 * d_model

# Dummy inputs and weights
X_dummy = np.random.randn(seq_len, d_model)
W1 = np.random.randn(d_model, hidden_dim)
b1 = np.random.randn(hidden_dim)
W2 = np.random.randn(hidden_dim, d_model)
b2 = np.random.randn(d_model)

# 1. Run FFN
ffn_out = feed_forward_network(X_dummy, W1, b1, W2, b2)

# 2. Apply the second Residual Connection & Layer Norm (just like after attention!)
# (Remember your `add_and_norm` function from earlier?)
# final_block_out = add_and_norm(X_dummy, ffn_out)

print("FFN Output Shape:", ffn_out.shape)