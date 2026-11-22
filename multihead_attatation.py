import numpy as np
import test as f

def multi_head_attention(X, W_q, W_k, W_v, W_o, num_heads):
    """
    X: Input matrix of shape (seq_len, d_model) -> e.g., (100, 8)
    W_q, W_k, W_v: Weight matrices of shape (d_model, d_model)
    W_o: Output projection weight matrix of shape (d_model, d_model)
    num_heads: Number of attention heads (e.g., 2)
    """
    seq_len, d_model = X.shape
    d_k = d_model // num_heads  # Dimension per head (e.g., 8 / 2 = 4)
    
    # 1. Project inputs to Q, K, V
    Q = np.matmul(X, W_q)  # (seq_len, d_model)
    K = np.matmul(X, W_k)  # (seq_len, d_model)
    V = np.matmul(X, W_v)  # (seq_len, d_model)
    
    # 2. Reshape and transpose to split into multiple heads
    # New shape: (num_heads, seq_len, d_k)
    Q = Q.reshape(seq_len, num_heads, d_k).transpose(1, 0, 2)
    K = K.reshape(seq_len, num_heads, d_k).transpose(1, 0, 2)
    V = V.reshape(seq_len, num_heads, d_k).transpose(1, 0, 2)
    
    head_outputs = []
    
    # 3. Run attention independently for each head
    for i in range(num_heads):
        # Pass each head's slice into your existing Attention function
        out_i = f.Attention(Q[i], K[i], V[i])  # Shape: (seq_len, d_k)
        head_outputs.append(out_i)
        
    # 4. Concatenate all head outputs back together along columns
    # Shape becomes: (seq_len, d_model)
    concatenated = np.concatenate(head_outputs, axis=-1)
    
    # 5. Final linear projection using W_o
    output = np.matmul(concatenated, W_o)
    
    return output

# --- Let's Test It ---
seq_len = 100
d_model = 8
num_heads = 2

# Dummy inputs to verify shapes
X_dummy = np.random.randn(seq_len, d_model)
W_q = np.random.randn(d_model, d_model)
W_k = np.random.randn(d_model, d_model)
W_v = np.random.randn(d_model, d_model)
W_o = np.random.randn(d_model, d_model)

mha_output = multi_head_attention(X_dummy, W_q, W_k, W_v, W_o, num_heads)
print("Multi-Head Attention Output Shape:", mha_output.shape)