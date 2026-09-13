import re
import numpy as np

def tokenize(text):
    token_map = {}
    pattern = r'^"(.+?)",(\d+)\s*$'

    with open("output.txt", "r", encoding="utf-8") as f:
        for line in f:
            match = re.match(pattern, line)
            if match:
                token_map[match.group(1)] = int(match.group(2))

    unk_id = token_map.get("<unk>", 3)
    tokens = []
    words = text.split(" ")
    
    for i, word in enumerate(words):
        if word in token_map:
            # 1. Exact match found
            tokens.append(token_map[word])
        elif word.lower() in token_map:
            # 2. Match found by turning it lowercase (handles "Hello" -> "hello")
            tokens.append(token_map[word.lower()])
        else:
            # 3. Fallback: break unknown words/punctuation down character-by-character!
            for char in word:
                tokens.append(token_map.get(char, unk_id))
        
        # Add space between words
        if i < len(words) - 1:
            tokens.append(token_map.get(" ", 98))

    return np.array(tokens)
def detokenize(tokens):
    token_map = {}
    pattern = r'^"(.+?)",(\d+)\s*$'

    # Read the vocabulary mapping token ID back to the text item
    with open("output.txt", "r", encoding="utf-8") as f:
        for line in f:
            match = re.match(pattern, line)
            if match:
                token_map[int(match.group(2))] = match.group(1)

    items = []
    for token in tokens:
        item = token_map.get(token, '')
        items.append(item)
        
    # Join everything back together smoothly
    return ''.join(items)
def padding(tokens, max_length):
    pad_id = 2  # Assuming <pad> has ID 2
    current_length = len(tokens)
    
    if current_length < max_length:
        padding_needed = max_length - current_length
        padded_tokens = np.concatenate([tokens, np.full(padding_needed, pad_id)])
    else:
        padded_tokens = tokens[:max_length]
    
    return padded_tokens
def get_positional_encoding(seq_len, d_model):
    # 1. Create an empty matrix for positions of shape (seq_len, d_model)
    pe = np.zeros((seq_len, d_model))
    
    # 2. Loop through every position (t) and every dimension (k)
    for t in range(seq_len):
        for k in range(d_model):
            # Formula denominator part: 10000^(k / d_model)
            # (Note: we use k // 2 to pair up sine and cosine channels)
            denominator = np.power(10000, (2 * (k // 2)) / d_model)
            
            if k % 2 == 0:
                # Even index -> Sine wave
                pe[t, k] = np.sin(t / denominator)
            else:
                # Odd index -> Cosine wave
                pe[t, k] = np.cos(t / denominator)
                
    return pe
def softmax(x):
    # Subtract max for numerical stability (prevents overflow)
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
def Attention(Queries, Keys, Values):
    # 1. Get the dimension size of the keys ($d_k$) for scaling
    d_k = Keys.shape[-1]
    
    # 2. Compute the dot product of Q and K transposed: Q * K^T
    # Shape: (seq_len, seq_len)
    scores = np.matmul(Queries, Keys.T) / np.sqrt(d_k)
    
    # 3. Apply softmax to get attention weights (probabilities that sum to 1)
    attention_weights = softmax(scores)
    
    # 4. Multiply weights by Values: Softmax(...) * V
    # Shape: (seq_len, embedding_dim)
    output = np.matmul(attention_weights, Values)
    
    return output
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
        out_i = Attention(Q[i], K[i], V[i])  # Shape: (seq_len, d_k)
        head_outputs.append(out_i)
        
    # 4. Concatenate all head outputs back together along columns
    # Shape becomes: (seq_len, d_model)
    concatenated = np.concatenate(head_outputs, axis=-1)
    
    # 5. Final linear projection using W_o
    output = np.matmul(concatenated, W_o)
    
    return output
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