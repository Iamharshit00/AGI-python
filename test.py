import re
import numpy as np
import os

"""this code is having all the function
that are usefull to make a AI first"""

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
def save_weights_to_txt(weights_dict, folder="transformer_weights"):
    """
    Saves each weight matrix into its own .txt file inside a folder.
    """
    os.makedirs(folder, exist_ok=True)
    for name, array in weights_dict.items():
        file_path = os.path.join(folder, f"{name}.txt")
        # Flatten multi-dimensional arrays if needed or save directly 
        # (np.savetxt handles 2D matrices easily)
        np.savetxt(file_path, array, delimiter=",")
    print(f"All weights successfully saved to folder: '{folder}/'")

# --- Function to Load All Weights from Text Files ---
def load_weights_from_txt(folder="transformer_weights"):
    """
    Loads weight matrices back from the .txt files.
    """
    weights_dict = {}
    weight_files = [
        "embedding_table", "pos_encoding", 
        "W_q", "W_k", "W_v", "W_o", 
        "W1", "b1", "W2", "b2", 
        "W_lm", "b_lm"
    ]
    
    for name in weight_files:
        file_path = os.path.join(folder, f"{name}.txt")
        if os.path.exists(file_path):
            weights_dict[name] = np.loadtxt(file_path, delimiter=",")
        else:
            raise FileNotFoundError(f"Could not find weight file: {file_path}")
            
    print(f"All weights successfully loaded from folder: '{folder}/'")
    return weights_dict
def save_weights_to_txt(weights_dict, folder="transformer_weights"):
    """Saves all weight matrices into .txt files inside a folder."""
    os.makedirs(folder, exist_ok=True)
    for name, array in weights_dict.items():
        file_path = os.path.join(folder, f"{name}.txt")
        np.savetxt(file_path, array, delimiter=",")
    print(f"Weights successfully saved to '{folder}/'")

def load_and_chat(folder="transformer_weights"):
    """
    Loads model weights from text files and starts an interactive chat loop 
    with multi-word generation (at least 10 words, up to 30 words).
    """
    print("Loading weights from text files...")
    weight_files = [
        "embedding_table", "pos_encoding", 
        "W_q", "W_k", "W_v", "W_o", 
        "W1", "b1", "W2", "b2", 
        "W_lm", "b_lm"
    ]
    
    w = {}
    for name in weight_files:
        file_path = os.path.join(folder, f"{name}.txt")
        if os.path.exists(file_path):
            w[name] = np.loadtxt(file_path, delimiter=",")
        else:
            raise FileNotFoundError(f"Missing weight file: {file_path}. Train and save weights first!")
            
    print("Weights loaded successfully!")
    print("\n--- 🤖 Transformer Chat Live! (Min 10+ words) Type 'exit' to quit. ---")
    
    seq_len = 100
    num_heads = 4  # Match your updated head count
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        if not user_input.strip():
            continue
            
        # Tokenize user input
        tokens = tokenize(user_input)
        current_tokens = list(tokens)
        
        generated_words = []
        min_words = 10
        max_words = 30  # Cap it so it doesn't loop infinitely
        
        # Autoregressive generation loop
        for step in range(max_words):
            # Pad the sequence to match seq_len
            padded_tokens = padding(current_tokens, seq_len)
            
            # 1. Embeddings + Positional Encoding
            x = w["embedding_table"][padded_tokens] + w["pos_encoding"]
            
            # 2. Multi-Head Attention + Add & Norm
            mha_out = multi_head_attention(x, w["W_q"], w["W_k"], w["W_v"], w["W_o"], num_heads)
            x = layer_norm(x + mha_out)
            
            # 3. Feed-Forward Network + Add & Norm
            ffn_out = feed_forward_network(x, w["W1"], w["b1"], w["W2"], w["b2"])
            x = layer_norm(x + ffn_out)
            
            # 4. Language Model Head Prediction
            logits = np.matmul(x, w["W_lm"]) + w["b_lm"]
            last_logits = logits[-1]
            
            # 5. Softmax & Argmax
            exp_logits = np.exp(last_logits - np.max(last_logits))
            probabilities = exp_logits / np.sum(exp_logits)
            next_token_id = np.argmax(probabilities)
            
            # Append token to current context and generated list
            current_tokens.append(next_token_id)
            # If sequence gets longer than seq_len, keep only the latest window
            if len(current_tokens) > seq_len:
                current_tokens = current_tokens[-seq_len:]
                
            response_word = detokenize([next_token_id])
            generated_words.append(response_word)
            
            # Stop condition: If we've hit at least 10 words and hit a period/stopping token
            if len(generated_words) >= min_words and (response_word == "." or response_word == "!"):
                break
                
        # Print the final multi-word sentence
        print(f"AI: {' '.join(generated_words)}")
def compute_loss_and_lm_gradient(logits_last_token, target_token_id):
    """
    Computes cross-entropy loss and the exact gradient for the logits.
    The combined gradient of Softmax + Cross-Entropy is simply: probabilities - one_hot
    """
    # Numerical stability softmax
    exp_logits = np.exp(logits_last_token - np.max(logits_last_token))
    probs = exp_logits / np.sum(exp_logits)
    
    # Cross-Entropy Loss
    loss = -np.log(probs[target_token_id] + 1e-9)
    
    # Gradient of loss w.r.t logits: shape (vocab_size,)
    d_logits = probs.copy()
    d_logits[target_token_id] -= 1.0
    
    return loss, d_logits


# --- 4. Training Step (Forward & Backward) ---
def train_step(input_text, target_token_id):
    global embedding_table, W_lm, b_lm, W_q, W_k, W_v, W_o, W1, W2, b1, b2
    
    # ==================== FORWARD PASS ====================
    tokens = test.tokenize(input_text)
    padded_tokens = test.padding(tokens, seq_len)
    
    # Word Embeddings + Positional Encoding
    word_embeds = embedding_table[padded_tokens]  # (100, 8)
    x = word_embeds + pos_encoding                # (100, 8)
    
    # Multi-Head Attention + Residual/Norm
    mha_out = test.multi_head_attention(x, W_q, W_k, W_v, W_o, num_heads)
    x_after_mha = test.layer_norm(x + mha_out)
    
    # Feed-Forward Network + Residual/Norm
    ffn_out = test.feed_forward_network(x_after_mha, W1, b1, W2, b2)
    x_final = test.layer_norm(x_after_mha + ffn_out)  # (100, 8)
    
    # Language Model Head
    logits = np.matmul(x_final, W_lm) + b_lm  # (100, vocab_size)
    
    # Take the prediction for the very last token position in the sequence
    last_token_logits = logits[-1]  # (vocab_size,)
    
    # ==================== LOSS & BACKPROP ====================
    loss, d_logits = compute_loss_and_lm_gradient(last_token_logits, target_token_id)
    
    # Backprop through LM Head (Linear Layer: Y = XW + b)
    # d_logits shape: (vocab_size,), x_final[-1] shape: (d_model,)
    dW_lm = np.outer(x_final[-1], d_logits)  # (d_model, vocab_size)
    db_lm = d_logits                        # (vocab_size,)
    
    # Gradient flowing back into the transformer output vector at the last position
    dx_final_last = np.matmul(W_lm, d_logits)  # (d_model,)
    
    # ==================== GRADIENT DESCENT UPDATE ====================
    # Update LM Head weights
    W_lm -= learning_rate * dW_lm
    b_lm -= learning_rate * db_lm
    
    # (Optional: In a full pipeline, you would continue backpropagating 
    #  dx_final_last backward through FFN, Attention, and Embeddings here)
    
    return loss