import numpy as np
import test

# --- Hyperparameters ---
seq_len = 100
d_model = 8
vocab_size = 900  # Match your output_2.txt vocabulary size
num_heads = 2
hidden_dim = 32

# --- Initialize Weights ---
embedding_table = np.random.randn(vocab_size, d_model)
pos_encoding = test.get_positional_encoding(seq_len, d_model)

# Attention Weights
W_q = np.random.randn(d_model, d_model)
W_k = np.random.randn(d_model, d_model)
W_v = np.random.randn(d_model, d_model)
W_o = np.random.randn(d_model, d_model)

# Feed-Forward Network Weights
W1 = np.random.randn(d_model, hidden_dim)
b1 = np.random.randn(hidden_dim)
W2 = np.random.randn(hidden_dim, d_model)
b2 = np.random.randn(d_model)

# Language Model Head Weights
W_lm = np.random.randn(d_model, vocab_size)
b_lm = np.random.randn(vocab_size)


def generate_next_word(input_text):
    """
    Passes text through your custom Transformer pipeline and 
    predicts the single next word/token.
    """
    # 1. Tokenize & Pad using your test module functions
    tokens = test.tokenize(input_text)
    padded_tokens = test.padding(tokens, seq_len)
    
    # 2. Word Embeddings + Positional Encoding
    word_embeds = embedding_table[padded_tokens]  # Shape: (100, 8)
    x = word_embeds + pos_encoding                # Shape: (100, 8)
    
    # 3. Multi-Head Attention + Residual/Norm
    mha_out = test.multi_head_attention(x, W_q, W_k, W_v, W_o, num_heads)
    x = test.layer_norm(x + mha_out)
    
    # 4. Feed-Forward Network + Residual/Norm
    ffn_out = test.feed_forward_network(x, W1, b1, W2, b2)
    x = test.layer_norm(x + ffn_out)
    
    # 5. Language Model Head (Project to Vocab Size)
    logits = np.matmul(x, W_lm) + b_lm  # Shape: (100, vocab_size)
    
    # 6. Extract logits for the LAST token position in the sequence
    last_token_logits = logits[-1]
    
    # 7. Apply Softmax to convert logits into probabilities
    exp_logits = np.exp(last_token_logits - np.max(last_token_logits))
    probabilities = exp_logits / np.sum(exp_logits)
    
    # 8. Pick the token ID with the highest probability
    next_token_id = np.argmax(probabilities)
    
    # 9. Convert the token ID back into text using your detokenizer
    next_word = test.detokenize([next_token_id])
    
    return next_word

# --- Test the Generation Pipeline ---
sample_text = "hello world! This is a test"
predicted = generate_next_word(sample_text)

print(f"Input Text: {sample_text}")
print(f"Predicted Next Word: {predicted}")