import numpy as np
import test as f  # Your tokenizer, padding, and positional encoding functions

# --- 1. Hyperparameters ---
seq_len = 100
d_model = 8
vocab_size = 900  # Match this to your actual output_2.txt vocabulary size
num_heads = 2
hidden_dim = 32

# --- 2. Initialize All Weights Randomly (In a real model, these get trained) ---
embedding_table = np.random.randn(vocab_size, d_model)
pos_encoding = f.get_positional_encoding(seq_len, d_model)

# Multi-head attention weights
W_q = np.random.randn(d_model, d_model)
W_k = np.random.randn(d_model, d_model)
W_v = np.random.randn(d_model, d_model)
W_o = np.random.randn(d_model, d_model)

# Feed-Forward Network weights
W1 = np.random.randn(d_model, hidden_dim)
b1 = np.random.randn(hidden_dim)
W2 = np.random.randn(hidden_dim, d_model)
b2 = np.random.randn(d_model)

# LM Head weights
W_lm = np.random.randn(d_model, vocab_size)
b_lm = np.random.randn(vocab_size)


# --- 3. The Full Forward Pipeline ---
def run_transformer_pipeline(text):
    # Step A: Tokenize & Pad to seq_len (100)
    tokens = f.tokenize(text)
    padded_tokens = f.padding(tokens, seq_len)
    
    # Step B: Word Embeddings + Positional Encoding
    word_embeds = embedding_table[padded_tokens]  # (100, 8)
    x = word_embeds + pos_encoding                # (100, 8)
    
    # Step C: Multi-Head Attention + First Add & Norm
    mha_out = f.multi_head_attention(x, W_q, W_k, W_v, W_o, num_heads)
    x = f.layer_norm(x + mha_out)                     # Residual + Norm
    
    # Step D: Feed-Forward Network + Second Add & Norm
    ffn_out = f.feed_forward_network(x, W1, b1, W2, b2)
    x = f.layer_norm(x + ffn_out)                     # Residual + Norm
    
    # Step E: Language Model Head (Project to Vocab Size)
    logits = np.matmul(x, W_lm) + b_lm              # (100, vocab_size)
    
    return logits

# --- Test the Whole Machine! ---
sample_text = "hello world! This is a test of the harshit tokenization system."
output_logits = run_transformer_pipeline(sample_text)

print("Pipeline executed successfully!")
print("Final Logits Shape (Sequence Length, Vocabulary Size):", output_logits.shape)