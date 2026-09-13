import test as f

a = f.tokenize("Hello world! This (π) is a test of the harshit tokenization system. Let's see how it handles various words, punctuation, and symbols.")
b = f.padding(a, 150)
print(a)
print(f"\n\nb: {b}")
print(f"\n\n{f.detokenize(b)}")

seq_length = 100
embedding_dim = 8
positional_encoding = f.get_positional_encoding(seq_length, embedding_dim)
print(f"\n\nPositional Encoding Shape: {positional_encoding.shape}")

import numpy as np

def create_transformer_input(token_ids, embedding_table, pos_encoding_matrix):
    """
    Takes token IDs, looks up their word embeddings, and adds the positional encoding.
    
    Shapes:
    - token_ids: (seq_len,)
    - embedding_table: (vocab_size, embedding_dim)
    - pos_encoding_matrix: (seq_len, embedding_dim)
    
    Returns:
    - final_input: (seq_len, embedding_dim) ready for the Transformer!
    """
    # 1. Lookup the word embeddings using the token IDs -> shape becomes (seq_len, embedding_dim)
    word_embeddings = embedding_table[token_ids]
    
    # 2. Add word embeddings and positional encoding element-wise
    final_input = word_embeddings + pos_encoding_matrix
    
    return final_input

# --- Let's test it with your setup ---
seq_len = 150
embedding_dim = 8
vocab_size = 900

# Dummy data to test the function
dummy_token_ids = np.zeros(seq_len, dtype=int) # e.g., your padded array `b`
dummy_embedding_table = np.random.randn(vocab_size, embedding_dim)
dummy_pos_encoding = np.random.randn(seq_len, embedding_dim) # (Or use your get_positional_encoding function!)

# Run the function
final_transformer_input = create_transformer_input(
    dummy_token_ids, 
    dummy_embedding_table, 
    dummy_pos_encoding
)

print("Final Transformer Input Shape:", final_transformer_input.shape)
print("First token's final mixed vector:\n", final_transformer_input[0])