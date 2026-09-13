import test as f
import numpy as np

a = f.tokenize("Hello world! This is a test of the tokenization system. Let's see how it handles various words, punctuation, and symbols.")
print(a)
print(f.detokenize(a))
b = f.tokenize("cat")
print(b)


# Step 1: Set your sizes
vocab_size = 896   # Total tokens from your output.txt
embedding_dim = 8  # How many numbers describe each word

# Step 2: Build the giant random table (Matrix)
# This creates a table of shape (896, 8) filled with random numbers
embedding_table = np.random.randn(vocab_size, embedding_dim)

# Step 3: Let's test it with token IDs from your tokenizer
# Imagine your tokenizer returned these IDs for a sentence:
sample_token_ids = np.array(f.tokenize("cat")) # e.g., cat, space, hello

# Step 4: LOOKUP - Grab the rows from the table using the token IDs!
word_vectors = embedding_table[sample_token_ids]

print("Shape of your embedding output:", word_vectors.shape)
print("The embedding vectors for your tokens:\n", word_vectors)