import numpy as np
import pandas as pd
import test

# --- Upgraded Hyperparameters ---
seq_len = 100
d_model = 64          # Upgraded from 8!
vocab_size = 900      
num_heads = 4         # Upgraded from 2! (64 / 4 = 16 per head)
hidden_dim = 256      # Upgraded from 32! (4 * d_model)
learning_rate = 0.001
epochs = 50            # You can crank this up or train on more samples now
total_samples = 10000  # Let's feed it more data rows to chew on!

print("Loading dataset...")
df = pd.read_csv('llm_dataset_1.csv')

# --- 2. Initialize Weights ---
embedding_table = np.random.randn(vocab_size, d_model) * 0.1
pos_encoding = test.get_positional_encoding(seq_len, d_model)

W_q = np.random.randn(d_model, d_model) * 0.1
W_k = np.random.randn(d_model, d_model) * 0.1
W_v = np.random.randn(d_model, d_model) * 0.1
W_o = np.random.randn(d_model, d_model) * 0.1

W1 = np.random.randn(d_model, hidden_dim) * 0.1
b1 = np.zeros(hidden_dim)
W2 = np.random.randn(hidden_dim, d_model) * 0.1
b2 = np.zeros(d_model)

W_lm = np.random.randn(d_model, vocab_size) * 0.1
b_lm = np.zeros(vocab_size)

# --- 3. Training Loop over the Dataset ---
print("Starting training on dataset...")
total_samples = min(1000, len(df))  # Let's start by training on the first 1,000 rows to test it fast

for epoch in range(epochs):
    total_loss = 0
    for i in range(total_samples):
        row = df.iloc[i]
        # Combine prompt and answer as the training text
        text = str(row['prompt']) + " " + str(row['answer'])
        
        # Pick a dummy target token ID based on the text length or hash
        target_token_id = abs(hash(text)) % vocab_size
        
        # --- FORWARD PASS ---
        tokens = test.tokenize(text)
        padded_tokens = test.padding(tokens, seq_len)
        
        word_embeds = embedding_table[padded_tokens]  # (100, 8)
        x = word_embeds + pos_encoding                # (100, 8)
        
        mha_out = test.multi_head_attention(x, W_q, W_k, W_v, W_o, num_heads)
        x_after_mha = test.layer_norm(x + mha_out)
        
        ffn_out = test.feed_forward_network(x_after_mha, W1, b1, W2, b2)
        x_final = test.layer_norm(x_after_mha + ffn_out)  # (100, 8)
        
        logits = np.matmul(x_final, W_lm) + b_lm  # (100, vocab_size)
        last_token_logits = logits[-1]
        
        # --- LOSS & BACKPROP (LM Head) ---
        exp_logits = np.exp(last_token_logits - np.max(last_token_logits))
        probs = exp_logits / np.sum(exp_logits)
        
        loss = -np.log(probs[target_token_id] + 1e-9)
        total_loss += loss
        
        d_logits = probs.copy()
        d_logits[target_token_id] -= 1.0
        
        dW_lm = np.outer(x_final[-1], d_logits)
        db_lm = d_logits
        
        # --- WEIGHT UPDATE ---
        W_lm -= learning_rate * dW_lm
        b_lm -= learning_rate * db_lm
        
    print(f"Epoch {epoch + 1} completed! Average Loss: {total_loss / total_samples:.4f}")

# --- 4. Save Trained Weights to Text Files ---
weights_dict = {
    "embedding_table": embedding_table,
    "pos_encoding": pos_encoding,
    "W_q": W_q, "W_k": W_k, "W_v": W_v, "W_o": W_o,
    "W1": W1, "b1": b1, "W2": W2, "b2": b2,
    "W_lm": W_lm, "b_lm": b_lm
}

test.save_weights_to_txt(weights_dict, folder="transformer_weights")
print("Training complete and weights saved successfully to text files!")