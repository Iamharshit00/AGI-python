import numpy as np
import os

# --- Function to Save All Weights to Text Files ---
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

# --- Example Usage ---
# 1. Gather your weights into a dictionary
my_weights = {
    "embedding_table": np.random.randn(900, 8),
    "pos_encoding": np.random.randn(100, 8),
    "W_q": np.random.randn(8, 8),
    "W_k": np.random.randn(8, 8),
    "W_v": np.random.randn(8, 8),
    "W_o": np.random.randn(8, 8),
    "W1": np.random.randn(8, 32),
    "b1": np.random.randn(32),
    "W2": np.random.randn(32, 8),
    "b2": np.random.randn(8),
    "W_lm": np.random.randn(8, 900),
    "b_lm": np.random.randn(900)
}

# 2. Save them
save_weights_to_txt(my_weights)

# 3. Load them back whenever you want!
loaded_weights = load_weights_from_txt()