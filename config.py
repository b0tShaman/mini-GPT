import os

# Flags
INSTRUCTION_SET = True 
CHAT_MODE = False

# Hyperparameters
CONTEXT_LEN = 64      
EMBED_DIM = 64        
NUM_LAYERS = 12        
NUM_HEADS = 4         
BATCH_SIZE = 128 
EPOCHS = 3
LR = 3e-4              
VAL_SPLIT = 0.1

# Data Parameters 
MAX_VOCAB_SIZE = 3000   # Max tokens BPE should create
STEP_SIZE = 16          # Sliding window step (smaller = more data)

# File Paths
CHECKPOINT_FILE = "assets/weights.pth"
TOKENIZER_FILE = "assets/tokenizer.json"
DATA_FILE = "input.txt"  # Your source text file
BIN_FILE = "assets/dataset.bin"
CLEAN_FILE = "assets/cleaned_data.txt"
PREVIEW_FILE = "assets/dataset_preview.csv"
SLIDING_WINDOW_PREVIEW_FILE = "assets/dataset.csv"

# Environment
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- AUTO-CREATE FOLDER ---
# This ensures the script doesn't crash on the first run
assets_dir = os.path.dirname(CHECKPOINT_FILE) # Gets "assets"
os.makedirs(assets_dir, exist_ok=True)
# --------------------------