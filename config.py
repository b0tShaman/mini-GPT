import os

# Flags
INSTRUCTION_SET = True  # Whether data uses instruction format
CHAT_MODE = True  # Whether to run in chat mode

# Hyperparameters
CONTEXT_LEN = 128
EMBED_DIM = 128
NUM_LAYERS = 4
NUM_HEADS = 4
BATCH_SIZE = 128
DROPOUT = 0.1
EPOCHS = 1
LR = 3e-4
VAL_SPLIT = 0.1

# Inference Parameters
MAX_NEW_TOKENS = 100  # Max tokens to generate in one go
NUM_SENTENCES = 5  # Sentences to generate in typewriter mode
TOP_K = 3  # Top-K sampling
TEMPERATURE = 0.6  # Sampling temperature

# Data Parameters
MAX_VOCAB_SIZE = 3000  # Max tokens BPE should create
STEP_SIZE = 1  # Sliding window step (smaller = more data)

# Your source text file
DATA_FILE = "input.txt"

# 1. Model Artifacts
CHECKPOINT_FILE = "outputs/model/weights.pth"
TOKENIZER_FILE = "outputs/model/tokenizer.json"

# 2. Data Cache Files
BIN_FILE = "outputs/data/dataset.bin"
CLEAN_FILE = "outputs/data/cleaned_data.txt"
PREVIEW_FILE = "outputs/data/dataset_preview.csv"
SLIDING_WINDOW_PREVIEW_FILE = "outputs/data/dataset.csv"

# --- AUTO-CREATE FOLDERS ---
# We now need to ensure BOTH sub-folders exist to prevent crashes
os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)  # Creates outputs/model
os.makedirs(os.path.dirname(BIN_FILE), exist_ok=True)  # Creates outputs/data
# ---------------------------
