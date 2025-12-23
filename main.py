import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import time
import sys
import matplotlib.pyplot as plt
import signal
import csv
from datetime import datetime
import numpy as np

# --- IMPORT TOKENIZERS ---
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

# ---------------------------------------------------------
# 1. SETUP DEVICE
# ---------------------------------------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ Using Mac GPU (MPS acceleration)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✅ Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("⚠️ Using CPU")

CHECKPOINT_FILE = "weights.pth"
TOKENIZER_FILE = "tokenizer.json"

# ---------------------------------------------------------
# 2. DATA CLEANING & PROCESSING (STREAMING / MEMMAP)
# ---------------------------------------------------------
class TextProcessor:
    def __init__(self, context_len=128, vocab_size=30000):
        self.context_len = context_len
        self.target_vocab_size = vocab_size
        self.tokenizer = None
        self.vocab_size = 0

    def train_tokenizer(self, filepath):
        print("🏗️  Initializing BPE Tokenizer...")
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=self.target_vocab_size, 
            special_tokens=["<pad>", "<unk>, <eos>"],
            min_frequency=2
        )

        print(f"🏋️  Training tokenizer on {filepath}...")
        # Tokenizer library handles file streaming internally
        tokenizer.train([filepath], trainer)
        
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.get_vocab_size()
        print(f"   ✅ BPE Training Complete. Vocab Size: {self.vocab_size}")

    def process_file(self, filepath):
        print(f"📖 Streaming {filepath}...")
        if not os.path.exists(filepath):
            print(f"❌ Error: '{filepath}' not found.")
            return None

        # 1. Train Tokenizer if not loaded
        if self.tokenizer is None:
            self.train_tokenizer(filepath)

        # 2. Stream, Clean, and Tokenize to Disk (Binary)
        # use a temporary binary file to store tokens instead of RAM
        bin_filename = "dataset.bin"
        clean_filename = "cleaned_data.txt"
        
        print(f"💾 Streaming cleaned text to '{clean_filename}' and tokens to '{bin_filename}'...")
        
        # Reset files
        open(clean_filename, 'w').close()
        open(bin_filename, 'wb').close()
        
        total_tokens = 0
        batch_buffer = []
        BATCH_WRITE_SIZE = 10000 # Write to disk every 10k tokens

        # Read line by line (Streaming)
        with open(filepath, 'r', encoding='utf-8') as f_in, \
             open(clean_filename, 'a', encoding='utf-8') as f_clean, \
             open(bin_filename, 'ab') as f_bin:
            
            for line in f_in:
                # Clean
                line_lower = line.lower()
                f_clean.write(line_lower)
                
                # Encode
                encoded = self.tokenizer.encode(line_lower)
                batch_buffer.extend(encoded.ids)
                
                # Flush buffer to disk if full
                if len(batch_buffer) >= BATCH_WRITE_SIZE:
                    # Save as int64 (matches torch.long)
                    np.array(batch_buffer, dtype=np.int64).tofile(f_bin)
                    total_tokens += len(batch_buffer)
                    batch_buffer = []

            # Flush remaining tokens
            if batch_buffer:
                np.array(batch_buffer, dtype=np.int64).tofile(f_bin)
                total_tokens += len(batch_buffer)

        print(f"   Total Tokens: {total_tokens}")

        # 3. Load using Memory Mapping (Zero RAM Copy)
        # This creates a tensor that reads directly from the hard drive
        # 'r' mode ensures we don't accidentally overwrite data
        print("🔗 creating memory map to dataset.bin...")
        np_data = np.memmap(bin_filename, dtype=np.int64, mode='r', shape=(total_tokens,))
        
        # We assign this to data_ids so the CSV logic below works unchanged
        data_ids = np_data 

        # --- GENERATE CSV PREVIEW ---
        preview_filename = "dataset_preview.csv"
        print(f"📊 Generating token preview to '{preview_filename}' (First 100 tokens)...")
        try:
            with open(preview_filename, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Position", "Token ID", "Decoded String"])
                
                # Preview first 100 tokens
                limit = min(100, len(data_ids))
                for i in range(limit):
                    tid = data_ids[i]
                    # cast numpy type to int for tokenizer
                    decoded_str = self.tokenizer.decode([int(tid)])
                    writer.writerow([i, tid, f"'{decoded_str}'"])
            print("   ✅ Preview saved.")
        except Exception as e:
            print(f"   ⚠️ Could not save preview: {e}")

    
        # --- GENERATE CSV PREVIEW ---
        preview_filename = "dataset.csv"
        print(f"📊 Generating sliding window preview to '{preview_filename}' (First 100 samples)...")
        try:
            with open(preview_filename, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Index", "Input Window", "Target Window"])
                
                # Preview first 100 sliding windows
                max_samples = len(data_ids) - self.context_len - 1
                limit = min(100, max_samples)
                
                if limit > 0:
                    for i in range(limit):
                        # Numpy memmap slicing works exactly like list slicing
                        input_ids = data_ids[i : i + self.context_len]
                        target_ids = data_ids[i + 1 : i + self.context_len + 1]
                        
                        input_decoded = self.tokenizer.decode(input_ids.tolist())
                        target_decoded = self.tokenizer.decode(target_ids.tolist())
                        
                        writer.writerow([i, f"'{input_decoded}'", f"'{target_decoded}'"])
            print("   ✅ Preview saved.")
        except Exception as e:
            print(f"   ⚠️ Could not save preview: {e}")
        
        # Return a PyTorch Tensor that wraps the Numpy Memmap
        # This does NOT load data into RAM, it shares memory with the memmap
        # return torch.from_numpy(data_ids)
        # New Code (Fixes Warning)
        return data_ids

class LazyWindowDataset(Dataset):
    def __init__(self, data_tensor, context_len):
        self.data = data_tensor
        self.context_len = context_len
        
    def __len__(self):
        # We need space for input + 1 shifted target
        return len(self.data) - self.context_len - 1

    def __getitem__(self, idx):
        # Slice window [idx : idx + context_len]
        input_seq = self.data[idx : idx + self.context_len]
        # Target is the SAME window shifted right by 1
        target_seq = self.data[idx + 1 : idx + self.context_len + 1]
        
        # If self.data is a memmap-backed tensor, slicing gives a tensor sharing that memory.
        # We ensure it's a standard LongTensor for the model training
        # .clone().detach() ensures we get a clean copy in RAM for the batch
        # preventing file-lock issues during multi-worker loading
        # .copy() creates a fresh, writable numpy array for just these few tokens
        return torch.from_numpy(input_seq.copy()), torch.from_numpy(target_seq.copy())

# ---------------------------------------------------------
# 3. NEURAL NETWORK (PRE-LN TRANSFORMER WITH CAUSAL MASK)
# ---------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, context_len):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4), 
            nn.GELU(), 
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.dropout = nn.Dropout(0.1)
        
        # We create a triangular mask to prevent looking at future tokens
        # 0 means "pay attention", -inf means "ignore"
        self.register_buffer("causal_mask", torch.triu(torch.ones(context_len, context_len) * float('-inf'), diagonal=1))

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Embed_Dim)
        batch, seq_len, _ = x.shape
        
        norm_x = self.ln1(x)
        
        # 1. Slice the mask to the current sequence length
        # This is crucial because the sequence grows during generation
        current_mask = self.causal_mask[:seq_len, :seq_len]
        
        # 2. Pass the mask explicitly (Standard approach, works on all devices)
        attn_out, _ = self.attention(norm_x, norm_x, norm_x, 
                                     attn_mask=current_mask, 
                                     need_weights=False)
        
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.feed_forward(self.ln2(x)))
        return x

class TextGenModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, context_len, num_layers=4, num_heads=4):
        super(TextGenModel, self).__init__()
        self.context_len = context_len
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(context_len, embed_dim)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, context_len)
            for _ in range(num_layers)
        ])
        
        self.ln_final = nn.LayerNorm(embed_dim)
        # self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.fc_out = nn.Linear(embed_dim, vocab_size, bias=False)
        self.fc_out.weight = self.embedding.weight

        self.register_buffer('pos_ids', torch.arange(context_len, dtype=torch.long))

    def forward(self, x):
        batch, seq_len = x.shape
        
        token_embeds = self.embedding(x)
        positions = self.pos_embedding(self.pos_ids[:seq_len])
        x = token_embeds + positions
        
        for block in self.blocks:
            x = block(x)
            
        x = self.ln_final(x)
        
        # Return logits for the ENTIRE sequence (Many-to-Many)
        logits = self.fc_out(x) # Shape: (Batch, Seq_Len, Vocab)
        return logits

# ---------------------------------------------------------
# 4. SAVE & LOAD
# ---------------------------------------------------------
def save_checkpoint(model, processor, filepath, embed_dim, num_layers, num_heads):
    print(f"\n💾 Saving model to {filepath}...")
    checkpoint = {
        'model_state': model.state_dict(),
        'context_len': processor.context_len,
        'vocab_size': processor.vocab_size,
        'embed_dim': embed_dim,
        'num_layers': num_layers,
        'num_heads': num_heads
    }
    torch.save(checkpoint, filepath)
    processor.tokenizer.save(TOKENIZER_FILE)
    print("   ✅ Model & Tokenizer saved.")

def load_checkpoint(filepath):
    if not os.path.exists(filepath) or not os.path.exists(TOKENIZER_FILE):
        return None, None, None
    
    print(f"\n📂 Found checkpoint. Loading...")
    
    checkpoint = torch.load(filepath, map_location=device)
    
    processor = TextProcessor(context_len=checkpoint['context_len'])
    processor.tokenizer = Tokenizer.from_file(TOKENIZER_FILE)
    processor.vocab_size = processor.tokenizer.get_vocab_size()

    embed_dim = checkpoint.get('embed_dim')
    num_layers = checkpoint.get('num_layers')
    num_heads = checkpoint.get('num_heads')

    if None in [embed_dim, num_layers, num_heads]:
        print("❌ Error: Checkpoint is missing architecture metadata!")
        return None, None, None

    model = TextGenModel(
        vocab_size=processor.vocab_size, 
        embed_dim=embed_dim, 
        context_len=processor.context_len,
        num_layers=num_layers,
        num_heads=num_heads
    )
    
    try:
        model.load_state_dict(checkpoint['model_state'])
    except RuntimeError as e:
        print(f"⚠️  Architecture mismatch! Missing keys: {e}")
        return None, None, None

    model.to(device)
    print("   ✅ Loaded successfully.")
    return model, processor, checkpoint

# ---------------------------------------------------------
# 5. TRAINING LOOP
# ---------------------------------------------------------
def train():
    CONTEXT_LEN = 64     
    EMBED_DIM = 64       
    NUM_LAYERS = 2        
    NUM_HEADS = 4         
    
    BATCH_SIZE = 128     
    EPOCHS = 200
    LR = 3e-4             

    model, processor, checkpoint_data = load_checkpoint(CHECKPOINT_FILE)
    
    if checkpoint_data:
        EMBED_DIM = checkpoint_data.get('embed_dim', EMBED_DIM)
        NUM_LAYERS = checkpoint_data.get('num_layers', NUM_LAYERS)
        NUM_HEADS = checkpoint_data.get('num_heads', NUM_HEADS)
    
    start_training = False 
    if model:
        choice = input("   Found saved model. (c)ontinue training, (n)ew training, or (s)kip to inference? [c/n/s]: ").lower()
        if choice == 's': return model, processor
        elif choice == 'n': 
            model = None         
            start_training = True
            processor = None 
        elif choice == 'c':
            start_training = True
    else:
        start_training = True

    if start_training:
        if processor is None:
            processor = TextProcessor(context_len=CONTEXT_LEN, vocab_size=30000)
            
        data_ids = processor.process_file("input.txt")
        if data_ids is None: return None, None
        
        dataset = LazyWindowDataset(data_ids, CONTEXT_LEN)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        if model is None:
            print(f"🏗️  Building new model: Layers={NUM_LAYERS}, Heads={NUM_HEADS}, Embed={EMBED_DIM}, Context={CONTEXT_LEN}")
            model = TextGenModel(
                vocab_size=processor.vocab_size, 
                embed_dim=EMBED_DIM, 
                context_len=CONTEXT_LEN,
                num_layers=NUM_LAYERS,
                num_heads=NUM_HEADS
            ).to(device)
        else:
            model.to(device)
        
        def safety_save(sig, frame):
            print("\n\n🛑 Interrupt! Saving model before exiting...")
            save_checkpoint(model, processor, CHECKPOINT_FILE, EMBED_DIM, NUM_LAYERS, NUM_HEADS)
            sys.exit(0)

        signal.signal(signal.SIGINT, safety_save)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        print(f"\n🚀 Starting Training on {device} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Total Windows: {len(dataset)} | Batch: {BATCH_SIZE}")
        
        # --- Plotting Setup ---
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        loss_line, = ax1.plot([], [], 'r-', label='Loss', marker='o')
        ax1.set_title('Training Metrics')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        acc_line, = ax2.plot([], [], 'b-', label='Accuracy', marker='o')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.grid(True)
        loss_history = []
        acc_history = []
        start_time = time.time()

        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            correct = 0
            total_tokens = 0
            
            for batch_idx, (x, y) in enumerate(loader):
                x, y = x.to(device), y.to(device)
                
                optimizer.zero_grad()
                logits = model(x) # Shape: (B, T, V)
                
                # Flatten logits and targets for CrossEntropy
                B, T, V = logits.shape
                loss = criterion(logits.view(B*T, V), y.view(B*T))
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                # Accuracy Calculation on the fly
                _, predicted = torch.max(logits, 2) # (B, T)
                correct += (predicted == y).sum().item()
                total_tokens += (B * T)

                if batch_idx % 50 == 0:
                   plt.pause(0.001)

            avg_loss = total_loss / len(loader)
            accuracy = 100 * correct / total_tokens
            
            loss_history.append(avg_loss)
            acc_history.append(accuracy)
            
            elapsed = time.time() - start_time
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = elapsed % 60
            time_str = f"{hours}h {minutes}m {seconds:.2f}s"
            
            # Update Plots
            epochs_x = range(1, epoch + 2)
            loss_line.set_xdata(epochs_x)
            loss_line.set_ydata(loss_history)
            ax1.relim(); ax1.autoscale_view()
            acc_line.set_xdata(epochs_x)
            acc_line.set_ydata(acc_history)
            ax2.relim(); ax2.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"{timestamp} | Epoch {epoch+1} | "
                f"Loss: {avg_loss:.4f} | "
                f"Perplexity: {math.exp(avg_loss):.2f} | "
                f"Acc: {accuracy:.2f}% | "
                f"Time: {time_str}"
            )

            scheduler.step()

        print("\n✅ Training Complete!")
        plt.ioff()
        plt.close()
        save_checkpoint(model, processor, CHECKPOINT_FILE, EMBED_DIM, NUM_LAYERS, NUM_HEADS)

    return model, processor

# ---------------------------------------------------------
# 6. TYPEWRITER GENERATION 
# ---------------------------------------------------------
def generate_text_typewriter(model, processor, seed_text, num_sentences=3, temperature=1, K=1):
    model.eval()
    
    print(f"\n🤖: ", end="", flush=True)
    
    # 1. Encode without manual padding
    current_ids = processor.tokenizer.encode(seed_text.lower()).ids
    
    # Convert list to tensor batch (1, seq_len)
    current_ids_tensor = torch.tensor([current_ids], dtype=torch.long).to(device)
    
    sentence_count = 0
    terminators = {'.', '!', '?', ';', '\n', '<eos>'}

    with torch.no_grad():
        for _ in range(1000): 
            # 2. Crop to context_len if input is too long
            cond_ids = current_ids_tensor[:, -processor.context_len:]

            logits = model(cond_ids)
            
            # 3. Focus only on the LAST token's prediction
            next_token_logits = logits[:, -1, :] # (1, Vocab)
            
            # temperature = 0.8
            scaled_logits = next_token_logits / temperature
            probs = torch.nn.functional.softmax(scaled_logits, dim=1)
            
            # Top-K Sampling
            # K = 1
            top_k_probs, top_k_indices = torch.topk(probs, K, dim=1)
            top_k_probs = top_k_probs / torch.sum(top_k_probs, dim=1, keepdim=True)
            
            next_token_index_in_k = torch.multinomial(top_k_probs, num_samples=1)
            next_id = top_k_indices.gather(1, next_token_index_in_k).item()
            
            word_chunk = processor.tokenizer.decode([next_id])
            print(word_chunk, end="", flush=True)
            time.sleep(0.02) 
            
            # Append new token
            current_ids_tensor = torch.cat([current_ids_tensor, torch.tensor([[next_id]], device=device)], dim=1)
            
            if any(term in word_chunk for term in terminators):
                sentence_count += 1
                if sentence_count >= num_sentences:
                    break
            
    print("\n") 
    return

if __name__ == "__main__":
    trained_model, proc = train()
    
    if trained_model:
        print("\n🤖 Model Ready for Inference!")
        while True:
            text = input("📝 Enter start text (or 'q'): ")
            if text == 'q': break
            generate_text_typewriter(trained_model, proc, text, num_sentences=10, temperature=0.8, K=1) # Greedy sampling for testing