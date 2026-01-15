import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
import config


class TextProcessor:
    def __init__(self, context_len=128, vocab_size=config.MAX_VOCAB_SIZE):
        self.context_len = context_len
        self.target_vocab_size = vocab_size
        self.tokenizer = None
        self.vocab_size = 0

    def train_tokenizer(self, filepath):
        print("🏗️  Initializing BPE Tokenizer...")
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()

        special_tokens = ["<pad>", "<unk>", "<eos>"]
        if config.INSTRUCTION_SET:
            special_tokens.extend(["<user>:", "<bot>:"])

        trainer = trainers.BpeTrainer(
            vocab_size=self.target_vocab_size,
            special_tokens=special_tokens,
            min_frequency=2,
        )

        print(f"🏋️  Training tokenizer on {filepath}...")
        tokenizer.train([filepath], trainer)
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.get_vocab_size()
        print(f"   ✅ BPE Training Complete. Vocab Size: {self.vocab_size}")

    def process_file(self, filepath):
        print(f"📖 Streaming {filepath}...")
        if not os.path.exists(filepath):
            print(f"❌ Error: '{filepath}' not found.")
            return None

        if self.tokenizer is None:
            self.train_tokenizer(filepath)

        # Use config paths
        print(f"💾 Streaming to '{config.CLEAN_FILE}' and '{config.BIN_FILE}'...")
        open(config.CLEAN_FILE, "w").close()
        open(config.BIN_FILE, "wb").close()

        total_tokens = 0
        batch_buffer = []
        BATCH_WRITE_SIZE = 10000

        with open(filepath, "r", encoding="utf-8") as f_in, open(
            config.CLEAN_FILE, "a", encoding="utf-8"
        ) as f_clean, open(config.BIN_FILE, "ab") as f_bin:

            prev_speaker = None
            for line in f_in:
                line = line.strip()

                if not line:  # Skip empty lines
                    continue
    
                # 1. Identify Speaker
                curr_speaker = "bot" if line.startswith("<bot>:") else "user"

                line_lower = line.lower()
                
                # 2. Process Line
                if config.INSTRUCTION_SET and line_lower.startswith("<bot>:"):
                    line_lower = line_lower + " <eos>"
                
                line_lower = line_lower + "\n"

                # 3. Check for Collision (Bot->Bot or User->User)
                if prev_speaker == curr_speaker:
                    # Fix the Text File: Write a distinct separator line
                    separator = "<eos>\n"
                    f_clean.write(separator) 
                    
                    sep_ids = self.tokenizer.encode("<eos>").ids 
                    batch_buffer.extend(sep_ids)

                f_clean.write(line_lower)
                encoded = self.tokenizer.encode(line_lower)
                batch_buffer.extend(encoded.ids)

                # 4. Update Tracker
                prev_speaker = curr_speaker

                if len(batch_buffer) >= BATCH_WRITE_SIZE:
                    np.array(batch_buffer, dtype=np.int64).tofile(f_bin)
                    total_tokens += len(batch_buffer)
                    batch_buffer = []

            if batch_buffer:
                np.array(batch_buffer, dtype=np.int64).tofile(f_bin)
                total_tokens += len(batch_buffer)

        print(f"   Total Tokens: {total_tokens}")
        print(f"🔗 creating memory map to {config.BIN_FILE}...")
        np_data = np.memmap(
            config.BIN_FILE, dtype=np.int64, mode="r", shape=(total_tokens,)
        )

        data_ids = np_data

        # --- GENERATE CSV PREVIEW ---
        preview_filename = config.PREVIEW_FILE
        print(
            f"📊 Generating token preview to '{config.PREVIEW_FILE}' (First 100 tokens)..."
        )
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
            print("   ✅ Token Preview saved.")
        except Exception as e:
            print(f"   ⚠️ Could not save preview: {e}")

        # --- GENERATE SLIDING WINDOW PREVIEW ---
        sw_preview_filename = config.SLIDING_WINDOW_PREVIEW_FILE
        print(
            f"📊 Generating sliding window preview to '{sw_preview_filename}' (First 100 samples)..."
        )
        try:
            with open(
                sw_preview_filename, "w", newline="", encoding="utf-8"
            ) as csvfile:
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

                        writer.writerow(
                            [i, f"'{input_decoded}'", f"'{target_decoded}'"]
                        )
            print("   ✅ Sliding Window Preview saved.")
        except Exception as e:
            print(f"   ⚠️ Could not save preview: {e}")

        return data_ids


class LazyWindowDataset(Dataset):
    def __init__(self, data_tensor, context_len, step_size=None):
        self.data = data_tensor
        self.context_len = context_len
        self.step_size = step_size if step_size is not None else context_len

    def __len__(self):
        if len(self.data) <= self.context_len:
            return 0
        return (len(self.data) - self.context_len - 1) // self.step_size

    def __getitem__(self, idx):
        start_idx = idx * self.step_size
        input_seq = self.data[start_idx : start_idx + self.context_len]
        target_seq = self.data[start_idx + 1 : start_idx + self.context_len + 1]
        return torch.from_numpy(input_seq.copy()), torch.from_numpy(target_seq.copy())
