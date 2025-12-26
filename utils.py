import torch
import os
from model import TextGenModel
from data import TextProcessor
import config


def get_device():
    if torch.backends.mps.is_available():
        print("✅ Using Mac GPU (MPS acceleration)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print(f"✅ Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    else:
        print("⚠️ Using CPU")
        return torch.device("cpu")


# Global device variable to be imported elsewhere
DEVICE = get_device()


def save_checkpoint(model, processor, filepath):
    print(f"\n💾 Saving model to {filepath}...")
    checkpoint = {
        "model_state": model.state_dict(),
        "context_len": processor.context_len,
        "vocab_size": processor.vocab_size,
        "embed_dim": model.embed_dim,
        "num_layers": len(model.blocks),
        "num_heads": model.blocks[0].attention.num_heads,
    }
    torch.save(checkpoint, filepath)
    processor.tokenizer.save(config.TOKENIZER_FILE)
    print("   ✅ Model & Tokenizer saved.")


def load_checkpoint(filepath):
    if not os.path.exists(filepath) or not os.path.exists(config.TOKENIZER_FILE):
        return None, None, None

    print(f"\n📂 Found checkpoint. Loading...")
    try:
        checkpoint = torch.load(filepath, map_location=DEVICE, weights_only=True)
    except Exception as e:
        print(f"❌ Error loading checkpoint file: {e}")
        return None, None, None

    # 1. Rebuild Processor
    from tokenizers import Tokenizer

    processor = TextProcessor(context_len=checkpoint["context_len"])
    processor.tokenizer = Tokenizer.from_file(config.TOKENIZER_FILE)
    processor.vocab_size = processor.tokenizer.get_vocab_size()

    # 2. Rebuild Model
    embed_dim = checkpoint.get("embed_dim")
    num_layers = checkpoint.get("num_layers")
    num_heads = checkpoint.get("num_heads")

    if None in [embed_dim, num_layers, num_heads]:
        print("❌ Error: Checkpoint is missing architecture metadata!")
        return None, None, None

    model = TextGenModel(
        vocab_size=processor.vocab_size,
        embed_dim=embed_dim,
        context_len=processor.context_len,
        num_layers=num_layers,
        num_heads=num_heads,
    )

    try:
        model.load_state_dict(checkpoint["model_state"])
    except RuntimeError as e:
        print(f"⚠️  Architecture mismatch! Missing keys: {e}")
        return None, None, None

    model.to(DEVICE)
    print("   ✅ Loaded successfully.")
    return model, processor, checkpoint
