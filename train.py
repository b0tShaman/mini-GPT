import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
import signal
import math
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split

# Local imports
import config
from utils import DEVICE, save_checkpoint, load_checkpoint
from model import TextGenModel
from data import TextProcessor, LazyWindowDataset
from inference import generate_text_string

def train_model():
    # 1. Load or Initialize
    model, processor, checkpoint_data = load_checkpoint(config.CHECKPOINT_FILE)
    
    # Defaults from config
    embed_dim = config.EMBED_DIM
    num_layers = config.NUM_LAYERS
    num_heads = config.NUM_HEADS
    
    if checkpoint_data:
        embed_dim = checkpoint_data.get('embed_dim', embed_dim)
        num_layers = checkpoint_data.get('num_layers', num_layers)
        num_heads = checkpoint_data.get('num_heads', num_heads)
    
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

    if not start_training:
        return model, processor

    # 2. Data Prep
    if processor is None:
        processor = TextProcessor(context_len=config.CONTEXT_LEN, vocab_size=config.MAX_VOCAB_SIZE)
        
    data_ids = processor.process_file(config.DATA_FILE)
    if data_ids is None: 
        print(f"Please ensure {config.DATA_FILE} exists.")
        return None, None
    
    full_dataset = LazyWindowDataset(data_ids, config.CONTEXT_LEN, step_size=config.STEP_SIZE)
    val_size = int(len(full_dataset) * config.VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # 3. Model Init
    if model is None:
        print(f"🏗️  Building new model: Layers={num_layers}, Heads={num_heads}, Embed={embed_dim}, Context={config.CONTEXT_LEN}")
        model = TextGenModel(
            vocab_size=processor.vocab_size, 
            embed_dim=embed_dim, 
            context_len=config.CONTEXT_LEN,
            num_layers=num_layers,
            num_heads=num_heads
        ).to(DEVICE)
    else:
        model.to(DEVICE)
    
    # 4. Safety Save
    def safety_save(sig, frame):
        print("\n\n🛑 Interrupt! Saving model before exiting...")
        save_checkpoint(model, processor, config.CHECKPOINT_FILE)
        sys.exit(0)
    signal.signal(signal.SIGINT, safety_save)

    # 5. Optimizer & Logging
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
    
    log_dir = f"runs/experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=log_dir)
    print(f"\n🚀 Logging to TensorBoard. Run: tensorboard --logdir=runs")
    
    print(f"\n🚀 Starting Training on {DEVICE} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Train Windows: {len(train_dataset)} | Val Windows: {len(val_dataset)}")
    start_time = time.time()

    global_step = 0

    # Get a dummy input batch
    dummy_input, _ = next(iter(train_loader))
    dummy_input = dummy_input.to(DEVICE)
    writer.add_graph(model, dummy_input)

    for epoch in range(config.EPOCHS):
        model.train()
        total_train_loss = 0
        train_correct = 0
        train_tokens = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            
            B, T, V = logits.shape
            loss = criterion(logits.view(B*T, V), y.view(B*T))
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            _, predicted = torch.max(logits, 2)
            train_correct += (predicted == y).sum().item()
            train_tokens += (B * T)

            if batch_idx % 10 == 0:
                writer.add_scalar('Batch/Loss', loss.item(), global_step)
                writer.add_scalar('Batch/LR', scheduler.get_last_lr()[0], global_step)
                global_step += 1

        avg_train_loss = total_train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_tokens
        
        # Validation
        model.eval()
        total_val_loss = 0
        val_correct = 0
        val_tokens = 0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(DEVICE), y_val.to(DEVICE)
                val_logits = model(x_val)
                B_v, T_v, V_v = val_logits.shape
                v_loss = criterion(val_logits.view(B_v*T_v, V_v), y_val.view(B_v*T_v))
                total_val_loss += v_loss.item()
                _, v_predicted = torch.max(val_logits, 2)
                val_correct += (v_predicted == y_val).sum().item()
                val_tokens += (B_v * T_v)
        
        avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_acc = 100 * val_correct / val_tokens if val_tokens > 0 else 0
        
        # Logging
        try: train_ppl = math.exp(avg_train_loss)
        except OverflowError: train_ppl = float('inf')
        try: val_ppl = math.exp(avg_val_loss)
        except OverflowError: val_ppl = float('inf')

         # 2. Add to TensorBoard
        writer.add_scalar('Epoch/Train Loss', avg_train_loss, epoch)
        writer.add_scalar('Epoch/Train Accuracy', train_acc, epoch)
        writer.add_scalar('Epoch/Train Perplexity', train_ppl, epoch)

        writer.add_scalar('Epoch/Val Loss', avg_val_loss, epoch)
        writer.add_scalar('Epoch/Val Accuracy', val_acc, epoch)
        writer.add_scalar('Epoch/Val Perplexity', val_ppl, epoch)

        # Log a sample generation to TensorBoard
        sample_prompt = "<user>: hello\n<bot>:"
        generated_sample = generate_text_string(model, processor, sample_prompt)

        writer.add_text(
            "Training/Sample_Generation", 
            f"**Epoch {epoch+1}**\n\n> Prompt: {sample_prompt}\n\n{generated_sample}", 
            epoch
        )

        # writer.add_scalars('Epoch/Loss', {'Train': avg_train_loss, 'Val': avg_val_loss}, epoch)
        # writer.add_scalars('Epoch/Accuracy', {'Train': train_acc, 'Val': val_acc}, epoch)
        # writer.add_scalars('Epoch/Perplexity', {'Train': train_ppl, 'Val': val_ppl}, epoch) 
        
        elapsed = time.time() - start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = elapsed % 60
        time_str = f"{hours}h {minutes}m {seconds:.2f}s"
    
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # 3. Update Print Statement
        print(
            f"{timestamp} | Epoch {epoch+1} | "
            f"Tr Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Tr PPL: {train_ppl:.2f} | "  
            f"Val PPL: {val_ppl:.2f} | " 
            f"Val Acc: {val_acc:.1f}% | "
            f"Time: {time_str}"
        )
        scheduler.step()

    print("\n✅ Training Complete!")
    writer.close()
    save_checkpoint(model, processor, config.CHECKPOINT_FILE)
    return model, processor