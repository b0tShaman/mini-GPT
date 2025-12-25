import torch
import time
from utils import DEVICE

def generate_text_typewriter(model, processor, seed_text, num_sentences=3, temperature=1, K=1):
    model.eval()
    print(f"\n🤖: ", end="", flush=True)
    
    current_ids = processor.tokenizer.encode(seed_text.lower()).ids
    current_ids_tensor = torch.tensor([current_ids], dtype=torch.long).to(DEVICE)
    
    sentence_count = 0
    terminators = {'.', '!', '?', ';', '\n', '<eos>'}

    with torch.no_grad():
        for _ in range(1000): 
            cond_ids = current_ids_tensor[:, -processor.context_len:]
            logits = model(cond_ids)
            next_token_logits = logits[:, -1, :] 
            
            scaled_logits = next_token_logits / temperature
            probs = torch.nn.functional.softmax(scaled_logits, dim=1)
            
            top_k_probs, top_k_indices = torch.topk(probs, K, dim=1)
            top_k_probs = top_k_probs / torch.sum(top_k_probs, dim=1, keepdim=True)
            
            next_token_index_in_k = torch.multinomial(top_k_probs, num_samples=1)
            next_id = top_k_indices.gather(1, next_token_index_in_k).item()

            if next_id == processor.tokenizer.token_to_id("<eos>"):
                break
            
            word_chunk = processor.tokenizer.decode([next_id])
            print(word_chunk, end="", flush=True)
            time.sleep(0.02) 
            
            current_ids_tensor = torch.cat([current_ids_tensor, torch.tensor([[next_id]], device=DEVICE)], dim=1)
            
            if any(term in word_chunk for term in terminators):
                sentence_count += 1
                if sentence_count >= num_sentences:
                    break
    print("\n") 

def generate_text_string(model, processor, seed_text, max_new_tokens=100, temperature=0.8, K=5):
    model.eval()
    ids = processor.tokenizer.encode(seed_text.lower().strip()).ids
    curr_tensor = torch.tensor([ids], dtype=torch.long).to(DEVICE)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            cond_ids = curr_tensor[:, -processor.context_len:]
            logits = model(cond_ids)
            next_token_logits = logits[:, -1, :]
            
            scaled_logits = next_token_logits / temperature
            probs = torch.nn.functional.softmax(scaled_logits, dim=1)
            
            top_k_probs, top_k_indices = torch.topk(probs, K, dim=1)
            top_k_probs = top_k_probs / torch.sum(top_k_probs, dim=1, keepdim=True)
            
            next_token_index = torch.multinomial(top_k_probs, 1)
            next_id = top_k_indices.gather(1, next_token_index).item()
            
            if next_id == processor.tokenizer.token_to_id("<eos>"):
                break
            
            curr_tensor = torch.cat([curr_tensor, torch.tensor([[next_id]], device=DEVICE)], dim=1)
            
    return processor.tokenizer.decode(curr_tensor[0].tolist())

def chat(model, processor, history, user_input, max_new_tokens=100, temperature=0.8, K=5):
    model.eval()
    if history == "":
        current_prompt = f"<user>: {user_input}\n<bot>:"
    else:
        current_prompt = f"{history}\n<user>: {user_input}\n<bot>:"

    current_ids = processor.tokenizer.encode(current_prompt.lower()).ids
    current_ids_tensor = torch.tensor([current_ids], dtype=torch.long).to(DEVICE)
    
    print(f"🤖: ", end="", flush=True)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            cond_ids = current_ids_tensor[:, -processor.context_len:]
            logits = model(cond_ids)
            next_token_logits = logits[:, -1, :]
            
            scaled_logits = next_token_logits / temperature
            probs = torch.nn.functional.softmax(scaled_logits, dim=1)
            
            top_k_probs, top_k_indices = torch.topk(probs, K, dim=1)
            top_k_probs = top_k_probs / torch.sum(top_k_probs, dim=1, keepdim=True)
            next_token_index = torch.multinomial(top_k_probs, 1)
            next_id = top_k_indices.gather(1, next_token_index).item()

            if next_id == processor.tokenizer.token_to_id("<eos>"):
                break
                
            word_chunk = processor.tokenizer.decode([next_id])
            print(word_chunk, end="", flush=True)
            time.sleep(0.02)
            
            current_ids_tensor = torch.cat([current_ids_tensor, torch.tensor([[next_id]], device=DEVICE)], dim=1)
            
    print("\n")
    updated_history = processor.tokenizer.decode(current_ids_tensor[0].tolist())
    return updated_history.replace("<eos>", "")