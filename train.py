import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
from model import SmallLanguageModel, ModelConfig
import random

def create_model_config(vocab_size):
    """Create a ~125M parameter model configuration"""
    return ModelConfig(
        vocab_size=vocab_size,
        block_size=512,        # Reduced from 1024
        n_layer=12,           # Reduced from 24
        n_head=12,            # Reduced from 16
        n_embd=768,           # Reduced from 1024
        dropout=0.1,
        bias=True
    )

def setup_training():
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create model configuration
    config = create_model_config(tokenizer.vocab_size)
    
    # Initialize model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SmallLanguageModel(config).to(device)
    
    return model, tokenizer, device

class TextDataset(Dataset):
    def __init__(self, tokenized_texts, block_size, tokenizer):
        self.examples = []
        self.block_size = block_size
        self.tokenizer = tokenizer
        
        # Group texts by exact length
        self.length_groups = {}  # Keep as instance variable
        
        for text in tokenized_texts["input_ids"]:
            if len(text) > 1:  # Ensure text is at least 2 tokens
                # Truncate if longer than block_size + 1
                if len(text) > block_size + 1:
                    text = text[:block_size + 1]
                
                length = len(text)
                if length not in self.length_groups:
                    self.length_groups[length] = []
                self.length_groups[length].append(torch.tensor(text, dtype=torch.long))
        
        # Sort lengths for more efficient batching
        self.lengths = sorted(self.length_groups.keys())
        
        # Create index mapping
        self.length_to_idx = {}
        start_idx = 0
        for length in self.lengths:
            group = self.length_groups[length]
            self.length_to_idx[length] = (start_idx, start_idx + len(group))
            start_idx += len(group)
            self.examples.extend(group)
        
        print(f"Created {len(self.examples)} sequences across {len(self.lengths)} different lengths")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

class BatchSchedulerSampler(torch.utils.data.Sampler):
    """Samples batches according to sequence length"""
    def __init__(self, dataset, batch_size):
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        
        # Create batches for each length
        self.batches = []
        for length in dataset.lengths:
            start_idx, end_idx = dataset.length_to_idx[length]
            # Create batches of indices for this length
            indices = list(range(start_idx, end_idx))
            for i in range(0, len(indices), batch_size):
                self.batches.append(indices[i:i + batch_size])
    
    def __iter__(self):
        # Shuffle batches
        random.shuffle(self.batches)
        for batch in self.batches:
            yield batch
    
    def __len__(self):
        return len(self.batches)

def prepare_dataset(tokenizer, block_size):
    # Load and tokenize dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    def tokenize_function(examples):
        # Remove empty strings and concatenate all texts
        texts = [text for text in examples["text"] if len(text.strip()) > 0]
        return tokenizer(texts, truncation=False, padding=False)
    
    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing texts"
    )
    
    # Create training dataset with tokenizer
    train_dataset = TextDataset(tokenized_dataset["train"], block_size=block_size, tokenizer=tokenizer)
    print(f"Created dataset with {len(train_dataset)} examples")
    return train_dataset

def collate_batch(batch):
    # All tensors in a batch should be the same length
    return torch.stack(batch)

def train_model(model, train_loader, optimizer, scheduler, device, num_epochs=3, gradient_accumulation_steps=4):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        optimizer.zero_grad()  # Zero gradients at start of epoch
        
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device)
            
            # Get input_ids and targets
            input_ids = batch[:, :-1].contiguous()
            targets = batch[:, 1:].contiguous()
            
            # Forward pass
            logits, loss = model(input_ids, targets)
            
            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            # Update weights every gradient_accumulation_steps
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * gradient_accumulation_steps
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item() * gradient_accumulation_steps:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, f'checkpoint_epoch_{epoch+1}.pt')

def main():
    # Setup
    model, tokenizer, device = setup_training()
    
    # Prepare dataset
    train_dataset = prepare_dataset(tokenizer, model.config.block_size)
    
    # Use custom sampler instead of shuffle
    train_loader = DataLoader(
        train_dataset, 
        batch_sampler=BatchSchedulerSampler(train_dataset, batch_size=4),  # Reduced batch size from 8 to 4
        num_workers=4
    )
    
    # Training setup with gradient accumulation
    optimizer = optim.AdamW(model.parameters(), 
                           lr=3e-4, 
                           weight_decay=0.1)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=len(train_loader) * 3,  # 3 epochs
        eta_min=1e-5
    )
    
    # Train the model
    train_model(model, train_loader, optimizer, scheduler, device)
    
    # Save the final model
    torch.save(model.state_dict(), "small_language_model.pt")
    
if __name__ == "__main__":
    main()
