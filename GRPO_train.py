import torch
import torch.optim as optim
from transformers import AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.data import DataLoader, Dataset
from model import SmallLanguageModel

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Load dataset and tokenize
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Convert dataset to PyTorch DataLoader
class TextDataset(Dataset):
    def __init__(self, tokenized_texts):
        self.data = tokenized_texts["input_ids"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx])

train_dataset = TextDataset(tokenized_dataset["train"])
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Initialize model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SmallLanguageModel(vocab_size=tokenizer.vocab_size).to(device)

# Apply LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Optimizer and Loss
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

# GRPO hyperparameter (controls regularization strength)
lambda_grpo = 0.01

def train_model(model, dataloader, epochs=3):
    model.train()
    for epoch in range(epochs):
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            output = model(batch)
            loss = criterion(output.view(-1, tokenizer.vocab_size), batch.view(-1))
            
            # Compute gradients
            loss.backward()
            
            # GRPO Regularization: Penalize large gradients
            with torch.no_grad():
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad += lambda_grpo * param  # GRPO Penalty

            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

train_model(model, train_loader)

# Save the LoRA adapter
model.save_pretrained("lora_adapter")
