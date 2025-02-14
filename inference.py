import torch
from transformers import AutoTokenizer
from peft import PeftModel
from model import SmallLanguageModel

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Load base model and LoRA adapter
device = "cuda" if torch.cuda.is_available() else "cpu"
base_model = SmallLanguageModel(vocab_size=tokenizer.vocab_size)
model = PeftModel.from_pretrained(base_model, "lora_adapter").to(device)
model.eval()

# Text generation function
def generate_text(prompt, model, tokenizer, max_length=50):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    output_ids = model.generate(input_ids, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Example usage
print(generate_text("Once upon a time", model, tokenizer))
