import torch
from transformers import AutoTokenizer
from model import SmallLanguageModel, ModelConfig

def create_model_config(vocab_size):
    """Create model configuration matching training"""
    return ModelConfig(
        vocab_size=vocab_size,
        block_size=512,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.1,
        bias=True
    )

def generate_text(prompt, model, tokenizer, max_length=100, temperature=0.8, top_k=50):
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    with torch.no_grad():
        for _ in range(max_length):
            # Get model predictions
            outputs = model(input_ids)
            next_token_logits = outputs[:, -1, :] / temperature
            
            # Apply top-k filtering
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
            next_token_logits[0, :] = float('-inf')
            next_token_logits[0, top_k_indices[0]] = top_k_logits[0]
            
            # Sample from the filtered distribution
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to input_ids
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            # Stop if we generate the EOS token
            if next_token[0].item() == tokenizer.eos_token_id:
                break
    
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create and load model
    config = create_model_config(tokenizer.vocab_size)
    model = SmallLanguageModel(config).to(device)
    
    # Load trained weights
    try:
        checkpoint = torch.load("small_language_model.pt", map_location=device)
        model.load_state_dict(checkpoint)
        print("Loaded model from small_language_model.pt")
    except FileNotFoundError:
        print("No saved model found. Please train the model first.")
        exit(1)
    
    # Generate some example texts
    prompts = [
        "Once upon a time",
        "The meaning of life is",
        "In the distant future",
        "The best way to learn programming is"
    ]
    
    print("\nGenerating text samples:\n")
    for prompt in prompts:
        print(f"Prompt: {prompt}")
        generated_text = generate_text(
            prompt, 
            model, 
            tokenizer, 
            max_length=100,
            temperature=0.8,
            top_k=50
        )
        print(f"Generated: {generated_text}\n")
