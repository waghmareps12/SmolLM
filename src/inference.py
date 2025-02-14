import torch
from transformers import AutoTokenizer
from model import SmallLanguageModel, ModelConfig  # Remove src. prefix

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
# ... rest of the code ... 