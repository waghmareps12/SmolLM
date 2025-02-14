import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                    .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch, sequence length, embedding dim
        
        # calculate query, key, values
        q, k, v = self.c_attn(x).split(self.config.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # causal self-attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.resid_dropout(self.c_proj(y))

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class ModelConfig:
    def __init__(self, vocab_size=50257, block_size=1024, n_layer=24, n_head=16, 
                 n_embd=1024, dropout=0.1, bias=True):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.bias = bias
def count_parameters(model):
    """Count number of trainable parameters in the model"""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate parameters for each component
    embedding_params = model.transformer.wte.weight.numel() + model.transformer.wpe.weight.numel()
    
    attention_params = 0
    mlp_params = 0
    layer_norm_params = 0
    
    for block in model.transformer.h:
        # Attention parameters
        attention_params += (
            block.attn.c_attn.weight.numel() + 
            (block.attn.c_attn.bias.numel() if block.attn.c_attn.bias is not None else 0) +
            block.attn.c_proj.weight.numel() +
            (block.attn.c_proj.bias.numel() if block.attn.c_proj.bias is not None else 0)
        )
        
        # MLP parameters
        mlp_params += (
            block.mlp.c_fc.weight.numel() + 
            (block.mlp.c_fc.bias.numel() if block.mlp.c_fc.bias is not None else 0) +
            block.mlp.c_proj.weight.numel() +
            (block.mlp.c_proj.bias.numel() if block.mlp.c_proj.bias is not None else 0)
        )
        
        # Layer norm parameters
        layer_norm_params += (
            block.ln_1.weight.numel() + 
            (block.ln_1.bias.numel() if block.ln_1.bias is not None else 0) +
            block.ln_2.weight.numel() +
            (block.ln_2.bias.numel() if block.ln_2.bias is not None else 0)
        )
    
    # Final layer norm
    layer_norm_params += (
        model.transformer.ln_f.weight.numel() + 
        (model.transformer.ln_f.bias.numel() if model.transformer.ln_f.bias is not None else 0)
    )
    
    # Print detailed breakdown
    print(f"\nParameter Count Breakdown:")
    print(f"Embeddings: {embedding_params:,} parameters")
    print(f"Attention Layers: {attention_params:,} parameters")
    print(f"MLP Layers: {mlp_params:,} parameters")
    print(f"Layer Normalization: {layer_norm_params:,} parameters")
    print(f"Total: {total:,} parameters")
    
    return total
class SmallLanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

        print("\nModel Configuration:")

        print(f"Layers: {config.n_layer}")

        print(f"Heads: {config.n_head}")

        print(f"Embedding Dimension: {config.n_embd}")

        print(f"Context Window: {config.block_size}")

        count_parameters(self)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, targets=None):
        device = input_ids.device
        b, t = input_ids.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        
        # forward the model
        tok_emb = self.transformer.wte(input_ids)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        if targets is not None:
            # Reshape logits and targets for loss calculation
            logits = logits.reshape(-1, logits.size(-1))
            targets = targets.reshape(-1)
            loss = F.cross_entropy(logits, targets)
            return logits, loss
        
        return logits
