import torch
import torch.nn as nn
import math

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super().__init__()
        # Split attention into query, key, value projections
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.attention_output = nn.Linear(embed_size, embed_size)
        
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.GELU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)
        self.heads = heads
        self.embed_size = embed_size

    def forward(self, x):
        B, N, C = x.shape
        head_dim = C // self.heads
        
        # Split heads
        q = self.query(x).view(B, N, self.heads, head_dim).transpose(1, 2)
        k = self.key(x).view(B, N, self.heads, head_dim).transpose(1, 2)
        v = self.value(x).view(B, N, self.heads, head_dim).transpose(1, 2)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Combine heads
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.attention_output(out)
        
        # Rest of the forward pass
        x = self.norm1(x + self.dropout(out))
        forward = self.feed_forward(x)
        return self.norm2(x + self.dropout(forward))

class SmallLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size=512, num_layers=6, heads=8, dropout=0.1, forward_expansion=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, heads, dropout, forward_expansion) for _ in range(num_layers)
        ])
        self.ln = nn.LayerNorm(embed_size)
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.fc_out(self.ln(x))
