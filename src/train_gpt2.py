from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

# Causal Self-Attention module
class CasualSelfAttention(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        # Ensure embedding dimension is divisible by the number of heads
        assert config.n_embd % config.n_head == 0
        # Linear layer to project input to query, key, and value matrices
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Linear layer to project the output of the attention mechanism
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # Number of attention heads
        self.n_head = config.n_head
        # Embedding dimension
        self.n_embd = config.n_embd
        # Causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        # Get input dimensions: Batch size, Time steps, Embedding size (n_embd)
        B, T, C = x.size()
        # Compute query, key, value matrices in a single pass
        qkv = self.c_attn(x)
        # Split the qkv matrix into query, key, and value matrices
        q, k, v = qkv.split(self.n_embd, dim=2)
        # Reshape and transpose query, key, and value for multi-head attention
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        # Compute attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # Apply causal mask to the attention scores
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        # Apply softmax to get attention weights
        att = F.softmax(att, dim=-1)
        # Compute the output of the attention mechanism
        y = att @ v  # (B, nh, T, hs)
        # Reshape and transpose the output
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # Project the output to the final embedding dimension
        y = self.c_proj(y)
        return y

# Multi-Layer Perceptron module
class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        # Linear layer to project input to a higher dimension
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        # GELU activation function
        self.gelu = nn.GELU(approximate='tanh')
        # Linear layer to project the output back to the original dimension
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        # Forward pass through the MLP
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

# Transformer Block module
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Layer normalization before the attention mechanism
        self.ln_1 = nn.LayerNorm(config.n_embd)
        # Causal self-attention mechanism
        self.attn = CasualSelfAttention(config)
        # Layer normalization before the MLP
        self.ln_2 = nn.LayerNorm(config.n_embd)
        # Multi-layer perceptron
        self.mlp = MLP(config)

    def forward(self, x):
        # Forward pass through the transformer block with residual connections
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

# GPT model configuration
@dataclass
class GPTConfig:
    block_size: int = 256  # Maximum sequence length
    vocab_size: int = 65   # Number of tokens in the vocabulary
    n_layer: int = 6       # Number of transformer blocks
    n_head: int = 6        # Number of attention heads
    n_embd: int = 384      # Embedding dimension

# GPT model
class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        # Transformer module dictionary
        self.transformer = nn.ModuleDict(dict(
            # Token embedding layer
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # Positional embedding layer
            wpe = nn.Embedding(config.block_size, config.n_embd),
            # List of transformer blocks
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # Final layer normalization
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        # Language model head to project the output to the vocabulary size
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying between token embedding and language model head
        self.transformer.wte.weight = self.lm_head.weight
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # Initialize weights for linear and embedding layers
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # Get input dimensions
        B, T = idx.size()
        # Ensure sequence length is within the block size
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # Get positional embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        # Get token embeddings
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        # Add token and positional embeddings
        x = tok_emb + pos_emb
        # Forward pass through the transformer blocks
        for block in self.transformer.h:
            x = block(x)
        # Final layer normalization
        x = self.transformer.ln_f(x)
        # Get logits from the language model head
        logits = self.lm_head(x) # (B, T, vocab_size)
        # Compute loss if targets are provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss