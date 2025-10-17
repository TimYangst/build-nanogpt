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
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
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
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
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
    block_size: int = 1024    # Maximum sequence length
    vocab_size: int = 50257   # Number of tokens in the vocabulary
    n_layer: int = 12          # Number of transformer blocks
    n_head: int = 12           # Number of attention heads
    n_embd: int = 768         # Embedding dimension

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
       
    @classmethod
    def from_pretrained(cls, model_type):
        """Load a pre-trained GPT model"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}, "Unsupported model type"
        from transformers import GPT2LMHeadModel
        print(f"Loading pre-trained model: {model_type}")

        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]

        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024

        config = GPTConfig(**config_args)
        model = GPT(config)

        sd = model.state_dict()
        sd_keys = set(sd.keys())
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]

        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_keys) == len(sd_keys_hf), f"State dict keys do not match in length: {len(sd_keys)} != {len(sd_keys_hf)}"

        for k in sd_keys_hf:
          if any(k.endswith(w) for w in transposed):
              assert sd[k].shape[::-1] == sd_hf[k].shape, f"Shape mismatch for {k}"
              with torch.no_grad():
                  sd[k].copy_(sd_hf[k].t())
          else:
              assert sd_hf[k].shape == sd[k].shape, f"Shape mismatch for {k}"
              with torch.no_grad():
                  sd[k].copy_(sd_hf[k])

        return model
    
model = GPT.from_pretrained('gpt2')
print("Model loaded successfully.")