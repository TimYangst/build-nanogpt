"""
GPT-2 Implementation from Scratch

This module implements a GPT-2 style transformer language model in PyTorch.
Includes model architecture, weight initialization, pre-trained model loading,
training loop, and text generation.

Architecture:
    - Token and positional embeddings
    - Stack of transformer blocks (multi-head attention + MLP)
    - Language modeling head with weight tying

Key Features:
    - Causal self-attention for autoregressive generation
    - Pre-normalization (LayerNorm before sub-layers)
    - Custom weight initialization with residual scaling
    - Compatible with HuggingFace GPT-2 pre-trained weights
"""

from dataclasses import dataclass
import inspect
import math
import torch
import torch.nn as nn
from torch.nn import functional as F


# ============================================================================
# Causal Self-Attention Module
# ============================================================================
# Multi-head causal self-attention with masking for autoregressive generation.
# Implements the standard scaled dot-product attention mechanism with a causal
# mask to prevent attending to future positions in the sequence.

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        # Validate that embedding dimension is evenly divisible by number of heads
        assert config.n_embd % config.n_head == 0

        # Combined linear projection for query, key, and value (more efficient than separate layers)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # Output projection layer
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # Mark for scaled initialization (std = (2 * n_layer)^-0.5)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        # Store attention configuration
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # Register causal mask as a buffer (not a trainable parameter)
        # Lower triangular matrix ensures position i can only attend to positions <= i
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        # Input shape: (B, T, C) where B=batch, T=sequence length, C=n_embd
        B, T, C = x.size()

        # Step 1: Compute Q, K, V in a single matrix multiplication
        qkv = self.c_attn(x)  # (B, T, 3*C)
        q, k, v = qkv.split(self.n_embd, dim=2)  # Each: (B, T, C)

        # Step 2: Reshape for multi-head attention
        # Split embedding dimension across heads: C = n_head * head_size
        # Transpose to get heads dimension before sequence dimension
        head_size = C // self.n_head
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, n_head, T, head_size)
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, n_head, T, head_size)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, n_head, T, head_size)

        # # Step 3: Compute scaled dot-product attention scores
        # # Scale by 1/sqrt(d_k) to prevent softmax saturation
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, n_head, T, T)

        # # Step 4: Apply causal mask (set future positions to -inf before softmax)
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))

        # # Step 5: Normalize attention scores to get attention weights
        # att = F.softmax(att, dim=-1)  # (B, n_head, T, T)

        # # Step 6: Apply attention weights to values
        # y = att @ v  # (B, n_head, T, head_size)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Step 7: Reassemble all head outputs
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)

        # Step 8: Final output projection
        y = self.c_proj(y)
        return y

# ============================================================================
# Multi-Layer Perceptron (MLP) Module
# ============================================================================
# Position-wise feed-forward network applied after attention in each transformer block.
# Uses a 4x expansion ratio (hidden dimension = 4 * n_embd) following GPT-2 architecture.

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        # Expansion layer: project from n_embd to 4*n_embd
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)

        # GELU activation with tanh approximation (matches GPT-2 implementation)
        self.gelu = nn.GELU(approximate='tanh')

        # Projection layer: project back from 4*n_embd to n_embd
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        # Mark for scaled initialization (std = (2 * n_layer)^-0.5)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        # MLP forward pass: Linear -> GELU -> Linear
        # Input/Output shape: (B, T, n_embd)
        x = self.c_fc(x)      # (B, T, 4*n_embd)
        x = self.gelu(x)      # (B, T, 4*n_embd)
        x = self.c_proj(x)    # (B, T, n_embd)
        return x

# ============================================================================
# Transformer Block Module
# ============================================================================
# Single transformer layer combining multi-head attention and feed-forward network.
# Uses pre-normalization (LayerNorm before sub-layers) and residual connections.

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        # Layer normalization applied before attention (pre-norm architecture)
        self.ln_1 = nn.LayerNorm(config.n_embd)

        # Multi-head causal self-attention
        self.attn = CausalSelfAttention(config)

        # Layer normalization applied before MLP (pre-norm architecture)
        self.ln_2 = nn.LayerNorm(config.n_embd)

        # Position-wise feed-forward network
        self.mlp = MLP(config)

    def forward(self, x):
        # Transformer block with residual connections:
        # x = x + Attention(LayerNorm(x))
        # x = x + MLP(LayerNorm(x))
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

# ============================================================================
# GPT Model Configuration
# ============================================================================
# Dataclass storing hyperparameters for the GPT model architecture.

@dataclass
class GPTConfig:
    block_size: int = 1024    # Maximum sequence length (context window)
    vocab_size: int = 50257   # Number of tokens (GPT-2 BPE vocabulary size)
    n_layer: int = 12         # Number of transformer blocks (depth)
    n_head: int = 12          # Number of attention heads per block
    n_embd: int = 768         # Embedding dimension (model width)


# ============================================================================
# GPT Model
# ============================================================================
# Full GPT-2 style language model with token/positional embeddings,
# transformer blocks, and language modeling head.

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Define all transformer components in a ModuleDict
        self.transformer = nn.ModuleDict(dict(
            # Token embedding: maps token IDs to embedding vectors
            wte = nn.Embedding(config.vocab_size, config.n_embd),

            # Positional embedding: learned position encodings
            wpe = nn.Embedding(config.block_size, config.n_embd),

            # Stack of transformer blocks
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),

            # Final layer normalization applied before output projection
            ln_f = nn.LayerNorm(config.n_embd)
        ))

        # Language model head: projects hidden states to vocabulary logits
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share weights between token embedding and output projection
        # This reduces parameters and improves performance
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize all weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialize weights for all layers in the model.
        Uses GPT-2 initialization scheme with special scaling for residual projections.
        """
        if isinstance(module, nn.Linear):
            # Default standard deviation for linear layers
            std = 0.02

            # Special scaled initialization for residual projection layers
            # Marked with NANOGPT_SCALE_INIT attribute (c_proj in attention and MLP)
            # Scale by 1/sqrt(2*n_layer) to account for residual path accumulation
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std = (2.0 * self.config.n_layer) ** -0.5

            # Initialize weights from normal distribution
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

            # Zero-initialize biases if present
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            # Initialize embedding weights from normal distribution
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        Forward pass through the GPT model.

        Args:
            idx: Input token indices, shape (B, T) where B=batch, T=sequence length
            targets: Optional target token indices for computing loss, shape (B, T)

        Returns:
            logits: Predicted logits for each position, shape (B, T, vocab_size)
            loss: Cross-entropy loss if targets provided, otherwise None
        """
        # Input shape: (B, T)
        B, T = idx.size()

        # Validate sequence length doesn't exceed model's context window
        assert T <= self.config.block_size, \
            f"Sequence length {T} exceeds block size {self.config.block_size}"

        # Step 1: Create embeddings
        token_embeddings = self.transformer.wte(idx)  # (B, T, n_embd)
        position_ids = torch.arange(0, T, dtype=torch.long, device=idx.device)  # (T,)
        position_embeddings = self.transformer.wpe(position_ids)  # (T, n_embd)

        # Step 2: Combine token and position information
        x = token_embeddings + position_embeddings  # (B, T, n_embd)

        # Step 3: Pass through all transformer blocks
        for block in self.transformer.h:
            x = block(x)  # (B, T, n_embd)

        # Step 4: Apply final layer normalization
        x = self.transformer.ln_f(x)  # (B, T, n_embd)

        # Step 5: Project to vocabulary space to get logits
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Step 6: Compute loss if targets are provided
        loss = None
        if targets is not None:
            # Flatten batch and sequence dimensions for cross-entropy
            # Shape: (B*T, vocab_size) and (B*T,)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
       
    @classmethod
    def from_pretrained(cls, model_type):
        """
        Load pre-trained GPT-2 weights from HuggingFace transformers library.

        Args:
            model_type: One of 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'

        Returns:
            GPT model initialized with pre-trained weights

        Note:
            Handles weight transposition for attention/MLP projection layers due to
            different weight matrix conventions between HuggingFace and this implementation.
        """
        # Validate model type
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}, \
            f"Unsupported model type: {model_type}"

        from transformers import GPT2LMHeadModel
        print(f"Loading pre-trained model: {model_type}")

        # Step 1: Create model configuration matching the requested model size
        config_args = {
            'gpt2':        dict(n_layer=12, n_head=12, n_embd=768),   # 117M params
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),  # 345M params
            'gpt2-large':  dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            'gpt2-xl':     dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]

        # Add vocabulary and context window size (same for all GPT-2 variants)
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024

        # Step 2: Initialize our model with random weights
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()

        # Get our model's parameter names (excluding attention bias buffer)
        sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')]

        # Step 3: Load pre-trained weights from HuggingFace
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # Get HuggingFace model's parameter names (excluding buffers we don't use)
        sd_keys_hf = [k for k in sd_hf.keys()
                      if not k.endswith('.attn.masked_bias')  # HF-specific buffer
                      and not k.endswith('.attn.bias')]        # Attention mask buffer

        # Step 4: Validate parameter counts match
        assert len(sd_keys) == len(sd_keys_hf), \
            f"Parameter count mismatch: {len(sd_keys)} != {len(sd_keys_hf)}"

        # Step 5: Copy weights from HuggingFace model to our model
        # Some weights need to be transposed due to different Conv1D vs Linear conventions
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight',
                      'mlp.c_fc.weight', 'mlp.c_proj.weight']

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # HuggingFace uses Conv1D which stores weights transposed
                assert sd[k].shape[::-1] == sd_hf[k].shape, \
                    f"Shape mismatch for {k}: {sd[k].shape} vs {sd_hf[k].shape}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # Direct copy for embeddings, layer norms, etc.
                assert sd[k].shape == sd_hf[k].shape, \
                    f"Shape mismatch for {k}: {sd[k].shape} vs {sd_hf[k].shape}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizer(self, weight_decay, learning_rate, device):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decay params tensors: {len(decay_params)}, total size: {num_decay_params:,} parameters")
        print(f"num no decay params tensors: {len(nodecay_params)}, total size: {num_nodecay_params:,} parameters")

        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device == 'cuda'
        print(f"Using fused AdamW: {use_fused} (fused available: {fused_available}, device: {device})")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

# ============================================================================
# Device Configuration
# ============================================================================
# Auto-detect best available device: CUDA GPU > Apple Silicon MPS > CPU

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
print(f"Using device: {device}")


# ============================================================================
# Data Loader
# ============================================================================

import tiktoken


class DataLoaderLite:
    """
    Lightweight data loader for language modeling.
    Loads text from a file, tokenizes it, and yields batches for training.
    """

    def __init__(self, B, T):
        """
        Initialize the data loader.

        Args:
            B: Batch size (number of sequences per batch)
            T: Sequence length (number of tokens per sequence)
        """
        self.B = B
        self.T = T
        self.current_pos = 0

        # Load and tokenize the training data
        with open("data/input.txt", "r") as f:
            text = f.read()

        # Use GPT-2 BPE tokenizer
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)

        print(f"Loaded {len(self.tokens)} tokens.")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches.")

    def next_batch(self):
        """
        Get the next batch of data.

        Returns:
            x: Input sequences, shape (B, T)
            y: Target sequences (shifted by 1), shape (B, T)
        """
        B, T = self.B, self.T

        # Extract chunk of B*T+1 tokens (extra token for target shift)
        buf = self.tokens[self.current_pos : self.current_pos + B * T + 1]

        # Input: tokens 0 to B*T-1
        x = buf[:-1].view(B, T)
        # Target: tokens 1 to B*T (shifted by 1 for next-token prediction)
        y = buf[1:].view(B, T)

        # Advance position for next batch
        self.current_pos += B * T

        # Loop back to beginning when reaching end of data
        if self.current_pos + B * T >= len(self.tokens):
            self.current_pos = 0

        return x, y


# ============================================================================
# Training Setup
# ============================================================================
import time

# Set random seeds for reproducibility
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

total_batch_size = 524288  # Total tokens per batch (B * T)
B = 32 # micro-batch size
T = 1024
assert total_batch_size % (B * T) == 0
grad_accum_steps = total_batch_size // (B * T)
print("Total batch size (tokens per batch):", total_batch_size)
print(f"Using batch size B={B}, sequence length T={T}, grad_accum_steps={grad_accum_steps}")

# Initialize data loader
# B=4: batch size, T=32: sequence length
train_loader = DataLoaderLite(B=B, T=T)

# Configure TF32 precision using new API (PyTorch 2.9+)
torch.backends.cuda.matmul.fp32_precision = 'tf32'
torch.backends.cudnn.conv.fp32_precision = 'tf32'

# Initialize model
# Option 1: Load pre-trained GPT-2 weights
# model = GPT.from_pretrained('gpt2')

# Option 2: Train from scratch with random initialization
model = GPT(GPTConfig(vocab_size=50304))
model = model.to(device)
model = torch.compile(model) if torch.__version__ >= "2.0.0" else model

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50
def get_lr(step):
    if step < warmup_steps:
        return max_lr * (step+1) / warmup_steps
    
    if step >= max_steps:
        return min_lr
    
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0.0 <= decay_ratio <= 1.0
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)   


# Initialize optimizer
optimizer = model.configure_optimizer(
    weight_decay=0.1,
    learning_rate=max_lr,
    device=device
)


# ============================================================================
# Training Loop
# ============================================================================

for steps in range(50):
    t0 = time.time()
   
    # Backward pass: compute gradients
    optimizer.zero_grad()
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16 if device == 'cuda' else torch.float32):
            # Forward pass: compute predictions and loss
            logits, loss = model(x, y)
            # import code; code.interact(local=locals())
        loss = loss / grad_accum_steps
        loss.backward()
        
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    lr = get_lr(steps)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    # Update weights
    optimizer.step()

    torch.cuda.synchronize() if device == 'cuda' else None

    t1 = time.time()
    dt = (t1 - t0) * 1000
    tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)

    # Log training progress
    print(f"Step {steps:4d}, Loss: {loss.item():.6f}, norm: {norm:.4f}, Time per batch: {dt:.2f} ms, tokens/sec: {tokens_per_sec:.2f}")

# Training complete - exit before generation code
import sys; sys.exit(0)


# ============================================================================
# Text Generation (Disabled - exits before reaching this code)
# ============================================================================

# Prepare model for inference
model.eval()
model.to(device)

# Generation parameters
num_return_squences = 5  # Number of different sequences to generate
max_length = 30          # Maximum length of each generated sequence

# Prepare prompt
import tiktoken
enc = tiktoken.get_encoding("gpt2")

# Encode the prompt text
prompt = "Hello, I'm a language model,"
tokens = enc.encode(prompt)
tokens = torch.tensor(tokens, dtype=torch.long)

# Create batch: repeat the prompt for each sequence we want to generate
tokens = tokens.unsqueeze(0).repeat(num_return_squences, 1)  # (num_sequences, prompt_length)
x = tokens.to(device)

# Set random seed for reproducible generation
torch.manual_seed(42)
if device == 'cuda':
    torch.cuda.manual_seed(42)

# Autoregressive generation loop
# Generate one token at a time until reaching max_length
while x.size(1) < max_length:
    with torch.no_grad():  # Disable gradient computation for inference
        # Step 1: Get model predictions
        logits = model(x)           # (B, current_length, vocab_size)
        logits = logits[:, -1, :]   # Only use logits for last position: (B, vocab_size)

        # Step 2: Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)  # (B, vocab_size)

        # Step 3: Top-k sampling (sample from top 50 most likely tokens)
        topk_probs, topk_indices = torch.topk(probs, k=50, dim=-1)  # (B, 50)

        # Step 4: Sample one token from the top-k distribution
        ix = torch.multinomial(topk_probs, num_samples=1)  # (B, 1)

        # Step 5: Map sampled index back to actual token ID
        xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)

        # Step 6: Append new token to sequence
        x = torch.cat((x, xcol), dim=1)  # (B, current_length + 1)

# Decode and display generated sequences
for i in range(num_return_squences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)