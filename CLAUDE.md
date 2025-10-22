# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a playground project for building a GPT-2 style language model from scratch (nanoGPT). The implementation closely follows the GPT-2 architecture with custom PyTorch modules for training and inference.

## Development Commands

### Environment Setup
```bash
# Python version
python --version  # Should be >=3.12

# Dependencies managed via uv
uv sync  # Install dependencies from uv.lock
```

### Running Training
```bash
# Main training script (executes src/train_gpt2.py implicitly via current setup)
python src/train_gpt2.py

# Note: Training requires data/input.txt to be present
```

### Package Management
The project uses `uv` for dependency management. Dependencies are declared in `pyproject.toml`:
- torch (>=2.9.0)
- tiktoken (>=0.12.0)
- transformers (>=4.57.1)

## Code Architecture

### Model Implementation (src/train_gpt2.py)

The entire GPT-2 implementation is contained in a single file with the following components:

**Core Model Classes:**
- `CausalSelfAttention`: Multi-head causal attention with masking for autoregressive generation
- `MLP`: Feed-forward network with GELU activation (4x expansion ratio)
- `Block`: Single transformer block combining LayerNorm + Attention + MLP with residual connections
- `GPT`: Main model class with token/positional embeddings and transformer blocks
- `GPTConfig`: Dataclass holding model hyperparameters (n_layer, n_head, n_embd, block_size, vocab_size)

**Key Architectural Details:**
- Weight sharing: Token embedding (`wte`) weights are tied with language model head (`lm_head`)
- Custom weight initialization: Uses `NANOGPT_SCALE_INIT` attribute on specific layers (c_proj) for scaled initialization based on model depth
- Standard config: 12 layers, 12 heads, 768 embedding dimension (GPT-2 small)

**Model Loading:**
- `GPT.from_pretrained(model_type)`: Loads pre-trained weights from HuggingFace transformers
  - Supports: 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'
  - Handles weight transposition for attention and MLP projection layers
  - Weight mapping validated via shape assertions

**Training Components:**
- `DataLoaderLite`: Simple data loader that reads from data/input.txt
  - Uses tiktoken GPT-2 encoding
  - Auto-loops when reaching end of data
- Training loop at bottom of file (currently 50 steps with AdamW optimizer, lr=3e-4)
- Device auto-detection: CUDA > MPS > CPU

**Generation Code (currently disabled):**
- Top-k sampling implementation (k=50)
- Temperature-based generation via softmax
- Located after `sys.exit(0)`

### Data Requirements

The training script expects:
- `data/input.txt`: Training corpus (currently ~1.1MB text file)
- Text is tokenized using tiktoken's GPT-2 BPE encoding

### Important Implementation Notes

1. **Weight initialization**: The `_init_weights` method applies custom scaling to layers marked with `NANOGPT_SCALE_INIT` attribute
2. **Forward pass**: Model returns `(logits, loss)` tuple when targets provided, just `logits` otherwise (though generation code expects single return)
3. **Training state**: Current script trains for 50 steps then exits (with`sys.exit(0)`)

### Testing

No formal test suite currently exists. Model verification is done via:
- Shape assertions during weight loading
- Training loss monitoring during optimization loop
- Generation quality (when generation code is enabled)
