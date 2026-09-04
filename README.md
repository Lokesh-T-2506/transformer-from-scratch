# Transformer from Scratch

A full encoder-decoder Transformer ("Attention Is All You Need"), implemented from
scratch in PyTorch — every module hand-built (no `nn.TransformerEncoder` /
`nn.MultiheadAttention` shortcuts). Trained on a small Python corpus for a
code-completion task.

## Progress

- [x] Positional encoding (`positional_encoding.py`)
- [x] Single-head scaled dot-product attention (`attention.py`)
- [x] Multi-head attention (`multi_head_attention.py`)
- [x] Position-wise feed-forward network (`feed_forward.py`)
- [x] Residual connections + LayerNorm (`add_norm.py`)
- [x] Encoder layer + encoder stack (`encoder.py`)
- [x] Masking (padding + causal) (`masks.py`)
- [x] Decoder layer + decoder stack (`decoder.py`)
- [ ] Embeddings + output projection
- [ ] Full model assembly
- [ ] Toy training loop (Colab GPU, code-completion corpus)

## Setup

Requires PyTorch. No other dependencies yet.