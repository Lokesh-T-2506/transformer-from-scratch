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
- [x] Embeddings + output projection (`embeddings.py`)
- [x] Full model assembly (`transformer.py`)
- [x] Toy training loop (`tokenizer.py`, `dataset.py`, `train.py`)

All core architecture components are built and verified. Current training setup uses a tiny
synthetic 5-pair corpus and a whitespace-level tokenizer to validate the pipeline end-to-end
(loss drops from ~75.8 to ~0.05 over 50 epochs). Next: swap in a real Python corpus and train
on Colab GPU (this machine has no CUDA-capable GPU).

## Setup

Requires PyTorch. No other dependencies yet.