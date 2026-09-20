import math

import torch
import torch.nn as nn

from positional_encoding import PositionalEncoding


class Embeddings(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int):
        super().__init__()
        # TODO 1: create the lookup table that maps an integer token ID to a
        # d_model-dimensional vector (PyTorch has a built-in layer for exactly
        # this -- one learnable row per vocab entry)
        self.embedding = nn.Embedding(vocab_size, d_model)
        # TODO 2: store an instance of your existing PositionalEncoding class
        # (reuse it, don't reimplement -- it already takes d_model and max_seq_len)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        # TODO 3: store whatever you'll need for the scaling step in forward()
        self.scale = math.sqrt(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len) -- integer token IDs, NOT one-hot or embeddings

        # TODO 4: look up the embeddings for x, then scale them
        # (think back to the sqrt(d_model) discussion -- why sqrt, not d_model itself)
        embeddings = self.embedding(x) * self.scale
        # TODO 5: add positional information to the scaled embeddings and return
        return self.positional_encoding(embeddings)


class OutputProjection(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, tied_embedding: nn.Embedding = None):
        super().__init__()
        # TODO 6: define the projection from d_model back to vocab-size logits
        self.proj = nn.Linear(d_model, vocab_size)
        # TODO 7: if tied_embedding was passed in, tie the projection's weight
        # to it -- shapes already match (as discussed), so this should be a
        # direct assignment of the SAME parameter object, no transpose needed
        if tied_embedding is not None:
            self.proj.weight = tied_embedding.weight
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, d_model) -- decoder's final output

        # TODO 8: apply the projection and return
        # output shape: (batch, seq_len, vocab_size) -- raw logits, no softmax here
        # (nn.CrossEntropyLoss expects raw logits during training)
        return self.proj(x)


if __name__ == "__main__":
    vocab_size, d_model, max_seq_len = 100, 16, 50

    embed = Embeddings(vocab_size, d_model, max_seq_len)
    out_proj = OutputProjection(d_model, vocab_size, tied_embedding=embed.embedding)

    tokens = torch.randint(0, vocab_size, (2, 10))  # (batch=2, seq_len=10) token IDs
    embedded = embed(tokens)
    print("embedded shape:", embedded.shape)  # expect torch.Size([2, 10, 16])

    logits = out_proj(embedded)
    print("logits shape:", logits.shape)  # expect torch.Size([2, 10, 100])

    # confirm weight tying actually worked -- same object, not just same values
    print("weights are the same object:", out_proj.proj.weight is embed.embedding.weight)
