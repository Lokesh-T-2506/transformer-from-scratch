import torch
import torch.nn as nn

from multi_head_attention import MultiHeadAttention
from feed_forward import FeedForward
from add_norm import AddNorm


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        # TODO 1: define self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        # TODO 2: define self.feed_forward = FeedForward(d_model, d_ff)
        self.ffd = FeedForward(d_model, d_ff)
        # TODO 3: define TWO separate AddNorm instances (one per sublayer)
        # e.g. self.add_norm1, self.add_norm2 = AddNorm(d_model), AddNorm(d_model)
        self.add_norm1 = AddNorm(d_model)
        self.add_norm2 = AddNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # x shape: (batch, seq_len, d_model)
        # mask: padding mask, broadcastable against (batch, num_heads, seq_len, seq_len)

        # TODO 4: x = self.add_norm1(x, self.self_attn)
        # self_attn now optionally takes a mask, but AddNorm's sublayer(x) call
        # only passes one arg — use a lambda closure to thread mask through:
        # self.add_norm1(x, lambda t: self.self_attn(t, mask))
        x = self.add_norm1(x, lambda t: self.self_attn(t, mask))
        # TODO 5: x = self.add_norm2(x, self.feed_forward)
        x = self.add_norm2(x, self.ffd)
        # TODO 6: return x
        return x


class Encoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        # TODO 7: define self.layers = nn.ModuleList([...]) containing
        # num_layers independent EncoderLayer instances
        # (careful: don't create ONE EncoderLayer and repeat the same instance
        # num_layers times in the list — each needs its own weights)
        self.layers  = nn.ModuleList(EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers))

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # x shape: (batch, seq_len, d_model)

        # TODO 8: pass x (and mask) through each layer in self.layers in sequence, return the result
        for layer in self.layers:
            x = layer(x, mask)
        return x


if __name__ == "__main__":
    encoder = Encoder(num_layers=6, d_model=16, num_heads=4, d_ff=64)
    x = torch.randn(2, 10, 16)  # (batch=2, seq_len=10, d_model=16)
    out = encoder(x)
    print(out.shape)  # expect torch.Size([2, 10, 16])
