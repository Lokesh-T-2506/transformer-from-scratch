import torch
import torch.nn as nn


class AddNorm(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        # TODO 1: define self.norm = nn.LayerNorm(d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, sublayer) -> torch.Tensor:
        # x shape: (batch, seq_len, d_model)
        # sublayer: a callable (module or function) that takes x and returns
        # a tensor of the SAME shape as x (e.g. self-attention, or feed-forward)

        # TODO 2: return self.norm(x + sublayer(x))
        return self.norm(x + sublayer(x))


if __name__ == "__main__":
    add_norm = AddNorm(d_model=16)
    x = torch.randn(2, 10, 16)  # (batch=2, seq_len=10, d_model=16)

    # sublayer here is just a stand-in identity-like function for the smoke test —
    # in the real encoder/decoder layers, this will be self-attention or the FFN module
    out = add_norm(x, lambda t: t * 2)
    print(out.shape)  # expect torch.Size([2, 10, 16])