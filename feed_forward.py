import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        # TODO 1: define self.net = nn.Sequential(...) containing, in order:
        # Linear(d_model -> d_ff), ReLU, Linear(d_ff -> d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, d_model)

        # TODO 2: return self.net(x)
        # output shape should match input: (batch, seq_len, d_model)
        return self.ffn(x)


if __name__ == "__main__":
    ffn = FeedForward(d_model=16, d_ff=64)
    x = torch.randn(2, 10, 16)  # (batch=2, seq_len=10, d_model=16)
    out = ffn(x)
    print(out.shape)  # expect torch.Size([2, 10, 16])