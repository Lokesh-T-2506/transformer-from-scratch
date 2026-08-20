import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int):
        super().__init__()
        # TODO 1: build the position column, shape (max_seq_len, 1)
        pos = torch.arange(0,max_seq_len).unsqueeze(1).float()
        # TODO 2: build the frequency term for i = 0, 2, 4, ..., d_model - 2
        i=torch.arange(0,d_model,2)
        freq = torch.exp(-torch.log(torch.tensor(10000.0)) * i / d_model).float()
        # TODO 3: outer-product position and frequency -> (max_seq_len, d_model/2)
        outer_product = pos * freq.unsqueeze(0)
        # TODO 4: allocate pe of shape (max_seq_len, d_model)
        pe = torch.zeros(max_seq_len, d_model)
        # TODO 5: fill even columns with sin, odd columns with cos
        pe[:,i] = torch.sin(outer_product)
        pe[:,i+1] = torch.cos(outer_product)
        # TODO 6: register pe as a buffer (not a parameter)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, d_model)
        # TODO 7: slice self.pe down to x's seq_len and add it to x
        x = x + self.pe[:x.size(1), :]
        return x


if __name__ == "__main__":
    pe = PositionalEncoding(d_model=16, max_seq_len=50)
    x = torch.zeros(2, 10, 16)  # (batch=2, seq_len=10, d_model=16)
    out = pe(x)
    print(out.shape)  # expect torch.Size([2, 10, 16])
