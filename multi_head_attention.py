import math

import torch
import torch.nn as nn

from attention import scaled_dot_product_attention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        # TODO 1: store d_model, num_heads, and d_k (= d_model // num_heads)
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        # TODO 2: define W_q, W_k, W_v as single Linear(d_model, d_model) layers
        # (NOT num_heads separate small layers)
        self.W_q= nn.Linear(d_model, d_model)
        self.W_k= nn.Linear(d_model, d_model)
        self.W_v= nn.Linear(d_model, d_model)

        # TODO 3: define the final output projection W_o: Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # x shape: (batch, seq_len, d_model)
        # mask shape (if provided): broadcastable against (batch, num_heads, seq_len, seq_len)
        batch, seq_len, d_model = x.shape

        # TODO 4: compute Q, K, V via W_q, W_k, W_v
        # each shape: (batch, seq_len, d_model)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # TODO 5: split each into heads
        # (batch, seq_len, d_model) -> (batch, seq_len, num_heads, d_k)
        # -> transpose -> (batch, num_heads, seq_len, d_k)
        Q = Q.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch, seq_len, self.num_heads, self.d_k).transpose(1,2)
        V = V.view(batch, seq_len, self.num_heads, self.d_k).transpose(1,2)

        # TODO 6: call scaled_dot_product_attention(Q, K, V, self.d_k) from attention.py
        # (imported above) — same function you already verified works on 4D
        # (batch, num_heads, seq_len, d_k) tensors, no need to reimplement the math
        # out shape: (batch, num_heads, seq_len, d_k)
        out = scaled_dot_product_attention(Q, K, V, self.d_k, mask)


        # TODO 7: merge heads back
        # transpose -> (batch, seq_len, num_heads, d_k)
        # .contiguous().view -> (batch, seq_len, d_model)
        out_reshaped = out.transpose(1,2).reshape(batch, seq_len, d_model)
        # TODO 8: apply W_o and return
        return self.W_o(out_reshaped)

if __name__ == "__main__":
    mha = MultiHeadAttention(d_model=16, num_heads=4)
    x = torch.randn(2, 10, 16)  # (batch=2, seq_len=10, d_model=16)
    out = mha(x)
    print(out.shape)  # expect torch.Size([2, 10, 16])