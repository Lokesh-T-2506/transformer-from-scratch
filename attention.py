import math

import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model: int, d_k: int):
        super().__init__()
        # TODO 1: define the three linear projections: W_q, W_k, W_v
        self.W_q = nn.Linear(d_model, d_k)
        self.W_k = nn.Linear(d_model, d_k)
        self.W_v = nn.Linear(d_model, d_k)
        self.d_k = d_k
        # each should map d_model -> d_k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, d_model)
         # TODO 2: compute Q, K, V by passing x through the projections
        # each shape: (batch, seq_len, d_k)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        return scaled_dot_product_attention(Q, K, V, self.d_k)

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, d_k) -> torch.Tensor: 
        # TODO 3: compute raw attention scores = Q @ K^T
        # careful: transpose only the last two dims of K
        # target shape: (batch, seq_len, seq_len)
        raw_scores = torch.matmul(Q, K.transpose(-2,-1))
        # TODO 4: scale scores by 1/sqrt(d_k)
        scaled_scores = raw_scores/torch.sqrt(torch.tensor(d_k))
        # TODO 5: apply softmax along the correct dimension
        # (which axis represents "distribution over keys for a fixed query"?)
        sm = nn.Softmax(dim=-1)
        sm_scores = sm(scaled_scores)
        # TODO 6: compute output = attention_weights @ V
        # target shape: (batch, seq_len, d_k)
        output = torch.matmul(sm_scores, V)
        return output    



if __name__ == "__main__":
    attn = ScaledDotProductAttention(d_model=16, d_k=16)
    x = torch.randn(2, 10, 16)  # (batch=2, seq_len=10, d_model=16)
    out = attn(x)
    print(out.shape)  # expect torch.Size([2, 10, 16])
