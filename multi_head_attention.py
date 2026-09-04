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

    def forward(self, query_input: torch.Tensor, key_value_input: torch.Tensor = None,
                mask: torch.Tensor = None) -> torch.Tensor:
        # query_input shape: (batch, seq_len_q, d_model)
        # key_value_input shape: (batch, seq_len_kv, d_model) -- for SELF-attention this
        # is the same tensor as query_input; for CROSS-attention (decoder attending to
        # encoder output) it's a different tensor, possibly a different seq_len entirely
        # mask shape (if provided): broadcastable against (batch, num_heads, seq_len_q, seq_len_kv)

        # TODO 4a: if key_value_input is None, this is self-attention -> set
        # key_value_input = query_input
        key_value_input = query_input if key_value_input is None else key_value_input
        # TODO 4b: pull batch, seq_len_q, d_model from query_input.shape
        # and seq_len_kv from key_value_input.shape (NOT the same variable as seq_len_q --
        # they can differ in the cross-attention case)
        batch, seq_len_q, d_model = query_input.shape
        seq_len_kv = key_value_input.shape[1]
        # TODO 5: compute Q from query_input via W_q
        # compute K, V from key_value_input via W_k, W_v (NOT from query_input)
        # Q shape: (batch, seq_len_q, d_model)   K, V shape: (batch, seq_len_kv, d_model)
        Q = self.W_q(query_input)
        K = self.W_k(key_value_input)
        V = self.W_v(key_value_input)
        # TODO 6: split each into heads, same view+transpose pattern as before --
        # just make sure Q's split uses seq_len_q and K/V's split uses seq_len_kv
        # Q -> (batch, num_heads, seq_len_q, d_k)   K, V -> (batch, num_heads, seq_len_kv, d_k)
        Q = Q.view(batch, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch, seq_len_kv, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch, seq_len_kv, self.num_heads, self.d_k).transpose(1, 2)
        # TODO 7: call scaled_dot_product_attention(Q, K, V, self.d_k, mask)
        # out shape: (batch, num_heads, seq_len_q, d_k) -- note it's seq_len_q, since
        # attention output always has the shape of the QUERY sequence, regardless of
        # how long the key/value sequence was
        out = scaled_dot_product_attention(Q, K, V, self.d_k, mask)
        # TODO 8: merge heads back -- transpose -> (batch, seq_len_q, num_heads, d_k)
        # -> .contiguous().reshape -> (batch, seq_len_q, d_model)
        out = out.transpose(1,2).reshape(batch, seq_len_q, self.d_model)
        # TODO 9: apply W_o and return
        return self.W_o(out)

if __name__ == "__main__":
    mha = MultiHeadAttention(d_model=16, num_heads=4)

    # self-attention case: query and key/value are the same sequence
    x = torch.randn(2, 10, 16)  # (batch=2, seq_len=10, d_model=16)
    out = mha(x)
    print("self-attention output shape:", out.shape)  # expect torch.Size([2, 10, 16])

    # cross-attention case: query (decoder, len 7) attends to key/value (encoder, len 12)
    decoder_x = torch.randn(2, 7, 16)
    encoder_memory = torch.randn(2, 12, 16)
    out_cross = mha(decoder_x, encoder_memory)
    print("cross-attention output shape:", out_cross.shape)  # expect torch.Size([2, 7, 16])
    # notice: output seq_len matches the QUERY (7), not the key/value source (12)