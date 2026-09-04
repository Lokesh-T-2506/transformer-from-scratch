import torch
import torch.nn as nn

from multi_head_attention import MultiHeadAttention
from feed_forward import FeedForward
from add_norm import AddNorm


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        # TODO 1: self.self_attn = MultiHeadAttention(d_model, num_heads)
        # (masked self-attention over the decoder's own sequence so far)
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        # TODO 2: self.cross_attn = MultiHeadAttention(d_model, num_heads)
        # a SEPARATE instance from self_attn -- different learned weights,
        # even though it's the same class
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        # TODO 3: self.feed_forward = FeedForward(d_model, d_ff)
        self.ffd = FeedForward(d_model, d_ff)
        # TODO 4: THREE separate AddNorm instances, one per sublayer

        self.add_norm1 = AddNorm(d_model)
        self.add_norm2 = AddNorm(d_model)
        self.add_norm3 = AddNorm(d_model)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                src_mask: torch.Tensor = None, tgt_mask: torch.Tensor = None) -> torch.Tensor:
        # x shape: (batch, tgt_len, d_model) -- the decoder's own sequence
        # encoder_output shape: (batch, src_len, d_model) -- the encoder's final output ("memory")
        # src_mask: padding mask over the SOURCE sequence, for cross-attention
        # tgt_mask: causal (+ padding) mask over the TARGET sequence, for self-attention

        # TODO 5: masked self-attention sublayer
        # x = self.add_norm1(x, lambda t: self.self_attn(t, mask=tgt_mask))
        x = self.add_norm1(x, lambda t: self.self_attn(t, mask=tgt_mask))
        # TODO 6: cross-attention sublayer
        # query comes from x (the decoder), key/value come from encoder_output
        
        x = self.add_norm2(x,lambda t: self.cross_attn(t,key_value_input=encoder_output, mask=src_mask))
        # TODO 7: feed-forward sublayer
        x = self.add_norm3(x, self.ffd)
        # TODO 8: return x
        return x


class Decoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        # TODO 9: self.layers = nn.ModuleList([...]) of num_layers independent
        # DecoderLayer instances (same pitfall as Encoder -- each needs its own weights)
        self.layers = nn.ModuleList(DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers))
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                src_mask: torch.Tensor = None, tgt_mask: torch.Tensor = None) -> torch.Tensor:
        # TODO 10: pass x through each layer in sequence, threading encoder_output,
        # src_mask, and tgt_mask into every layer call. Return the final x.
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return x


if __name__ == "__main__":
    from masks import make_causal_mask

    decoder = Decoder(num_layers=6, d_model=16, num_heads=4, d_ff=64)

    batch, tgt_len, src_len, d_model = 2, 7, 10, 16
    x = torch.randn(batch, tgt_len, d_model)              # decoder input
    encoder_output = torch.randn(batch, src_len, d_model)  # stand-in for real encoder output
    tgt_mask = make_causal_mask(tgt_len)                    # (1, 1, tgt_len, tgt_len)

    out = decoder(x, encoder_output, src_mask=None, tgt_mask=tgt_mask)
    print(out.shape)  # expect torch.Size([2, 7, 16])
