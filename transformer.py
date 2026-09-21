import torch
import torch.nn as nn

from embeddings import Embeddings, OutputProjection
from encoder import Encoder
from decoder import Decoder
from masks import make_padding_mask, make_causal_mask


class Transformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, d_ff: int,
                 num_layers: int, max_seq_len: int, pad_token_id: int):
        super().__init__()
        # TODO 1: create ONE Embeddings instance -- shared for both source and
        # target sides, since they use the same vocabulary and you tied weights
        self.embeddings = Embeddings(vocab_size, d_model, max_seq_len)
        # TODO 2: create the Encoder stack
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff)
        # TODO 3: create the Decoder stack
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff)

        # TODO 4: create the OutputProjection, tied to the SAME embedding
        # instance from TODO 1 (not a new one)
        self.output_proj = OutputProjection(d_model, vocab_size, tied_embedding=self.embeddings.embedding)
        # TODO 5: store pad_token_id -- you'll need it to build padding masks
        # from raw token IDs inside forward()
        self.pad_token_id = pad_token_id

    def forward(self, src_tokens: torch.Tensor, tgt_tokens: torch.Tensor) -> torch.Tensor:
        # src_tokens shape: (batch, src_len) -- raw integer token IDs
        # tgt_tokens shape: (batch, tgt_len) -- raw integer token IDs

        # TODO 6: build the source padding mask from src_tokens
        src_mask = make_padding_mask(src_tokens, self.pad_token_id)
        # TODO 7: build the target padding mask from tgt_tokens
        tgt_pad_mask = make_padding_mask(tgt_tokens, self.pad_token_id)
                                         
        # TODO 8: build the causal mask for the target sequence length
        tgt_cas_mask = make_causal_mask(tgt_tokens.shape[-1])
        # TODO 9: combine the target padding mask and causal mask into one
        # mask (think back to the AND-via-multiplication discussion)
        combined_tgt_mask = tgt_pad_mask * tgt_cas_mask
        # TODO 10: embed src_tokens, run through the encoder with the source
        # mask, to get the encoder output ("memory")
        memory = self.encoder(self.embeddings(src_tokens), src_mask)
        # TODO 11: embed tgt_tokens (same embedding instance), run through the
        # decoder with encoder_output, source mask, and the combined target mask
        decoder_output = self.decoder(self.embeddings(tgt_tokens), memory, src_mask, combined_tgt_mask)
        # TODO 12: project the decoder output to vocabulary logits and return
        return self.output_proj(decoder_output)


if __name__ == "__main__":
    vocab_size, d_model, num_heads, d_ff = 100, 16, 4, 64
    num_layers, max_seq_len, pad_token_id = 2, 50, 0

    model = Transformer(vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len, pad_token_id)

    batch, src_len, tgt_len = 2, 10, 8
    # include some padding (token 0) near the end of each sequence, like a real batch would
    src_tokens = torch.randint(1, vocab_size, (batch, src_len))
    src_tokens[:, -2:] = pad_token_id
    tgt_tokens = torch.randint(1, vocab_size, (batch, tgt_len))
    tgt_tokens[:, -1:] = pad_token_id

    logits = model(src_tokens, tgt_tokens)
    print("logits shape:", logits.shape)  # expect torch.Size([2, 8, 100])
