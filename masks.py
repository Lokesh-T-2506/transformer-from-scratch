import torch


def make_padding_mask(seq: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    # seq shape: (batch, seq_len) — token IDs, NOT embeddings (this runs before
    # the embedding layer, on the raw integer token sequence)

    # TODO 1: build a mask that's 1 where seq != pad_token_id, 0 where it IS padding
    # target shape: (batch, 1, 1, seq_len) — the two extra singleton dims are for
    # broadcasting against attention scores (batch, num_heads, seq_len_q, seq_len_k):
    # this mask should apply the SAME per-batch-item key-validity pattern to every
    # head and every query position
    pad_mask = (seq != pad_token_id).unsqueeze(1).unsqueeze(2).to(torch.float32)
    return pad_mask 


def make_causal_mask(seq_len: int) -> torch.Tensor:
    # TODO 2: build a (seq_len, seq_len) mask where position [i, j] is 1 if
    # j <= i (key j is allowed for query i), 0 if j > i (future position, forbidden)
    # hint: look at torch.tril or torch.triu
    # target shape for broadcasting: (1, 1, seq_len, seq_len)
    cas_masking = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.float32)).unsqueeze(0).unsqueeze(0)
    return cas_masking


if __name__ == "__main__":
    # padding mask smoke test
    seq = torch.tensor([[5, 8, 3, 0, 0], [7, 2, 9, 4, 0]])  # 0 = <pad>
    pad_mask = make_padding_mask(seq, pad_token_id=0)
    print("padding mask shape:", pad_mask.shape)  # expect (2, 1, 1, 5)
    print(pad_mask)

    # causal mask smoke test
    causal_mask = make_causal_mask(seq_len=5)
    print("causal mask shape:", causal_mask.shape)  # expect (1, 1, 5, 5)
    print(causal_mask)
