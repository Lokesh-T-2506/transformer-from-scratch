import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformer import Transformer
from tokenizer import SimpleTokenizer
from dataset import CodeCompletionDataset, collate_fn, TOY_PAIRS


def shift_for_teacher_forcing(tgt_batch: torch.Tensor, sos_id: int):
    # tgt_batch shape: (batch, seq_len) -- e.g. [t1, t2, t3, <eos>, <pad>, <pad>]

    # TODO 1: build decoder_input -- prepend a column filled with sos_id to
    decoder_input = torch.cat([torch.full_like(tgt_batch[:, :1], fill_value=sos_id), tgt_batch[:, :-1]], dim=1)

    # TODO 2: loss_target is just tgt_batch itself, completely unchanged
    loss_target = tgt_batch
    # TODO 3: return (decoder_input, loss_target)
    return (decoder_input, loss_target)


if __name__ == "__main__":
    pad_token_id = 0

    # TODO 4: build the tokenizer from all source+target text in TOY_PAIRS
    # (same pattern as dataset.py's own __main__ block)
    all_tokens = [src for src,_ in TOY_PAIRS] + [tgt for _,tgt in TOY_PAIRS]
    tokenizer = SimpleTokenizer(all_tokens)
    # TODO 5: build the dataset and a DataLoader (small batch_size is fine --
    # there are only 5 examples total)
    dataset = CodeCompletionDataset(TOY_PAIRS, tokenizer)
    dataloader  = DataLoader(dataset, batch_size=2, collate_fn=lambda batch:collate_fn(batch, pad_token_id))
    # TODO 6: build the Transformer model. Keep dimensions small for a fast
    model = Transformer(
        d_model=32, num_heads=4, d_ff=64, num_layers=2, vocab_size=tokenizer.vocab_size, pad_token_id=pad_token_id, max_seq_len=25
    )

    # TODO 7: build the loss (nn.CrossEntropyLoss, ignore_index=pad_token_id)
    loss = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    optimizer = torch.optim.Adam(model.parameters())

    sos_id = 1  # TODO 8: look up the real <sos> token ID from the tokenizer

    num_epochs = 50
    for epoch in range(num_epochs):
        total_loss = 0.0
        for src_batch, tgt_batch in dataloader:  # TODO 9: iterate over your DataLoader
            # TODO 10: build decoder_input and loss_target via
            # shift_for_teacher_forcing
            decoder_input, loss_target = shift_for_teacher_forcing(tgt_batch, sos_id)
            # TODO 11: zero gradients, run the model forward on
            # (src_batch, decoder_input) to get logits
            optimizer.zero_grad()
            logits = model(src_batch, decoder_input)
            # TODO 12: compute the loss -- remember to flatten logits to
            # (N, vocab_size) and loss_target to (N,) first
            train_loss = loss(logits.view(-1, logits.size(-1)), loss_target.view(-1))
            # TODO 13: backward() and optimizer step()
            train_loss.backward()
            optimizer.step()
            # TODO 14: accumulate total_loss (e.g. += loss.item()) so you can
            # print an average per epoch
            total_loss += train_loss.item()

        print(f"epoch {epoch+1}/{num_epochs}  loss: {total_loss:.4f}")
