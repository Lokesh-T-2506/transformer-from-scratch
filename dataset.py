import torch
from torch.utils.data import Dataset, DataLoader

from tokenizer import SimpleTokenizer

# Tiny synthetic (source, target) pairs -- just enough to prove the training
# loop mechanics work end-to-end. Not meant to teach the model anything real yet.
TOY_PAIRS = [
    ("def add(a, b):", "return a + b"),
    ("def greet(name):", "print(name)"),
    ("for i in range(10):", "print(i)"),
    ("if x > 0:", "print(x)"),
    ("while True:", "break"),
]


class CodeCompletionDataset(Dataset):
    def __init__(self, pairs: list, tokenizer: SimpleTokenizer):
        # TODO 1: store the tokenizer and the pairs (encoding lazily in
        # __getitem__ is fine -- no need to pre-encode everything up front
        # for a dataset this small)
        self.pairs = pairs
        self.tokenizer = tokenizer

    def __len__(self):
        # TODO 2: return the number of (source, target) pairs
        return len(self.pairs)

    def __getitem__(self, idx: int):
        # TODO 3: encode the source and target text at this index using the
        # tokenizer, return them as a (src_ids, tgt_ids) tuple of torch.long
        # tensors (IDs need to be long dtype -- both nn.Embedding lookups and
        # CrossEntropyLoss targets require it)
        src_text, tgt_text = self.pairs[idx]
        self.src_ids = torch.tensor(self.tokenizer.encode(src_text), dtype=torch.long)
        self.tgt_ids = torch.tensor(self.tokenizer.encode(tgt_text), dtype=torch.long)
        return self.src_ids, self.tgt_ids


def collate_fn(batch: list, pad_token_id: int):
    # batch: a list of (src_ids, tgt_ids) tuples with DIFFERENT lengths per example

    # TODO 4: separate the batch into a list of src_ids tensors and a list of
    # tgt_ids tensors
    src_ids_list, tgt_ids_list = zip(*batch)
    # TODO 5: pad each list to that list's own max length in this batch
    # (hint: torch.nn.utils.rnn.pad_sequence exists for exactly this, with a
    # padding_value argument -- check its `batch_first` argument too, you want
    # output shaped (batch, seq_len), not (seq_len, batch))
    src_batch = torch.nn.utils.rnn.pad_sequence(src_ids_list, batch_first=True, padding_value = pad_token_id)
    tgt_batch = torch.nn.utils.rnn.pad_sequence(tgt_ids_list, batch_first=True, padding_value = pad_token_id)
    # TODO 6: return (src_batch, tgt_batch) -- two padded tensors,
    # each shape (batch, max_len_for_that_side)
    return src_batch, tgt_batch


if __name__ == "__main__":
    all_text = [s for pair in TOY_PAIRS for s in pair]
    tokenizer = SimpleTokenizer(all_text)

    dataset = CodeCompletionDataset(TOY_PAIRS, tokenizer)
    print("dataset size:", len(dataset))

    loader = DataLoader(
        dataset, batch_size=2, shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, pad_token_id=0),
    )

    src_batch, tgt_batch = next(iter(loader))
    print("src_batch shape:", src_batch.shape)
    print("tgt_batch shape:", tgt_batch.shape)
    print(src_batch)
    print(tgt_batch)
