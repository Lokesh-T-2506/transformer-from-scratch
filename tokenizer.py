class SimpleTokenizer:
    PAD_TOKEN, SOS_TOKEN, EOS_TOKEN = "<pad>", "<sos>", "<eos>"

    def __init__(self, corpus: list):
        # corpus: a list of raw strings (both source and target text combined,
        # so one shared vocabulary covers both -- matches the tied-embedding
        # design from earlier, which assumed one shared vocab)

        # TODO 1: build token_to_id and id_to_token mappings.
        # Special tokens go first at fixed IDs: PAD=0, SOS=1, EOS=2
        # (PAD must be 0 -- it needs to match the pad_token_id already used
        # everywhere else in the model). Then assign IDs to every unique
        # whitespace-split token found across the corpus.
        self.token_to_id = {self.PAD_TOKEN: 0, self.SOS_TOKEN: 1, self.EOS_TOKEN: 2}
        self.id_to_token = {0: self.PAD_TOKEN, 1: self.SOS_TOKEN, 2: self.EOS_TOKEN}
        corpus  = [token for sen in corpus for token in sen.split()]
        tokens = set(corpus)
        for token in tokens:
            if token not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[token] = idx
                self.id_to_token[idx] = token
        # TODO 2: store vocab_size (needed to construct the Transformer later)
        self.vocab_size = len(self.token_to_id)

    def encode(self, text: str) -> list:
        # TODO 3: split text on whitespace, look up each token's ID,
        # append the EOS token's ID at the end. Return a list of ints.
        self.tokens = text.split()
        self.ids = [self.token_to_id[token] for token in self.tokens]
        self.ids.append(self.token_to_id[self.EOS_TOKEN])
        return self.ids

    def decode(self, ids: list) -> str:
        # TODO 4: map each ID back to its token string and join with spaces
        # (stop at EOS if you want cleaner output, but not required for now)
        self.tokens = [self.id_to_token[id] for id in ids]
        return " ".join(self.tokens)


if __name__ == "__main__":
    corpus = ["def add(a, b):", "return a + b", "for i in range(10):", "print(i)"]
    tok = SimpleTokenizer(corpus)
    print("vocab size:", tok.vocab_size)

    ids = tok.encode("return a + b")
    print("encoded:", ids)
    print("decoded:", tok.decode(ids))
