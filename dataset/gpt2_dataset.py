import tiktoken
import torch
from torch.utils.data import Dataset


class GPT2Dataset(Dataset):
    def __init__(self, data_path: str, seq_len: int = 32):
        super(GPT2Dataset, self).__init__()
        self.tokenizer = tiktoken.get_encoding("gpt2")
        with open(data_path, "r") as fi:
            self.data = fi.read()
        self.tokens = self.tokenizer.encode(self.data)

        print(f"Load {len(self.tokens)} tokens.")

        self.seq_len = seq_len

        print(f"1 epoch has {len(self.tokens) /seq_len} steps.")

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        T = self.seq_len
        buf = torch.tensor(self.tokens[index : index + T + 1])
        x = buf[:-1].view(-1)
        label = buf[1:].view(-1)
        # padding x, label to T
        if x.size(0) < T:
            x = torch.cat([x, torch.zeros(T - x.size(0), dtype=torch.long)])
            label = torch.cat([label, torch.zeros(T - label.size(0), dtype=torch.long)])
        return x, label
