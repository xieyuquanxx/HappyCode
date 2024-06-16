from dataclasses import dataclass

import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dataset.gpt2_dataset import GPT2Dataset
from utils import count_parameters, seed_everything


@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50257  # number of tokens
    n_layer: int = 12  # number of layers
    n_head: int = 12  # number of heads
    n_embed: int = 768  # embedding dimension


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)
        # https://pytorch.org/docs/stable/generated/torch.nn.GELU.html
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)

        self.c_proj.GPT_INIT_SCALE = torch.tensor(1)

    def forward(self, x: torch.Tensor):
        return self.c_proj(self.gelu(self.c_fc(x)))


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_embed = config.n_embed
        self.head_dim = config.n_embed // config.n_head
        self.scale = self.head_dim**-0.5
        # key, query, value projections for all heads, int batch
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)
        self.attn_drop = nn.Dropout(0.1)
        self.resid_drop = nn.Dropout(0.1)
        # output projection
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.c_proj.GPT_INIT_SCALE = torch.tensor(1)

        # define a buffer to store the bias, like constant
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x: torch.Tensor):
        # batch_size, sequence length, embedding dimension
        # nh: number of heads, hs: head size, C = nh * ns
        B, T, C = x.shape
        qkv = self.c_attn(x)  # B, T, 3C
        # https://pytorch.org/docs/stable/generated/torch.split.html#torch.split
        # q,k,v shape: B, T, C
        q, k, v = qkv.split(self.n_embed, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # B, nh, T, hs
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # B, nh, T, hs
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # B, nh, T, hs
        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, nh, T, T
        attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        y = attn @ v  # B, nh, T, hs
        y = y.transpose(1, 2).contiguous().reshape(B, T, C)
        y = self.c_proj(y)  # B, T, C
        return y


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embed),
                wpe=nn.Embedding(config.block_size, config.n_embed),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embed),
            )
        )
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

        # weight sharing 124.44M  vs not weight sharing: 163.04M
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "GPT_INIT_SCALE"):
                std *= (2 * self.config.n_layer) ** -0.5
            nn.init.normal_(module.weight, mean=0.00, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.00, std=0.02)

    def forward(self, x: torch.Tensor, label: torch.Tensor | None = None):
        # input x shape: B, T
        B, T = x.shape
        pos = torch.arange(T, dtype=torch.long, device=x.device)
        input_h = self.transformer.wte(x) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            input_h = block(input_h)
        x = self.transformer.ln_f(input_h)
        lm_logits = self.lm_head(x)
        loss = None
        if label is not None:
            # F.cross_entropy will do softmax
            loss = F.cross_entropy(
                lm_logits.view(-1, self.config.vocab_size), label.view(-1)
            )
        return lm_logits, loss

    def generate(self, prompt: str, max_length: int = 5):
        # auto-regressive generation

        tokenizer = tiktoken.get_encoding("gpt2")
        tokens = tokenizer.encode(prompt)
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = torch.unsqueeze(tokens, 0).repeat(5, 1)  # B, T
        x = tokens.to("cuda")
        while x.size(1) < max_length:
            with torch.no_grad():
                logits, _ = self.forward(x)  # B, T, vocab_size
                logits = logits[:, -1, :]  # B, vocab_size
                probs = F.softmax(logits, dim=-1)
                # top-k sampling
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                idx = torch.multinomial(topk_probs, 1)  # (B, 1) sampling
                # gather the crossponding token index, input_id
                xcol = torch.gather(topk_indices, -1, idx)  # (B, 1)
                x = torch.cat([x, xcol], dim=1)
        for i in range(5):
            tokens = x[i, :max_length].tolist()
            decoded = tokenizer.decode(tokens)
            print(">", decoded)


seed_everything(42)
gpt2 = GPT(GPTConfig())
# gpt2.eval()
gpt2.to("cuda")

print(f"Model has {count_parameters(gpt2) / 1e6: .2f}M parameters.")


data_path = "/data/Users/xyq/developer/happy_code/data/gpt2/shakespeare_input.txt"
train_dataset = GPT2Dataset(data_path)
dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
# gpt2的词表大小时50257，在最开始的相当于乱猜，因此loss差不多在-ln(1/50257)=10.8左右
optimizer = torch.optim.AdamW(gpt2.parameters(), lr=3e-4)
for i in range(1):
    for x, label in dataloader:
        x = x.to("cuda")
        label = label.to("cuda")
        optimizer.zero_grad()
        logits, loss = gpt2(x, label)
        loss.backward()
        optimizer.step()
        print(f"loss: {loss.item()}")
