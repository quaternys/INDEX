"""
Causal language model with decoder-only Transformer
2023/08/13

Reference: Attention Is All You Need (Vaswani+, 2017)
https://arxiv.org/abs/1706.03762
"""

import torch

class PositionalEncoding(torch.nn.Module):
    def __init__(self, dim=512, dropout=0.1):
        super().__init__()
        θ = torch.pi/2 - torch.arange(3535).outer(10000**(-torch.arange(0, dim, 2)/dim))
        pe = torch.view_as_real(torch.exp(θ*1j)).reshape(-1, dim)
        self.register_buffer("pe", pe)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):   # (bsz, seqlen, dim)
        return self.dropout(x + self.pe[:x.shape[1]])

class Attention(torch.nn.Module):
    def __init__(self, dim=512, n_heads=8, bias=False):
        super().__init__()
        self.n_heads = n_heads
        self.fcs = torch.nn.ModuleList([torch.nn.Linear(dim, dim, bias=bias) for _ in range(4)])

    def forward(self, x, mask=None):    # (bsz, seqlen, dim)
        Q, K, V = [f(x).view(*x.shape[:-1], self.n_heads, -1).transpose(1, 2) for f in self.fcs[:-1]]
        scores = Q @ K.transpose(-2, -1) / Q.shape[-1]**0.5 # (bsz, n_heads, seqlen, seqlen)
        if mask is not None:
            scores += mask
        out = scores.softmax(-1) @ V    # (bsz, n_heads, seqlen, head_dim)
        out = out.transpose(1, 2).contiguous().view_as(x)   # (bsz, seqlen, dim)
        return self.fcs[-1](out), scores

class FeedForward(torch.nn.Module):
    def __init__(self, dim=512, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(self.fc1(x).relu()))

class TransformerBlock(torch.nn.Module):
    def __init__(self, dim=512, n_heads=8, hidden_dim=512, dropout=0.1, bias=False):
        super().__init__()
        self.attention = Attention(dim, n_heads, bias)
        self.feed_forward = FeedForward(dim, hidden_dim, dropout)
        self.atn_norm = torch.nn.LayerNorm(dim)
        self.ffn_norm = torch.nn.LayerNorm(dim)
        self.atn_dropout = torch.nn.Dropout(dropout)
        self.ffn_dropout = torch.nn.Dropout(dropout)

    def forward(self, x, mask=None):    # (bsz, seqlen, dim)
        h = self.atn_norm(x + self.atn_dropout(self.attention(x, mask)[0]))
        out = self.ffn_norm(h + self.ffn_dropout(self.feed_forward(h)))
        return out

class Transformer(torch.nn.Module):
    def __init__(self, V=8000, n_layers=6, dim=512, n_heads=8, hidden_dim=512, dropout=0.1, bias=False):
        super().__init__()
        self.dim = dim
        self.embed = torch.nn.Embedding(V, dim)
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.positional_encoding = PositionalEncoding(dim, dropout)
        self.layers = torch.nn.ModuleList([TransformerBlock(dim, n_heads, hidden_dim, dropout, bias) for _ in range(n_layers)])
        self.norm = torch.nn.LayerNorm(dim)
        self.output = torch.nn.Linear(dim, V)

    def forward(self, tokens: torch.Tensor):
        _bsz, seqlen = tokens.shape
        mask = torch.triu(torch.full((seqlen, seqlen), -torch.inf), 1).cuda()
        h = self.embed(tokens) * self.dim**0.5
        h = self.positional_encoding(h)
        for layer in self.layers:
            h = layer(h, mask)
        return self.output(self.norm(h))


@torch.no_grad()
def generate(model: Transformer, tokens: list[int], temperature=0., genlen=64) -> list[int]:
    tokens = tokens[:]
    model.eval()
    for _ in range(genlen):
        logits = model(torch.tensor([tokens]).cuda())[0, -1]
        if temperature == 0:    # greedy
            next_word = logits.argmax()
        else:                   # probabilistic
            next_word = torch.multinomial((logits/temperature).softmax(0), 1)
        tokens.append(int(next_word))
    return tokens


if __name__ == "__main__":
    model = Transformer().cuda()
    input_sequence = [1, 23, 456, 7890]
    output_sequence = generate(model, input_sequence, genlen=10)
    print(output_sequence)
