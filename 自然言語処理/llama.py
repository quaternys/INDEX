"""
2023/08/14

Reference: Llama 2 (Meta, 2023/07)
https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/
"""

import torch
from dataclasses import dataclass

@dataclass
class ModelArgs:
    V:              int = -1
    n_layers:       int = 6     # 32
    dim:            int = 512   # 4096
    n_heads:        int = 8     # 32
    multiple_of:    int = 256
    norm_eps:     float = 1e-5
    dropout:      float = 0.1

    max_bsz:        int = 32
    max_seq_len:    int = 128   # 2048

class RMSNorm(torch.nn.Module):
    def __init__(self, dim=512, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class Attention(torch.nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads
        self.wq = torch.nn.Linear(dim, dim, bias=False)
        self.wk = torch.nn.Linear(dim, dim, bias=False)
        self.wv = torch.nn.Linear(dim, dim, bias=False)
        self.wo = torch.nn.Linear(dim, dim, bias=False)

    def forward(self, x, freqs_cis, mask=None):
        bsz, seqlen, _dim = x.shape
        xq = self.wq(x).view(bsz, seqlen, -1, self.head_dim)
        xk = self.wk(x).view(bsz, seqlen, -1, self.head_dim)
        xv = self.wv(x).view(bsz, seqlen, -1, self.head_dim)
        xq, xk = self.apply_rotary_emb(xq, xk, freqs_cis)
        
        # Scaled dot-product
        xq = xq.transpose(1, 2) # (bs, n_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        scores = xq @ xk.transpose(2, 3) / self.head_dim**0.5
        if mask is not None:
            scores += mask
        scores = scores.float().softmax(-1).type_as(xq)
        output = scores @ xv
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)

    @staticmethod
    def apply_rotary_emb(xq, xk, freqs_cis):
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.view(1, xq_.shape[1], 1, xq_.shape[-1])
        xq_out = torch.view_as_real(xq_*freqs_cis).flatten(3).type_as(xq)
        xk_out = torch.view_as_real(xk_*freqs_cis).flatten(3).type_as(xk)
        return xq_out, xk_out

class FeedForward(torch.nn.Module):
    def __init__(self, dim, hidden_dim, multiple_of, dropout):
        super().__init__()
        hidden_dim = 2 * hidden_dim // 3
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = torch.nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = torch.nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = torch.nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(torch.nn.functional.silu(self.w1(x))) * self.w3(x))

class TransformerBlock(torch.nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention = Attention(args.dim, args.n_heads)
        self.feed_forward = FeedForward(args.dim, 4*args.dim, args.multiple_of, args.dropout)
        self.atn_norm = RMSNorm(args.dim, args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)
        self.atn_dropout = torch.nn.Dropout(args.dropout)
        self.ffn_dropout = torch.nn.Dropout(args.dropout)

    def forward(self, x, freqs_cis, mask):
        h = x + self.atn_dropout(self.attention(self.atn_norm(x), freqs_cis, mask))
        out = h + self.ffn_dropout(self.feed_forward(self.ffn_norm(h)))
        return out

class Transformer(torch.nn.Module):
    def __init__(self, params: ModelArgs, pad_id=-1):
        super().__init__()
        self.embed = torch.nn.Embedding(params.V, params.dim, pad_id)
        self.layers = torch.nn.ModuleList([TransformerBlock(params) for _ in range(params.n_layers)])
        self.norm = RMSNorm(params.dim, params.norm_eps)
        self.output = torch.nn.Linear(params.dim, params.V, bias=False)
        # RoPE
        head_dim = params.dim // params.n_heads
        _f = 10000 ** (-torch.arange(0, head_dim, 2)[:head_dim//2]/head_dim)
        freqs_cis = torch.polar(torch.ones_like(_f), torch.arange(5000).outer(_f))
        self.register_buffer("freqs_cis", freqs_cis)

    def forward(self, tokens: torch.Tensor):
        _bsz, seqlen = tokens.shape
        h = self.embed(tokens)
        mask = torch.triu(torch.full((seqlen, seqlen), -torch.inf), 1).cuda()
        for layer in self.layers:
            h = layer(h, self.freqs_cis[:seqlen], mask)
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
    model = Transformer(ModelArgs(8000)).cuda()
    input_sequence = [1, 23, 456, 7890]
    output_sequence = generate(model, input_sequence, genlen=10)
    print(output_sequence)
