import math
import numbers
from typing import Optional, Tuple, Union, Callable

import torch
from torch import nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """LayerNorm with optional bias."""

    def __init__(
        self,
        shape: Union[int, Tuple[int, ...], torch.Size],
        eps: float = 1e-5,
        bias: bool = True,
    ):
        super().__init__()
        self.eps = eps
        if isinstance(shape, numbers.Integral):
            self.normalized_shape = (shape,)
        else:
            self.normalized_shape = tuple(shape)

        self.weight = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        activation: Optional[Callable] = None,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = activation if activation is not None else nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features, bias=bias)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SelfAttention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout: float,
        causal: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.causal = causal

        self.qkv = nn.Linear(embedding_dim, 3 * embedding_dim, bias=bias)
        self.proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.uses_flash = hasattr(F, "scaled_dot_product_attention")

        if causal and not self.uses_flash:
            self.register_buffer("mask", torch.empty((1, 1, 0, 0), dtype=bool))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, dim = x.shape
        head_dim = dim // self.num_heads

        q, k, v = self.qkv(x).split(dim, dim=2)
        q = q.view(bsz, seqlen, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.num_heads, head_dim).transpose(1, 2)

        if self.uses_flash:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.causal,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
            if self.causal:
                if self.mask.shape[-1] < seqlen:
                    self.mask = torch.triu(torch.ones(seqlen, seqlen, device=x.device), diagonal=1).bool()
                    self.mask = self.mask.view(1, 1, seqlen, seqlen)
                att = att.masked_fill(self.mask[:, :, :seqlen, :seqlen], float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_drop(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, dim)
        y = self.resid_drop(self.proj(y))
        return y


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout: float,
        causal: bool = False,
        bias: bool = True,
        mlp_expansion: int = 4,
    ):
        super().__init__()
        self.ln1 = LayerNorm(embedding_dim, bias=bias)
        self.attn = SelfAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            causal=causal,
            bias=bias,
        )
        self.ln2 = LayerNorm(embedding_dim, bias=bias)
        self.mlp = MLP(
            in_features=embedding_dim,
            hidden_features=mlp_expansion * embedding_dim,
            activation=nn.GELU(),
            dropout=dropout,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


def init_linear_by_depth(module: nn.Module, depth: float) -> None:
    if isinstance(module, nn.Linear):
        fan_in = module.weight.size(-1)
        std = 1.0 / math.sqrt(2 * fan_in * depth)
        nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)