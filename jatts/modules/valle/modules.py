#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2025 Nagoya University (Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""VALL-E related modules."""

import torch
import math
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, einsum, nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.checkpoint import checkpoint
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

def _create_mask(l, device):
    """1 is valid region and 0 is invalid."""
    seq = torch.arange(max(l), device=device).unsqueeze(0)  # (1 t)
    stop = torch.tensor(l, device=device).unsqueeze(1)  # (b 1)
    return (seq < stop).float()  # (b t)


def list_to_tensor(x_list: list[Tensor], pattern="t b c -> b t c"):
    """
    Args:
        x_list: [(t d)]
    Returns:
        x: (? ? ?)
        m: (? ? ?), same as x
    """
    l = list(map(len, x_list))
    x = rearrange(pad_sequence(x_list), pattern)
    m = _create_mask(l, x_list[0].device)
    m = m.t().unsqueeze(-1)  # (t b 1)
    m = rearrange(m, pattern)
    m = m.to(x)
    return x, m


class SinusodialEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        exponent = torch.arange(self.d_half, dtype=torch.float32)
        exponent = exponent / self.d_half
        omega = torch.exp(-math.log(1e4) * exponent)
        self.omega: torch.Tensor
        self.register_buffer("omega", omega, persistent=False)

    @property
    def d_half(self):
        assert self.d_model % 2 == 0, "Only support even d_model."
        return self.d_model // 2

    def forward(self, x):
        """
        Args:
            x: (...)
        Returns:
            pe: (... d)
        """
        omega = self.omega

        while omega.dim() <= x.dim():
            omega = omega.unsqueeze(0)  # (... d)

        x = x.unsqueeze(-1)  # (... 1)
        x = omega * x
        x = torch.cat([x.sin(), x.cos()], dim=-1)

        return x

    def get_pe(self, n: int):
        """
        Args:
            n: int
        Returns:
            pe: (n d)
        """
        device = self.omega.device
        return self.forward(torch.arange(n, device=device))

    def add_pe(self, x):
        """
        Args:
            x: (b t c)
        """
        e = self.get_pe(x.shape[1])  # t d
        e = e[None]  # b t d
        x = x + e
        return x


class Attention(nn.Module):
    def __init__(self, d_model, n_heads, causal):
        super().__init__()
        assert d_model % n_heads == 0
        dim_head = d_model // n_heads
        self.causal = causal
        self.n_heads = n_heads
        self.scale = dim_head**-0.5
        self.to_qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.to_out = nn.Linear(d_model, d_model)

    def forward(self, x, m):
        """
        Args:
            x: (b t c)
            m: (b t c), 1 is data, 0 is padding
        Returns:
            x: (b t c)
        """
        h = self.n_heads

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b t (h d) -> b t h d", h=h), (q, k, v))

        e = einsum("b i h d, b j h d -> b i j h", q, k)
        e = e * self.scale

        kpm = m.unsqueeze(1) * m.unsqueeze(2)  # b i j 1

        if self.causal:
            kpm = kpm.squeeze(-1).tril().unsqueeze(-1)  # b i j 1

        e = e.masked_fill(kpm == 0, -torch.finfo(e.dtype).max)
        a = e.softmax(dim=2)  # Normalize on j, i.e. key

        o = einsum("b i j h, b j h d -> b i h d", a, v)
        o = o.flatten(-2)
        o = self.to_out(o)  # b t c

        o = o * m

        return o


class AdaLN(nn.Module):
    def __init__(self, d_model, n_levels, eps=1e-5, k=0.1, c=2):
        super().__init__()
        self.eps = eps
        self.emb = nn.Embedding(n_levels, d_model * 2)
        self.k = k
        self.c = c
        nn.init.zeros_(self.emb.weight)

    def forward(self, x, l):
        logγ, β = self.emb(l).unsqueeze(1).chunk(2, dim=-1)

        h = F.layer_norm(x, x.shape[-1:], eps=self.eps)

        # The initial implementation (https://github.com/enhuiz/vall-e/blob/fbf023448c08e55c0422eefed7fc234cf8b76680/vall_e/vall_e/base.py#L135)
        # performed worse than vanilla LayerNorm.
        # The authors mentioned another AdaNorm paper (https://openreview.net/pdf?id=HyxndNrxLB) as they introduce AdaLN.
        # Did they use AdaNorm inside AdaLN? (as follows)
        h = self.c * (1 - (self.k * h).detach()) * h

        y = logγ.exp() * h + β

        return y


class PrenormResidual(nn.Module):
    def __init__(
        self,
        block,
        d_model,
        p_dropout,
        requires_mask=False,
        norm_type="ln",
        n_levels: int | None = None,
    ):
        super().__init__()
        self.block = block
        self.requires_mask = requires_mask
        self.norm_type = norm_type
        if norm_type == "ln":
            self.norm = nn.LayerNorm(d_model)
        elif norm_type == "adaln":
            assert n_levels is not None
            self.norm = AdaLN(d_model, n_levels)
        else:
            raise NotImplementedError(norm_type)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x, m, l):
        """
        Args:
            x: input (b t d)
            m: mask (b t 1), 1 is valuable and 0 is padding
            l: level to use, required only for AdaLN
        """
        nopts = {"l": l} if self.norm_type == "adaln" else {}
        bopts = {"m": m} if self.requires_mask else {}
        x = x + self.dropout(self.block(self.norm(x, **nopts) * m, **bopts))
        return x * m


class Block(nn.Sequential):
    def __init__(self, d_model, n_heads, p_dropout, causal, norm_type, n_levels):
        super().__init__()
        self.attn = PrenormResidual(
            Attention(d_model, n_heads, causal),
            d_model=d_model,
            p_dropout=p_dropout,
            requires_mask=True,
            norm_type=norm_type,
            n_levels=n_levels,
        )
        self.attn = checkpoint_wrapper(self.attn)
        self.ffn = PrenormResidual(
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(p_dropout),
                nn.Linear(d_model * 4, d_model),
            ),
            d_model=d_model,
            p_dropout=p_dropout,
            norm_type=norm_type,
            n_levels=n_levels,
        )

    def forward(self, x, m, l):
        """
        Args:
            x: (b t c)
            m: (b t 1)
            l: (b)
        """
        poor_in_vram = True
        if x.requires_grad and poor_in_vram:
            x = checkpoint(self.attn, x, m, l, use_reentrant=False)
        else:
            x = self.attn(x, m, l)
        x = self.ffn(x, m, l)
        return x


class Embedding(nn.Embedding):
    def forward(self, x_list: list[Tensor]) -> list[Tensor]:
        if len(x_list) == 0:
            return []
        return super().forward(torch.cat(x_list)).split([*map(len, x_list)])


class MultiEmbedding(nn.Module):
    """
    This embedding sums embeddings on different levels.
    """

    def __init__(self, max_n_levels, n_tokens, token_dim):
        super().__init__()
        # Store the maximum number of levels
        self.max_n_levels = max_n_levels
        # Store the number of tokens
        self.n_tokens = n_tokens
        # Create a learnable parameter for embeddings
        # Shape: (max_n_levels, n_tokens, token_dim)
        self.weight = nn.Parameter(torch.randn(max_n_levels, n_tokens, token_dim))

    def forward(self, x_list: list[Tensor]) -> list[Tensor]:
        # If the input list is empty, return an empty list
        if len(x_list) == 0:
            return []

        # Alias for the embedding weights
        w = self.weight

        padded_x_list = []

        for xi in x_list:
            # Convert input tensor to one-hot encoding
            # Shape: (time, levels, n_tokens)
            if xi.dtype != torch.long:
                # Convert to integer indices if not already
                xi = xi.long()

            xi = F.one_hot(xi, num_classes=self.n_tokens)  # t l' k
            # Pad the levels dimension to match max_n_levels
            xi = F.pad(xi, (0, 0, 0, w.shape[0] - xi.shape[1]))  # t l k
            # Add the padded tensor to the list, ensuring it's on the same device as w
            padded_x_list.append(xi.to(w))

        # Concatenate all padded tensors
        # Shape: (total_time, max_n_levels, n_tokens)
        x = torch.cat(padded_x_list)  # n l k
        # Perform einsum operation to sum embeddings across levels
        # Shape: (total_time, token_dim)
        x = einsum("l k d, n l k -> n d", w, x)

        # Split the result back into a list of tensors with original lengths
        x_list = x.split([*map(len, x_list)])

        return x_list


def _join(x: tuple[Tensor], sep: Tensor):
    """
    Args:
        x: (k t d)
        sep: (d)
    """
    ret = x[0]
    for i in range(1, len(x)):
        ret = torch.cat((ret, sep[None], x[i]), dim=0)
    return ret