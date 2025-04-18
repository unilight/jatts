#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2025 Nagoya University (Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging
import math
from functools import partial
from typing import Literal, overload

import torch
import torch.nn.functional as F
from jatts.modules.valle.modules import (
    Block,
    Embedding,
    MultiEmbedding,
    SinusodialEmbedding,
    _join,
    list_to_tensor,
)
from jatts.utils.prompt import prepare_prompt
from torch import Tensor, einsum, nn
from torch.distributions import Categorical


class VALLEBase(nn.Module):
    @property
    def causal(self) -> bool:
        raise NotImplementedError

    @property
    def n_resp_levels(self) -> int:
        raise NotImplementedError

    @property
    def use_stop_token(self) -> bool:
        raise NotImplementedError

    @property
    def norm_type(self):
        raise NotImplementedError

    @property
    def n_prom_levels(self) -> int:
        return 8

    @property
    def resp_loss_only(self):
        raise NotImplementedError

    def __init__(
        self,
        idim: int,
        n_tokens: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 12,
        p_dropout: float = 0.1,
        # prompt related
        prompt_prefix_mode: int = 1,
        prompt_max_frame_length: int = 225,
    ):
        super().__init__()
        self.idim = idim  # not really used
        self.n_tokens = n_tokens

        # prompt preparation related
        self.prompt_prefix_mode = prompt_prefix_mode
        self.prompt_max_frame_length = prompt_max_frame_length

        causal = self.causal

        # +1 to include the stop token
        n_stop_tokens = 1 if self.use_stop_token else 0
        n_resp_tokens = n_tokens + n_stop_tokens

        self.text_emb = Embedding(n_tokens, d_model)
        self.proms_emb = MultiEmbedding(self.n_prom_levels, n_tokens, d_model)
        self.resps_emb = MultiEmbedding(self.n_resp_levels, n_resp_tokens, d_model)

        self.sin_emb = SinusodialEmbedding(d_model)

        self.sep = nn.Parameter(torch.randn(d_model))

        blocks = [
            Block(
                d_model=d_model,
                n_heads=n_heads,
                p_dropout=p_dropout,
                causal=causal,
                norm_type=self.norm_type,
                n_levels=self.n_resp_levels,
            )
            for _ in range(n_layers)
        ]

        self.blocks = nn.ModuleList(blocks)

        self.classifier = nn.Linear(d_model, n_resp_tokens)

    @property
    def stop_token(self):
        if not self.use_stop_token:
            raise ValueError("Not using stop token!")
        return self.n_tokens

    @property
    def ignore_index(self):
        return -100

    @staticmethod
    def _samplewise_merge_tensors(*l, sep: Tensor | None):
        if sep is None:
            cat = torch.cat
        else:
            cat = partial(_join, sep=sep)
        return [*map(cat, zip(*l))]

    @overload
    def forward(
        self,
        text_list: list[Tensor],
        proms_list: list[Tensor],
        resps_list: list[Tensor],
        targ_list: list[Tensor] | None = None,
        quant_levels: Tensor | None = None,
        shift_targ_list: bool = False,
        return_all_resp: Literal[False] = False,
        sampling_temperature: float = 1.0,
    ) -> Tensor: ...

    @overload
    def forward(
        self,
        text_list: list[Tensor],
        proms_list: list[Tensor],
        resps_list: list[Tensor],
        targ_list: list[Tensor] | None = None,
        quant_levels: Tensor | None = None,
        shift_targ_list: bool = False,
        return_all_resp: Literal[True] = True,
        sampling_temperature: float = 1.0,
    ) -> list[Tensor]: ...

    def forward(
        self,
        text_list: list[Tensor],
        proms_list: list[Tensor],
        resps_list: list[Tensor],
        targ_list: list[Tensor] | None = None,
        quant_levels: Tensor | None = None,
        shift_targ_list: bool = False,
        return_all_resp: bool = False,
        sampling_temperature: float = 1.0,
    ):
        """
        Args:
            text_list: List of tensors, each containing token indices for input text. Shape: [sequence_length] * batch_size
            proms_list: List of tensors containing prompt tokens. Shape: [prompt_length, num_quantization_levels] * batch_size
            resps_list: List of tensors containing response tokens. Shape: [response_length, num_quantization_levels] * batch_size
            targ_list: Optional list of target tensors for loss computation. Shape: [target_length] * batch_size. If provided, loss will be computed.
            quant_levels: Optional tensor specifying which quantization levels to use in non-autoregressive (NAR) mode.
            shift_targ_list: Boolean flag. If True (as in autoregressive mode), shifts the target list when computing loss.
            return_all_resp: Boolean flag. If True (as in NAR mode), returns all responses instead of sampling.
            sampling_temperature: Float controlling randomness in sampling. Lower values make output more deterministic but less diverse.
        Returns:
            y: Tensor of sampled token indices if return_all_resp is False, otherwise a list of response tensors.
        """
        # Crop prompt
        proms_list = [
            prepare_prompt(self.prompt_prefix_mode, p, self.prompt_max_frame_length)
            for p in proms_list
        ]

        # Merge text, prompt, and response embeddings
        # _samplewise_merge_tensors concatenates the embeddings for each sample,
        # separating them with self.sep if provided
        x_list = self._samplewise_merge_tensors(
            self.text_emb(text_list),  # Convert text tokens to embeddings
            self.proms_emb(proms_list),  # Convert prompt tokens to embeddings
            self.resps_emb(resps_list),  # Convert response tokens to embeddings
            sep=self.sep,
        )

        # Convert list of tensors to padded tensor and create attention mask
        # list_to_tensor pads the sequences to the same length and creates a mask
        x, m = list_to_tensor(x_list)
        # Add positional encoding to the input
        # add_pe adds sinusoidal positional encodings to the input
        x = self.sin_emb.add_pe(x)

        # Pass through transformer blocks
        for block in self.blocks:
            # Each block applies self-attention and feed-forward layers
            # m is the attention mask, quant_levels is used in NAR mode
            x = block(x, m, quant_levels)

        # Apply final classification layer and mask
        # The classifier projects the hidden states to token probabilities
        h = self.classifier(x) * m

        # Remove padding from output
        # This step is necessary because the sequences were padded earlier
        h_list = [hi[:li] for hi, li in zip(h, map(len, x_list))]

        # Compute loss if target is provided
        if targ_list is not None:
            if any([l == 0 for l in map(len, targ_list)]):
                raise ValueError("Cannot compute loss given empty targ_list.")

            device = h.device

            ignore_sep = torch.tensor(self.ignore_index, device=device)

            # Ignore prompt in the target by setting it to ignore_index
            prom_list = [
                torch.full_like(t[..., 0], self.ignore_index) for t in proms_list
            ]

            # Merge text and prompt, separating with ignore_sep
            text_prom_list = self._samplewise_merge_tensors(
                text_list, prom_list, sep=ignore_sep
            )

            # Prepare targets for loss computation
            for i in range(len(text_prom_list)):
                if self.resp_loss_only:
                    # If only computing loss on response, ignore all text and prompt
                    text_prom_list[i][:] = self.ignore_index
                else:
                    # Shift targets to align with predictions
                    text_prom_list[i] = text_prom_list[i].roll(-1, dims=0)
                    text_prom_list[i][-1] = self.ignore_index

            if shift_targ_list:
                # Shift target for autoregressive mode
                # This makes the model predict the next token given the current one
                targ_list = [*targ_list]
                for i in range(len(targ_list)):
                    targ_list[i] = targ_list[i].roll(-1, dims=0)
                    targ_list[i][-1] = self.stop_token

            # Merge prepared text_prom and target lists
            y_list = self._samplewise_merge_tensors(
                text_prom_list, targ_list, sep=ignore_sep
            )

            # Compute cross-entropy loss
            # This measures how well the model's predictions match the targets
            self.loss = dict(
                nll=F.cross_entropy(
                    torch.cat(h_list),
                    torch.cat(y_list),
                    ignore_index=self.ignore_index,
                )
            )
            # loss = sum(loss.values())
            # return loss

        # Sample from output distribution
        if return_all_resp:
            # For NAR (Non-Autoregressive) mode, return all responses
            # This generates the entire sequence in one step
            logits = [hi[-li:] for hi, li in zip(h_list, map(len, resps_list))]
            ret = [
                Categorical(logits=hi / sampling_temperature).sample() for hi in logits
            ]
        else:
            # For AR (Autoregressive) mode, return only the last token
            # This generates one token at a time
            logits = torch.stack([hi[-1] for hi in h_list])
            ret = Categorical(logits=logits / sampling_temperature).sample()

        return ret
