#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2025 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

import math

import torch
from jatts.modules.utils import make_non_pad_mask


class CFMLoss(object):
    """Just a dummy class for the CFM loss. Calculate in `modules/matchatts/flow_matching.py"""

    def __call__(self, *args, **kwargs):
        return None


class EncoderPriorLoss(torch.nn.Module):
    """
    Encoder prior loss.
    Original implementation: https://github.com/shivammehta25/Matcha-TTS/blob/main/matcha/models/matcha_tts.py#L240-L241

    """

    def __init__(self, use_masking=True, reduction="mean"):
        """Initialize encoder prior loss module.

        Args:
            use_masking (bool):
                Whether to apply masking for padded part in loss calculation.

        """
        super(EncoderPriorLoss, self).__init__()
        self.use_masking = use_masking

        # define criterion
        self.mse = torch.nn.MSELoss(reduction=reduction)

    def forward(self, hs, ys, olens):
        """Calculate forward propagation.

        Args:
            hs (Tensor): Batch of encoder outputs (B, Lmax, dim).
            ys (Tensor): Batch of target features (B, Lmax, dim).
            olens (LongTensor): Batch of the lengths of each target (B,).

        Returns:
            Tensor: Encoder prior loss value.

        """
        # apply mask to remove padded part
        if self.use_masking:
            out_masks = make_non_pad_mask(olens).unsqueeze(-1).to(ys.device)
            hs = hs.masked_select(out_masks)
            ys = ys.masked_select(out_masks)

        # calculate loss
        encoder_prior_loss = 0.5 * self.mse(hs, ys) + math.log(2 * math.pi)

        # encoder_prior_loss = torch.mean(0.5 * ((ys - hs) ** 2 + math.log(2 * math.pi)), dim=-1) # [B, T]
        # encoder_prior_loss = torch.sum(encoder_prior_loss, dim=-1) / torch.sum(ret["h_masks"].squeeze(1), dim=-1) # B
        # encoder_prior_loss = torch.sum(encoder_prior_loss)

        return encoder_prior_loss
