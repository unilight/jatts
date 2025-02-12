#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2025 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

import torch
from jatts.modules.utils import make_non_pad_mask


class PitchLoss(torch.nn.Module):
    """Pitch Loss function module (for FastSpeech2)"""

    def __init__(self, use_masking=True, reduction="mean"):
        """Initialize pitch loss module.

        Args:
            use_masking (bool):
                Whether to apply masking for padded part in loss calculation.

        """
        super(PitchLoss, self).__init__()
        self.use_masking = use_masking

        # define criterion
        self.criterion = torch.nn.MSELoss(reduction=reduction)

    def forward(self, p_outs, ps, olens):
        """Calculate forward propagation.

        Args:
            p_outs (Tensor): Batch of predicted pitch (B, Lmax).
            ps (Tensor): Batch of ground truth pitch (B, Lmax).
            olens (LongTensor): Batch of the lengths of each target (B,).

        Returns:
            Tensor: L2 loss value.

        """
        # apply mask to remove padded part
        if self.use_masking:
            out_masks = make_non_pad_mask(olens).unsqueeze(-1).to(ps.device)
            p_outs = p_outs.masked_select(out_masks)
            ps = ps.masked_select(out_masks)

        # print("p_outs", p_outs)
        # print("ps", ps)

        # calculate loss
        loss = self.criterion(p_outs, ps)

        return loss


class EnergyLoss(torch.nn.Module):
    """Energy Loss function module (for FastSpeech2)"""

    def __init__(self, use_masking=True, reduction="mean"):
        """Initialize energy loss module.

        Args:
            use_masking (bool):
                Whether to apply masking for padded part in loss calculation.

        """
        super(EnergyLoss, self).__init__()
        self.use_masking = use_masking

        # define criterion
        self.criterion = torch.nn.MSELoss(reduction=reduction)

    def forward(self, e_outs, es, olens):
        """Calculate forward propagation.

        Args:
            e_outs (Tensor): Batch of predicted energy (B, Lmax).
            es (Tensor): Batch of ground truth energy (B, Lmax).
            olens (LongTensor): Batch of the lengths of each target (B,).

        Returns:
            Tensor: L2 loss value.

        """
        # apply mask to remove padded part
        if self.use_masking:
            out_masks = make_non_pad_mask(olens).unsqueeze(-1).to(es.device)
            e_outs = e_outs.masked_select(out_masks)
            es = es.masked_select(out_masks)

        # print("e_outs", e_outs)
        # print("es", es)

        # calculate loss
        loss = self.criterion(e_outs, es)

        return loss
