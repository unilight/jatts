#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2025 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

import torch
from jatts.modules.utils import make_non_pad_mask


class MelLoss(torch.nn.Module):
    """Mel Loss function module (for feed-forward Transformer)"""

    def __init__(self, _type="L1Loss", params={}):
        """Initialize feed-forward Transformer loss module.

        Args:
            use_masking (bool):
                Whether to apply masking for padded part in loss calculation.

        """
        super(MelLoss, self).__init__()
        self._type = _type
        if _type == "L1Loss":
            self.criterion = L1Loss(**params)

    def forward(self, after_outs, before_outs, ys, olens):
        if self._type == "L1Loss":
            return self.criterion(after_outs, before_outs, ys, olens)


class L1Loss(torch.nn.Module):
    """L1 Loss function module (for feed-forward Transformer)"""

    def __init__(self, use_masking=True, reduction="mean"):
        """Initialize feed-forward Transformer loss module.

        Args:
            use_masking (bool):
                Whether to apply masking for padded part in loss calculation.

        """
        super(L1Loss, self).__init__()
        self.use_masking = use_masking

        # define criterion
        self.l1_criterion = torch.nn.L1Loss(reduction=reduction)

    def forward(self, after_outs, before_outs, ys, olens):
        """Calculate forward propagation.

        Args:
            after_outs (Tensor): Batch of outputs after postnets (B, Lmax, odim).
            before_outs (Tensor): Batch of outputs before postnets (B, Lmax, odim).
            ys (Tensor): Batch of target features (B, Lmax, odim).
            olens (LongTensor): Batch of the lengths of each target (B,).

        Returns:
            Tensor: L1 loss value.

        """
        # apply mask to remove padded part
        if self.use_masking:
            out_masks = make_non_pad_mask(olens).unsqueeze(-1).to(ys.device)
            before_outs = before_outs.masked_select(out_masks)
            after_outs = (
                after_outs.masked_select(out_masks) if after_outs is not None else None
            )
            ys = ys.masked_select(out_masks)

        # calculate loss
        l1_loss = self.l1_criterion(before_outs, ys)
        if after_outs is not None:
            l1_loss += self.l1_criterion(after_outs, ys)

        return l1_loss
