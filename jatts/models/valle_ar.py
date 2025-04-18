#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2025 Nagoya University (Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging

import torch
from einops import rearrange
from torch import Tensor

from .valle_base import VALLEBase


class VALLEAR(VALLEBase):
    @property
    def n_resp_levels(self):
        return 1

    @property
    def causal(self):
        return True

    @property
    def use_stop_token(self):
        return True

    @property
    def norm_type(self):
        return "ln"

    @property
    def resp_loss_only(self):
        return False

    def _prune(self, l):
        indices = (l == self.stop_token).nonzero()
        if len(indices) == 0:
            return l
        return l[: indices.min().item()]

    @staticmethod
    def _unsqueeze_list(x_list, axis=-1):
        return [x.unsqueeze(dim=axis) for x in x_list]

    def forward(
        self,
        text_list,
        proms_list,
        resp_list=None,
        max_steps=1000,
        sampling_temperature=1.0,
    ):
        # training
        if resp_list is not None:
            return super().forward(
                text_list,
                proms_list,
                self._unsqueeze_list(resp_list),
                resp_list,
                quant_levels=None,
                shift_targ_list=True,
                return_all_resp=False,
            )
        # inference
        else:
            return self._generate(
                text_list,
                proms_list,
                max_steps,
                sampling_temperature,
            )

    def _generate(
        self,
        text_list,
        proms_list,
        max_steps,
        sampling_temperature,
    ):
        """This is executed during inference."""
        device = text_list[0].device
        resp_list = [torch.zeros(0, device=device).long() for _ in text_list]
        stopped = torch.zeros(len(text_list), device=device).bool()
        for _ in range(max_steps):
            r = super().forward(
                text_list,
                proms_list,
                self._unsqueeze_list(resp_list),
                sampling_temperature=sampling_temperature,
            )
            stopped |= r == self.stop_token
            for i, ri in enumerate(r):
                resp_list[i] = torch.cat([resp_list[i], ri[None]])
            if stopped.all().item():
                break
        pruned = [self._prune(r) for r in resp_list]
        return pruned
