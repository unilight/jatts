#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2025 Nagoya University (Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""VALL-E Non AR model."""

import torch
from torch import Tensor

from .valle_base import VALLEBase


class VALLENAR(VALLEBase):
    @property
    def n_resp_levels(self):
        return 7

    @property
    def causal(self):
        return False

    @property
    def use_stop_token(self):
        return False

    @property
    def norm_type(self):
        return "adaln"

    @property
    def resp_loss_only(self):
        return True

    def forward(
        self,
        text_list: list[Tensor],
        proms_list: list[Tensor],
        resps_list: list[Tensor],
        sampling_temperature: float = 0.2,
    ):
        """
        Args:
            text_list: [t] * b
            proms_list: [t' l] * b, l=8
            resps_list: [t'' l] * b, l=1 or 8, 1 for testing and 8 for training.
        Returns:
            [t'' l], l=8 if testing. empty list will be returned during training.
        """

        n_levels_set = {r.shape[-1] for r in resps_list}

        if len(n_levels_set) > 1:
            raise ValueError(f"Please give only one level, got {n_levels_set}.")

        n_levels = next(iter(n_levels_set))

        device = text_list[0].device

        if n_levels == self.n_resp_levels + 1:
            assert resps_list is not None

            quant_levels = torch.randint(0, self.n_resp_levels, (len(resps_list),))

            prev_list = [o[..., : l + 1] for o, l in zip(resps_list, quant_levels)]
            targ_list = [o[..., l + 1] for o, l in zip(resps_list, quant_levels)]

            quant_levels = quant_levels.to(device=device)

            _ = super().forward(
                text_list,
                proms_list,
                prev_list,
                targ_list,
                return_all_resp=True,
                shift_targ_list=False,
                quant_levels=quant_levels,
            )

            # Yes, just nothing as we are training
            prev_list = []
        else:
            prev_list = resps_list

            while True:
                level = prev_list[0].shape[-1] - 1

                if level >= self.n_resp_levels:
                    break

                quant_levels = torch.full((len(text_list),), level, device=device)

                resp_list = super().forward(
                    text_list,
                    proms_list,
                    prev_list,
                    return_all_resp=True,
                    shift_targ_list=False,
                    quant_levels=quant_levels,
                    sampling_temperature=sampling_temperature,
                )

                prev_list = [
                    torch.cat([rs, r.unsqueeze(-1)], dim=-1)
                    for rs, r in zip(prev_list, resp_list)
                ]

        return prev_list
