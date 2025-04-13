#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2025 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

import numpy as np
import torch
import torch.nn.functional as F


class ValleCollater(object):
    """Customized collater for Pytorch DataLoader in language-model based TTS training."""

    def __init__(self):
        """Initialize customized collater for PyTorch DataLoader."""
        pass

    def __call__(self, batch):
        """Convert into batch tensors."""

        xs = []
        ys = []
        pm = []

        for b in batch:
            xs.append(torch.from_numpy(b["token_indices"]))
            ys.append(torch.from_numpy(b["encodec"]))
            pm.append(torch.from_numpy(b["prompts"]))

        items = {
            "xs": xs,
            "ys": ys,
            "pm": pm,
        }

        return items
