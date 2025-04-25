#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2025 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

import numpy as np
import torch
import torch.nn.functional as F


class VALLECollater(object):
    """Customized collater for Pytorch DataLoader in language-model based TTS training."""

    def __init__(self):
        """Initialize customized collater for PyTorch DataLoader."""
        pass

    def __call__(self, batch):
        """Convert into batch tensors."""

        # get encodec names
        prompt_encodec_key = None
        y_encodec_key = None
        for key in batch[0].keys():
            if key.startswith("prompt_"):
                prompt_encodec_key = key
            if key.startswith("encodec"):
                y_encodec_key = key
        assert prompt_encodec_key is not None, "prompt_encodec_key is None"
        assert y_encodec_key is not None, "y_encodec_key is None"
        
        xs = []
        ys = []
        pm = []

        for b in batch:
            xs.append(torch.from_numpy(b["token_indices"]))
            ys.append(torch.from_numpy(b[y_encodec_key]))
            pm.append(torch.from_numpy(b[prompt_encodec_key]))

        items = {
            "xs": xs,
            "ys": ys,
            "pm": pm,
        }

        return items
