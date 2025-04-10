#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2025 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

import numpy as np
import torch


class FastSpeech2Collater(object):
    """Customized collater for Pytorch DataLoader in FastSpeech2 training."""

    def __init__(self):
        """Initialize customized collater for PyTorch DataLoader."""

    def __call__(self, batch):
        """Convert into batch tensors."""

        def pad_list(xs, pad_value):
            """Perform padding for the list of tensors.

            Args:
                xs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
                pad_value (float): Value for padding.

            Returns:
                Tensor: Padded tensor (B, Tmax, `*`).

            Examples:
                >>> x = [torch.ones(4), torch.ones(2), torch.ones(1)]
                >>> x
                [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
                >>> pad_list(x, 0)
                tensor([[1., 1., 1., 1.],
                        [1., 1., 0., 0.],
                        [1., 0., 0., 0.]])

            """
            n_batch = len(xs)
            max_len = max(x.size(0) for x in xs)
            pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)

            for i in range(n_batch):
                pad[i, : xs[i].size(0)] = xs[i]

            return pad

        xs = []
        ys = []

        for b in batch:
            xs.append(b["token_indices"])
            ys.append(b["mel"])

        # get list of lengths (must be tensor for DataParallel)
        ilens = torch.from_numpy(np.array([x.shape[0] for x in xs])).long()
        olens = torch.from_numpy(np.array([y.shape[0] for y in ys])).long()

        # perform padding and conversion to tensor
        xs = pad_list([torch.from_numpy(x).long() for x in xs], 0)
        ys = pad_list([torch.from_numpy(y).float() for y in ys], 0)

        items = {
            "xs": xs,
            "ilens": ilens,
            "ys": ys,
            "olens": olens,
            "spkembs": None,
        }

        if "pitch" in batch[0]:
            pitches = [b["pitch"] for b in batch]
            pitch_lens = torch.from_numpy(
                np.array([pitch.shape[0] for pitch in pitches])
            ).long()
            pitches = pad_list(
                [torch.from_numpy(pitch).float() for pitch in pitches], 0
            )
            items["pitch"] = pitches
            items["pitch_lens"] = pitch_lens

        if "energy" in batch[0]:
            energys = [b["energy"] for b in batch]
            energy_lens = torch.from_numpy(
                np.array([energy.shape[0] for energy in energys])
            ).long()
            energys = pad_list(
                [torch.from_numpy(energy).float() for energy in energys], 0
            )
            items["energys"] = energys
            items["energy_lens"] = energy_lens

        if "durations_int" in batch[0]:
            durations = [b["durations_int"] for b in batch]
            durations_lens = torch.from_numpy(
                np.array([d.shape[0] for d in durations])
            ).long()
            durations = pad_list([torch.from_numpy(d).long() for d in durations], 0)
            items["durations"] = durations
            items["durations_lens"] = durations_lens

        if "spkemb" in batch[0]:
            spkembs = [b["spkemb"] for b in batch]
            items["spkembs"] = torch.from_numpy(np.array(spkembs)).float()

        return items
