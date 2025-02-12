#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2025 Wen-Chin Huang (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""F0 extractor using DIO + Stonemask algorithm."""

import logging
from typing import Any, Dict, Tuple, Union

import numpy as np
import pyworld
import torch
import torch.nn.functional as F
from jatts.utils.types import int_or_none
from scipy.interpolate import interp1d
from typeguard import typechecked


class Dio:
    """F0 estimation with dio + stonemask algorithm.

    This is f0 extractor based on dio + stonmask algorithm introduced in `WORLD:
    a vocoder-based high-quality speech synthesis system for real-time applications`_.

    .. _`WORLD: a vocoder-based high-quality speech synthesis system for real-time
        applications`: https://doi.org/10.1587/transinf.2015EDP7457

    Note:
        This module is based on NumPy implementation. Therefore, the computational graph
        is not connected.

    Todo:
        Replace this module with PyTorch-based implementation.

    """

    @typechecked
    def __init__(
        self,
        fs: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 256,
        use_token_averaged_f0: bool = True,
        use_continuous_f0: bool = True,
        use_log_f0: bool = True,
        reduction_factor: int_or_none = None,
    ):
        super().__init__()
        self.fs = fs
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.frame_period = 1000 * hop_length / fs
        self.use_token_averaged_f0 = use_token_averaged_f0
        self.use_continuous_f0 = use_continuous_f0
        self.use_log_f0 = use_log_f0
        if use_token_averaged_f0:
            assert reduction_factor >= 1
        self.reduction_factor = reduction_factor

    def output_size(self) -> int:
        return 1

    def get_parameters(self) -> Dict[str, Any]:
        return dict(
            fs=self.fs,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            use_token_averaged_f0=self.use_token_averaged_f0,
            use_continuous_f0=self.use_continuous_f0,
            use_log_f0=self.use_log_f0,
            reduction_factor=self.reduction_factor,
        )

    def forward(
        self,
        input: np.array,
        f0min: int = 80,
        f0max: int = 400,
        feat_length: int = None,
        durations: np.array = None,
    ) -> np.array:

        # F0 extraction
        pitch = self._calculate_f0(input, f0min, f0max)

        # (Optional): Adjust length to match with the mel-spectrogram
        if feat_length is not None:
            pitch = self._adjust_num_frames(pitch, feat_length)

        # (Optional): Average by duration to calculate token-wise f0
        if self.use_token_averaged_f0:
            durations = durations * self.reduction_factor
            pitch = self._average_by_duration(pitch, durations)

        # Return with the shape (B, T, 1)
        return pitch

    def _calculate_f0(self, input, f0min, f0max):
        x = input.astype(np.double)
        f0, timeaxis = pyworld.dio(
            x,
            self.fs,
            f0_floor=f0min,
            f0_ceil=f0max,
            frame_period=self.frame_period,
        )
        f0 = pyworld.stonemask(x, f0, timeaxis, self.fs)
        if self.use_continuous_f0:
            f0 = self._convert_to_continuous_f0(f0)
        if self.use_log_f0:
            nonzero_idxs = np.where(f0 != 0)[0]
            f0[nonzero_idxs] = np.log(f0[nonzero_idxs])
        return f0.reshape(-1)

    @staticmethod
    def _adjust_num_frames(x: np.array, num_frames: int) -> np.array:
        if num_frames > len(x):
            x = np.pad(x, (0, num_frames - len(x)))
        elif num_frames < len(x):
            x = x[:num_frames]
        return x

    @staticmethod
    def _convert_to_continuous_f0(f0: np.array) -> np.array:
        if (f0 == 0).all():
            logging.warning("All frames seems to be unvoiced.")
            return f0

        # padding start and end of f0 sequence
        start_f0 = f0[f0 != 0][0]
        end_f0 = f0[f0 != 0][-1]
        start_idx = np.where(f0 == start_f0)[0][0]
        end_idx = np.where(f0 == end_f0)[0][-1]
        f0[:start_idx] = start_f0
        f0[end_idx:] = end_f0

        # get non-zero frame index
        nonzero_idxs = np.where(f0 != 0)[0]

        # perform linear interpolation
        interp_fn = interp1d(nonzero_idxs, f0[nonzero_idxs])
        f0 = interp_fn(np.arange(0, f0.shape[0]))

        return f0

    def _average_by_duration(self, x: np.array, d: np.array) -> np.array:
        assert 0 <= len(x) - d.sum() < self.reduction_factor, f"{len(x)}, {d.sum()}"
        d_cumsum = np.pad(np.cumsum(d, axis=0), (1, 0))
        x_avg = [
            (
                x[start:end][x[start:end] > 0.0].mean(axis=0)
                if len(x[start:end][x[start:end] > 0.0]) != 0
                else 0.0
            )
            for start, end in zip(d_cumsum[:-1], d_cumsum[1:])
        ]
        return np.stack(x_avg)
