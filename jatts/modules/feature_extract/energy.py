#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2025 Wen-Chin Huang (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Energy extractor."""

from typing import Any, Dict, Optional, Tuple, Union

import librosa
import numpy as np
from typeguard import typechecked


class Energy:
    """Energy extractor."""

    @typechecked
    def __init__(
        self,
        fs: int = 22050,
        n_fft: int = 1024,
        win_length: Optional[int] = None,
        hop_length: int = 256,
        window: str = "hann",
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
        use_token_averaged_energy: bool = True,
        reduction_factor: Optional[int] = None,
    ):
        self.fs = fs
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.use_token_averaged_energy = use_token_averaged_energy
        if use_token_averaged_energy:
            assert reduction_factor >= 1
        self.reduction_factor = reduction_factor

        # self.stft = Stft(
        #     n_fft=n_fft,
        #     win_length=win_length,
        #     hop_length=hop_length,
        #     window=window,
        #     center=center,
        #     normalized=normalized,
        #     onesided=onesided,
        # )

    def output_size(self) -> int:
        return 1

    def get_parameters(self) -> Dict[str, Any]:
        return dict(
            fs=self.fs,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            win_length=self.win_length,
            center=self.stft.center,
            normalized=self.stft.normalized,
            use_token_averaged_energy=self.use_token_averaged_energy,
            reduction_factor=self.reduction_factor,
        )

    def forward(
        self,
        input: np.array,
        feat_length: int = None,
        durations: np.array = None,
    ) -> np.array:

        # Domain-conversion: e.g. Stft: time -> time-freq
        x_stft = librosa.stft(
            input,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            pad_mode="reflect",
        )
        spc = np.abs(x_stft).T  # (#frames, #bins)
        input_power = spc**2

        # sum over frequency (B, N, F) -> (B, N)
        energy = np.sqrt(np.maximum(input_power.sum(axis=1), 1.0e-10))

        # (Optional): Adjust length to match with the mel-spectrogram
        if feat_length is not None:
            energy = self._adjust_num_frames(energy, feat_length)

        # (Optional): Average by duration to calculate token-wise energy
        if self.use_token_averaged_energy:
            durations = durations * self.reduction_factor
            energy = self._average_by_duration(energy, durations)

        # Return with the shape (B, T, 1)
        return energy

    def _average_by_duration(self, x: np.array, d: np.array) -> np.array:
        assert 0 <= len(x) - d.sum() < self.reduction_factor
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

    @staticmethod
    def _adjust_num_frames(x: np.array, num_frames: int) -> np.array:
        if num_frames > len(x):
            x = np.pad(x, (0, num_frames - len(x)))
        elif num_frames < len(x):
            x = x[:num_frames]
        return x
