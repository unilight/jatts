#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2025 Wen-Chin Huang (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Encodec model."""

from encodec import EncodecModel
from encodec.utils import convert_audio
from einops import rearrange
from pathlib import Path

import torch
import soundfile as sf


class EnCodec:
    def __init__(self, fs=24, bandwidth=6.0, device="cuda"):
        self.fs = fs
        self.device = device

        # Instantiate a pretrained EnCodec model
        if fs == 24:
            self.model = EncodecModel.encodec_model_24khz()
            self._rescale = False
        elif fs == 48:
            self.model = EncodecModel.encodec_model_48khz()
            self._rescale = True
        else:
            raise ValueError(f"Unsupported sampling rate: {fs}. Supported rates are 24kHz and 48kHz.")
        self.model.set_target_bandwidth(bandwidth)
        self.model.to(device)
        self.model.eval()

    @property
    def rescale(self):
        return self._rescale


    @torch.no_grad()
    def encode(self, wav, sr):
        """
        Encode the input waveform into quantized codes.
        
        Args:
            wav: (t)
            sr: int
        Returns:
            qnt: (b q t)

        """
        wav = torch.tensor(wav).unsqueeze(0)
        wav = convert_audio(wav, sr, self.model.sample_rate, self.model.channels)
        wav = wav.to(self.device) # [1/2, t]

        # unsqueeze the batch dimension due to encodec's signature requirement
        if wav.dim() == 2:
            wav = wav.unsqueeze(0) # [1, 1/2, t]
        elif wav.dim() == 1:
            wav = wav.unsqueeze(0).unsqueeze(0)  # [1, 1, t]

        encoded_frames = self.model.encode(wav)
        qnt = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # (b q t)
        return qnt

    def unload_model(self):
        return self.model.cache_clear()

    @torch.no_grad()
    def decode(self, codes):
        """
        Decode the quantized codes into waveform.
        
        Args:
            codes: (b q t)
        Returns:
            wav: (b 1/2 t)
            sr: int

        """

        assert codes.dim() == 3
        codes = codes.to(self.device)
        return self.model.decode([(codes, None)]), self.model.sample_rate

    def decode_to_file(self, resps, path: Path):
        assert resps.dim() == 2, f"Require shape (t q), but got {resps.shape}."
        resps = rearrange(resps, "t q -> 1 q t")
        wavs, sr = self.decode(resps)
        sf.write(str(path), wavs.cpu()[0, 0], sr)
